#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Retry and model-fallback policy shared by every LLM provider.

``503 UNAVAILABLE`` means the serving pool of *that model* is saturated. Retrying
the same model forever only burns wall-clock time, so the policy is:

1. Retry the current model a bounded number of times with exponential backoff
   and jitter.
2. On persistent saturation, flag the model for an hour and move to the next
   model in the chain.
3. Once a model answers, keep using it for the rest of the run.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger

from code_reviewer.model_health import ModelHealthStore
from code_reviewer.utils import _safe_str

# Fallback chains are provider-specific: a Gemini model name means nothing to
# OpenAI, so only Gemini ships a default and every other provider needs the
# configured model to name its own fallbacks.
DEFAULT_MODEL_CHAINS: dict[str, list[str]] = {
    "gemini": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"],
}

_SATURATION_STATUSES = {"UNAVAILABLE", "INTERNAL", "SERVICE_UNAVAILABLE", "UNKNOWN"}
_TRANSIENT_STATUSES = {"DEADLINE_EXCEEDED", "ABORTED"}


class ErrorKind(StrEnum):
    """How an LLM API failure should be handled."""

    SATURATION = "saturation"
    RATE_LIMIT = "rate_limit"
    MODEL_NOT_FOUND = "model_not_found"
    TRANSIENT = "transient"
    FATAL = "fatal"


@dataclass
class RetryPolicy:
    """Bounded retry budget for a single model."""

    max_attempts: int = 3
    initial_wait: float = 2.0
    max_wait: float = 20.0
    fail_fast_on_daily_quota: bool = True

    @classmethod
    def from_env(cls) -> RetryPolicy:
        """Build the policy from the ``LLM_*`` environment variables."""
        return cls(
            max_attempts=_env_int("LLM_MAX_ATTEMPTS", 3, minimum=1),
            initial_wait=_env_float("LLM_INITIAL_BACKOFF_SECONDS", 2.0),
            max_wait=_env_float("LLM_MAX_BACKOFF_SECONDS", 20.0),
            fail_fast_on_daily_quota=os.getenv("LLM_FAIL_FAST_ON_DAILY_QUOTA", "1") == "1",
        )


@dataclass
class CallResult:
    """Outcome of running a callable across the model chain."""

    text: str | None = None
    model: str = ""
    attempts: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether a model produced text."""
        return bool(self.text)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, int(str(raw).strip()))
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}; using {default}")
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, float(str(raw).strip()))
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}; using {default}")
        return default


def _status_code(error: BaseException) -> int | None:
    """Best-effort HTTP status code from a provider exception."""
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _status_name(error: BaseException) -> str:
    status = getattr(error, "status", None)
    return status.upper() if isinstance(status, str) else ""


def _is_transport_error(error: BaseException) -> bool:
    if isinstance(error, TimeoutError | ConnectionError):
        return True
    name = type(error).__name__.lower()
    return "timeout" in name or "connect" in name


def classify(error: BaseException) -> ErrorKind:
    """Map a provider exception to the handling policy.

    Providers expose the HTTP status differently (``google.genai`` uses ``code``,
    OpenAI and Anthropic use ``status_code``), so the status is read defensively.
    Unknown failures stay retryable but never flag a model: only an explicit 5xx
    is evidence of provider-side saturation.
    """
    code = _status_code(error)
    if code is not None:
        if code >= 500:
            return ErrorKind.SATURATION
        if code == 429:
            return ErrorKind.RATE_LIMIT
        if code == 404:
            return ErrorKind.MODEL_NOT_FOUND
        if 400 <= code < 500:
            return ErrorKind.FATAL
    if _is_transport_error(error):
        return ErrorKind.TRANSIENT
    status = _status_name(error)
    if status in _SATURATION_STATUSES:
        return ErrorKind.SATURATION
    if status in _TRANSIENT_STATUSES:
        return ErrorKind.TRANSIENT
    return ErrorKind.TRANSIENT


def backoff_delay(
    attempt: int,
    initial_wait: float,
    max_wait: float,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with equal jitter.

    Half of the exponential delay is fixed so retries never hammer the API
    immediately; the other half is random so parallel jobs desynchronize.
    """
    if initial_wait <= 0 or max_wait <= 0:
        return 0.0
    generator = rng or random
    base = min(max_wait, initial_wait * (2 ** max(0, attempt)))
    half = base / 2.0
    # Jitter does not need cryptographic randomness (bandit B311).
    return half + generator.uniform(0.0, half)  # nosec B311


def _split_models(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def resolve_chain(
    primary: str | None,
    fallbacks: str | None = None,
    override: str | None = None,
    provider: str = "gemini",
) -> list[str]:
    """Build the ordered, de-duplicated list of models to try."""
    override_models = _split_models(override)
    if override_models:
        return list(dict.fromkeys(override_models))

    chain = _split_models(primary)
    explicit_fallbacks = _split_models(fallbacks)
    if explicit_fallbacks:
        chain.extend(explicit_fallbacks)
    elif provider == "gemini":
        chain.extend(DEFAULT_MODEL_CHAINS["gemini"])
    return list(dict.fromkeys(chain))


def prioritize(chain: Sequence[str], model: str) -> list[str]:
    """Move a model that just worked to the front of the chain."""
    rest = [candidate for candidate in chain if candidate != model]
    return [model, *rest]


def health_key(provider: str, model: str) -> str:
    """Key under which a model's health is tracked."""
    return f"{provider}:{model}"


def call_with_fallback(
    chain: Sequence[str],
    call: Callable[[str], str | None],
    *,
    health: ModelHealthStore,
    provider: str = "gemini",
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> CallResult:
    """Run *call* against each usable model until one answers.

    Args:
        chain: Ordered model names, best first.
        call: ``call(model) -> text | None``; may raise provider exceptions.
        health: Store used to skip and flag saturated models.
        provider: Provider name, used to namespace the health keys.
        policy: Retry budget per model.
        sleep: Injectable sleep, used by tests.
        rng: Injectable randomness for jitter, used by tests.

    Returns:
        The last successful result, or an empty result when every model failed.
    """
    policy = policy or RetryPolicy()
    candidates = _usable_candidates(chain, health, provider)

    result = CallResult()
    for model in candidates:
        text, kind, error, used = _attempt_model(model, call, policy=policy, sleep=sleep, rng=rng, errors=result.errors)
        result.attempts += used
        if text:
            result.text = text
            result.model = model
            return result
        if kind is ErrorKind.SATURATION:
            health.block(
                health_key(provider, model),
                code=_status_code(error) if error is not None else None,
                reason=_safe_str(error) if error is not None else "",
            )
        if kind is ErrorKind.FATAL:
            logger.error(f"Aborting model fallback: {model} failed with a non model-specific error")
            break

    return result


def _usable_candidates(chain: Sequence[str], health: ModelHealthStore, provider: str) -> list[str]:
    """Drop flagged models, falling back to the full chain when all are flagged."""
    blocked = health.blocked_models()
    candidates = [model for model in chain if health_key(provider, model) not in blocked]
    if blocked:
        blocked_label = ", ".join(sorted(blocked))
        logger.info(f"Skipping flagged models: {blocked_label}")
    if not candidates:
        logger.warning("Every candidate model is flagged as saturated; trying the full chain anyway")
        return list(chain)
    return candidates


def _attempt_model(
    model: str,
    call: Callable[[str], str | None],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None],
    rng: random.Random | None,
    errors: list[str],
) -> tuple[str | None, ErrorKind | None, BaseException | None, int]:
    """Retry one model up to the policy budget; return text, last kind, error and attempts used."""
    last_kind: ErrorKind | None = None
    last_error: BaseException | None = None
    for attempt in range(policy.max_attempts):
        text, kind, error = _try_once(model, call, policy, errors, attempt)
        if text:
            logger.info(f"Model {model} answered after {attempt + 1} attempt(s)")
            return text, None, None, attempt + 1
        last_kind, last_error = kind, error
        if kind in (ErrorKind.FATAL, ErrorKind.MODEL_NOT_FOUND):
            return None, kind, error, attempt + 1
        if attempt + 1 < policy.max_attempts:
            _sleep_before_retry(model, attempt, policy, sleep, rng)
    return None, last_kind, last_error, policy.max_attempts


def _try_once(
    model: str,
    call: Callable[[str], str | None],
    policy: RetryPolicy,
    errors: list[str],
    attempt: int,
) -> tuple[str | None, ErrorKind | None, BaseException | None]:
    """Run a single attempt and classify its failure, if any."""
    try:
        text = call(model)
    except Exception as error:  # pylint: disable=broad-exception-caught
        kind = classify(error)
        errors.append(f"{model}: {_safe_str(error)}")
        logger.warning(
            f"Model {model} attempt {attempt + 1}/{policy.max_attempts} failed " f"({kind.value}): {_safe_str(error)}"
        )
        if (
            kind is ErrorKind.RATE_LIMIT
            and policy.fail_fast_on_daily_quota
            and _looks_like_daily_quota_exhausted(_safe_str(error))
        ):
            logger.error("Daily quota exhausted and fail-fast is enabled.")
            raise
        return None, kind, error
    if text:
        return text, None, None
    errors.append(f"{model}: empty response")
    logger.warning(f"Model {model} attempt {attempt + 1}/{policy.max_attempts} returned an empty response")
    return None, ErrorKind.TRANSIENT, None


def _sleep_before_retry(
    model: str,
    attempt: int,
    policy: RetryPolicy,
    sleep: Callable[[float], None],
    rng: random.Random | None,
) -> None:
    """Wait before the next attempt, unless the computed delay is zero."""
    delay = backoff_delay(attempt, policy.initial_wait, policy.max_wait, rng)
    if delay > 0:
        logger.warning(f"Retrying {model} in {delay:.1f}s")
        sleep(delay)


def _looks_like_daily_quota_exhausted(message: str) -> bool:
    """Detect daily quota exhaustion from error text."""
    msg = (message or "").lower()
    return any(
        needle in msg
        for needle in (
            "per day",
            "daily",
            "requests per day",
            "rpd",
            "day quota",
            "insufficient_quota",
        )
    )
