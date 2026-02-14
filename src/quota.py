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
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from google.api_core import exceptions as google_exceptions
from loguru import logger

from src.utils import _get_usage_metadata, _safe_str


class NoQuotaAvailableError(RuntimeError):
    """Raised when we detect there is no quota available (fail-fast)."""


def _looks_like_daily_quota_exhausted(message: str) -> bool:
    """Heuristic: detect daily quota exhaustion from error text."""
    msg = (message or "").lower()
    return any(
        needle in msg
        for needle in (
            "per day",
            "daily",
            "requests per day",
            "rpd",
            "day quota",
        )
    )


def _sleep_with_jitter(seconds: float) -> None:
    """Sleep with a small random jitter to avoid synchronized retries."""
    # Jitter in [0, 1.0) seconds, capped so it never dominates the delay.
    jitter = min(1.0, random.random())
    time.sleep(max(0.0, seconds + jitter))


def _handle_api_error(
    error,
    *,
    attempt: int,
    max_attempts: int,
    initial_wait: float,
    max_wait: float,
    fail_fast_on_no_quota: bool,
) -> bool:
    """Handle API errors with exponential backoff + jitter.

    Returns True if the caller should retry (and we already waited), else False.
    """
    is_last_attempt = (attempt + 1) >= max_attempts

    if isinstance(error, google_exceptions.ResourceExhausted):
        # Rate limit / quota exceeded.
        err_text = _safe_str(error)
        logger.warning(f"Rate limit / quota exceeded details: {err_text}")

        if fail_fast_on_no_quota and _looks_like_daily_quota_exhausted(err_text):
            logger.error("Daily quota exhausted and fail-fast is enabled; aborting without retries.")
            raise error

        if is_last_attempt:
            logger.error("Rate limit hit. No retries remaining.")
            return False
        wait_time = min(max_wait, initial_wait * (2**attempt))
        logger.warning(f"Rate limit hit. Waiting {wait_time:.0f}s before retry...")
        _sleep_with_jitter(wait_time)
        return True

    if isinstance(error, google_exceptions.DeadlineExceeded):
        logger.error("API request timed out")
        return not is_last_attempt

    if isinstance(error, google_exceptions.InvalidArgument):
        logger.error(f"Invalid API request: {str(error)}")
        return False

    # Default: do not spin forever on unexpected errors.
    logger.error(f"Unexpected API error: {str(error)}")
    return False


@dataclass
class QuotaTracker:
    """Track run-local usage and provide best-effort remaining-quota hints.

    Note: The Gemini API (AI Studio) does not reliably expose "remaining quota"
    counters to clients. This tracker logs:
    - actual per-request usage (when ``usage_metadata`` is provided)
    - run-estimated remaining RPM/TPM/RPD from *completed* requests if you
      provide limits via env vars (in-flight/pending requests are not counted)
    """

    window_seconds: int = 60
    request_timestamps: deque = field(default_factory=deque)
    token_events: deque = field(default_factory=deque)  # (timestamp, total_tokens)
    requests_total: int = 0
    tokens_total: int = 0
    last_pruned_at: float = 0.0
    prune_interval_seconds: float = 1.0  # Only prune if this much time has elapsed

    quota_rpm: Optional[int] = None
    quota_tpm: Optional[int] = None
    quota_rpd: Optional[int] = None

    @staticmethod
    def from_env() -> "QuotaTracker":
        def _parse_int(name: str) -> Optional[int]:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return None
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer value for environment variable {name!r}: {raw!r}. "
                    "Please set it to a valid integer (e.g., '60') or leave it unset."
                ) from exc
            if value < 0:
                raise ValueError(
                    f"Invalid non-negative value for environment variable {name!r}: {value!r}. "
                    "Please set it to a non-negative integer or leave it unset."
                )
            return value

        return QuotaTracker(
            quota_rpm=_parse_int("GEMINI_QUOTA_RPM"),
            quota_tpm=_parse_int("GEMINI_QUOTA_TPM"),
            quota_rpd=_parse_int("GEMINI_QUOTA_RPD"),
        )

    def _prune(self, now: float) -> None:
        # Optimization: only prune if enough time has passed since last prune
        if now - self.last_pruned_at < self.prune_interval_seconds:
            return
        cutoff = now - self.window_seconds
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        while self.token_events and self.token_events[0][0] < cutoff:
            self.token_events.popleft()
        self.last_pruned_at = now

    def note_request(self, now: float) -> None:
        self.requests_total += 1
        self.request_timestamps.append(now)
        self._prune(now)

    def note_tokens(self, now: float, total_tokens: int) -> None:
        self.tokens_total += int(total_tokens)
        self.token_events.append((now, int(total_tokens)))
        self._prune(now)

    def recent_rpm(self, now: float) -> int:
        self._prune(now)
        return len(self.request_timestamps)

    def recent_tpm(self, now: float) -> int:
        self._prune(now)
        return sum(t for _, t in self.token_events)

    def remaining_estimate(self, now: float) -> dict:
        """Return run-estimated remaining quota (if limits are configured)."""
        rem: dict = {}
        if self.quota_rpm is not None:
            rem["rpm_remaining"] = max(0, self.quota_rpm - self.recent_rpm(now))
            rem["rpm_limit"] = self.quota_rpm
        if self.quota_tpm is not None:
            rem["tpm_remaining"] = max(0, self.quota_tpm - self.recent_tpm(now))
            rem["tpm_limit"] = self.quota_tpm
        if self.quota_rpd is not None:
            rem["rpd_remaining"] = max(0, self.quota_rpd - self.requests_total)
            rem["rpd_limit"] = self.quota_rpd
        return rem

    def has_all_quotas_set_to_zero(self) -> bool:
        """Check if all quota limits are explicitly configured and set to zero."""
        return (
            self.quota_rpm is not None
            and self.quota_tpm is not None
            and self.quota_rpd is not None
            and self.quota_rpm == 0
            and self.quota_tpm == 0
            and self.quota_rpd == 0
        )

    def log_after_response(self, response, label: str) -> None:
        now = time.time()
        usage = _get_usage_metadata(response)
        total_tokens = usage.get("total_tokens")
        if total_tokens is not None:
            self.note_tokens(now, total_tokens)

        remaining = self.remaining_estimate(now)
        usage_bits = []
        if usage:
            prompt_tokens = usage.get("prompt_tokens", "?")
            output_tokens = usage.get("output_tokens", "?")
            total_tokens_val = usage.get("total_tokens", "?")
            usage_bits.append(
                "usage_tokens="
                f"prompt={prompt_tokens},"
                f"output={output_tokens},"
                f"total={total_tokens_val}"
            )
        if remaining:
            usage_bits.append(
                "run_estimated_remaining="
                + ",".join(f"{k}={v}" for k, v in remaining.items())
            )
        if not usage_bits:
            usage_bits.append("usage_metadata=<not provided by API>")
        joined_usage_bits = ", ".join(usage_bits)
        logger.info(f"{label} {joined_usage_bits}")
