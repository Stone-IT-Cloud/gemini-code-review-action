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
"""Tests for provider-agnostic retry and model-fallback behaviour."""

from __future__ import annotations

import random

import pytest
from google.genai import errors as genai_errors

from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.resilience import (
    DEFAULT_MODEL_CHAINS,
    ErrorKind,
    RetryPolicy,
    backoff_delay,
    call_with_fallback,
    classify,
    health_key,
    prioritize,
    resolve_chain,
)
from code_reviewer.llm.review import run_review
from code_reviewer.model_health import ModelHealthStore


def gemini_server_error(code: int = 503, status: str = "UNAVAILABLE") -> Exception:
    """Build a real ``google.genai`` 5xx error."""
    return genai_errors.ServerError(
        code,
        {
            "error": {
                "code": code,
                "status": status,
                "message": "This model is currently experiencing high demand.",
            }
        },
    )


class FakeHTTPError(Exception):
    """Stand-in for an OpenAI/Anthropic style error exposing ``status_code``."""

    def __init__(self, status_code: int, message: str | None = None) -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


def no_sleep(_seconds: float) -> None:
    """Skip real backoff waits in tests."""


class FakeClient(LLMClient):
    """Minimal ``LLMClient`` that scripts one behaviour per model name."""

    def __init__(self, behavior, context_limit: int = 1_000_000) -> None:
        self.behavior = behavior
        self.calls: list[str] = []
        self.context_limit = context_limit

    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        self.calls.append(config.model)
        result = self.behavior(config.model)
        if isinstance(result, Exception):
            raise result
        return LLMResponse(text=result)

    def get_context_limit(self, model: str) -> int:
        return self.context_limit


def _policy() -> RetryPolicy:
    return RetryPolicy(max_attempts=3, initial_wait=0.0, max_wait=0.0)


class TestClassify:
    """Provider error shapes map to the same policy."""

    def test_gemini_503_is_saturation(self):
        assert classify(gemini_server_error(503)) is ErrorKind.SATURATION

    def test_gemini_500_is_saturation(self):
        assert classify(gemini_server_error(500, "INTERNAL")) is ErrorKind.SATURATION

    def test_status_code_attribute_is_used_for_other_providers(self):
        assert classify(FakeHTTPError(503)) is ErrorKind.SATURATION
        assert classify(FakeHTTPError(429)) is ErrorKind.RATE_LIMIT
        assert classify(FakeHTTPError(400)) is ErrorKind.FATAL

    def test_gemini_429_is_rate_limit(self):
        error = genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
        assert classify(error) is ErrorKind.RATE_LIMIT

    def test_gemini_404_is_model_not_found(self):
        error = genai_errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND"}})
        assert classify(error) is ErrorKind.MODEL_NOT_FOUND

    def test_gemini_400_is_fatal(self):
        error = genai_errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
        assert classify(error) is ErrorKind.FATAL

    def test_timeout_is_transient(self):
        assert classify(TimeoutError("read timed out")) is ErrorKind.TRANSIENT

    def test_unknown_error_stays_retryable(self):
        assert classify(ValueError("weird provider payload")) is ErrorKind.TRANSIENT


class TestBackoff:
    """Exponential growth with equal jitter and a hard ceiling."""

    def test_delay_grows_with_attempts(self):
        # Deterministic seeds only: jitter quality is irrelevant in a test.
        low = backoff_delay(0, initial_wait=2.0, max_wait=20.0, rng=random.Random(0))  # noqa: S311
        high = backoff_delay(3, initial_wait=2.0, max_wait=20.0, rng=random.Random(0))  # noqa: S311
        assert 1.0 <= low <= 2.0
        assert 8.0 <= high <= 16.0

    def test_delay_never_exceeds_max_wait(self):
        for attempt in range(10):
            delay = backoff_delay(
                attempt,
                initial_wait=2.0,
                max_wait=20.0,
                rng=random.Random(attempt),  # noqa: S311
            )
            assert delay <= 20.0


class TestChainResolution:
    """Chains are provider-aware and de-duplicated."""

    def test_gemini_appends_default_fallbacks(self):
        assert resolve_chain("gemini-2.5-pro", None, None, "gemini") == [
            "gemini-2.5-pro",
            *DEFAULT_MODEL_CHAINS["gemini"],
        ]

    def test_other_providers_get_no_gemini_fallbacks(self):
        assert resolve_chain("gpt-4o", None, None, "openai") == ["gpt-4o"]

    def test_explicit_fallbacks_override_defaults(self):
        assert resolve_chain("a", "b, c ,", None, "gemini") == ["a", "b", "c"]

    def test_full_override_wins(self):
        assert resolve_chain("ignored", "also-ignored", "x,y", "gemini") == ["x", "y"]

    def test_duplicates_are_removed(self):
        chain = resolve_chain("gemini-3.8-flash", "gemini-3.8-flash,gemini-3.6-flash", None, "gemini")
        assert chain == ["gemini-3.8-flash", "gemini-3.6-flash"]

    def test_prioritize_moves_model_to_front(self):
        assert prioritize(["a", "b", "c"], "c") == ["c", "a", "b"]

    def test_health_key_is_namespaced(self):
        assert health_key("gemini", "gemini-3.8-flash") == "gemini:gemini-3.8-flash"


class TestCallWithFallback:
    """Retry budget, fallback order and flagging semantics."""

    def test_saturation_retries_then_switches_and_flags(self):
        calls: list[str] = []
        health = ModelHealthStore(path=None, ttl_seconds=3600, clock=lambda: 1000.0)

        def call(model: str) -> str | None:
            calls.append(model)
            if model == "primary":
                raise gemini_server_error()
            return "review"

        result = call_with_fallback(
            ["primary", "fallback"],
            call,
            health=health,
            policy=_policy(),
            sleep=no_sleep,
        )

        assert calls == ["primary", "primary", "primary", "fallback"]
        assert result.text == "review"
        assert result.model == "fallback"
        assert health.is_blocked("gemini:primary")

    def test_rate_limit_retries_but_never_flags(self):
        health = ModelHealthStore(path=None, ttl_seconds=3600, clock=lambda: 1000.0)

        def call(model: str) -> str | None:
            raise FakeHTTPError(429)

        result = call_with_fallback(
            ["primary", "fallback"],
            call,
            health=health,
            policy=_policy(),
            sleep=no_sleep,
        )

        assert result.text is None
        assert health.blocked_models() == set()
        assert result.attempts == 6

    def test_fatal_error_stops_the_chain(self):
        calls: list[str] = []
        health = ModelHealthStore(path=None, ttl_seconds=3600, clock=lambda: 1000.0)

        def call(model: str) -> str | None:
            calls.append(model)
            raise FakeHTTPError(401)

        result = call_with_fallback(["primary", "fallback"], call, health=health, policy=_policy(), sleep=no_sleep)

        assert calls == ["primary"]
        assert result.text is None

    def test_flagged_models_are_skipped(self):
        health = ModelHealthStore(path=None, ttl_seconds=3600, clock=lambda: 1000.0)
        health.block("gemini:primary", code=503)
        calls: list[str] = []

        def call(model: str) -> str | None:
            calls.append(model)
            return "review"

        result = call_with_fallback(["primary", "fallback"], call, health=health, policy=_policy(), sleep=no_sleep)

        assert calls == ["fallback"]
        assert result.model == "fallback"

    def test_all_flagged_falls_back_to_full_chain(self):
        health = ModelHealthStore(path=None, ttl_seconds=3600, clock=lambda: 1000.0)
        health.block("gemini:primary", code=503)
        health.block("gemini:fallback", code=503)

        result = call_with_fallback(
            ["primary", "fallback"],
            lambda model: "review",
            health=health,
            policy=_policy(),
            sleep=no_sleep,
        )

        assert result.text == "review"

    def test_daily_quota_failure_fails_fast(self):
        health = ModelHealthStore(path=None, ttl_seconds=3600, clock=lambda: 1000.0)

        def call(model: str) -> str | None:
            raise FakeHTTPError(429, "requests per day exceeded")

        with pytest.raises(FakeHTTPError):
            call_with_fallback(
                ["primary"],
                call,
                health=health,
                policy=_policy(),
                sleep=no_sleep,
            )


class TestRunReviewFallback:
    """End-to-end wiring through ``run_review``."""

    def _config(self, **overrides) -> dict:
        config = {
            "diff": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
            "extra_prompt": "",
            "model": "primary",
            "fallback_models": "fallback",
            "prompt_chunk_size": 10000,
            "comments_text": "",
            "temperature": 0.1,
            "top_p": 0.95,
            "max_output_tokens": 256,
        }
        config.update(overrides)
        return config

    def test_saturated_model_falls_back_and_is_flagged(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_ATTEMPTS", "2")
        monkeypatch.setenv("LLM_INITIAL_BACKOFF_SECONDS", "0")
        monkeypatch.setenv("LLM_MAX_BACKOFF_SECONDS", "0")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        client = FakeClient(lambda model: gemini_server_error() if model == "primary" else "review")
        health = ModelHealthStore(path=None, ttl_seconds=3600)

        chunks, summary = run_review(client, self._config(), health=health)

        assert client.calls == ["primary", "primary", "fallback"]
        assert chunks == ["review"]
        assert summary == "review"
        assert health.is_blocked("gemini:primary")

    def test_chunked_run_stays_on_the_model_that_answered(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_ATTEMPTS", "2")
        monkeypatch.setenv("LLM_INITIAL_BACKOFF_SECONDS", "0")
        monkeypatch.setenv("LLM_MAX_BACKOFF_SECONDS", "0")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        # A tiny context limit forces the chunked path instead of the single call.
        client = FakeClient(
            lambda model: gemini_server_error() if model == "primary" else "review",
            context_limit=10,
        )
        health = ModelHealthStore(path=None, ttl_seconds=3600)

        chunks, _ = run_review(
            client,
            self._config(diff="x" * 120, prompt_chunk_size=40),
            health=health,
        )

        assert len(chunks) == 3
        assert client.calls[:4] == ["primary", "primary", "fallback", "fallback"]
