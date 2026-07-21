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
"""Gemini provider — implements LLMClient via google.genai SDK."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from google.genai import errors as gemini_errors
from google.genai import types as gemini_types
from loguru import logger

from src.llm.base import LLMClient, LLMConfig, LLMResponse
from src.llm.provider_registry import register_provider
from src.prompts import get_review_prompt, get_summarize_prompt
from src.quota import QuotaTracker
from src.utils import _extract_model_text, _get_usage_metadata, _safe_str, calculate_char_budget, chunk_string

DEFAULT_TOKEN_LIMIT = 1_000_000


# ── Gemini-specific error handling (moved from src/quota.py) ────────────────


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


def _handle_api_error(
    error,
    *,
    attempt: int,
    max_attempts: int,
    initial_wait: float,
    max_wait: float,
    fail_fast_on_no_quota: bool,
) -> bool:
    """Handle Gemini API errors with exponential backoff + jitter.

    Returns True if the caller should retry (and we already waited), else False.
    """
    is_last_attempt = (attempt + 1) >= max_attempts

    if not isinstance(error, gemini_errors.APIError):
        logger.error(f"Unexpected non-API error: {_safe_str(error)}")
        return False

    code = getattr(error, "code", None)
    err_text = _safe_str(error)

    # 429 — ResourceExhausted (rate limit / quota exceeded)
    if code == 429:
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

    # 504 — DeadlineExceeded (timeout)
    if code == 504:
        logger.error("API request timed out")
        return not is_last_attempt

    # 400 — InvalidArgument (bad request, do not retry)
    if code == 400:
        logger.error(f"Invalid API request: {err_text}")
        return False

    # Any other APIError — do not retry on unknown codes
    logger.error(f"Unexpected API error (code={code}): {err_text}")
    return False


def _sleep_with_jitter(seconds: float) -> None:
    """Sleep with a small random jitter to avoid synchronized retries."""
    import secrets

    jitter = min(1.0, secrets.randbelow(1_000_000_000) / 1_000_000_000.0)
    time.sleep(max(0.0, seconds + jitter))


# ── Gemini client class ─────────────────────────────────────────────────────


class GeminiClient(LLMClient):
    """LLMClient implementation for Google Gemini (google.genai SDK)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> GeminiClient:
        """Create a GeminiClient from environment variables.

        Requires ``GEMINI_API_KEY`` to be set.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for the 'gemini' LLM provider. "
                "Set it as an environment variable or choose a different provider."
            )
        # Delayed import: only needed when Gemini is actually used
        from google import genai  # pylint: disable=import-outside-toplevel

        return cls(genai.Client(api_key=api_key))

    @property
    def client(self) -> Any:
        """Expose the underlying ``genai.Client`` for advanced usage (learner, etc.)."""
        return self._client

    # ── LLMClient interface ──────────────────────────────────────────────

    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Single generate_content call (non-chunked).

        Args:
            prompt: The user message / diff content.
            config: Standardised ``LLMConfig``.

        Returns:
            ``LLMResponse`` with generated text and token usage.
        """
        gconfig = gemini_types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_output_tokens,
            response_mime_type="application/json",
            system_instruction=config.system_instruction,
        )
        response = self._client.models.generate_content(
            model=config.model,
            contents=prompt,
            config=gconfig,
        )
        text = _extract_model_text(response)
        usage = _get_usage_metadata(response)
        return LLMResponse(text=text, usage=usage if usage else {})

    def get_context_limit(self, model: str) -> int:
        """Query the model's ``input_token_limit`` via ``client.models.get()``.

        Falls back to ``DEFAULT_TOKEN_LIMIT`` (1_000_000) on any failure.
        """
        try:
            model_info = self._client.models.get(model=model)
            limit = getattr(model_info, "input_token_limit", None)
            if limit is not None and limit > 0:
                return int(limit)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning(f"get_model failed for {model}, falling back to {DEFAULT_TOKEN_LIMIT}")
        return DEFAULT_TOKEN_LIMIT

    # ── High-level review workflow (backward-compatible entry point) ──────

    def get_review(self, config: dict) -> tuple[list[str], str]:
        """Run a full code review using Gemini — budget check, single or chunked.

        This is the legacy entry point used by ``main.py``. Accepts an
        ``AiReviewConfig``-compatible dict.

        Returns:
            ``(chunked_reviews, summarized_review)``.
        """
        model = config["model"]
        diff = config["diff"]
        extra_prompt = config["extra_prompt"]
        prompt_chunk_size = config["prompt_chunk_size"]
        comments_text = config.get("comments_text", "")
        temperature = config.get("temperature", 1)
        top_p = config.get("top_p", 0.95)
        max_output_tokens = config.get("max_output_tokens", 8192)
        review_prompt = get_review_prompt(extra_prompt=extra_prompt)
        llm_config = LLMConfig(
            model=model,
            system_instruction=review_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

        # Budget check: send full diff in a single call if it fits within the model's context.
        token_limit = self.get_context_limit(model)
        char_budget = calculate_char_budget(token_limit)
        if len(diff) <= char_budget:
            logger.info(f"Diff fits within budget ({len(diff)} <= {char_budget}), sending single request")
            resp = self.generate_content(diff, llm_config)
            if resp.text:
                return ([resp.text], resp.text)
            return ([], "")

        # Diff exceeds budget — fall through to chunked processing.
        chunked_diff_list = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)
        logger.info(f"Created {len(chunked_diff_list)} chunks from diff")

        return self._process_chunks(
            model=model,
            chunked_diff_list=chunked_diff_list,
            llm_config=llm_config,
            comments_text=comments_text,
        )

    # ── Chunked processing (migrated from old gemini_client.py) ───────────

    def _process_chunks(
        self,
        model: str,
        chunked_diff_list: list[str],
        llm_config: LLMConfig,
        comments_text: str,
    ) -> tuple[list[str], str]:
        """Process diff chunks with retry logic, then summarize if needed."""
        max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "6"))
        initial_wait = float(os.getenv("GEMINI_INITIAL_BACKOFF_SECONDS", "15"))
        max_wait = float(os.getenv("GEMINI_MAX_BACKOFF_SECONDS", "240"))
        min_request_interval = float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "6"))
        fail_fast_on_no_quota = os.getenv("GEMINI_FAIL_FAST_ON_NO_QUOTA", "1") == "1"

        tracker = QuotaTracker.from_env(prefix="GEMINI")
        if tracker.has_all_quotas_set_to_zero():
            raise NoQuotaAvailableError("Configured quota is 0 (GEMINI_QUOTA_RPM/TPM/RPD). Refusing to start.")

        chunked_reviews: list[str] = []
        last_request_at = 0.0
        for idx, chunked_diff in enumerate(chunked_diff_list, start=1):
            since_last = time.time() - last_request_at
            if since_last < min_request_interval:
                time.sleep(min_request_interval - since_last)

            chunk_result = self._process_single_chunk(
                model=model,
                idx=idx,
                total=len(chunked_diff_list),
                chunked_diff=chunked_diff,
                comments_text=comments_text,
                max_attempts=max_attempts,
                initial_wait=initial_wait,
                max_wait=max_wait,
                min_request_interval=min_request_interval,
                fail_fast_on_no_quota=fail_fast_on_no_quota,
                tracker=tracker,
            )
            if chunk_result is not None:
                chunked_reviews.append(chunk_result)
            last_request_at = time.time()
            time.sleep(min_request_interval)

        if len(chunked_reviews) <= 1:
            return self._single_chunk_result(chunked_reviews)

        return self._summarize_chunks(
            model=model,
            chunked_reviews=chunked_reviews,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            max_output_tokens=llm_config.max_output_tokens,
            max_attempts=max_attempts,
            initial_wait=initial_wait,
            max_wait=max_wait,
            min_request_interval=min_request_interval,
            fail_fast_on_no_quota=fail_fast_on_no_quota,
            tracker=tracker,
        )

    def _process_single_chunk(
        self,
        model: str,
        idx: int,
        total: int,
        chunked_diff: str,
        comments_text: str,
        max_attempts: int,
        initial_wait: float,
        max_wait: float,
        min_request_interval: float,
        fail_fast_on_no_quota: bool,
        tracker: QuotaTracker,
    ) -> str | None:
        """Process a single diff chunk with retry logic. Returns the review text or None."""
        for attempt in range(max_attempts):
            try:
                prompt_parts: list[str] = [
                    f"[Pull request diff chunk {idx}/{total}]\n{chunked_diff}",
                ]
                if comments_text.strip():
                    prompt_parts.append(
                        "\n\n[Existing PR comments context]\n"
                        "Take these into consideration when performing your review.\n\n" + comments_text
                    )
                # Note: memory_context and supplemental_context are injected
                # by main.py into the AiReviewConfig, but the chunk path
                # reads those from the generate_content system_instruction.
                prompt_parts.append("\n\nNow provide your review according to the earlier instructions.")

                response = self._client.models.generate_content(
                    model=model,
                    contents="\n".join(prompt_parts),
                )
                now = time.time()
                tracker.note_request(now)
                review_result = _extract_model_text(response)
                tracker.log_after_response(response, label=f"Gemini call success (review chunk {idx}/{total})", prefix="GEMINI")
                if review_result:
                    return review_result
            except gemini_errors.APIError as e:
                logger.error(f"Chunk {idx}/{total} attempt {attempt + 1}/{max_attempts} failed: {_safe_str(e)}")
                should_retry = _handle_api_error(
                    e,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    initial_wait=initial_wait,
                    max_wait=max_wait,
                    fail_fast_on_no_quota=fail_fast_on_no_quota,
                )
                if should_retry:
                    continue
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Chunk {idx}/{total} attempt {attempt + 1}/{max_attempts} unexpected error: {_safe_str(e)}")
                if attempt + 1 < max_attempts:
                    wait_time = min(max_wait, initial_wait * (2**attempt))
                    _sleep_with_jitter(wait_time)
                    continue
            logger.error(f"Failed to get model response for chunk {idx}/{total}")
            return None
        return None

    @staticmethod
    def _single_chunk_result(chunked_reviews: list[str]) -> tuple:
        """Return result when there's 0 or 1 chunk reviews."""
        if len(chunked_reviews) == 1:
            return chunked_reviews, chunked_reviews[0]
        return (
            [],
            "Unable to generate review (Gemini rate limit/quota exceeded). "
            "Please rerun later or reduce request volume.",
        )

    def _summarize_chunks(
        self,
        model: str,
        chunked_reviews: list[str],
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        max_attempts: int,
        initial_wait: float,
        max_wait: float,
        min_request_interval: float,
        fail_fast_on_no_quota: bool,
        tracker: QuotaTracker,
    ) -> tuple:
        """Summarize multiple chunk reviews into a single review."""
        summarize_prompt = get_summarize_prompt()
        chunked_reviews_join = "\n".join(chunked_reviews)
        summarized_review: str | None = None
        last_request_at = time.time()

        for attempt in range(max_attempts):
            try:
                since_last = time.time() - last_request_at
                if since_last < min_request_interval:
                    time.sleep(min_request_interval - since_last)

                summarize_config = gemini_types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                )
                response = self._client.models.generate_content(
                    model=model,
                    contents=summarize_prompt + "\n\n" + chunked_reviews_join,
                    config=summarize_config,
                )
                now = time.time()
                tracker.note_request(now)
                summarized_review = _extract_model_text(response)
                tracker.log_after_response(response, label="Gemini call success (summary)", prefix="GEMINI")
                if summarized_review:
                    break
            except gemini_errors.APIError as e:
                logger.error(f"Summary attempt {attempt + 1}/{max_attempts} failed: {_safe_str(e)}")
                should_retry = _handle_api_error(
                    e,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    initial_wait=initial_wait,
                    max_wait=max_wait,
                    fail_fast_on_no_quota=fail_fast_on_no_quota,
                )
                if not should_retry:
                    break

        if not summarized_review:
            summarized_review = "Unable to generate summary (Gemini API rate limit/quota exceeded)."
        logger.debug(f"Response AI: {summarized_review}")
        return chunked_reviews, summarized_review


register_provider("gemini", GeminiClient)
