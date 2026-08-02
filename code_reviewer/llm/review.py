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
"""Provider-agnostic code review workflow — budget check, chunking, summarization.

This module contains the core review logic that works with any ``LLMClient``.
It handles:
- Budget check: send full diff in one call if it fits the model's context window.
- Chunked processing: split large diffs, review each chunk, then summarize.
- Retry with exponential backoff for transient API errors.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from loguru import logger

from src.prompts import get_review_prompt, get_summarize_prompt
from src.utils import calculate_char_budget, chunk_string, _safe_str

if TYPE_CHECKING:
    from src.llm.base import LLMClient, LLMConfig

DEFAULT_TOKEN_LIMIT = 1_000_000


def run_review(client: LLMClient, config: dict) -> tuple[list[str], str]:
    """Run a code review using any LLM provider.

    Handles budget check, chunking, and optional summarization.

    Args:
        client: An ``LLMClient`` instance (any provider).
        config: An ``AiReviewConfig``-compatible dict with keys:
            ``model``, ``diff``, ``extra_prompt``, ``prompt_chunk_size``,
            ``comments_text``, ``temperature``, ``top_p``, ``max_output_tokens``.

    Returns:
        ``(chunked_reviews, summarized_review)`` — the per-chunk reviews
        and a final summary (or the single review if no chunking was needed).
    """
    model = config["model"]
    diff = config["diff"]
    extra_prompt = config.get("extra_prompt", "")
    prompt_chunk_size = config.get("prompt_chunk_size", 500000)
    comments_text = config.get("comments_text", "")
    temperature = config.get("temperature", 0.1)
    top_p = config.get("top_p", 0.95)
    max_output_tokens = config.get("max_output_tokens", 8192)

    review_prompt = get_review_prompt(extra_prompt=extra_prompt)

    # ── Budget check: send full diff in a single call if it fits ──────
    token_limit = _get_context_limit(client, model)
    char_budget = calculate_char_budget(token_limit)
    if len(diff) <= char_budget:
        logger.info(f"Diff fits within budget ({len(diff)} <= {char_budget}), sending single request")
        from src.llm.base import LLMConfig  # pylint: disable=import-outside-toplevel

        llm_config = LLMConfig(
            model=model,
            system_instruction=review_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        resp = client.generate_content(diff, llm_config)
        if resp.text:
            return ([resp.text], resp.text)
        return ([], "")

    # ── Diff exceeds budget — chunked processing ──────────────────────
    chunked_diff_list = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)
    logger.info(f"Created {len(chunked_diff_list)} chunks from diff")

    chunked_reviews: list[str] = []
    for idx, chunked_diff in enumerate(chunked_diff_list, start=1):
        chunk_result = _review_chunk(
            client=client,
            model=model,
            review_prompt=review_prompt,
            idx=idx,
            total=len(chunked_diff_list),
            chunked_diff=chunked_diff,
            comments_text=comments_text,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        if chunk_result is not None:
            chunked_reviews.append(chunk_result)

    if len(chunked_reviews) <= 1:
        if len(chunked_reviews) == 1:
            return chunked_reviews, chunked_reviews[0]
        return (
            [],
            "Unable to generate review (LLM rate limit/quota exceeded). "
            "Please rerun later or reduce request volume.",
        )

    # ── Summarize multiple chunk reviews ──────────────────────────────
    summarized = _summarize_chunks(
        client=client,
        model=model,
        chunked_reviews=chunked_reviews,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )
    return chunked_reviews, summarized


def _get_context_limit(client: LLMClient, model: str) -> int:
    """Query the model's context limit with a fallback."""
    try:
        return client.get_context_limit(model)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning(f"get_context_limit failed for {model}, falling back to {DEFAULT_TOKEN_LIMIT}")
        return DEFAULT_TOKEN_LIMIT


def _build_prompt_parts(
    chunked_diff: str,
    idx: int,
    total: int,
    comments_text: str,
) -> str:
    """Build the prompt parts for a single chunk review call."""
    parts: list[str] = [
        f"[Pull request diff chunk {idx}/{total}]\n{chunked_diff}",
    ]
    if comments_text.strip():
        parts.append(
            "\n\n[Existing PR comments context]\n"
            "Take these into consideration when performing your review.\n\n"
            + comments_text
        )
    parts.append("\n\nNow provide your review according to the earlier instructions.")
    return "\n".join(parts)


def _review_chunk(
    client: LLMClient,
    model: str,
    review_prompt: str,
    idx: int,
    total: int,
    chunked_diff: str,
    comments_text: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> str | None:
    """Review a single diff chunk with retry logic."""
    from src.llm.base import LLMConfig  # pylint: disable=import-outside-toplevel

    max_attempts = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
    initial_wait = float(os.getenv("LLM_INITIAL_BACKOFF_SECONDS", "10"))
    max_wait = float(os.getenv("LLM_MAX_BACKOFF_SECONDS", "60"))

    for attempt in range(max_attempts):
        try:
            prompt = _build_prompt_parts(chunked_diff, idx, total, comments_text)
            chunk_config = LLMConfig(
                model=model,
                system_instruction=review_prompt,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
            resp = client.generate_content(prompt, chunk_config)
            if resp.text:
                logger.info(f"Chunk {idx}/{total} reviewed successfully")
                return resp.text
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Chunk {idx}/{total} attempt {attempt + 1}/{max_attempts} failed: {_safe_str(e)}")
            if attempt + 1 < max_attempts:
                wait_time = min(max_wait, initial_wait * (2**attempt))
                time.sleep(wait_time)

        logger.warning(f"Chunk {idx}/{total} attempt {attempt + 1}/{max_attempts} returned empty, retrying...")

    logger.error(f"Failed to get valid response for chunk {idx}/{total} after {max_attempts} attempts")
    return None


def _summarize_chunks(
    client: LLMClient,
    model: str,
    chunked_reviews: list[str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> str:
    """Summarize multiple chunk reviews into a single review."""
    from src.llm.base import LLMConfig  # pylint: disable=import-outside-toplevel

    summarize_prompt = get_summarize_prompt()
    chunked_reviews_join = "\n".join(chunked_reviews)
    max_attempts = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
    initial_wait = float(os.getenv("LLM_INITIAL_BACKOFF_SECONDS", "10"))
    max_wait = float(os.getenv("LLM_MAX_BACKOFF_SECONDS", "60"))

    for attempt in range(max_attempts):
        try:
            summary_config = LLMConfig(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
            resp = client.generate_content(summarize_prompt + "\n\n" + chunked_reviews_join, summary_config)
            if resp.text:
                logger.info("Summarization successful")
                return resp.text
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Summary attempt {attempt + 1}/{max_attempts} failed: {_safe_str(e)}")
            if attempt + 1 < max_attempts:
                time.sleep(min(max_wait, initial_wait * (2**attempt)))

    return "Unable to generate summary (LLM rate limit/quota exceeded)."
