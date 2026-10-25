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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from code_reviewer.llm.resilience import (
    CallResult,
    RetryPolicy,
    call_with_fallback,
    prioritize,
    resolve_chain,
)
from code_reviewer.model_health import ModelHealthStore
from code_reviewer.prompts import get_review_prompt, get_summarize_prompt
from code_reviewer.utils import calculate_char_budget, chunk_string

if TYPE_CHECKING:
    from code_reviewer.llm.base import LLMClient

DEFAULT_TOKEN_LIMIT = 1_000_000


@dataclass
class _CallContext:
    """Retry/fallback state shared by every request of one review run."""

    provider: str
    chain: list[str]
    health: ModelHealthStore
    policy: RetryPolicy
    errors: list[str] = field(default_factory=list)

    def call(
        self,
        client: LLMClient,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
    ) -> CallResult:
        """Send one prompt through the fallback chain, keeping the winner sticky."""
        result = call_with_fallback(
            self.chain,
            lambda model: _generate(
                client,
                model,
                prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            ),
            health=self.health,
            provider=self.provider,
            policy=self.policy,
        )
        self.errors.extend(result.errors)
        if result.ok:
            self.chain = prioritize(self.chain, result.model)
        return result


def _generate(
    client: LLMClient,
    model: str,
    prompt: str,
    *,
    system_instruction: str | None,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> str | None:
    """One request to one model, in the provider's own terms."""
    from code_reviewer.llm.base import LLMConfig  # pylint: disable=import-outside-toplevel

    response = client.generate_content(
        prompt,
        LLMConfig(
            model=model,
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text


def _build_context(config: Mapping[str, Any], health: ModelHealthStore | None = None) -> _CallContext:
    """Resolve the provider, model chain, health store and retry policy."""
    provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    chain = resolve_chain(
        config.get("model"),
        config.get("fallback_models"),
        os.getenv("LLM_MODEL_CHAIN"),
        provider,
    )
    logger.info(f"Model chain for provider '{provider}': {' -> '.join(chain)}")
    return _CallContext(
        provider=provider,
        chain=chain,
        health=health or ModelHealthStore.from_env(),
        policy=RetryPolicy.from_env(),
    )


def run_review(
    client: LLMClient,
    config: Mapping[str, Any],
    *,
    health: ModelHealthStore | None = None,
) -> tuple[list[str], str]:
    """Run a code review using any LLM provider.

    Handles budget check, chunking, and optional summarization.

    Args:
        client: An ``LLMClient`` instance (any provider).
        config: An ``AiReviewConfig``-compatible dict with keys:
            ``model``, ``diff``, ``extra_prompt``, ``prompt_chunk_size``,
            ``comments_text``, ``temperature``, ``top_p``, ``max_output_tokens``,
            ``fallback_models``.
        health: Health store to use. When omitted, one is built from the
            environment; callers that persist it across runs pass their own.

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
    ctx = _build_context(config, health)

    # ── Budget check: send full diff in a single call if it fits ──────
    token_limit = _get_context_limit(client, model)
    char_budget = calculate_char_budget(token_limit)
    if len(diff) <= char_budget:
        logger.info(f"Diff fits within budget ({len(diff)} <= {char_budget}), sending single request")
        result = ctx.call(
            client,
            diff,
            system_instruction=review_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        if result.ok:
            text = result.text or ""
            return ([text], text)
        return ([], "")

    # ── Diff exceeds budget — chunked processing ──────────────────────
    chunked_diff_list = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)
    logger.info(f"Created {len(chunked_diff_list)} chunks from diff")

    chunked_reviews: list[str] = []
    for idx, chunked_diff in enumerate(chunked_diff_list, start=1):
        chunk_result = _review_chunk(
            ctx,
            client=client,
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
            "Unable to generate review (LLM unavailable, rate limited or quota exceeded "
            f"after trying: {', '.join(ctx.chain)}). "
            "Please rerun later or reduce request volume.",
        )

    # ── Summarize multiple chunk reviews ──────────────────────────────
    summarized = _summarize_chunks(
        ctx,
        client=client,
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
            "Take these into consideration when performing your review.\n\n" + comments_text
        )
    parts.append("\n\nNow provide your review according to the earlier instructions.")
    return "\n".join(parts)


def _review_chunk(
    ctx: _CallContext,
    client: LLMClient,
    review_prompt: str,
    idx: int,
    total: int,
    chunked_diff: str,
    comments_text: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> str | None:
    """Review a single diff chunk through the retry/fallback policy."""
    prompt = _build_prompt_parts(chunked_diff, idx, total, comments_text)
    result = ctx.call(
        client,
        prompt,
        system_instruction=review_prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )
    if result.ok:
        logger.info(f"Chunk {idx}/{total} reviewed successfully with {result.model}")
        return result.text
    logger.error(f"Failed to get valid response for chunk {idx}/{total}")
    return None


def _summarize_chunks(
    ctx: _CallContext,
    client: LLMClient,
    chunked_reviews: list[str],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> str:
    """Summarize multiple chunk reviews into a single review."""
    summarize_prompt = get_summarize_prompt()
    chunked_reviews_join = "\n".join(chunked_reviews)
    result = ctx.call(
        client,
        summarize_prompt + "\n\n" + chunked_reviews_join,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )
    if result.ok and result.text:
        logger.info(f"Summarization successful with {result.model}")
        return result.text
    return "Unable to generate summary (LLM unavailable, rate limited or quota exceeded)."
