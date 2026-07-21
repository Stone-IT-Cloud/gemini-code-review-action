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
"""Application configuration types and validation."""

from __future__ import annotations

import os
from typing import NotRequired, TypedDict


class AiReviewConfig(TypedDict):
    """Configuration for an AI review request (provider-agnostic)."""

    model: str
    diff: str
    extra_prompt: str
    prompt_chunk_size: int
    comments_text: str
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    max_output_tokens: NotRequired[int]
    review_memory_context: NotRequired[str]
    supplemental_context: NotRequired[str]


# ── Provider-specific API key env vars ──────────────────────────────────────

_PROVIDER_API_KEYS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
    "kimi": "KIMI_API_KEY",
    "baichuan": "BAICHUAN_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_CI_REQUIRED_VARS = [
    "GITHUB_TOKEN",
    "GITHUB_REPOSITORY",
    "GITHUB_PULL_REQUEST_NUMBER",
    "GIT_COMMIT_HASH",
]


def _detect_provider() -> str:
    """Read the active provider from env var, defaulting to ``"gemini"``."""
    return (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()


def check_required_env_vars() -> None:
    """Check required environment variables based on the active provider.

    Raises:
        ValueError: If any required variable is missing or empty.
    """
    provider = _detect_provider()

    required: list[str] = []

    # Provider API key
    api_key_var = _PROVIDER_API_KEYS.get(provider)
    if api_key_var:
        required.append(api_key_var)

    # CI-only vars (not needed in local mode)
    if os.getenv("LOCAL") is None:
        required.extend(_CI_REQUIRED_VARS)

    for var in required:
        value = os.getenv(var)
        if value is None or not value.strip():
            raise ValueError(
                f"{var} is not set or is empty. "
                f"This variable is required when LLM_PROVIDER={provider!r}."
            )
