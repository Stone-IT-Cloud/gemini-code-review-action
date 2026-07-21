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
"""Backward-compatible re-exports for the Gemini client.

New code should import from ``src.llm`` and use the provider abstraction.
This file remains for backward compatibility with any external code
that imports directly from ``src.gemini_client``.
"""

# ruff: noqa: F401

from typing import Any

from src.llm.gemini_client import (
    DEFAULT_TOKEN_LIMIT,
    GeminiClient,
    NoQuotaAvailableError,
    _handle_api_error,
    _looks_like_daily_quota_exhausted,
)

# Wrap module-level functions for backward compat
# Old signature: get_review(client, config) → client was a genai.Client
# New: GeminiClient wraps it.


def get_review(client: Any, config: dict) -> tuple[list[str], str]:
    """Backward-compat wrapper for get_review().

    Accepts a raw genai.Client and an AiReviewConfig dict,
    creates a GeminiClient around it, and delegates to
    ``GeminiClient.get_review()``.
    """
    gc = GeminiClient(client)
    return gc.get_review(config)


def get_model_context_limit(client: Any, model_name: str) -> int:
    """Backward-compat wrapper for get_model_context_limit()."""
    gc = GeminiClient(client)
    return gc.get_context_limit(model_name)
