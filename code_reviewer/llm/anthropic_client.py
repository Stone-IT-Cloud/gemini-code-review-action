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
"""Anthropic provider — Claude models via the Messages API.

Uses the Anthropic Messages API directly via ``requests``.
Configure via environment variables:
  - ``ANTHROPIC_API_KEY`` (required)
  - ``ANTHROPIC_BASE_URL`` (optional, defaults to ``https://api.anthropic.com/v1``)
  - ``ANTHROPIC_VERSION`` (optional, defaults to ``2023-06-01``)
"""

from __future__ import annotations

import os
from typing import Any

import requests
from loguru import logger

from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import register_provider

DEFAULT_TOKEN_LIMIT = 200_000
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

# Known context limits for Anthropic models
_KNOWN_CONTEXT_LIMITS: dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-sonnet-v2": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-sonnet-20240229": 200_000,
}


class AnthropicClient(LLMClient):
    """LLMClient for Anthropic Claude models via the Messages API."""

    def __init__(self, api_key: str, base_url: str, api_version: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._session = requests.Session()
        self._session.headers.update({
            "x-api-key": api_key,
            "anthropic-version": api_version,
            "Content-Type": "application/json",
        })

    @classmethod
    def from_env(cls) -> AnthropicClient:
        """Create an AnthropicClient from environment variables.

        Reads ``ANTHROPIC_API_KEY`` (required), ``ANTHROPIC_BASE_URL``
        (defaults to Anthropic API), and ``ANTHROPIC_VERSION``.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for the 'anthropic' provider. "
                "Set it as an environment variable."
            )
        base_url = os.getenv("ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL)
        api_version = os.getenv("ANTHROPIC_VERSION", ANTHROPIC_VERSION)
        return cls(api_key=api_key, base_url=base_url, api_version=api_version)

    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Send a prompt to Claude via the Messages API.

        Args:
            prompt: The user message content.
            config: ``LLMConfig`` with model, system_instruction, temperature, etc.

        Returns:
            ``LLMResponse`` with generated text and token usage.
        """
        payload = self._build_payload(prompt, config)
        resp = self._session.post(
            f"{self._base_url}/messages",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text from response content blocks
        text: str | None = None
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block.get("text")
                break

        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": (usage.get("input_tokens", 0) or 0)
                + (usage.get("output_tokens", 0) or 0),
            },
        )

    def get_context_limit(self, model: str) -> int:
        """Return the model's context limit from a known lookup table.

        Falls back to ``DEFAULT_TOKEN_LIMIT`` (200K) for unknown models.
        """
        return _KNOWN_CONTEXT_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)

    def _build_payload(self, prompt: str, config: LLMConfig) -> dict[str, Any]:
        """Build the request payload for the Anthropic Messages API."""
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_output_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.system_instruction:
            payload["system"] = config.system_instruction

        return payload


register_provider("anthropic", AnthropicClient)
