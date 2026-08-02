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
"""OpenAI-compatible provider — supports OpenAI, OpenRouter, DeepSeek, Kimi, Qwen.

Uses the OpenAI-compatible ``/chat/completions`` REST API via ``requests``.
Configure via environment variables:
  - ``OPENAI_API_KEY`` or ``LLM_API_KEY`` (required)
  - ``OPENAI_BASE_URL`` (optional, defaults to ``https://api.openai.com/v1``)

Registered as ``"openai"`` (defaults to OpenAI) and ``"openrouter"`` (defaults to
OpenRouter). Both use the same ``OpenAIClient`` class under the hood.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from loguru import logger

from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import register_provider

DEFAULT_TOKEN_LIMIT = 128_000
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Known context limits for common models (tokens)
_KNOWN_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "o1": 200_000,
    "o3-mini": 200_000,
    # OpenRouter-specific / common models
    "openai/gpt-4o": 128_000,
    "openai/o1": 200_000,
    "openai/o3-mini": 200_000,
    "anthropic/claude-sonnet-4": 200_000,
    "anthropic/claude-3.5-sonnet": 200_000,
    "google/gemini-2.5-flash": 1_000_000,
    "google/gemini-2.5-pro": 1_000_000,
    "meta-llama/llama-3.1-405b": 128_000,
    "deepseek/deepseek-chat": 1_000_000,
    "deepseek/deepseek-r1": 128_000,
    "qwen/qwen-max": 32_000,
    "qwen/qwen-plus": 131_072,
    "qwen/qwen-turbo": 1_000_000,
    "mistral/mistral-large": 128_000,
    "cohere/command-r": 128_000,
}


class OpenAIClient(LLMClient):
    """LLMClient for OpenAI-compatible APIs (OpenRouter, OpenAI, DeepSeek, etc.)."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    @classmethod
    def from_env(cls) -> OpenAIClient:
        """Create an OpenAIClient from environment variables.

        Reads ``OPENAI_API_KEY`` (or ``LLM_API_KEY`` as fallback)
        and ``OPENAI_BASE_URL`` (defaults to OpenAI API).
        """
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY or LLM_API_KEY is required for the 'openai' provider. "
                "Set it as an environment variable."
            )
        base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)

    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Send a prompt to the model via the chat completions API.

        Args:
            prompt: The user message content.
            config: ``LLMConfig`` with model, system_instruction, temperature, etc.

        Returns:
            ``LLMResponse`` with generated text and token usage.

        Raises:
            requests.RequestException: On HTTP errors or network failures.
        """
        payload = self._build_payload(prompt, config)
        resp = self._session.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

    def get_context_limit(self, model: str) -> int:
        """Return the model's context limit from a known lookup table.

        Falls back to ``DEFAULT_TOKEN_LIMIT`` (128K) for unknown models.
        """
        return _KNOWN_CONTEXT_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)

    def _build_payload(self, prompt: str, config: LLMConfig) -> dict[str, Any]:
        """Build the request payload for the chat completions API."""
        messages: list[dict[str, str]] = []
        if config.system_instruction:
            messages.append({"role": "system", "content": config.system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_output_tokens,
        }

        # Request JSON mode for structured output
        payload["response_format"] = {"type": "json_object"}

        return payload


class OpenRouterClient(OpenAIClient):
    """OpenAIClient subclass that defaults to OpenRouter's base URL.

    Registered as ``"openrouter"`` provider.
    """

    @classmethod
    def from_env(cls) -> OpenRouterClient:
        """Create an OpenRouter client from environment variables.

        Reads ``OPENAI_API_KEY`` (or ``LLM_API_KEY``) and ``OPENAI_BASE_URL``
        (defaults to ``https://openrouter.ai/api/v1``).
        """
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY or LLM_API_KEY is required for the 'openrouter' provider. "
                "Set it as an environment variable."
            )
        base_url = os.getenv("OPENAI_BASE_URL", OPENROUTER_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)


register_provider("openai", OpenAIClient)
register_provider("openrouter", OpenRouterClient)
