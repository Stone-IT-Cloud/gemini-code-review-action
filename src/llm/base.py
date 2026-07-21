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
"""Abstract base class and types for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Standardised configuration for an LLM generate request."""

    model: str
    system_instruction: str | None = None
    temperature: float = 0.1
    top_p: float = 0.95
    max_output_tokens: int = 8192
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Standardised response from any LLM provider."""

    text: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)  # prompt_tokens, output_tokens, total_tokens


class LLMClient(ABC):
    """Abstract interface for an LLM provider client.

    All providers (Gemini, OpenAI, Anthropic, etc.) must implement
    ``generate_content`` and ``get_context_limit``.
    """

    @abstractmethod
    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Send a prompt to the model and return the response.

        Args:
            prompt: The user message / diff content.
            config: LLMConfig with model, temperature, top_p, etc.

        Returns:
            LLMResponse with generated text and token usage.
        """

    @abstractmethod
    def get_context_limit(self, model: str) -> int:
        """Return the model's input token limit.

        Falls back to a sensible default (e.g. 128K) on failure.
        """
