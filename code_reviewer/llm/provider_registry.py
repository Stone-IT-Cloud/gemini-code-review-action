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
"""Provider registration and factory — maps provider names to client classes."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.llm.base import LLMClient

_PROVIDERS: dict[str, type[LLMClient]] = {}


def register_provider(name: str, client_class: type[LLMClient]) -> None:
    """Register an LLM provider class by name.

    Args:
        name: Short provider name (e.g. ``"gemini"``, ``"openai"``).
        client_class: The class implementing ``LLMClient``.
    """
    _PROVIDERS[name] = client_class
    logger.debug(f"Registered LLM provider: {name}")


def list_providers() -> list[str]:
    """Return all registered provider names."""
    return list(_PROVIDERS.keys())


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory: instantiate the right LLM client based on provider name.

    Args:
        provider: Explicit provider name. If ``None``, reads the
            ``LLM_PROVIDER`` environment variable. Defaults to ``"gemini"``.

    Returns:
        An initialised ``LLMClient`` instance.

    Raises:
        ValueError: If the provider is unknown or not configured.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Available providers: {', '.join(list_providers())}. "
            "Set LLM_PROVIDER env var or pass --provider CLI arg."
        )

    client_class = _PROVIDERS[provider]
    logger.info(f"Using LLM provider: {provider} ({client_class.__name__})")
    return client_class.from_env()
