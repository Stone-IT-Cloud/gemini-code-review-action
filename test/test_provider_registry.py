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
"""Tests for code_reviewer/llm/provider_registry.py — provider registration and factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import get_llm_client, list_providers, register_provider


# ── Mock provider for testing ───────────────────────────────────────────────


class _MockProvider(LLMClient):
    """Minimal mock provider for registry tests."""

    @classmethod
    def from_env(cls) -> _MockProvider:
        return cls()

    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        return LLMResponse(text="mock response")

    def get_context_limit(self, model: str) -> int:
        return 128_000


# ── Tests ───────────────────────────────────────────────────────────────────


class TestProviderRegistry:
    """Tests for register_provider and list_providers."""

    def test_register_and_list(self):
        register_provider("mock_test", _MockProvider)
        providers = list_providers()
        assert "mock_test" in providers
        assert "gemini" in providers  # registered by gemini_client.py import

    def test_get_llm_client_returns_mock(self):
        register_provider("mock_test2", _MockProvider)
        client = get_llm_client("mock_test2")
        assert isinstance(client, _MockProvider)

    def test_get_llm_client_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_client("nonexistent_provider")

    @patch.dict("os.environ", {"LLM_PROVIDER": "mock_test3"}, clear=True)
    def test_get_llm_client_from_env(self):
        register_provider("mock_test3", _MockProvider)
        client = get_llm_client()  # reads LLM_PROVIDER env
        assert isinstance(client, _MockProvider)

    def test_get_llm_client_defaults_to_gemini(self):
        """When LLM_PROVIDER is unset, should default to 'gemini'."""
        import importlib

        # Re-import with clean env to test the default
        with patch.dict("os.environ", {}, clear=True):
            importlib.reload(__import__("code_reviewer.llm.provider_registry"))
            from code_reviewer.llm.provider_registry import get_llm_client as glic

            with pytest.raises(ValueError, match=r".*gemini.*"):  # no GEMINI_API_KEY
                glic()
