"""Tests for src/llm/openai_client.py — OpenAI-compatible provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.llm.base import LLMConfig, LLMResponse
from src.llm.openai_client import OpenAIClient


class TestOpenAIClientRegistration:
    """Tests that the OpenAI provider is properly registered."""

    def test_openai_client_is_registered(self):
        """OpenAI provider debe estar registrado en el registry."""
        from src.llm import list_providers

        assert "openai" in list_providers()


class TestOpenAIClientGenerateContent:
    """Tests for OpenAIClient.generate_content()."""

    def test_generate_content_success(self):
        """POST exitoso retorna LLMResponse con text."""
        client = OpenAIClient(api_key="test-key", base_url="https://fake.api.test")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Great review!"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }
        mock_resp.raise_for_status.return_value = None

        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            config = LLMConfig(model="openai/gpt-4o", temperature=0.1)
            response = client.generate_content("Review this code", config)

        assert response.text == "Great review!"
        assert response.usage["prompt_tokens"] == 50
        assert response.usage["total_tokens"] == 60
        mock_post.assert_called_once()

    def test_generate_content_includes_system_prompt(self):
        """system_instruction se envía como mensaje system."""
        client = OpenAIClient(api_key="test-key", base_url="https://fake.api.test")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        mock_resp.raise_for_status.return_value = None

        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            config = LLMConfig(
                model="openai/gpt-4o",
                system_instruction="You are a code reviewer.",
                temperature=0.1,
            )
            client.generate_content("Review this", config)

        # Verify the system message was included in the payload
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        messages = payload["messages"]
        assert messages[0] == {"role": "system", "content": "You are a code reviewer."}
        assert messages[1] == {"role": "user", "content": "Review this"}

    def test_generate_content_http_error_raises(self):
        """HTTP 401/403 lanza excepción."""
        client = OpenAIClient(api_key="bad-key", base_url="https://fake.api.test")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")

        with patch.object(client._session, "post", return_value=mock_resp):
            config = LLMConfig(model="openai/gpt-4o")
            with pytest.raises(requests.exceptions.HTTPError):
                client.generate_content("test", config)

    def test_generate_content_empty_choices(self):
        """Si choices está vacío, text es None."""
        client = OpenAIClient(api_key="test-key", base_url="https://fake.api.test")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [], "usage": {}}
        mock_resp.raise_for_status.return_value = None

        with patch.object(client._session, "post", return_value=mock_resp):
            config = LLMConfig(model="openai/gpt-4o")
            with pytest.raises(IndexError):
                client.generate_content("test", config)


class TestOpenAIClientContextLimit:
    """Tests for OpenAIClient.get_context_limit()."""

    def test_known_model_returns_limit(self):
        """Modelo conocido retorna su límite específico."""
        client = OpenAIClient(api_key="test", base_url="https://fake.api.test")
        assert client.get_context_limit("gpt-4o") == 128_000
        assert client.get_context_limit("openai/o1") == 200_000
        assert client.get_context_limit("deepseek/deepseek-chat") == 1_000_000

    def test_unknown_model_falls_back(self):
        """Modelo desconocido retorna DEFAULT_TOKEN_LIMIT."""
        client = OpenAIClient(api_key="test", base_url="https://fake.api.test")
        assert client.get_context_limit("nonexistent-model-v99") == 128_000


class TestOpenAIClientFactory:
    """Tests for OpenAIClient.from_env()."""

    def test_from_env_with_api_key(self):
        """OPENAI_API_KEY crea cliente con default OpenAI."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            client = OpenAIClient.from_env()
            assert client._api_key == "sk-test123"
            assert client._base_url == "https://api.openai.com/v1"

    def test_from_env_with_llm_api_key_fallback(self):
        """LLM_API_KEY funciona como fallback si OPENAI_API_KEY no está."""
        with patch.dict("os.environ", {"LLM_API_KEY": "sk-fallback"}):
            client = OpenAIClient.from_env()
            assert client._api_key == "sk-fallback"

    def test_from_env_with_custom_base_url(self):
        """OPENAI_BASE_URL personalizada se usa correctamente."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "https://api.deepseek.com/v1"}):
            client = OpenAIClient.from_env()
            assert client._base_url == "https://api.deepseek.com/v1"

    def test_from_env_missing_key_raises(self):
        """Sin API key lanza ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIClient.from_env()


class TestOpenRouterClient:
    """Tests for OpenRouterClient (openrouter provider alias)."""

    def test_openrouter_is_registered(self):
        """openrouter provider debe estar registrado."""
        from src.llm import list_providers
        assert "openrouter" in list_providers()

    def test_openrouter_defaults_to_openrouter_url(self):
        """OpenRouterClient usa https://openrouter.ai/api/v1 por defecto."""
        from src.llm.openai_client import OpenRouterClient
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            client = OpenRouterClient.from_env()
            assert client._base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_respects_custom_base_url(self):
        """OpenRouterClient respeta OPENAI_BASE_URL si se setea."""
        from src.llm.openai_client import OpenRouterClient
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "https://custom.api/v1"}):
            client = OpenRouterClient.from_env()
            assert client._base_url == "https://custom.api/v1"

    def test_openrouter_missing_key_raises(self):
        """Sin API key lanza ValueError."""
        from src.llm.openai_client import OpenRouterClient
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenRouterClient.from_env()
