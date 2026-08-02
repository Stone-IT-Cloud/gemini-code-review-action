"""Tests for code_reviewer/llm/anthropic_client.py — Anthropic Claude provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from code_reviewer.llm.anthropic_client import AnthropicClient
from code_reviewer.llm.base import LLMConfig


class TestAnthropicRegistration:
    def test_anthropic_is_registered(self):
        from code_reviewer.llm import list_providers
        assert "anthropic" in list_providers()


class TestAnthropicClientGenerateContent:
    def test_generate_content_success(self):
        client = AnthropicClient(api_key="sk-ant-test", base_url="https://fake.api.test", api_version="2023-06-01")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Great review!"}],
            "usage": {"input_tokens": 50, "output_tokens": 10},
        }
        mock_resp.raise_for_status.return_value = None
        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            config = LLMConfig(model="claude-sonnet-4", temperature=0.1)
            response = client.generate_content("Review this", config)
        assert response.text == "Great review!"
        assert response.usage["prompt_tokens"] == 50
        assert response.usage["total_tokens"] == 60
        mock_post.assert_called_once()

    def test_generate_content_includes_system(self):
        client = AnthropicClient(api_key="sk-ant-test", base_url="https://fake.api.test", api_version="2023-06-01")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"type": "text", "text": "ok"}], "usage": {}}
        mock_resp.raise_for_status.return_value = None
        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            config = LLMConfig(model="claude-sonnet-4", system_instruction="You are a reviewer.", temperature=0.1)
            client.generate_content("Review this", config)
        payload = mock_post.call_args[1]["json"]
        assert payload["system"] == "You are a reviewer."
        assert payload["messages"] == [{"role": "user", "content": "Review this"}]

    def test_http_error_raises(self):
        client = AnthropicClient(api_key="bad-key", base_url="https://fake.api.test", api_version="2023-06-01")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        with patch.object(client._session, "post", return_value=mock_resp):
            config = LLMConfig(model="claude-sonnet-4")
            with pytest.raises(requests.exceptions.HTTPError):
                client.generate_content("test", config)

    def test_empty_content(self):
        client = AnthropicClient(api_key="sk-ant-test", base_url="https://fake.api.test", api_version="2023-06-01")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [], "usage": {}}
        mock_resp.raise_for_status.return_value = None
        with patch.object(client._session, "post", return_value=mock_resp):
            config = LLMConfig(model="claude-sonnet-4")
            response = client.generate_content("test", config)
        assert response.text is None


class TestAnthropicContextLimit:
    def test_known_model(self):
        client = AnthropicClient(api_key="test", base_url="https://fake.api.test", api_version="2023-06-01")
        assert client.get_context_limit("claude-sonnet-4") == 200_000
        assert client.get_context_limit("claude-3-5-sonnet-20241022") == 200_000

    def test_unknown_model_falls_back(self):
        client = AnthropicClient(api_key="test", base_url="https://fake.api.test", api_version="2023-06-01")
        assert client.get_context_limit("unknown-model") == 200_000


class TestAnthropicFactory:
    def test_from_env_with_api_key(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            client = AnthropicClient.from_env()
            assert client._api_key == "sk-ant-test123"
            assert client._base_url == "https://api.anthropic.com/v1"

    def test_from_env_with_custom_url(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "ANTHROPIC_BASE_URL": "https://custom.anthropic.com/v1"}):
            client = AnthropicClient.from_env()
            assert client._base_url == "https://custom.anthropic.com/v1"

    def test_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicClient.from_env()
