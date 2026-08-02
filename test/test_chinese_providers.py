"""Tests for code_reviewer/llm/chinese_providers.py — DeepSeek, Qwen, Kimi, Baichuan, Zhipu."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from code_reviewer.llm.chinese_providers import (
    BaichuanClient,
    DeepSeekClient,
    KimiClient,
    QwenClient,
    ZhipuClient,
)


class TestChineseProvidersRegistration:
    """All Chinese providers must be registered."""

    def test_all_providers_registered(self):
        from code_reviewer.llm import list_providers
        providers = list_providers()
        for name in ("deepseek", "qwen", "kimi", "baichuan", "zhipu"):
            assert name in providers, f"{name} not registered"


class TestDeepSeekClient:
    def test_default_url(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            c = DeepSeekClient.from_env()
            assert c._base_url == "https://api.deepseek.com/v1"

    def test_fallsback_to_openai_api_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            c = DeepSeekClient.from_env()
            assert c._api_key == "sk-test"

    def test_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                DeepSeekClient.from_env()


class TestQwenClient:
    def test_default_url(self):
        with patch.dict("os.environ", {"QWEN_API_KEY": "sk-test"}):
            c = QwenClient.from_env()
            assert c._base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_fallsback_to_dashscope_key(self):
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "sk-dash"}):
            c = QwenClient.from_env()
            assert c._api_key == "sk-dash"

    def test_fallsback_to_openai_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai"}):
            c = QwenClient.from_env()
            assert c._api_key == "sk-openai"

    def test_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="QWEN_API_KEY"):
                QwenClient.from_env()


class TestKimiClient:
    def test_default_url(self):
        with patch.dict("os.environ", {"KIMI_API_KEY": "sk-test"}):
            c = KimiClient.from_env()
            assert c._base_url == "https://api.moonshot.cn/v1"

    def test_fallsback_to_moonshot_key(self):
        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "sk-moon"}):
            c = KimiClient.from_env()
            assert c._api_key == "sk-moon"

    def test_fallsback_to_openai_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai"}):
            c = KimiClient.from_env()
            assert c._api_key == "sk-openai"

    def test_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="KIMI_API_KEY"):
                KimiClient.from_env()


class TestBaichuanClient:
    def test_default_url(self):
        with patch.dict("os.environ", {"BAICHUAN_API_KEY": "sk-test"}):
            c = BaichuanClient.from_env()
            assert c._base_url == "https://api.baichuan-ai.com/v1"

    def test_fallsback_to_openai_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-oa"}):
            c = BaichuanClient.from_env()
            assert c._api_key == "sk-oa"

    def test_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="BAICHUAN_API_KEY"):
                BaichuanClient.from_env()


class TestZhipuClient:
    def test_default_url(self):
        with patch.dict("os.environ", {"ZHIPU_API_KEY": "sk-test"}):
            c = ZhipuClient.from_env()
            assert c._base_url == "https://open.bigmodel.cn/api/paas/v4"

    def test_fallsback_to_openai_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-oa"}):
            c = ZhipuClient.from_env()
            assert c._api_key == "sk-oa"

    def test_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
                ZhipuClient.from_env()
