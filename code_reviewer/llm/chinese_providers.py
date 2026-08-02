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
"""Chinese LLM providers — DeepSeek, Qwen, Kimi, Baichuan, ZhipuAI (GLM).

All use OpenAI-compatible ``/chat/completions`` REST API.
Each provider is a lightweight ``OpenAIClient`` subclass that only overrides
``from_env()`` to set its default API key env var and base URL.
"""

from __future__ import annotations

import os

from code_reviewer.llm.openai_client import OpenAIClient
from code_reviewer.llm.provider_registry import register_provider

# ── Base URLs ───────────────────────────────────────────────────────────────

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
BAICHUAN_BASE_URL = "https://api.baichuan-ai.com/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


# ── Provider subclasses ─────────────────────────────────────────────────────


class DeepSeekClient(OpenAIClient):
    """DeepSeek provider (https://platform.deepseek.com).

    Registered as ``"deepseek"``.
    Reads ``DEEPSEEK_API_KEY`` (or ``OPENAI_API_KEY`` as fallback).
    """

    @classmethod
    def from_env(cls) -> DeepSeekClient:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY or OPENAI_API_KEY is required "
                "for the 'deepseek' provider."
            )
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)


class QwenClient(OpenAIClient):
    """Qwen (Alibaba Cloud) provider (https://help.aliyun.com/model-studio).

    Registered as ``"qwen"``.
    Reads ``QWEN_API_KEY`` (or ``DASHSCOPE_API_KEY`` or ``OPENAI_API_KEY``).
    """

    @classmethod
    def from_env(cls) -> QwenClient:
        api_key = (
            os.getenv("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "QWEN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY is required "
                "for the 'qwen' provider."
            )
        base_url = os.getenv("QWEN_BASE_URL", QWEN_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)


class KimiClient(OpenAIClient):
    """Kimi (Moonshot AI) provider (https://platform.moonshot.cn).

    Registered as ``"kimi"``.
    Reads ``KIMI_API_KEY`` (or ``MOONSHOT_API_KEY`` or ``OPENAI_API_KEY``).
    """

    @classmethod
    def from_env(cls) -> KimiClient:
        api_key = (
            os.getenv("KIMI_API_KEY")
            or os.getenv("MOONSHOT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "KIMI_API_KEY, MOONSHOT_API_KEY, or OPENAI_API_KEY is required "
                "for the 'kimi' provider."
            )
        base_url = os.getenv("KIMI_BASE_URL", KIMI_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)


class BaichuanClient(OpenAIClient):
    """Baichuan AI provider (https://platform.baichuan-ai.com).

    Registered as ``"baichuan"``.
    Reads ``BAICHUAN_API_KEY`` (or ``OPENAI_API_KEY``).
    """

    @classmethod
    def from_env(cls) -> BaichuanClient:
        api_key = os.getenv("BAICHUAN_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "BAICHUAN_API_KEY or OPENAI_API_KEY is required "
                "for the 'baichuan' provider."
            )
        base_url = os.getenv("BAICHUAN_BASE_URL", BAICHUAN_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)


class ZhipuClient(OpenAIClient):
    """Zhipu AI (GLM) provider (https://open.bigmodel.cn).

    Registered as ``"zhipu"``.
    Reads ``ZHIPU_API_KEY`` (or ``OPENAI_API_KEY``).
    """

    @classmethod
    def from_env(cls) -> ZhipuClient:
        api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "ZHIPU_API_KEY or OPENAI_API_KEY is required "
                "for the 'zhipu' provider."
            )
        base_url = os.getenv("ZHIPU_BASE_URL", ZHIPU_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)


# ── Register ────────────────────────────────────────────────────────────────

register_provider("deepseek", DeepSeekClient)
register_provider("qwen", QwenClient)
register_provider("kimi", KimiClient)
register_provider("baichuan", BaichuanClient)
register_provider("zhipu", ZhipuClient)
