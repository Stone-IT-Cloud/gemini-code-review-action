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
"""LLM provider abstraction layer — multi-model support for code review."""

from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import get_llm_client, list_providers, register_provider

# Import provider modules to trigger register_provider() calls at module level.
# Each provider module calls register_provider() at import time.
from code_reviewer.llm import gemini_client  # noqa: F401
from code_reviewer.llm import openai_client  # noqa: F401
from code_reviewer.llm import chinese_providers  # noqa: F401
from code_reviewer.llm import anthropic_client  # noqa: F401

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "get_llm_client",
    "list_providers",
    "register_provider",
]
