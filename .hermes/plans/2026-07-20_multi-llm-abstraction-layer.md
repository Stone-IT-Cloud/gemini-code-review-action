# Multi-LLM Abstraction Layer Implementation Plan

> **Goal:** Refactor `gemini-code-review-action` from a Gemini-only action to a multi-provider LLM code review action, supporting OpenAI, Anthropic, DeepSeek, Kimi, Qwen, OpenRouter, and any OpenAI-compatible API.

**Architecture:** Introduce a lightweight provider abstraction (`code_reviewer/llm/`) with a common `LLMClient` interface. Each provider implements it. The action selects the provider at runtime via an env var (`LLM_PROVIDER`). The existing Gemini code is refactored into a `GeminiClient` class, and new `OpenAIClient` (covers OpenAI, DeepSeek, Kimi, Qwen, OpenRouter via compatibility) and `AnthropicClient` are added.

**Tech Stack:** Python 3.12, `requests` for OpenAI-compatible APIs, `anthropic` SDK, `google-genai` SDK (kept for Gemini), `pytest`, `ruff`

---

## Task 1: Create provider base class and registry

**Objective:** Define the abstract interface that all LLM providers must implement.

**Files:**
- Create: `code_reviewer/llm/__init__.py`
- Create: `code_reviewer/llm/base.py`
- Create: `code_reviewer/llm/provider_registry.py`

**Step 1: Write `code_reviewer/llm/base.py`**

```python
"""Abstract base class and types for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMConfig:
    """Standardised config passed to every generate_content call."""
    model: str
    system_instruction: str | None = None
    temperature: float = 0.1
    top_p: float = 0.95
    max_output_tokens: int = 8192
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""
    text: str | None = None
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, output_tokens, total_tokens


class LLMClient(ABC):
    """Abstract interface for an LLM provider client."""

    @abstractmethod
    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Send a prompt to the model and return the response."""
        ...

    @abstractmethod
    def get_context_limit(self, model: str) -> int:
        """Return the model's input token limit. Fallback to a sensible default."""
        ...
```

**Step 2: Write `code_reviewer/llm/provider_registry.py`**

```python
"""Provider registration and factory."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from code_reviewer.llm.base import LLMClient


_PROVIDERS: dict[str, type[LLMClient]] = {}


def register_provider(name: str, client_class: type[LLMClient]) -> None:
    """Register an LLM provider class by name."""
    _PROVIDERS[name] = client_class
    logger.debug(f"Registered LLM provider: {name}")


def list_providers() -> list[str]:
    """Return all registered provider names."""
    return list(_PROVIDERS.keys())


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Factory: instantiate the right LLM client based on provider name or env var.

    Args:
        provider: Explicit provider name. If None, reads ``LLM_PROVIDER`` env var.

    Returns:
        An initialised LLMClient instance.

    Raises:
        ValueError: If the provider is unknown or not set.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Available: {', '.join(list_providers())}. "
            "Set LLM_PROVIDER env var or pass --provider CLI arg."
        )

    client_class = _PROVIDERS[provider]
    logger.info(f"Using LLM provider: {provider} ({client_class.__name__})")
    return client_class.from_env()
```

**Step 3: Write `code_reviewer/llm/__init__.py`**

```python
"""LLM provider abstraction layer — multi-model support for code review."""
from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import get_llm_client, list_providers, register_provider

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "get_llm_client",
    "list_providers",
    "register_provider",
]
```

**Step 4: Verify imports work**

```bash
cd /root/work/gemini-code-review-action
python -c "from code_reviewer.llm import LLMClient, LLMConfig; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add code_reviewer/llm/
git commit -m "feat: add LLM provider base class, types, and registry"
```

---

## Task 2: Refactor Gemini into a provider class

**Objective:** Move the existing Gemini logic from `code_reviewer/gemini_client.py` into `code_reviewer/llm/gemini_client.py` implementing `LLMClient`.

**Files:**
- Create: `code_reviewer/llm/gemini_client.py`
- Modify: `code_reviewer/gemini_client.py` → thin re-export shim for backward compat
- Modify: `code_reviewer/llm/__init__.py` → register provider

**Step 1: Write `code_reviewer/llm/gemini_client.py`**

The class wraps the existing logic but implements `LLMClient`. Key points:
- `from_env()` reads `GEMINI_API_KEY` and instantiates `genai.Client`
- `generate_content()` handles both single and chunked flows internally
- `get_context_limit()` uses the existing `client.models.get()` approach
- `_process_chunks()`, `_summarize_chunks()`, `_process_single_chunk()` become private methods
- Rate limiting / quota logic stays (but needs the `google.genai.errors` import)

```python
"""Gemini provider — implements LLMClient via google.genai SDK."""

from __future__ import annotations

import os
import time
from typing import Any

from google.genai import errors as gemini_errors, types as gemini_types
from loguru import logger

from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import register_provider
from code_reviewer.prompts import get_review_prompt, get_summarize_prompt
from code_reviewer.quota import NoQuotaAvailableError, QuotaTracker, _handle_api_error
from code_reviewer.utils import _extract_model_text, _safe_str, calculate_char_budget, chunk_string

DEFAULT_TOKEN_LIMIT = 1_000_000


class GeminiClient(LLMClient):
    """LLMClient implementation for Google Gemini (google.genai SDK)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> GeminiClient:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini provider")
        from google import genai
        return cls(genai.Client(api_key=api_key))

    @property
    def client(self) -> Any:
        return self._client

    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Single generate_content call (non-chunked)."""
        gconfig = gemini_types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_output_tokens,
            response_mime_type="application/json",
            system_instruction=config.system_instruction,
        )
        response = self._client.models.generate_content(
            model=config.model,
            contents=prompt,
            config=gconfig,
        )
        text = _extract_model_text(response)
        usage = _extract_usage(response)
        return LLMResponse(text=text, usage=usage)

    def get_context_limit(self, model: str) -> int:
        try:
            model_info = self._client.models.get(model=model)
            limit = getattr(model_info, "input_token_limit", None)
            if limit is not None and limit > 0:
                return int(limit)
        except Exception:
            logger.warning(f"get_model failed for {model}, falling back to {DEFAULT_TOKEN_LIMIT}")
        return DEFAULT_TOKEN_LIMIT

    # ── Existing chunked/summarize logic moved here ──
    def get_review(self, config: dict) -> tuple[list[str], str]:
        """Legacy entry point — get_review interface for backward compatibility.

        This is the main review flow (budget check → single call or chunked).
        """
        model = config["model"]
        diff = config["diff"]
        extra_prompt = config["extra_prompt"]
        prompt_chunk_size = config["prompt_chunk_size"]
        comments_text = config.get("comments_text", "")
        temperature = config.get("temperature", 1)
        top_p = config.get("top_p", 0.95)
        max_output_tokens = config.get("max_output_tokens", 8192)
        review_prompt = get_review_prompt(extra_prompt=extra_prompt)
        llm_config = LLMConfig(
            model=model,
            system_instruction=review_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

        token_limit = self.get_context_limit(model)
        char_budget = calculate_char_budget(token_limit)
        if len(diff) <= char_budget:
            logger.info(f"Diff fits within budget ({len(diff)} <= {char_budget}), sending single request")
            resp = self.generate_content(diff, llm_config)
            if resp.text:
                return ([resp.text], resp.text)
            return ([], "")

        chunked_diff_list = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)
        logger.info(f"Created {len(chunked_diff_list)} chunks from diff")
        return self._process_chunks(
            model, chunked_diff_list, llm_config, comments_text,
        )

    def _process_chunks(self, model: str, chunked_diff_list: list[str],
                        llm_config: LLMConfig, comments_text: str) -> tuple[list[str], str]:
        max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "6"))
        initial_wait = float(os.getenv("GEMINI_INITIAL_BACKOFF_SECONDS", "15"))
        max_wait = float(os.getenv("GEMINI_MAX_BACKOFF_SECONDS", "240"))
        min_request_interval = float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "6"))
        fail_fast_on_no_quota = os.getenv("GEMINI_FAIL_FAST_ON_NO_QUOTA", "1") == "1"
        # ... (rest of the chunked processing logic from _process_chunks)
        # NOTE: This will be the full migrated code from the existing gemini_client.py
        # See gemini_client.py lines 28-234 for the complete chunking/summary logic
        ...
        return chunked_reviews, summarized_review


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a Gemini response."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    prompt_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)
    out: dict[str, int] = {}
    if prompt_tokens is not None:
        out["prompt_tokens"] = int(prompt_tokens)
    if output_tokens is not None:
        out["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        out["total_tokens"] = int(total_tokens)
    return out


register_provider("gemini", GeminiClient)
```

**Step 2: Re-export from `code_reviewer/gemini_client.py` for backward compat**

The existing file becomes a thin shim that imports from `code_reviewer.llm.gemini_client`:

```python
"""Backward-compat re-exports. New code should import from code_reviewer.llm."""
from code_reviewer.llm.gemini_client import GeminiClient, get_review  # noqa: F401
```

**Step 3: Verify**

```bash
python -c "from code_reviewer.llm import get_llm_client; c = get_llm_client('gemini'); print(type(c).__name__)"
```

Expected: `GeminiClient`

**Step 4: Commit**

```bash
git add code_reviewer/llm/gemini_client.py code_reviewer/gemini_client.py
git commit -m "refactor: extract Gemini provider into code_reviewer/llm/gemini_client.py"
```

---

## Task 3: Create OpenAI-compatible provider

**Objective:** Implement `OpenAIClient` supporting OpenAI, DeepSeek, Kimi, Qwen, and any OpenAI-compatible API (including OpenRouter) via configurable base URL.

**Files:**
- Create: `code_reviewer/llm/openai_client.py`
- Modify: `code_reviewer/llm/__init__.py`

**Step 1: Write `code_reviewer/llm/openai_client.py`**

```python
"""OpenAI-compatible provider — supports OpenAI, DeepSeek, Kimi, Qwen, OpenRouter."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from loguru import logger

from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import register_provider

DEFAULT_TOKEN_LIMIT = 128_000


class OpenAIClient(LLMClient):
    """LLMClient for OpenAI-compatible APIs.

    Supports: OpenAI, DeepSeek, Kimi (Moonshot), Qwen (Tongyi), OpenRouter, etc.
    Configure via env vars:
      - ``OPENAI_API_KEY`` (required for OpenAI)
      - ``OPENAI_BASE_URL`` (optional, defaults to ``https://api.openai.com/v1``)
      - For other providers, set both the provider-specific key AND base URL.
    """

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
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY or LLM_API_KEY is required for OpenAI-compatible provider"
            )
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(api_key=api_key, base_url=base_url)

    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_output_tokens,
        }
        if config.system_instruction:
            payload["messages"].append({
                "role": "system",
                "content": config.system_instruction,
            })
        payload["messages"].append({"role": "user", "content": prompt})

        # Support response_format for JSON mode
        payload["response_format"] = {"type": "json_object"}

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
        # OpenAI model context sizes are well-known; use a lookup or default
        known_limits = {
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "gpt-4-turbo": 128_000,
            "o1": 200_000,
            "o3-mini": 200_000,
            "deepseek-chat": 1_000_000,
            "deepseek-reasoner": 64_000,
            "moonshot-v1-8k": 8_192,
            "moonshot-v1-32k": 32_768,
            "moonshot-v1-128k": 131_072,
            "qwen-max": 32_000,
            "qwen-plus": 131_072,
            "qwen-turbo": 1_000_000,
        }
        return known_limits.get(model, DEFAULT_TOKEN_LIMIT)

    def _build_chunked_config(self, base_config: LLMConfig) -> LLMConfig:
        """Build a config without system_instruction for chunk continuation prompts."""
        return LLMConfig(
            model=base_config.model,
            system_instruction=None,
            temperature=base_config.temperature,
            top_p=base_config.top_p,
            max_output_tokens=base_config.max_output_tokens,
        )


register_provider("openai", OpenAIClient)
```

**Step 2: Verify**

```bash
python -c "
from code_reviewer.llm import get_llm_client
c = get_llm_client('openai')
print(type(c).__name__)
print(f'Base URL: {c._base_url}')
"
```

Expected: `OpenAIClient`, `Base URL: https://api.openai.com/v1`

**Step 3: Commit**

```bash
git add code_reviewer/llm/openai_client.py
git commit -m "feat: add OpenAI-compatible LLM provider (OpenAI, DeepSeek, Kimi, Qwen, OpenRouter)"
```

---

## Task 4: Create Anthropic provider

**Objective:** Implement `AnthropicClient` using the official `anthropic` SDK.

**Files:**
- Create: `code_reviewer/llm/anthropic_client.py`
- Modify: `code_reviewer/llm/__init__.py`

**Step 1: Write `code_reviewer/llm/anthropic_client.py`**

```python
"""Anthropic provider — implements LLMClient via anthropic SDK."""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from loguru import logger

from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import register_provider

DEFAULT_TOKEN_LIMIT = 200_000


class AnthropicClient(LLMClient):
    """LLMClient implementation for Anthropic (Claude) via anthropic SDK."""

    def __init__(self, client: Anthropic) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> AnthropicClient:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
        return cls(Anthropic(api_key=api_key))

    def generate_content(self, prompt: str, config: LLMConfig) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
        }
        if config.system_instruction:
            kwargs["system"] = config.system_instruction

        message = self._client.messages.create(**kwargs)
        text = message.content[0].text if message.content else None

        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            usage={
                "prompt_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": (getattr(usage, "input_tokens", 0) or 0)
                                + (getattr(usage, "output_tokens", 0) or 0),
            } if usage else {},
        )

    def get_context_limit(self, model: str) -> int:
        known = {
            "claude-sonnet-4-20250514": 200_000,
            "claude-sonnet-4": 200_000,
            "claude-3-5-sonnet-20241022": 200_000,
            "claude-3-5-haiku-20241022": 200_000,
            "claude-3-opus-20240229": 200_000,
            "claude-3-haiku-20240307": 200_000,
        }
        return known.get(model, DEFAULT_TOKEN_LIMIT)


register_provider("anthropic", AnthropicClient)
```

**Step 2: Add `anthropic` to requirements.txt**

```
anthropic>=0.40,<1
```

**Step 3: Verify**

```bash
python -c "from code_reviewer.llm import get_llm_client; c = get_llm_client('anthropic'); print(type(c).__name__)"
```

Expected: `AnthropicClient`

**Step 4: Commit**

```bash
git add code_reviewer/llm/anthropic_client.py requirements.txt
git commit -m "feat: add Anthropic LLM provider (Claude)"
```

---

## Task 5: Decouple quota.py from google.genai imports

**Objective:** Remove the hard dependency on `google.genai.errors` in `code_reviewer/quota.py` so it can be reused across providers.

**Files:**
- Modify: `code_reviewer/quota.py` — make `_handle_api_error` provider-agnostic or move Gemini-specific handling to `code_reviewer/llm/gemini_client.py`

**Step 1: Analyze** — The `_handle_api_error` function references `errors.APIError` from `google.genai`. This is only used in `code_reviewer/gemini_client.py`'s chunk processing. Solution: Move `_handle_api_error` into `code_reviewer/llm/gemini_client.py` as a private helper, and keep `QuotaTracker` in `code_reviewer/quota.py` clean.

**Step 2: Refactor** — Remove `_handle_api_error` and `NoQuotaAvailableError` from `code_reviewer/quota.py`. Move them to `code_reviewer/llm/gemini_client.py`.

**Step 3: Commit**

```bash
git add code_reviewer/quota.py code_reviewer/llm/gemini_client.py
git commit -m "refactor: decouple quota.py from google.genai, move error handling to Gemini provider"
```

---

## Task 6: Make utils.py provider-agnostic

**Objective:** Remove Gemini-specific `_extract_model_text()` from utils.py (or keep it as a general utility).

The `_extract_model_text(response)` function accesses `response.text` which is Gemini-specific. Since the new `LLMResponse` already carries `.text`, this function is no longer needed in the new code paths. Keep it in utils for backward compat but mark as deprecated.

**Files:**
- Modify: `code_reviewer/utils.py` — add deprecation note, no functional change needed
- No immediate code change required — the GeminiClient internally uses `_extract_model_text` which is fine.

---

## Task 7: Wire providers into main.py

**Objective:** Update `code_reviewer/main.py` to select the LLM provider at startup and use the abstraction layer.

**Files:**
- Modify: `code_reviewer/main.py`

**Key changes:**

1. Add `--provider` CLI option (default: `gemini`)
2. Replace `genai.Client(api_key=api_key)` with `get_llm_client(provider)`
3. Pass the `LLMClient` instance to `get_review()` instead of a `genai.Client`
4. Update `check_required_env_vars()` to be provider-aware
5. Update `AiReviewConfig` in `code_reviewer/config.py` to be provider-agnostic

**Step 1: Update `code_reviewer/config.py`**

```python
class AiReviewConfig(TypedDict):
    """Configuration for an AI review request."""
    provider: NotRequired[str]
    model: str
    diff: str
    extra_prompt: str
    prompt_chunk_size: int
    comments_text: str
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    max_output_tokens: NotRequired[int]
    review_memory_context: NotRequired[str]
    supplemental_context: NotRequired[str]


def check_required_env_vars():
    """Check required environment variables based on provider."""
    provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()

    required_env_vars = ["GITHUB_TOKEN", "GITHUB_REPOSITORY",
                         "GITHUB_PULL_REQUEST_NUMBER", "GIT_COMMIT_HASH"]

    if provider == "gemini":
        required_env_vars.insert(0, "GEMINI_API_KEY")
    elif provider == "openai":
        required_env_vars.insert(0, "OPENAI_API_KEY")
    elif provider == "anthropic":
        required_env_vars.insert(0, "ANTHROPIC_API_KEY")

    if os.getenv("LOCAL") is not None:
        # Local mode only needs the LLM API key
        required_env_vars = [v for v in required_env_vars if v.startswith(("GEMINI_", "OPENAI_", "ANTHROPIC_")) or v == "LLM_API_KEY"]

    for required_env_var in required_env_vars:
        value = os.getenv(required_env_var)
        if value is None or not value.strip():
            raise ValueError(f"{required_env_var} is not set or is empty")
```

**Step 2: Update `code_reviewer/main.py` main() function**

Replace:
```python
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
```

With:
```python
from code_reviewer.llm import get_llm_client
client = get_llm_client(provider=provider)  # 'provider' comes from --provider CLI arg
```

Add `--provider` click option:
```python
@click.option(
    "--provider",
    type=click.Choice(["gemini", "openai", "anthropic"], case_sensitive=False),
    required=False,
    default=None,
    help="LLM provider (gemini, openai, anthropic). Defaults to LLM_PROVIDER env var or 'gemini'.",
)
```

**Step 3: Update `get_review()` call** — The GeminiClient has `get_review()` which currently uses the old signature. We need to make it consistent. Either:
- Add a `get_review(client, config)` wrapper in `code_reviewer/llm/__init__.py` that dispatches to the right client
- Or modify main.py to call `client.get_review(config)` directly

Option B is cleaner. The `get_review()` interface becomes part of the `LLMClient` protocol, but since chunking is specific to code review, it's better as a standalone function that takes an `LLMClient`:

```python
def run_review(client: LLMClient, config: AiReviewConfig) -> tuple[list[str], str]:
    """Run a code review using any LLM provider."""
    # Check budget, chunk if needed, call client.generate_content()
    ...
```

**Step 4: Commit**

```bash
git add code_reviewer/main.py code_reviewer/config.py
git commit -m "feat: wire LLM provider selection into CLI --provider flag and config"
```

---

## Task 8: Update action.yml with new inputs

**Objective:** Add new action inputs for multi-provider support.

**Files:**
- Modify: `action.yml`

**Changes:**

1. Add `llm_provider` input (optional, default: `gemini`)
2. Add `openai_api_key` input (optional, secret)
3. Add `anthropic_api_key` input (optional, secret)
4. Rename `gemini_api_key` → make it optional (only required when provider=gemini)
5. Update model input description to reflect multi-provider

```yaml
inputs:
  llm_provider:
    description: "LLM provider to use (gemini, openai, anthropic)."
    required: false
    default: "gemini"
  gemini_api_key:
    description: "The Gemini API key (required when provider=gemini)"
    required: false
  openai_api_key:
    description: "The OpenAI API key (required when provider=openai). Also used for DeepSeek, Kimi, Qwen, OpenRouter via OPENAI_BASE_URL env."
    required: false
  anthropic_api_key:
    description: "The Anthropic API key (required when provider=anthropic)"
    required: false
  openai_base_url:
    description: "Base URL for OpenAI-compatible APIs (for DeepSeek, Kimi, Qwen, OpenRouter)."
    required: false
    default: ""
  model:
    description: "Model name (e.g., gemini-2.5-flash, gpt-4o, claude-sonnet-4, deepseek-chat)"
    required: true
    default: "gemini-2.5-flash"
```

Update the docker run section to pass the new env vars:

```yaml
- name: "Run action"
  env:
    LLM_PROVIDER: ${{ inputs.llm_provider }}
    GEMINI_API_KEY: ${{ inputs.gemini_api_key }}
    OPENAI_API_KEY: ${{ inputs.openai_api_key }}
    ANTHROPIC_API_KEY: ${{ inputs.anthropic_api_key }}
    OPENAI_BASE_URL: ${{ inputs.openai_base_url }}
    ...
```

**Commit:**

```bash
git add action.yml
git commit -m "feat: add multi-provider inputs to action.yml"
```

---

## Task 9: Update tests

**Objective:** Update existing tests and add new tests for the provider abstraction.

**Files:**
- Modify: `test/test_gemini_client.py` → adapt to new GeminiClient
- Create: `test/test_provider_registry.py`
- Create: `test/test_openai_client.py`
- Create: `test/test_anthropic_client.py`

**Key test scenarios:**

1. **Provider registry**: registering, listing, factory creation for each provider
2. **GeminiClient**: same as existing tests but via new interface
3. **OpenAIClient**: mock HTTP responses, verify payload format
4. **AnthropicClient**: mock SDK calls, verify message format
5. **Provider selection by env var**: `os.environ["LLM_PROVIDER"] = "openai"` → correct client

**Commit:**

```bash
git add test/
git commit -m "test: add provider abstraction tests, adapt existing Gemini tests"
```

---

## Task 10: Update documentation

**Objective:** Update README.md with multi-provider usage instructions.

**Files:**
- Modify: `README.md`

**Key sections to update:**

1. Rename title from "Gemini Code Review" to "AI Code Review"
2. Update pre-requisites to list supported providers
3. Add provider configuration section with examples for each
4. Update example workflows to show multiple providers
5. Update project structure diagram to include `code_reviewer/llm/`

**Commit:**

```bash
git add README.md
git commit -m "docs: update README with multi-provider support documentation"
```

---

## Risks & Tradeoffs

1. **OpenAI JSON mode**: Not all OpenAI-compatible APIs support `response_format: {"type": "json_object"}`. DeepSeek does, but some (Kimi older models, Qwen) may not. We may need a fallback: either drop the requirement or use prompt-only JSON extraction for those.

2. **Anthropic tool use for structured output**: The current system uses `response_mime_type="application/json"` (Gemini) and `response_format: {"type": "json_object"}` (OpenAI). For Anthropic, we should use its native tool-use or structured output feature. The current implementation just prompts for JSON, which works but is less reliable.

3. **Chunked review flow**: The chunking logic (split diff → review each chunk → summarize) is currently in `gemini_client.py`. We should either:
   - Make it a shared function that any provider can use
   - Or keep it in the Gemini provider and implement similar logic for others

   **Decision**: Make `run_review()` a standalone function in `code_reviewer/llm/` that any provider can use, since chunking logic is provider-agnostic.

4. **OpenRouter**: Can be used via the `openai` provider by setting `OPENAI_BASE_URL=https://openrouter.ai/api/v1`. This gives access to ALL models (including Gemini, Claude, etc.) through one API key. Document this as the simplest path.

5. **Backward compatibility**: The existing `GEMINI_API_KEY` env var keeps working. Old configs (just `GEMINI_API_KEY`, no `LLM_PROVIDER`) default to Gemini behavior. Zero breaking changes for existing users.

---

## Summary of Files Changed

| File | Action | Purpose |
|---|---|---|
| `code_reviewer/llm/__init__.py` | **Create** | Module exports |
| `code_reviewer/llm/base.py` | **Create** | LLMClient ABC, LLMConfig, LLMResponse |
| `code_reviewer/llm/provider_registry.py` | **Create** | Provider registration and factory |
| `code_reviewer/llm/gemini_client.py` | **Create** | Gemini provider (refactored from code_reviewer/gemini_client.py) |
| `code_reviewer/llm/openai_client.py` | **Create** | OpenAI-compatible provider |
| `code_reviewer/llm/anthropic_client.py` | **Create** | Anthropic provider |
| `code_reviewer/gemini_client.py` | **Modify** | Thin re-export shim (backward compat) |
| `code_reviewer/config.py` | **Modify** | Provider-aware env var validation |
| `code_reviewer/main.py` | **Modify** | --provider flag, factory-based client creation |
| `code_reviewer/quota.py` | **Modify** | Remove google.genai dependency |
| `action.yml` | **Modify** | New inputs for API keys and provider selection |
| `requirements.txt` | **Modify** | Add `anthropic` SDK |
| `README.md` | **Modify** | Multi-provider docs and examples |
| `test/test_gemini_client.py` | **Modify** | Adapt to GeminiClient |
| `test/test_provider_registry.py` | **Create** | Registry tests |
| `test/test_openai_client.py` | **Create** | OpenAI client tests |
| `test/test_anthropic_client.py` | **Create** | Anthropic client tests |

---

## Verification Checklist

- [ ] `python -m code_reviewer.main --help` shows `--provider` option
- [ ] With `LLM_PROVIDER=gemini` + `GEMINI_API_KEY`: works exactly as before
- [ ] With `LLM_PROVIDER=openai` + `OPENAI_API_KEY`: uses OpenAI-compatible API
- [ ] With `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`: uses Anthropic
- [ ] With `LLM_PROVIDER=openai` + `OPENAI_BASE_URL=https://openrouter.ai/api/v1`: works via OpenRouter
- [ ] All existing `test/test_gemini_client.py` tests pass
- [ ] New provider tests pass
- [ ] `ruff check code_reviewer/` passes
- [ ] Docker build succeeds
