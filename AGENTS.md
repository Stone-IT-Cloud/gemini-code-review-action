<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, **always prefix with `rtk`**. This reduces context
usage by 60-90% with zero behavior change. If rtk has no filter for a command,
it passes through unchanged — so it is always safe to use.

## Key Commands
```bash
# Git (59-80% savings)
rtk git status          rtk git diff            rtk git log

# Files & Search (60-75% savings)
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk find <pattern>      rtk diff <file>

# Test (90-99% savings) — shows failures only
rtk pytest tests/       rtk cargo test          rtk test <cmd>

# Build & Lint (80-90% savings) — shows errors only
rtk tsc                 rtk lint                rtk cargo build
rtk prettier --check    rtk mypy                rtk ruff check

# Analysis (70-90% savings)
rtk err <cmd>           rtk log <file>          rtk json <file>
rtk summary <cmd>       rtk deps                rtk env

# GitHub (26-87% savings)
rtk gh pr view <n>      rtk gh run list         rtk gh issue list

# Infrastructure (85% savings)
rtk docker ps           rtk kubectl get         rtk docker logs <c>

# Package managers (70-90% savings)
rtk pip list            rtk pnpm install        rtk npm run <script>
```

## Rules
- In command chains, prefix each segment: `rtk git add . && rtk git commit -m "msg"`
- For debugging, use raw command without rtk prefix
- `rtk proxy <cmd>` runs command without filtering but tracks usage
<!-- /headroom:rtk-instructions -->

## LLM Provider Architecture

The action supports multiple LLM providers via a clean abstraction layer in `code_reviewer/llm/`:

- **`code_reviewer/llm/base.py`**: `LLMClient` ABC — implement `generate_content()` and `get_context_limit()`
- **`code_reviewer/llm/provider_registry.py`**: `register_provider(name, class)` + `get_llm_client(provider)`
- **`code_reviewer/llm/review.py`**: `run_review(client, config)` — provider-agnostic chunking + summarization
- **`code_reviewer/llm/gemini_client.py`**: Gemini via `google.genai` SDK
- **`code_reviewer/llm/openai_client.py`**: OpenAI-compatible (OpenRouter, DeepSeek, etc.) via `requests`

### Adding a new provider

```python
# code_reviewer/llm/myclient.py
from code_reviewer.llm.base import LLMClient, LLMConfig, LLMResponse
from code_reviewer.llm.provider_registry import register_provider

class MyClient(LLMClient):
    @classmethod
    def from_env(cls) -> 'MyClient': ...
    def generate_content(self, prompt, config) -> LLMResponse: ...
    def get_context_limit(self, model) -> int: ...

register_provider("myclient", MyClient)
```

Then add `from code_reviewer.llm import myclient` to `code_reviewer/llm/__init__.py`.

### Provider env vars

| Provider | API key env | Base URL env | Default URL |
|----------|------------|-------------|-------------|
| gemini | `GEMINI_API_KEY` | — | — |
| openai | `OPENAI_API_KEY` or `LLM_API_KEY` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| openrouter | `OPENAI_API_KEY` or `LLM_API_KEY` | `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` |
| deepseek | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| qwen | `QWEN_API_KEY` or `DASHSCOPE_API_KEY` | `QWEN_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| kimi | `KIMI_API_KEY` or `MOONSHOT_API_KEY` | `KIMI_BASE_URL` | `https://api.moonshot.cn/v1` |
| baichuan | `BAICHUAN_API_KEY` | `BAICHUAN_BASE_URL` | `https://api.baichuan-ai.com/v1` |
| zhipu | `ZHIPU_API_KEY` | `ZHIPU_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` |
| (yours) | `YOUR_API_KEY` | `YOUR_BASE_URL` | — |

### Testing

```bash
# All tests
rtk pytest tests/

# Provider-specific
rtk pytest test/test_openai_client.py -v
rtk pytest test/test_provider_registry.py -v
```
