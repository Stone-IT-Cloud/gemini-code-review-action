# OpenRouter / OpenAI-Compatible Driver — SDD

**Goal:** Implement `OpenAIClient` (registrado como `"openai"`) que permita usar cualquier modelo vía OpenRouter (y cualquier API OpenAI-compatible como DeepSeek, Kimi, Qwen).

**Arquitectura:** Un solo `OpenAIClient` implementando `LLMClient`. Usa `requests` HTTP directamente (sin SDK adicional). Configurable vía `OPENAI_API_KEY` + `OPENAI_BASE_URL`. Por defecto apunta a `https://openrouter.ai/api/v1`.

**Design decisions:**
- No SDK externo — `requests` ya está en requirements.txt
- `OPENAI_BASE_URL` permite apuntar a cualquier API compatible (OpenRouter, OpenAI directo, DeepSeek, etc.)
- `get_context_limit()` usa lookup table de modelos conocidos, con fallback a 128K
- `register_provider("openai", OpenAIClient)` — se auto-registra al importar el módulo

---

## Task 1: Escribir test que falla — OpenAIClient no existe

**RED** — test que falla porque `OpenAIClient` no está implementado.

**Archivo:** `test/test_openai_client.py`

```python
"""Tests for src/llm/openai_client.py — OpenAI-compatible provider."""

from src.llm.openai_client import OpenAIClient
from src.llm.base import LLMConfig


def test_openai_client_is_registered():
    """OpenAI provider debe estar registrado."""
    from src.llm import list_providers
    assert "openai" in list_providers()
```

## Task 2: Crear OpenAIClient + registro

**GREEN** — implementación mínima.

**Archivo:** `src/llm/openai_client.py`

- Clase `OpenAIClient(LLMClient)` con `from_env()`, `generate_content()`, `get_context_limit()`
- `from_env()` lee `OPENAI_API_KEY` (o `LLM_API_KEY`) y `OPENAI_BASE_URL` (default `https://openrouter.ai/api/v1`)
- `generate_content()` hace POST a `/chat/completions`, retorna `LLMResponse`
- `get_context_limit()` lookup table con defaults conocidos
- `register_provider("openai", OpenAIClient)` al final del módulo

## Task 3: Tests de generate_content

**RED → GREEN** — ciclo TDD por cada comportamiento:

1. `test_generate_content_success` — POST exitoso → `LLMResponse.text` no vacío
2. `test_generate_content_includes_system_prompt` — system_instruction se envía como mensaje system
3. `test_generate_content_http_error` — HTTP 401/403 → raise
4. `test_get_context_limit_known_model` — modelo conocido retorna su límite
5. `test_get_context_limit_unknown_model` — modelo desconocido retorna default 128K
6. `test_from_env_with_custom_base_url` — `OPENAI_BASE_URL` personalizada se usa

## Task 4: Wire en __init__.py

Agregar `from src.llm import openai_client` en `src/llm/__init__.py`.

## Task 5: Verificar full suite

Correr `pytest test/ -q` — todo verde, sin regresiones.

## Task 6: Actualizar README

Documentar uso de OpenRouter y OpenAI-compatible providers.

---

## Archivos modificados

| Archivo | Acción |
|---|---|
| `src/llm/openai_client.py` | **Crear** |
| `src/llm/__init__.py` | Modificar (import openai_client) |
| `test/test_openai_client.py` | **Crear** |
| `README.md` | Modificar |
