# SPEC: Post-PR learning — retroalimentación del bot desde discusiones humanas

## Resumen

Cuando un PR se cierra o mergea, el action corre en modo "learn" para analizar los comentarios del bot que recibieron respuesta humana y persistir las decisiones en Engram.

Esto cierra el loop: cada discusión humana alimenta la memoria del bot, evitando que repita falsos positivos.

## Comportamiento

```
PR mergeado/cerrado
  ↓
Workflow "learn" se dispara
  ↓
1. Fetch todos los review comments del bot en el PR
2. Para cada comment, buscar si tiene respuestas humanas
3. Clasificar la decisión: rejected, accepted, acknowledged
4. Store cada decisión en Engram (type=review-decision)
  ↓
Cache de Engram actualizada
  ↓
Próximo PR → bot consulta Engram → evita repetir errores
```

## Clasificación de decisiones

| Decisión | Indicadores | Ejemplo de respuesta humana |
|----------|------------|-----------------------------|
| **rejected** | "no aplica", "falso", "no", "wont fix", "not relevant", "trivial", "cerrando como no aplica" | "Esto es un falso positivo" |
| **accepted** | "fixed", "done", "good catch", "thanks", "applied", "implementado" | "Buen punto, lo corrijo" |
| **acknowledged** | Respuesta que no rechaza ni acepta explícitamente | "Lo voy a revisar" |

Para la clasificación, usar el mismo LLM (Gemini) en lugar de keyword matching — es más preciso y menos propenso a falsos.

## Modo "learn"

El action tendrá un nuevo input `mode: "learn"` que:

1. No revisa el diff (no consume tokens de review)
2. Solo procesa discusiones cerradas
3. Usa Gemini para clasificar decisiones (un solo prompt, barato)

### Prompt para clasificación

```
Analiza la siguiente discusión de code review y determina si la sugerencia
fue aceptada, rechazada o acknowledgada.

Sugerencia original: "{suggestion}"
Respuesta humana: "{reply}"

Responde con UNO de estos tres valores exactos:
- rejected
- accepted
- acknowledged
```

## Workflow

```yaml
name: Learn from PR

on:
  pull_request:
    types: [closed]

jobs:
  learn:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
    steps:
      - uses: actions/checkout@v4
      - name: Restore Engram cache
        uses: actions/cache@v4
        with:
          path: .engram
          key: review-memory-${{ github.repository }}
      - uses: ./
        with:
          mode: learn
          gemini_api_key: ${{ secrets.GEMINI_API_KEY }}
          github_token: ${{ secrets.GITHUB_TOKEN }}
          github_repository: ${{ github.repository }}
          github_pull_request_number: ${{ github.event.pull_request.number }}
          review_memory_path: .engram
          log_level: INFO
      - name: Save Engram cache
        uses: actions/cache@v4
        with:
          path: .engram
          key: review-memory-${{ github.repository }}
```

## Implementación

### Nuevos archivos

- `src/learner.py` — lógica de análisis de discusiones post-PR
- `test/test_learner.py` — tests

### Cambios en archivos existentes

- `src/main.py` — nuevo branch para `mode: "learn"`
- `action.yml` — nuevo input `mode` (values: "review", "learn")

### `src/learner.py`

```python
def fetch_bot_comments(github_token, repo, pr_number) -> list[dict]:
    """Fetch all review comments by github-actions[bot] con sus respuestas."""

def classify_decision(suggestion: str, reply: str, llm_client) -> str:
    """Usa Gemini para clasificar: rejected | accepted | acknowledged."""

def store_decisions(engram_dir, repo, pr_number, decisions):
    """Persiste decisiones en Engram."""

def run(github_token, repo, pr_number, engram_dir, llm_client):
    """Orquestador del modo learn."""
```

### Tests

| Test | Descripción |
|------|-------------|
| `fetch_bot_comments` returns only bot comments | Filtra por autor |
| `fetch_bot_comments` includes human replies | Incluye respuestas |
| `classify_decision` rejected | "no aplica" → rejected |
| `classify_decision` accepted | "good catch" → accepted |
| `classify_decision` acknowledged | "lo voy a revisar" → acknowledged |
| `store_decisions` persiste en Engram | DB tiene los datos correctos |
| Integration: learn run → Engram updated | Flujo completo |

## Token consumption

El modo learn es barato:
- 1 llamada a Gemini por cada bot comment con respuesta humana
- Prompt: ~200 tokens por decisión
- PR típico: 0-3 decisiones → ~600 tokens
- Comparado con un review que puede gastar 10K-50K tokens
