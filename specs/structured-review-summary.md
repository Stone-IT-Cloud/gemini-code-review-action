# SPEC: Structured Review Summary

## Resumen

Agregar un encabezado estructurado al inicio del review que resuma en un vistazo el contenido de la revisión: archivos tocados, issues encontrados por severidad, y áreas de preocupación.

## Comportamiento

Cada review generado debe incluir un bloque de resumen al inicio con este formato:

```
📋 Review: +#additions / -#deletions líneas, #archivos archivos
🔴 # CRITICAL — descripción corta (si aplica)
🟡 # IMPORTANT — descripción corta (si aplica)
🔵 # TRIVIAL — descripción corta (si aplica)
```

Ejemplo real:
```
📋 Review: +350 / -120 líneas, 8 archivos
🔴 2 CRITICAL — SQL injection en query builder, null pointer sin guard
🟡 5 IMPORTANT — lógica de pricing incorrecta, falta validación de input
🔵 3 TRIVIAL — imports sin usar, naming inconsistente
```

Si no hay issues de una severidad, esa línea se omite. Si no hay ningún issue, el resumen dice:

```
📋 Review: +42 / -10 líneas, 2 archivos
✅ No se encontraron issues.
```

## Implementación

**Archivo:** `code_reviewer/review_parser.py` — nueva función `build_review_summary()`

```python
def build_review_summary(
    all_items: list[dict],
    diff_stats: dict[str, int] | None = None,
) -> str:
    """Build a structured summary header for the review."""
```

**Archivo:** `code_reviewer/main.py` — donde se arma `review_comment`, agregar el summary al principio.

**Archivo:** `code_reviewer/review_parser.py` — función `format_review_comment()` existente, insertar summary al inicio.

### Inputs necesarios

El summary necesita:
- `all_items` — lista de items reviewados (con `severity` y `comment`)
- `diff_additions` / `diff_deletions` — líneas agregadas/eliminadas (desde el diff)
- `diff_files` — cantidad de archivos tocados (desde el diff)

### Cómo obtener diff_stats

Ya tenemos `diff` en `main()` — es el string del diff completo. Podemos extraer los stats con:

```python
import re

def _parse_diff_stats(diff: str) -> dict[str, int]:
    """Parse diff stats: additions, deletions, files changed."""
    additions = len(re.findall(r'^\+', diff, re.MULTILINE))
    deletions = len(re.findall(r'^-', diff, re.MULTILINE))
    files = len(re.findall(r'^\+\+\+ b/', diff, re.MULTILINE))
    return {"additions": additions, "deletions": deletions, "files": files}
```

## Tests

| Test | Input | Esperado |
|------|-------|----------|
| Mixed severities | 2 CRITICAL, 3 IMPORTANT, 1 TRIVIAL | Summary con 🔴🟡🔵 |
| Only IMPORTANT | 0 CRITICAL, 2 IMPORTANT, 0 TRIVIAL | Solo 🟡 |
| No issues | Lista vacía | ✅ No se encontraron issues |
| Single CRITICAL | 1 CRITICAL | 🔴 1 CRITICAL |
| Diff stats included | diff con +50/-30, 2 files | Review: +50 / -30 líneas, 2 archivos |
| Summary + formatted review | Summary + body | Review completo con encabezado |

## Archivos a modificar

- `code_reviewer/review_parser.py` — `build_review_summary()` + `_parse_diff_stats()`
- `code_reviewer/main.py` — integrar summary en `review_comment`
- `test/test_review_parser.py` — tests del summary

## No-rompimiento

- Si el summary falla (error parseando stats), se ignora y se muestra el review completo como siempre
- El formato actual (`review_comment`) es un string plano — agregar el summary al inicio es transparente para los consumidores
- Backward compatible: workflows existentes que parsean `review_comment` van a recibir texto extra al inicio, pero el contenido del review sigue siendo el mismo
