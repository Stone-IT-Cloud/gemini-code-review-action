# SPEC: Auto-review-level por tamaño de diff

## Resumen

El `review_level` se ajusta automáticamente según la cantidad de líneas del diff, para que el volumen de feedback sea proporcional al tamaño del cambio. Si el usuario lo fuerza explícitamente, se respeta su elección.

## Comportamiento

```
review_level forzado por el usuario?
  ├── Sí → usar ese (backward compatible, prioridad máxima)
  └── No → auto según líneas del diff:
         ├── < 50   líneas → TRIVIAL
         ├── 50-500 líneas → IMPORTANT  (default actual)
         └── > 500  líneas → CRITICAL
```

## Implementación

**Archivo:** `src/main.py`, antes de la línea donde se determina `min_severity`.

```python
# Auto-adjust review level based on diff size (unless user explicitly set it)
if not review_level and not os.getenv("REVIEW_LEVEL"):
    diff_lines = diff.count("\n")
    if diff_lines < 50:
        review_level = "TRIVIAL"
        logger.info(f"Diff is small ({diff_lines} lines), auto-setting review_level=TRIVIAL")
    elif diff_lines > 500:
        review_level = "CRITICAL"
        logger.info(f"Diff is large ({diff_lines} lines), auto-setting review_level=CRITICAL")
```

**Variables involucradas:**
- `diff` — string con el diff completo (ya resuelto en `main()`)
- `review_level` — CLI argument (`--review-level`), `None` si no se pasó
- `REVIEW_LEVEL` — env var, opcional
- `min_severity` — se calcula después con prioridad: CLI arg > env var > auto

## Tests

| Test | Diff | review_level input | Esperado |
|------|------|-------------------|----------|
| Small diff | 10 líneas | None | TRIVIAL |
| Medium diff | 200 líneas | None | IMPORTANT |
| Large diff | 1000 líneas | None | CRITICAL |
| Boundary low | 50 líneas | None | IMPORTANT |
| Boundary high | 500 líneas | None | IMPORTANT |
| Explicit override | 10 líneas | "CRITICAL" | CRITICAL |
| Env var override | 1000 líneas | None + REVIEW_LEVEL=TRIVIAL | TRIVIAL |

## Archivos a modificar

- `src/main.py` — lógica de auto-adjust
- `test/test_review_level_cli.py` — nuevos tests TDD

## No-rompimiento

- Si el usuario ya tiene `review_level: IMPORTANT` en su workflow → mismo comportamiento que hoy
- Si no configura nada → gana inteligencia sin tocar nada
- El valor auto-ajustado se loggea para transparencia
