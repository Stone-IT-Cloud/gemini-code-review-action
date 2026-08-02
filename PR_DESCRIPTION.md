# Fix inline suggestion validation, pylint, default model, and Python 3.11 compatibility

## Summary

- Improves inline suggestion handling so GitHub PR suggestions contain only valid code (no prose or raw diff).
- Fixes all pylint findings by changing code (no rule disables).
- Sets default Gemini model to `gemini-2.5-flash` and fixes Python < 3.12 compatibility in local review output.
- Adds a local Docker test script.

## Changes

### 1. Inline suggestion validation (`code_reviewer/review_parser.py`)

- **System prompt:** Clarified that `suggestion` must be exact replacement code only; never prose or unified diff; omit or set to null if no concrete fix.
- **New helpers:** `_looks_like_prose()`, `_extract_diff_additions()`, `_sanitize_suggestion()` — reject prose, normalize diff to added lines only, return `None` for invalid.
- **Validation:** `_validate_review_item()` uses `_sanitize_suggestion(suggestion_val)` instead of a simple non-empty string check.

**Tests:** New `TestSanitizeSuggestion` and tests for prose rejection and diff sanitization in `test/test_review_parser.py` and `test/test_inline_suggestions.py`.

### 2. Pylint fixes (no disables)

- **code_reviewer/context/scanner.py:** Double-quoted strings; `PARSER_EXCEPTIONS` instead of broad `Exception`; `_read_file_limited` renamed to `read_file_limited`.
- **code_reviewer/github_client.py:** Variables in f-strings; `except (requests.RequestException, OSError)`.
- **code_reviewer/main.py:** Module-level `_ANSI` dict; `print_local_review` uses a `SimpleNamespace` for ANSI codes (attribute access in f-strings, no nested quotes).
- **Tests:** Unused import removed, `assert not result` / `assert not parse_review_response(...)`, `read_file_limited`, double-quoted strings, `*mocks` for too many args, `_mock_*` for unused args.

### 3. Default model and docs

- **Default model:** `gpt-3.5-turbo` / `text-davinci-003` → **`gemini-2.5-flash`** in `code_reviewer/main.py`, `action.yml`, README, `.github/workflows/test-action.yml`, and test examples.
- **action.yml:** Descriptions updated (e.g. "Gemini model name", "Extra prompt for Gemini").

### 4. Python < 3.12 compatibility (`code_reviewer/main.py`)

- **Local review output:** Replaced f-strings with nested double-quoted dict keys (e.g. `f"{a["BOLD"]}"`) by a `SimpleNamespace` built from `_ANSI` with lowercase keys (`c.bold`, `c.cyan`, etc.). F-strings now use attribute access only (e.g. `f"{c.bold}{c.cyan}"`), so no nested quotes and no SyntaxError on Python 3.11 and below (PEP 701).

### 5. Local Docker test script

- **scripts/test-docker-local.sh:** Builds the action image, creates a diff from the last commit (`HEAD~1..HEAD`), and runs the container with `LOCAL=1` and `--diff-file=/tmp/pr.diff` so the review is printed locally. Requires `GEMINI_API_KEY` in the environment.

## Files touched

- `action.yml`
- `README.md`
- `.github/workflows/test-action.yml`
- `scripts/test-docker-local.sh` (new)
- `code_reviewer/context/scanner.py`
- `code_reviewer/github_client.py`
- `code_reviewer/main.py`
- `code_reviewer/review_parser.py`
- `test/test_context_scanner.py`
- `test/test_github_inline_comments.py`
- `test/test_inline_suggestions.py`
- `test/test_local_mode.py`
- `test/test_review_level_cli.py`
- `test/test_review_parser.py`
- `test/test_suggestion_fence.py`
- `test/long-diff.txt`

## Testing

- `pre-commit run pylint --all-files` passes.
- Run `./scripts/test-docker-local.sh` (with `GEMINI_API_KEY` set) to test the Docker image against this repo.
- Run `pytest` locally to confirm all tests pass.

## Checklist

- [x] No pylint rule disables; all issues fixed in code.
- [x] Inline suggestions validated (prose/diff sanitized or rejected).
- [x] Default model is `gemini-2.5-flash`.
- [x] Local review output compatible with Python 3.11 (no nested quotes in f-strings).
- [x] Local Docker test script added.
