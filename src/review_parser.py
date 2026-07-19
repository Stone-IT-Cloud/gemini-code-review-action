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
"""Review response parser — JSON extraction, validation, and suggestion sanitization."""

import json
import re
from enum import Enum, auto

from loguru import logger

VALID_SEVERITIES = frozenset({"critical", "important", "trivial"})

REVIEW_SYSTEM_PROMPT = (
    "You are an expert code reviewer. Your task is to analyze the provided "
    "code changes.\n"
    "You must output your review strictly as a JSON array of objects.\n"
    "Do not include any markdown formatting (like ```json).\n"
    "\n"
    "Severity Classification:\n"
    "- TRIVIAL: Style issues, formatting, minor refactoring, missing "
    "docstrings.\n"
    "- IMPORTANT: Logic errors, potential bugs, performance inefficiencies "
    "(e.g., O(n^2)), bad practices.\n"
    "- CRITICAL: Security vulnerabilities (SQLi, XSS), potential crashes, "
    "breaking changes, data loss risks.\n"
    "\n"
    "IMPORTANT — Avoiding Repeated Suggestions:\n"
    "- The PR comments section below contains EXISTING comments and discussion "
    "from this pull request.\n"
    "- If a suggestion was already made in a previous comment and the author "
    "rejected it, explained why it doesn't apply, or marked it as resolved, "
    "DO NOT repeat it.\n"
    "- If a conversation was resolved without action, that means the involved "
    "parties agreed the issue was addressed, not applicable, or not worth "
    "pursuing. Respect that decision.\n"
    "- Focus your review on NEW issues that have not been raised before.\n"
    "- Only flag an issue if it adds meaningful value beyond what was already "
    "discussed.\n"
    "\n"
    "Use the following schema for each review item:\n"
    "["
    "  {\n"
    '    "file": "filename.py",\n'
    '    "line": <line_number_as_integer>,\n'
    '    "severity": "TRIVIAL | IMPORTANT | CRITICAL",\n'
    '    "comment": "Your review comment here",\n'
    '    "suggestion": "optional fixed code snippet"\n'
    "  }\n"
    "]\n"
    "The 'suggestion' field is optional. When present it MUST contain only "
    "the exact replacement code for the line(s) at the given location. Never "
    "put natural language descriptions, explanations, or advice in suggestion. "
    "Never use unified diff format (no ---, +++, @@, or +/- line prefixes). "
    "If you cannot provide a concrete code fix, omit suggestion or set it to null.\n"
    "If you have no comments, return an empty JSON array: []\n"
    "Do not add any text before or after the JSON array."
)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (e.g. ```json ... ```) wrapping JSON content."""
    stripped = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


# Common prose sentence starters that indicate natural language, not code
_PROSE_STARTERS = (
    "verify",
    "review",
    "ensure",
    "update",
    "check",
    "consider",
    "identify",
    "investigate",
    "add",
    "remove",
    "fix",
    "change",
    "thoroughly",
    "please",
    "recommended",
    "suggest",
    "avoid",
)


def _count_code_characters(text: str) -> int:
    """Count code-like structural characters in text."""
    code_chars = "=(){};:[]<>"
    return sum(1 for c in text if c in code_chars)


def _has_high_alpha_ratio(text: str, code_count: int) -> bool:
    """Return True if text has high alphabetic/space ratio suggesting prose."""
    if code_count != 0:
        return False
    alpha = sum(1 for c in text if c.isalpha() or c.isspace())
    return alpha / len(text) > 0.85


def _looks_like_prose(text: str) -> bool:
    """Return True if text appears to be natural language prose, not code."""
    stripped = text.strip()
    if not stripped or len(stripped) < 20:
        return False
    if "#" in stripped:
        return False
    code_count = _count_code_characters(stripped)
    if code_count >= 2:
        return False
    first_word = stripped.split()[0].lower() if stripped.split() else ""
    if first_word in _PROSE_STARTERS:
        return True
    return _has_high_alpha_ratio(stripped, code_count)


def _extract_diff_additions(text: str) -> str | None:
    """Extract added lines from a unified diff.

    If text looks like unified diff format, extract only the added lines
    (lines starting with single '+'), stripping the '+' prefix.

    Returns:
        The added lines as a string, or None if no valid additions found.
    """
    lines = text.splitlines()
    added = []
    for line in lines:
        if line.startswith("+") and not line.startswith("++"):
            content = line[1:]
            # Skip diff metadata lines that look like file paths
            if content.strip().startswith(("---", "+++", "diff ")):
                continue
            added.append(content)
    if not added:
        return None
    result = "\n".join(added).rstrip()
    return result if result else None


class DiffFormat(Enum):
    """Classification of a suggestion string's format for sanitization."""

    NONE = auto()
    UNIFIED = auto()
    SINGLE_LINE = auto()
    PLAIN = auto()


def _has_diff_headers(raw: str) -> bool:
    """Check if text contains unified diff header markers (---/+++ or @@)."""
    for marker in ("--- a/", "--- b/", "+++ a/", "+++ b/"):
        if marker in raw:
            return True
    return "\n@@ " in raw or raw.startswith("@@")


def _is_diff_like_line(ln: str) -> bool:
    """Check if a line looks like a diff addition/removal (non-metadata)."""
    stripped = ln.strip()
    return stripped.startswith(("+", "-")) and not stripped.startswith(("++", "--"))


def _classify_diff_format(value: str) -> DiffFormat:
    """Classify a suggestion string's format.

    Returns the format category without performing prose validation.
    Returns NONE for empty, whitespace-only, or non-string values.
    """
    if not isinstance(value, str) or not value.rstrip():
        return DiffFormat.NONE
    raw = value.rstrip()
    lines = raw.splitlines()

    if _has_diff_headers(raw):
        return DiffFormat.UNIFIED

    # Single line starting with + (diff addition)
    if len(lines) == 1 and _is_diff_like_line(lines[0]):
        return DiffFormat.SINGLE_LINE

    # Multi-line with at least one diff-like line
    diff_like_lines = sum(1 for ln in lines if _is_diff_like_line(ln))
    if diff_like_lines >= 1:
        return DiffFormat.UNIFIED

    return DiffFormat.PLAIN


def _sanitize_from_unified(raw: str) -> str | None:
    """Extract and validate additions from a unified diff.

    Strips diff headers/metadata and returns only added lines.
    Returns None if the extracted content is prose or empty.
    """
    extracted = _extract_diff_additions(raw)
    if not extracted:
        return None
    if _looks_like_prose(extracted):
        return None
    return extracted


def _sanitize_from_single_line(raw: str) -> str | None:
    """Extract and validate content from a single-line diff (+ prefix).

    Strips the leading '+' and preserves original indentation.
    Returns None if the content is prose or empty.
    """
    content = raw.splitlines()[0][1:]  # preserve original indentation after +
    stripped = content.strip()
    if not stripped or _looks_like_prose(stripped):
        return None
    return content


def _sanitize_suggestion(value: str) -> str | None:
    """Validate and sanitize a suggestion to contain only valid replacement code.

    Rejects natural language prose and strips unified diff format to additions only.

    Returns:
        The sanitized suggestion, or None if not valid code.
    """
    fmt = _classify_diff_format(value)

    if fmt == DiffFormat.NONE:
        return None

    raw = value.rstrip()

    if fmt == DiffFormat.UNIFIED:
        return _sanitize_from_unified(raw)

    if fmt == DiffFormat.SINGLE_LINE:
        return _sanitize_from_single_line(raw)

    # PLAIN format — reject prose, otherwise return as-is
    if _looks_like_prose(raw):
        return None

    return raw


def _normalize_line(line_val: object) -> int:
    """Coerce line value to int; return 0 for missing/invalid values."""
    if line_val is None:
        return 0
    if isinstance(line_val, int | str):
        try:
            return int(line_val)
        except (TypeError, ValueError):
            return 0
    return 0


def _normalize_severity(severity_val: object) -> str:
    """Normalize severity string; default to 'important'."""
    normalized: str = "important"
    if isinstance(severity_val, str):
        stripped = severity_val.strip().lower()
        if stripped in VALID_SEVERITIES:
            normalized = stripped
    return normalized


def _validate_review_item(item: dict) -> dict | None:
    """Validate and normalize a single review item.

    Returns the normalized item or None if the item is invalid.
    """
    if not isinstance(item, dict):
        return None

    file_val = item.get("file")
    line_val = item.get("line")
    severity_val = item.get("severity")
    comment_val = item.get("comment")
    suggestion_val = item.get("suggestion")

    if not isinstance(file_val, str) or not file_val.strip():
        return None
    if not isinstance(comment_val, str) or not comment_val.strip():
        return None

    # Validate and sanitize suggestion: must be valid code, not prose or raw diff
    normalized_suggestion = None
    if suggestion_val is not None and isinstance(suggestion_val, str):
        normalized_suggestion = _sanitize_suggestion(suggestion_val)

    result: dict = {
        "file": file_val.strip(),
        "line": _normalize_line(line_val),
        "severity": _normalize_severity(severity_val),
        "comment": comment_val.strip(),
    }

    if normalized_suggestion is not None:
        result["suggestion"] = normalized_suggestion

    return result


def _unwrap_review_dict(parsed: dict) -> list:
    """Unwrap a model response dict into a list of review items.

    Handles: {"reviews": [...]}, {"comments": [...]}, or single item dicts.
    """
    for key in ("reviews", "comments", "items", "review"):
        if key in parsed and isinstance(parsed[key], list):
            return parsed[key]
    return [parsed]


def parse_review_response(text: str | None) -> list[dict]:
    """Parse a Gemini review response into a list of validated review items.

    Handles:
    - Valid JSON arrays
    - Markdown-wrapped JSON (```json ... ```)
    - Malformed / non-JSON text (returns empty list)
    """
    if not text or not text.strip():
        logger.warning("Empty response from model; returning no review items.")
        return []

    cleaned = strip_markdown_fences(text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse model response as JSON; returning no review items. " "Response preview: %s",
            cleaned[:200],
        )
        return []

    if isinstance(parsed, dict):
        parsed = _unwrap_review_dict(parsed)

    if not isinstance(parsed, list):
        logger.warning("Model response JSON is not an array; returning no review items.")
        return []

    results: list[dict] = []
    for item in parsed:
        validated = _validate_review_item(item)
        if validated is not None:
            results.append(validated)

    logger.info("Parsed %d valid review item(s) from model response.", len(results))
    return results
