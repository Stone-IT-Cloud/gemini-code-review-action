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
import json
import re
from typing import List, Optional

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
    "Use the following schema for each review item:\n"
    "[\n"
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
    "verify", "review", "ensure", "update", "check", "consider",
    "identify", "investigate", "add", "remove", "fix", "change",
    "thoroughly", "please", "recommended", "suggest", "avoid",
)


def _looks_like_prose(text: str) -> bool:
    """Return True if text appears to be natural language prose, not code."""
    stripped = text.strip()
    if not stripped or len(stripped) < 20:
        return False
    # Code comment character strongly indicates code, not prose
    if "#" in stripped:
        return False
    # Code-like characters that prose rarely has in density
    code_chars = "=(){};:[]<>"
    code_count = sum(1 for c in stripped if c in code_chars)
    if code_count >= 2:
        return False
    first_word = stripped.split()[0].lower() if stripped.split() else ""
    if first_word in _PROSE_STARTERS:
        return True
    # High ratio of alphabetic + space (no operators) suggests prose
    alpha = sum(1 for c in stripped if c.isalpha() or c.isspace())
    if alpha / len(stripped) > 0.85 and code_count == 0:
        return True
    return False


def _extract_diff_additions(text: str) -> Optional[str]:
    """
    If text looks like unified diff format, extract only the added lines
    (lines starting with single '+'), stripping the '+' prefix.
    Return None if no valid additions found.
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


def _sanitize_suggestion(value: str) -> Optional[str]:
    """
    Validate and sanitize a suggestion so it contains only valid replacement code.
    Rejects natural language prose and strips unified diff format to additions only.
    Returns None if the suggestion is not valid code.
    """
    if not value or not isinstance(value, str):
        return None
    raw = value.rstrip()
    if not raw:
        return None

    # Detect unified diff format (headers or lines starting with + / -)
    has_diff_headers = (
        "--- a/" in raw or "--- b/" in raw
        or "+++ a/" in raw or "+++ b/" in raw
        or "\n@@ " in raw or raw.startswith("@@ ")
    )
    lines = raw.splitlines()
    diff_like_lines = sum(
        1 for ln in lines
        if ln.strip().startswith(("+", "-"))
        and not ln.strip().startswith(("++", "--"))
    )
    if has_diff_headers or (len(lines) > 1 and diff_like_lines >= 1):
        extracted = _extract_diff_additions(raw)
        if not extracted:
            return None
        if _looks_like_prose(extracted):
            return None
        return extracted

    # Single line starting with + (diff addition)
    if len(lines) == 1 and lines[0].startswith("+") and not lines[0].startswith("++"):
        content = lines[0][1:].strip()
        if not content or _looks_like_prose(content):
            return None
        return lines[0][1:]  # preserve original indentation after +

    # Reject prose-only suggestions
    if _looks_like_prose(raw):
        return None

    return raw


def _validate_review_item(item: dict) -> Optional[dict]:
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

    # Coerce line to int; accept missing/null as 0 (file-level comment).
    if line_val is None:
        line_val = 0
    try:
        line_val = int(line_val)
    except (TypeError, ValueError):
        line_val = 0

    # Normalize severity; default to "important" if unrecognized.
    if isinstance(severity_val, str):
        severity_val = severity_val.strip().lower()
    if severity_val not in VALID_SEVERITIES:
        severity_val = "important"

    # Validate and sanitize suggestion: must be valid code, not prose or raw diff
    normalized_suggestion = None
    if suggestion_val is not None and isinstance(suggestion_val, str):
        normalized_suggestion = _sanitize_suggestion(suggestion_val)

    result = {
        "file": file_val.strip(),
        "line": line_val,
        "severity": severity_val,
        "comment": comment_val.strip(),
    }

    if normalized_suggestion is not None:
        result["suggestion"] = normalized_suggestion

    return result


def parse_review_response(text: Optional[str]) -> List[dict]:
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
            "Failed to parse model response as JSON; returning no review items. "
            "Response preview: %s",
            cleaned[:200],
        )
        return []

    if isinstance(parsed, dict):
        # Some models wrap the array in an object like {"reviews": [...]}
        for key in ("reviews", "comments", "items", "review"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # Single item wrapped in an object
            parsed = [parsed]

    if not isinstance(parsed, list):
        logger.warning(
            "Model response JSON is not an array; returning no review items."
        )
        return []

    results: List[dict] = []
    for item in parsed:
        validated = _validate_review_item(item)
        if validated is not None:
            results.append(validated)

    logger.info("Parsed %d valid review item(s) from model response.", len(results))
    return results
