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
    "You are an expert code reviewer. Your task is to analyze the provided code changes.\n"
    "You must output your review strictly as a JSON array of objects.\n"
    "Do not include any markdown formatting (like ```json).\n"
    "\n"
    "Severity Classification:\n"
    "- TRIVIAL: Style issues, formatting, minor refactoring, missing docstrings.\n"
    "- IMPORTANT: Logic errors, potential bugs, performance inefficiencies (e.g., O(n^2)), bad practices.\n"
    "- CRITICAL: Security vulnerabilities (SQLi, XSS), potential crashes, breaking changes, data loss risks.\n"
    "\n"
    "Use the following schema for each review item:\n"
    "[\n"
    "  {\n"
    '    "file": "filename.py",\n'
    '    "line": <line_number_as_integer>,\n'
    '    "severity": "TRIVIAL | IMPORTANT | CRITICAL",\n'
    '    "comment": "Your review comment here"\n'
    "  }\n"
    "]\n"
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

    return {
        "file": file_val.strip(),
        "line": line_val,
        "severity": severity_val,
        "comment": comment_val.strip(),
    }


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
