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
"""Review formatting helpers — severity filtering and comment rendering."""

import json
import re

from loguru import logger

from code_reviewer.review_parser import parse_review_response, strip_markdown_fences
from code_reviewer.utils import create_suggestion_fence

# Severity mapping for filtering
SEVERITY_MAP = {"trivial": 1, "important": 2, "critical": 3}


def filter_by_severity(items: list[dict], min_severity: str) -> list[dict]:
    """Filter review items based on minimum severity threshold.

    Args:
        items: List of review items with 'severity' field
        min_severity: Minimum severity level (trivial, important, critical)

    Returns:
        Filtered list of items that meet the threshold
    """
    min_severity_normalized = min_severity.strip().lower()
    if min_severity_normalized not in SEVERITY_MAP:
        logger.warning(f"Unknown severity threshold '{min_severity}', defaulting to 'important'")
        min_severity_normalized = "important"

    min_level = SEVERITY_MAP[min_severity_normalized]
    filtered: list[dict] = []

    for item in items:
        item_severity = item.get("severity", "important").lower()
        item_level = SEVERITY_MAP.get(item_severity, SEVERITY_MAP["important"])

        if item_level >= min_level:
            filtered.append(item)
        else:
            file_name = item.get("file", "unknown")
            logger.info(f"Skipping {item_severity.upper()} comment on file {file_name}")

    return filtered


def _format_item_line(item: dict) -> str:
    """Format a single review item into a Markdown string with optional suggestion."""
    severity = item["severity"].upper()
    file_name = item["file"]
    line_num = item["line"]
    comment = item["comment"]
    suggestion = item.get("suggestion")

    loc = f"{file_name}:{line_num}" if line_num != 0 else file_name
    formatted = f"**[{severity}]** `{loc}`: {comment}"
    if suggestion:
        formatted += create_suggestion_fence(suggestion)
    return formatted


def _collect_review_items(chunked_reviews: list[str]) -> tuple[list[dict], bool]:
    """Parse chunked review texts into validated items.

    Returns:
        (all_items, any_parsed) where any_parsed is True if at least one chunk
        contained valid JSON (even if the JSON array was empty).
    """
    all_items: list[dict] = []
    any_parsed = False
    for chunk_text in chunked_reviews:
        parsed = parse_review_response(chunk_text)
        if parsed:
            all_items.extend(parsed)
            any_parsed = True
        else:
            cleaned = strip_markdown_fences(chunk_text) if chunk_text else ""
            try:
                json.loads(cleaned)
                any_parsed = True
            except (json.JSONDecodeError, TypeError):
                pass
    return all_items, any_parsed


# ---------------------------------------------------------------------------
# Structured review summary
# ---------------------------------------------------------------------------

SEVERITY_ICONS = {"critical": "🔴", "important": "🟡", "trivial": "🔵"}


def _parse_diff_stats(diff: str) -> dict[str, int]:
    """Parse diff stats from a unified diff string.

    Returns:
        Dict with ``additions``, ``deletions``, and ``files`` counts.
    """
    # Count lines starting with + (but not +++ which is file headers)
    additions = len(re.findall(r"^\+[^+]", diff, re.MULTILINE))
    # Count lines starting with - (but not --- which is file headers)
    deletions = len(re.findall(r"^\-[^-]", diff, re.MULTILINE))
    files = len(re.findall(r"^\+\+\+ b/", diff, re.MULTILINE))
    return {"additions": additions, "deletions": deletions, "files": files}


def _count_by_severity(all_items: list[dict]) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Count items per severity and keep up to two short descriptions each."""
    counts: dict[str, int] = {}
    short_descriptions: dict[str, list[str]] = {}
    for item in all_items:
        sev = item.get("severity", "important").lower()
        counts.setdefault(sev, 0)
        short_descriptions.setdefault(sev, [])
        counts[sev] += 1
        if len(short_descriptions[sev]) < 2:
            short_descriptions[sev].append((item.get("comment") or "")[:60])
    return counts, short_descriptions


def _diff_stats_line(diff_stats: dict[str, int]) -> str:
    """Render the diff-stats header line."""
    additions = diff_stats.get("additions", 0)
    deletions = diff_stats.get("deletions", 0)
    files = diff_stats.get("files", 0)
    return f"📋 Review: +{additions} / -{deletions} líneas, {files} archivos"


def build_review_summary(
    all_items: list[dict],
    diff_stats: dict[str, int] | None = None,
) -> str:
    """Build a structured summary header for the review.

    Args:
        all_items: List of review items with ``severity`` and ``comment`` keys.
        diff_stats: Optional dict with ``additions``, ``deletions``, ``files``.

    Returns:
        A markdown summary string, empty if there's no data.
    """
    if not all_items:
        if diff_stats:
            return f"{_diff_stats_line(diff_stats)}\n✅ No se encontraron issues."
        return "✅ No se encontraron issues."

    counts, short_descriptions = _count_by_severity(all_items)
    lines = [_diff_stats_line(diff_stats) if diff_stats else "📋 Review"]
    for sev in ("critical", "important", "trivial"):
        if sev in counts:
            icon = SEVERITY_ICONS.get(sev, "•")
            descs = short_descriptions.get(sev, [])
            desc_str = f" — {', '.join(descs)}" if descs else ""
            lines.append(f"{icon} {counts[sev]} {sev.upper()}{desc_str}")
    return "\n".join(lines)


def _structured_body(
    chunked_reviews: list[str],
    summarized_review: str,
    min_severity: str,
) -> tuple[str, list[dict]]:
    """Resolve the per-item body, falling back to raw text when nothing parses."""
    all_items, any_parsed = _collect_review_items(chunked_reviews)
    if all_items and min_severity:
        all_items = filter_by_severity(all_items, min_severity)
    if all_items:
        return "\n\n".join(_format_item_line(item) for item in all_items), all_items
    if any_parsed:
        return _fallback_from_summary(summarized_review, min_severity)
    return ("\n".join(chunked_reviews) if chunked_reviews else ""), all_items


def _fallback_from_summary(summarized_review: str, min_severity: str) -> tuple[str, list[dict]]:
    """Parse the summary itself when the chunk payload was valid JSON but unusable."""
    stripped = summarized_review.strip()
    if not (stripped.startswith("[") or stripped.startswith("{")):
        return "", []
    from code_reviewer.review_parser import parse_review_response  # pylint: disable=import-outside-toplevel

    items = parse_review_response(summarized_review)
    if items and min_severity:
        items = filter_by_severity(items, min_severity)
    if not items:
        return "", []
    return "\n\n".join(_format_item_line(item) for item in items), items


def format_review_comment(
    summarized_review: str,
    chunked_reviews: list[str],
    min_severity: str = "trivial",
    diff: str | None = None,
) -> str:
    """Format reviews, parsing structured JSON when possible.

    Args:
        summarized_review: Summarized review text.
        chunked_reviews: List of chunked review texts.
        min_severity: Minimum severity threshold.
        diff: Optional full diff string. When provided, a structured summary
              header is prepended.

    Returns:
        Formatted review comment string.
    """
    structured_body, all_items = _structured_body(chunked_reviews, summarized_review, min_severity)
    body = _wrap_body(structured_body, summarized_review, chunked_reviews)

    if diff is not None:
        summary = build_review_summary(all_items=all_items, diff_stats=_parse_diff_stats(diff))
        if summary:
            body = f"{summary}\n\n---\n\n{body}"

    if not body.strip() or body.strip().startswith("["):
        level_label = min_severity.upper() if min_severity else "TRIVIAL"
        body = f"✅ Code review complete — no issues found at **{level_label}** level."
    return body


def _wrap_body(
    structured_body: str,
    summarized_review: str,
    chunked_reviews: list[str],
) -> str:
    """Pick between the structured body, the raw summary and a details wrapper."""
    if len(chunked_reviews) > 1:
        return f"<details>\n    <summary>{summarized_review}</summary>\n    {structured_body}\n    </details>"
    if structured_body:
        return structured_body
    stripped = summarized_review.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return ""
    return summarized_review
