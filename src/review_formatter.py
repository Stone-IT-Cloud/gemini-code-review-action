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
from typing import List

from loguru import logger

from src.review_parser import parse_review_response, strip_markdown_fences

# Severity mapping for filtering
SEVERITY_MAP = {"trivial": 1, "important": 2, "critical": 3}


def filter_by_severity(items: List[dict], min_severity: str) -> List[dict]:
    """Filter review items based on minimum severity threshold.

    Args:
        items: List of review items with 'severity' field
        min_severity: Minimum severity level (trivial, important, critical)

    Returns:
        Filtered list of items that meet the threshold
    """
    min_severity_normalized = min_severity.strip().lower()
    if min_severity_normalized not in SEVERITY_MAP:
        logger.warning(
            f"Unknown severity threshold '{min_severity}', defaulting to 'important'"
        )
        min_severity_normalized = "important"

    min_level = SEVERITY_MAP[min_severity_normalized]
    filtered: List[dict] = []

    for item in items:
        item_severity = item.get("severity", "important").lower()
        item_level = SEVERITY_MAP.get(item_severity, SEVERITY_MAP["important"])

        if item_level >= min_level:
            filtered.append(item)
        else:
            file_name = item.get("file", "unknown")
            logger.info(
                f"Skipping {item_severity.upper()} comment on file {file_name}"
            )

    return filtered


def format_review_comment(
    summarized_review: str, chunked_reviews: List[str], min_severity: str = "trivial"
) -> str:
    """Format reviews, parsing structured JSON when possible."""
    all_items: list = []
    any_parsed = False
    for chunk_text in chunked_reviews:
        parsed = parse_review_response(chunk_text)
        if parsed:
            all_items.extend(parsed)
            any_parsed = True
        else:
            # Check if it was valid JSON that just had no items (e.g. []).
            cleaned = strip_markdown_fences(chunk_text) if chunk_text else ""
            try:
                json.loads(cleaned)
                any_parsed = True
            except (json.JSONDecodeError, TypeError):
                pass

    # Apply severity filtering
    if all_items and min_severity:
        all_items = filter_by_severity(all_items, min_severity)

    if all_items:
        lines: List[str] = []
        for item in all_items:
            severity = item["severity"].upper()
            file_name = item["file"]
            line_num = item["line"]
            comment = item["comment"]
            loc = f"{file_name}:{line_num}" if line_num != 0 else file_name
            lines.append(f"**[{severity}]** `{loc}`: {comment}")
        structured_body = "\n\n".join(lines)
    elif any_parsed:
        # All chunks were valid JSON but had no review items.
        structured_body = ""
    else:
        # Fallback: use raw text if JSON parsing yielded nothing.
        structured_body = "\n".join(chunked_reviews) if chunked_reviews else ""

    if len(chunked_reviews) <= 1:
        return structured_body or summarized_review

    review = (
        f"<details>\n"
        f"    <summary>{summarized_review}</summary>\n"
        f"    {structured_body}\n"
        f"    </details>"
    )
    return review
