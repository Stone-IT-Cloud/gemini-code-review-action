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

from src.review_formatter import filter_by_severity, format_review_comment


class TestFilterBySeverity:
    """Test severity filtering logic."""

    def test_filter_critical_only(self):
        """Test that CRITICAL filter only includes critical items."""
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Security issue"},
            {"file": "b.py", "line": 2, "severity": "important", "comment": "Logic error"},
            {"file": "c.py", "line": 3, "severity": "trivial", "comment": "Style issue"},
        ]
        result = filter_by_severity(items, "critical")
        assert len(result) == 1
        assert result[0]["severity"] == "critical"

    def test_filter_important_includes_critical(self):
        """Test that IMPORTANT filter includes both important and critical."""
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Security issue"},
            {"file": "b.py", "line": 2, "severity": "important", "comment": "Logic error"},
            {"file": "c.py", "line": 3, "severity": "trivial", "comment": "Style issue"},
        ]
        result = filter_by_severity(items, "important")
        assert len(result) == 2
        assert {item["severity"] for item in result} == {"critical", "important"}

    def test_filter_trivial_includes_all(self):
        """Test that TRIVIAL filter includes all items."""
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Security issue"},
            {"file": "b.py", "line": 2, "severity": "important", "comment": "Logic error"},
            {"file": "c.py", "line": 3, "severity": "trivial", "comment": "Style issue"},
        ]
        result = filter_by_severity(items, "trivial")
        assert len(result) == 3

    def test_filter_case_insensitive(self):
        """Test that severity filtering is case-insensitive."""
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Issue"},
        ]
        result = filter_by_severity(items, "CRITICAL")
        assert len(result) == 1

    def test_filter_unknown_severity_defaults_to_important(self):
        """Test that unknown severity threshold defaults to important."""
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Issue"},
            {"file": "b.py", "line": 2, "severity": "trivial", "comment": "Style"},
        ]
        result = filter_by_severity(items, "unknown_level")
        assert len(result) == 1  # Only critical passes important threshold

    def test_filter_empty_list(self):
        """Test filtering an empty list."""
        result = filter_by_severity([], "critical")
        assert result == []

    def test_filter_missing_severity_defaults_to_important(self):
        """Test that items with missing severity are treated as important."""
        items = [
            {"file": "a.py", "line": 1, "comment": "No severity"},
            {"file": "b.py", "line": 2, "severity": "trivial", "comment": "Style"},
        ]
        result = filter_by_severity(items, "important")
        assert len(result) == 1  # Item without severity is treated as important


class TestFormatReviewComment:
    def test_single_chunk_with_valid_json(self):
        items = [{"file": "a.py", "line": 10, "severity": "critical", "comment": "Bug found"}]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=[chunk], min_severity="trivial"
        )
        assert "**[CRITICAL]**" in result
        assert "`a.py:10`" in result
        assert "Bug found" in result

    def test_single_chunk_fallback_to_raw_text(self):
        raw = "This is plain text review."
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=[raw], min_severity="trivial"
        )
        assert result == raw

    def test_multiple_chunks_with_valid_json(self):
        chunk1 = json.dumps([{"file": "a.py", "line": 1, "severity": "trivial", "comment": "Style"}])
        chunk2 = json.dumps([{"file": "b.py", "line": 2, "severity": "important", "comment": "Logic error"}])
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk1, chunk2],
            min_severity="trivial",
        )
        assert "<details>" in result
        assert "Summary" in result
        assert "**[TRIVIAL]**" in result
        assert "**[IMPORTANT]**" in result

    def test_multiple_chunks_fallback_to_raw(self):
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=["plain text 1", "plain text 2"],
            min_severity="trivial",
        )
        assert "<details>" in result
        assert "plain text 1" in result
        assert "plain text 2" in result

    def test_empty_chunks_returns_summary(self):
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=[], min_severity="trivial"
        )
        assert result == "Summary"

    def test_file_level_comment_no_line(self):
        items = [{"file": "a.py", "line": 0, "severity": "trivial", "comment": "Consider refactoring"}]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=[chunk], min_severity="trivial"
        )
        assert "`a.py`" in result
        assert "a.py:0" not in result

    def test_empty_json_array_falls_back_to_summary(self):
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=["[]"], min_severity="trivial"
        )
        assert result == "Summary"

    def test_severity_filtering_critical(self):
        """Test that only critical items are included when filter is CRITICAL."""
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Security issue"},
            {"file": "b.py", "line": 2, "severity": "important", "comment": "Logic error"},
            {"file": "c.py", "line": 3, "severity": "trivial", "comment": "Style issue"},
        ]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=[chunk], min_severity="critical"
        )
        assert "**[CRITICAL]**" in result
        assert "**[IMPORTANT]**" not in result
        assert "**[TRIVIAL]**" not in result

    def test_severity_filtering_important(self):
        """Test that important and critical items are included when filter is IMPORTANT."""
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Security issue"},
            {"file": "b.py", "line": 2, "severity": "important", "comment": "Logic error"},
            {"file": "c.py", "line": 3, "severity": "trivial", "comment": "Style issue"},
        ]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=[chunk], min_severity="important"
        )
        assert "**[CRITICAL]**" in result
        assert "**[IMPORTANT]**" in result
        assert "**[TRIVIAL]**" not in result

    def test_severity_filtering_trivial(self):
        """Test that all items are included when filter is TRIVIAL."""
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Security issue"},
            {"file": "b.py", "line": 2, "severity": "important", "comment": "Logic error"},
            {"file": "c.py", "line": 3, "severity": "trivial", "comment": "Style issue"},
        ]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=[chunk], min_severity="trivial"
        )
        assert "**[CRITICAL]**" in result
        assert "**[IMPORTANT]**" in result
        assert "**[TRIVIAL]**" in result

    def test_all_items_filtered_returns_summary(self):
        """Test that when all items are filtered, summary is returned."""
        items = [
            {"file": "a.py", "line": 1, "severity": "trivial", "comment": "Style"},
            {"file": "b.py", "line": 2, "severity": "important", "comment": "Logic"},
        ]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary", chunked_reviews=[chunk], min_severity="critical"
        )
        # When all items filtered out and single chunk, should return summary
        assert result == "Summary"
