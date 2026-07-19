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
"""Tests for structured review summary in review_formatter.py."""

from src.review_formatter import _parse_diff_stats, build_review_summary


# ---------------------------------------------------------------------------
# _parse_diff_stats
# ---------------------------------------------------------------------------


class TestParseDiffStats:
    def test_empty_diff(self):
        assert _parse_diff_stats("") == {"additions": 0, "deletions": 0, "files": 0}

    def test_additions_and_deletions(self):
        diff = """--- a/a.py
+++ b/a.py
@@ -1 +1,2 @@
 old
+new line
+another new
-removed line"""
        stats = _parse_diff_stats(diff)
        assert stats["additions"] == 2
        assert stats["deletions"] == 1

    def test_files_counted(self):
        diff = """--- a/a.py
+++ b/b.py
@@ -1 +1,2 @@
+added
--- a/c.py
+++ b/d.py
@@ -1 +1,2 @@
+added"""
        stats = _parse_diff_stats(diff)
        assert stats["files"] == 2

    def test_no_changes(self):
        diff = """--- a/a.py
+++ b/a.py
@@ -1 +1 @@
 same"""
        stats = _parse_diff_stats(diff)
        assert stats["additions"] == 0
        assert stats["deletions"] == 0


# ---------------------------------------------------------------------------
# build_review_summary
# ---------------------------------------------------------------------------


class TestBuildReviewSummary:
    def test_mixed_severities(self):
        items = [
            {"severity": "critical", "comment": "SQL injection"},
            {"severity": "critical", "comment": "Null pointer"},
            {"severity": "important", "comment": "Wrong logic"},
            {"severity": "trivial", "comment": "Unused import"},
        ]
        result = build_review_summary(items, {"additions": 50, "deletions": 30, "files": 2})
        assert "📋 Review:" in result
        assert "+50 / -30" in result
        assert "2 archivos" in result
        assert "🔴" in result
        assert "🟡" in result
        assert "🔵" in result

    def test_only_important(self):
        items = [
            {"severity": "important", "comment": "Bug"},
            {"severity": "important", "comment": "Performance"},
        ]
        result = build_review_summary(items, {"additions": 10, "deletions": 5, "files": 1})
        assert "🔴" not in result
        assert "🟡" in result
        assert "🔵" not in result

    def test_no_items(self):
        result = build_review_summary([], {"additions": 0, "deletions": 0, "files": 0})
        assert "✅ No se encontraron issues" in result

    def test_single_critical(self):
        items = [{"severity": "critical", "comment": "Security hole"}]
        result = build_review_summary(items, {"additions": 1, "deletions": 0, "files": 1})
        assert "🔴 1 CRITICAL" in result
        assert "🟡" not in result
        assert "🔵" not in result

    def test_without_diff_stats(self):
        items = [{"severity": "important", "comment": "Bug"}]
        result = build_review_summary(items)
        assert "📋 Review" in result
        assert "no stats provided" not in result.lower()

    def test_filtered_items_only_include_severity(self):
        """Items without severity are treated as important."""
        items = [{"severity": "critical", "comment": "X"}, {"comment": "Y"}]
        result = build_review_summary(items, {"additions": 10, "deletions": 0, "files": 1})
        assert "🔴 1 CRITICAL" in result
        assert "🟡 1 IMPORTANT" in result


# ---------------------------------------------------------------------------
# Integration with format_review_comment
# ---------------------------------------------------------------------------


class TestFormatReviewCommentWithSummary:
    def test_summary_prepended(self):
        from src.review_formatter import format_review_comment

        result = format_review_comment(
            summarized_review="Summary text",
            chunked_reviews=[
                '[{"file": "a.py", "line": 1, "severity": "critical", "comment": "X"}]'
            ],
            min_severity="trivial",
            diff="--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n+new\n",
        )
        assert "📋 Review:" in result
        assert "🔴 1 CRITICAL" in result

    def test_summary_not_added_when_no_diff_provided(self):
        from src.review_formatter import format_review_comment

        result = format_review_comment(
            summarized_review="Summary text",
            chunked_reviews=[
                '[{"file": "a.py", "line": 1, "severity": "critical", "comment": "X"}]'
            ],
            min_severity="trivial",
        )
        # Without diff param, backward compatible — no summary
        assert "📋 Review:" not in result

    def test_fallback_parses_summarized_review_json(self):
        """When items fail validation but summarized_review has valid JSON,
        the fallback in any_parsed branch should parse and format them."""
        from src.review_formatter import format_review_comment
        from src.review_parser import _validate_review_item

        # Craft JSON that's valid but items fail validation (empty comment)
        chunk = '[{"file": "a.py", "line": 1, "severity": "trivial", "comment": ""}]'
        # Verify items are invalid on their own
        import json
        assert _validate_review_item(json.loads(chunk)[0]) is None

        # summarized_review has the same items, but shorter
        result = format_review_comment(
            summarized_review='[{"file": "a.py", "line": 1, "severity": "trivial", "comment": "X"}]',
            chunked_reviews=[chunk],
            min_severity="trivial",
        )
        # Without diff param, no summary header
        # The fallback should have parsed summarized_review
        assert "TRIVIAL" in result
        assert "a.py" in result
