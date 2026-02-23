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

from src.review_formatter import format_review_comment
from src.review_parser import (REVIEW_SYSTEM_PROMPT, _validate_review_item,
                               parse_review_response)

# ---------------------------------------------------------------------------
# _validate_review_item - suggestion field validation
# ---------------------------------------------------------------------------

class TestValidateReviewItemWithSuggestion:
    """Test validation of review items with suggestion field."""

    def test_valid_item_with_suggestion(self):
        """Test that valid item with suggestion is properly validated."""
        item = {
            "file": "main.py",
            "line": 10,
            "severity": "critical",
            "comment": "Use list comprehension",
            "suggestion": "results = [x * 2 for x in items]"
        }
        result = _validate_review_item(item)
        assert result is not None
        assert result["file"] == "main.py"
        assert result["line"] == 10
        assert result["severity"] == "critical"
        assert result["comment"] == "Use list comprehension"
        assert result["suggestion"] == "results = [x * 2 for x in items]"

    def test_valid_item_without_suggestion(self):
        """Test that valid item without suggestion works as before."""
        item = {
            "file": "main.py",
            "line": 10,
            "severity": "critical",
            "comment": "Bug here"
        }
        result = _validate_review_item(item)
        assert result is not None
        assert result["file"] == "main.py"
        assert result["comment"] == "Bug here"
        assert "suggestion" not in result

    def test_suggestion_trailing_whitespace_trimmed(self):
        """Test that suggestion trailing whitespace is trimmed."""
        item = {
            "file": "main.py",
            "line": 10,
            "severity": "critical",
            "comment": "Fix this",
            "suggestion": "result = x + 1  \n  "
        }
        result = _validate_review_item(item)
        assert result is not None
        assert result["suggestion"] == "result = x + 1"

    def test_suggestion_leading_whitespace_preserved(self):
        """Test that suggestion leading whitespace is preserved."""
        item = {
            "file": "main.py",
            "line": 10,
            "severity": "critical",
            "comment": "Fix indented code",
            "suggestion": "    indented = True"
        }
        result = _validate_review_item(item)
        assert result is not None
        assert result["suggestion"] == "    indented = True"
        # Verify leading spaces are preserved
        assert result["suggestion"].startswith("    ")

    def test_empty_suggestion_excluded(self):
        """Test that empty suggestion string is excluded from result."""
        item = {
            "file": "main.py",
            "line": 10,
            "severity": "critical",
            "comment": "Fix this",
            "suggestion": "  "
        }
        result = _validate_review_item(item)
        assert result is not None
        assert "suggestion" not in result

    def test_null_suggestion_excluded(self):
        """Test that null suggestion is excluded from result."""
        item = {
            "file": "main.py",
            "line": 10,
            "severity": "critical",
            "comment": "Fix this",
            "suggestion": None
        }
        result = _validate_review_item(item)
        assert result is not None
        assert "suggestion" not in result

    def test_non_string_suggestion_excluded(self):
        """Test that non-string suggestion types are excluded."""
        for invalid_suggestion in [123, [], {}, True]:
            item = {
                "file": "main.py",
                "line": 10,
                "severity": "critical",
                "comment": "Fix this",
                "suggestion": invalid_suggestion
            }
            result = _validate_review_item(item)
            assert result is not None
            assert "suggestion" not in result

    def test_prose_suggestion_excluded(self):
        """Test that natural language prose in suggestion is excluded from result."""
        prose = (
            "Verify that all existing glob patterns and their usage throughout "
            "the codebase are compatible with glob v10. Thorough testing is recommended."
        )
        item = {
            "file": "package-lock.json",
            "line": 7239,
            "severity": "critical",
            "comment": "Major version update of glob",
            "suggestion": prose
        }
        result = _validate_review_item(item)
        assert result is not None
        assert result["comment"] == "Major version update of glob"
        assert "suggestion" not in result

    def test_diff_suggestion_sanitized_to_code_only(self):
        """Test that raw unified diff in suggestion is sanitized to added lines only."""
        raw_diff = (
            "--- a/package.json\n"
            "+++ b/package.json\n"
            "+ \"overrides\": {\n"
            "+ \"ajv\": \">=6.14.0\"\n"
            "+ }\n"
        )
        item = {
            "file": "package.json",
            "line": 45,
            "severity": "critical",
            "comment": "Add overrides for vulnerable deps",
            "suggestion": raw_diff
        }
        result = _validate_review_item(item)
        assert result is not None
        assert "suggestion" in result
        assert "---" not in result["suggestion"]
        assert "+++" not in result["suggestion"]
        assert '"ajv": ">=6.14.0"' in result["suggestion"]

    def test_multiline_suggestion(self):
        """Test that multiline suggestions are preserved."""
        multiline_code = "def foo():\n    return 42"
        item = {
            "file": "main.py",
            "line": 10,
            "severity": "critical",
            "comment": "Fix this function",
            "suggestion": multiline_code
        }
        result = _validate_review_item(item)
        assert result is not None
        assert result["suggestion"] == multiline_code

    def test_suggestion_with_special_characters(self):
        """Test that suggestions with special characters are handled."""
        code_with_special_chars = 'result = f"Hello {name}!"'
        item = {
            "file": "main.py",
            "line": 10,
            "severity": "trivial",
            "comment": "Use f-string",
            "suggestion": code_with_special_chars
        }
        result = _validate_review_item(item)
        assert result is not None
        assert result["suggestion"] == code_with_special_chars


# ---------------------------------------------------------------------------
# parse_review_response - with suggestions
# ---------------------------------------------------------------------------

class TestParseReviewResponseWithSuggestions:
    """Test parsing of review responses that include suggestions."""

    def test_single_item_with_suggestion(self):
        """Test parsing single review item with suggestion."""
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Use list comprehension",
            "suggestion": "result = [x * 2 for x in items]"
        }]
        text = json.dumps(items)
        result = parse_review_response(text)
        assert len(result) == 1
        assert result[0]["file"] == "a.py"
        assert result[0]["suggestion"] == "result = [x * 2 for x in items]"

    def test_multiple_items_mixed_suggestions(self):
        """Test parsing multiple items where some have suggestions and some don't."""
        items = [
            {
                "file": "a.py",
                "line": 1,
                "severity": "critical",
                "comment": "Security issue",
                "suggestion": "sanitized = html.escape(user_input)"
            },
            {
                "file": "b.py",
                "line": 20,
                "severity": "trivial",
                "comment": "Consider adding docstring"
            },
            {
                "file": "c.py",
                "line": 30,
                "severity": "important",
                "comment": "Fix the loop",
                "suggestion": "for item in items:\n    process(item)"
            }
        ]
        text = json.dumps(items)
        result = parse_review_response(text)
        assert len(result) == 3
        assert result[0]["suggestion"] == "sanitized = html.escape(user_input)"
        assert "suggestion" not in result[1]
        assert result[2]["suggestion"] == "for item in items:\n    process(item)"

    def test_markdown_wrapped_with_suggestion(self):
        """Test that markdown-wrapped JSON with suggestions is parsed correctly."""
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Improve code",
            "suggestion": "x = 1"
        }]
        text = f'```json\n{json.dumps(items)}\n```'
        result = parse_review_response(text)
        assert len(result) == 1
        assert result[0]["suggestion"] == "x = 1"

    def test_suggestion_with_code_block_content(self):
        """Test suggestion containing code with indentation and newlines."""
        suggestion_code = """def calculate(x, y):
    result = x + y
    return result"""
        items = [{
            "file": "calc.py",
            "line": 10,
            "severity": "important",
            "comment": "Add proper function",
            "suggestion": suggestion_code
        }]
        text = json.dumps(items)
        result = parse_review_response(text)
        assert len(result) == 1
        assert result[0]["suggestion"] == suggestion_code

    def test_empty_suggestion_filtered_out(self):
        """Test that items with empty suggestions don't include the field."""
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Fix this",
            "suggestion": "   "
        }]
        text = json.dumps(items)
        result = parse_review_response(text)
        assert len(result) == 1
        assert "suggestion" not in result[0]


# ---------------------------------------------------------------------------
# format_review_comment - with suggestions
# ---------------------------------------------------------------------------

class TestFormatReviewCommentWithSuggestions:
    """Test formatting of review comments with GitHub inline suggestions."""

    def test_single_item_with_suggestion(self):
        """Test formatting single item with suggestion."""
        items = [{
            "file": "a.py",
            "line": 10,
            "severity": "critical",
            "comment": "Use list comprehension",
            "suggestion": "results = [x * 2 for x in items]"
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert "**[CRITICAL]**" in result
        assert "`a.py:10`" in result
        assert "Use list comprehension" in result
        assert (
            "```suggestion\nresults = [x * 2 for x in items]\n```" in result
        )

    def test_single_item_without_suggestion(self):
        """Test that items without suggestions are formatted normally."""
        items = [{
            "file": "a.py",
            "line": 10,
            "severity": "critical",
            "comment": "Bug found"
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert "**[CRITICAL]**" in result
        assert "`a.py:10`" in result
        assert "Bug found" in result
        assert "```suggestion" not in result

    def test_multiple_items_with_mixed_suggestions(self):
        """Test formatting multiple items where some have suggestions."""
        items = [
            {
                "file": "a.py",
                "line": 1,
                "severity": "trivial",
                "comment": "Use f-string",
                "suggestion": 'message = f"Hello {name}"'
            },
            {
                "file": "b.py",
                "line": 2,
                "severity": "important",
                "comment": "Logic error - needs investigation"
            },
            {
                "file": "c.py",
                "line": 3,
                "severity": "critical",
                "comment": "Fix SQL injection",
                "suggestion": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
            }
        ]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        # Check all items are present
        assert "**[TRIVIAL]**" in result
        assert "**[IMPORTANT]**" in result
        assert "**[CRITICAL]**" in result

        # Check suggestions are formatted correctly
        assert '```suggestion\nmessage = f"Hello {name}"\n```' in result
        sql_suggestion = (
            "```suggestion\ncursor.execute"
            "('SELECT * FROM users WHERE id = ?', (user_id,))\n```"
        )
        assert sql_suggestion in result

        # Check item without suggestion doesn't have suggestion block
        lines = result.split("\n\n")
        important_line = [line for line in lines if "IMPORTANT" in line][0]
        assert "```suggestion" not in important_line

    def test_multiline_suggestion_formatting(self):
        """Test that multiline suggestions are formatted correctly."""
        multiline_code = "def foo():\n    x = 1\n    return x"
        items = [{
            "file": "a.py",
            "line": 10,
            "severity": "important",
            "comment": "Refactor function",
            "suggestion": multiline_code
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert "```suggestion" in result
        assert multiline_code in result
        assert "```" in result
        # Ensure proper markdown formatting
        expected_suggestion_block = f"```suggestion\n{multiline_code}\n```"
        assert expected_suggestion_block in result

    def test_multiple_chunks_with_suggestions(self):
        """Test formatting multiple chunks where items have suggestions."""
        chunk1 = json.dumps([{
            "file": "a.py",
            "line": 1,
            "severity": "trivial",
            "comment": "Style improvement",
            "suggestion": "x = 1"
        }])
        chunk2 = json.dumps([{
            "file": "b.py",
            "line": 2,
            "severity": "important",
            "comment": "Fix bug",
            "suggestion": "if x is not None:"
        }])
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk1, chunk2],
            min_severity="trivial"
        )
        assert "<details>" in result
        assert "```suggestion\nx = 1\n```" in result
        assert "```suggestion\nif x is not None:\n```" in result

    def test_file_level_comment_with_suggestion(self):
        """Test file-level comment (line 0) with suggestion."""
        items = [{
            "file": "a.py",
            "line": 0,
            "severity": "trivial",
            "comment": "Add module docstring",
            "suggestion": '"""Module for handling user data."""'
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert "`a.py`" in result
        assert "a.py:0" not in result  # Line 0 should not show :0
        assert '```suggestion\n"""Module for handling user data."""\n```' in result

    def test_suggestion_with_special_markdown_characters(self):
        """Test that suggestions with markdown chars are preserved."""
        suggestion = "# This is a comment\nresult = [x for x in range(10)]"
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Add comment",
            "suggestion": suggestion
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert suggestion in result
        assert "```suggestion" in result

    def test_severity_filtering_preserves_suggestions(self):
        """Test that severity filtering doesn't affect suggestions."""
        items = [
            {
                "file": "a.py",
                "line": 1,
                "severity": "critical",
                "comment": "Security issue",
                "suggestion": "sanitized = html.escape(input)"
            },
            {
                "file": "b.py",
                "line": 2,
                "severity": "trivial",
                "comment": "Style issue",
                "suggestion": "x = 1"
            }
        ]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="critical"
        )
        # Only critical should be included
        assert "**[CRITICAL]**" in result
        assert "**[TRIVIAL]**" not in result
        # Critical's suggestion should be present
        suggestion_block = (
            "```suggestion\nsanitized = html.escape(input)\n```"
        )
        assert suggestion_block in result
        # Trivial's suggestion should not be present
        assert "x = 1" not in result


# ---------------------------------------------------------------------------
# REVIEW_SYSTEM_PROMPT - suggestion field
# ---------------------------------------------------------------------------

class TestReviewSystemPromptWithSuggestion:
    """Test that system prompt includes suggestion field documentation."""

    def test_contains_suggestion_field(self):
        """Test that system prompt documents the suggestion field."""
        assert '"suggestion"' in REVIEW_SYSTEM_PROMPT

    def test_indicates_suggestion_is_optional(self):
        """Test that system prompt indicates suggestion is optional."""
        assert "optional" in REVIEW_SYSTEM_PROMPT.lower()

    def test_mentions_all_required_fields(self):
        """Test that all fields are still documented."""
        assert '"file"' in REVIEW_SYSTEM_PROMPT
        assert '"line"' in REVIEW_SYSTEM_PROMPT
        assert '"severity"' in REVIEW_SYSTEM_PROMPT
        assert '"comment"' in REVIEW_SYSTEM_PROMPT
        assert '"suggestion"' in REVIEW_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestSuggestionEdgeCases:
    """Test edge cases for suggestion handling."""

    def test_suggestion_with_backticks(self):
        """Test suggestion containing backticks."""
        suggestion = "result = `command`"
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Fix command",
            "suggestion": suggestion
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert suggestion in result

    def test_suggestion_with_triple_quotes(self):
        """Test suggestion containing triple quotes in code."""
        suggestion = 'docstring = """Example"""'
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Add docstring",
            "suggestion": suggestion
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert suggestion in result

    def test_suggestion_with_actual_triple_backticks(self):
        """Test suggestion containing triple backticks (markdown fence)."""
        suggestion = 'code = "```python\\nprint()\\n```"'
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Fix markdown",
            "suggestion": suggestion
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        # Should still contain the suggestion despite backticks
        assert suggestion in result

    def test_suggestion_with_unicode(self):
        """Test suggestion containing unicode characters."""
        suggestion = "message = 'Hello 世界 🌍'"
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Internationalization",
            "suggestion": suggestion
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert suggestion in result

    def test_empty_array_with_suggestion_schema(self):
        """Test that empty array response works with new schema."""
        result = parse_review_response("[]")
        assert not result

    def test_very_long_suggestion(self):
        """Test that very long suggestions are handled."""
        long_suggestion = "\n".join([f"line_{i} = {i}" for i in range(100)])
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "important",
            "comment": "Refactor this",
            "suggestion": long_suggestion
        }]
        chunk = json.dumps(items)
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=[chunk],
            min_severity="trivial"
        )
        assert "```suggestion" in result
        assert "line_0 = 0" in result
        assert "line_99 = 99" in result

    def test_suggestion_only_whitespace_and_newlines(self):
        """Test suggestion with only whitespace and newlines is excluded."""
        items = [{
            "file": "a.py",
            "line": 5,
            "severity": "trivial",
            "comment": "Fix this",
            "suggestion": "\n\n   \n  \n"
        }]
        text = json.dumps(items)
        result = parse_review_response(text)
        assert len(result) == 1
        assert "suggestion" not in result[0]
