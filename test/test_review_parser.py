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

import pytest

from src.review_parser import (
    REVIEW_SYSTEM_PROMPT,
    _validate_review_item,
    parse_review_response,
    strip_markdown_fences,
)


# ---------------------------------------------------------------------------
# strip_markdown_fences
# ---------------------------------------------------------------------------

class TestStripMarkdownFences:
    def test_no_fences(self):
        raw = '[{"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"}]'
        assert strip_markdown_fences(raw) == raw

    def test_json_fences(self):
        raw = '```json\n[{"file": "a.py"}]\n```'
        assert strip_markdown_fences(raw) == '[{"file": "a.py"}]'

    def test_plain_fences(self):
        raw = '```\n[{"file": "a.py"}]\n```'
        assert strip_markdown_fences(raw) == '[{"file": "a.py"}]'

    def test_fences_with_surrounding_whitespace(self):
        raw = '  \n```json\n  [{"file": "a.py"}]  \n```  \n'
        assert strip_markdown_fences(raw) == '[{"file": "a.py"}]'

    def test_no_fences_returns_stripped(self):
        raw = '  [{"file": "a.py"}]  '
        assert strip_markdown_fences(raw) == '[{"file": "a.py"}]'


# ---------------------------------------------------------------------------
# _validate_review_item
# ---------------------------------------------------------------------------

class TestValidateReviewItem:
    def test_valid_item(self):
        item = {"file": "main.py", "line": 10, "severity": "critical", "comment": "Bug here"}
        result = _validate_review_item(item)
        assert result == {"file": "main.py", "line": 10, "severity": "critical", "comment": "Bug here"}

    def test_missing_file(self):
        assert _validate_review_item({"line": 1, "severity": "minor", "comment": "x"}) is None

    def test_empty_file(self):
        assert _validate_review_item({"file": "  ", "line": 1, "severity": "minor", "comment": "x"}) is None

    def test_missing_comment(self):
        assert _validate_review_item({"file": "a.py", "line": 1, "severity": "minor"}) is None

    def test_empty_comment(self):
        assert _validate_review_item({"file": "a.py", "line": 1, "severity": "minor", "comment": "  "}) is None

    def test_null_line_defaults_to_zero(self):
        result = _validate_review_item({"file": "a.py", "line": None, "severity": "minor", "comment": "ok"})
        assert result["line"] == 0

    def test_missing_line_defaults_to_zero(self):
        result = _validate_review_item({"file": "a.py", "severity": "minor", "comment": "ok"})
        assert result["line"] == 0

    def test_string_line_coerced(self):
        result = _validate_review_item({"file": "a.py", "line": "42", "severity": "minor", "comment": "ok"})
        assert result["line"] == 42

    def test_invalid_line_defaults_to_zero(self):
        result = _validate_review_item({"file": "a.py", "line": "abc", "severity": "minor", "comment": "ok"})
        assert result["line"] == 0

    def test_unknown_severity_defaults_to_suggestion(self):
        result = _validate_review_item({"file": "a.py", "line": 1, "severity": "unknown", "comment": "ok"})
        assert result["severity"] == "suggestion"

    def test_missing_severity_defaults_to_suggestion(self):
        result = _validate_review_item({"file": "a.py", "line": 1, "comment": "ok"})
        assert result["severity"] == "suggestion"

    def test_severity_case_insensitive(self):
        result = _validate_review_item({"file": "a.py", "line": 1, "severity": "MAJOR", "comment": "ok"})
        assert result["severity"] == "major"

    def test_non_dict_returns_none(self):
        assert _validate_review_item("not a dict") is None
        assert _validate_review_item(42) is None
        assert _validate_review_item(None) is None

    def test_whitespace_trimmed(self):
        result = _validate_review_item({"file": "  a.py  ", "line": 1, "severity": " minor ", "comment": " ok "})
        assert result == {"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"}

    def test_all_valid_severities(self):
        for severity in ("critical", "major", "minor", "suggestion"):
            result = _validate_review_item({"file": "a.py", "line": 1, "severity": severity, "comment": "ok"})
            assert result["severity"] == severity


# ---------------------------------------------------------------------------
# parse_review_response – valid JSON
# ---------------------------------------------------------------------------

class TestParseReviewResponseValidJson:
    def test_single_item_array(self):
        text = json.dumps([{"file": "a.py", "line": 5, "severity": "minor", "comment": "Looks odd"}])
        result = parse_review_response(text)
        assert len(result) == 1
        assert result[0]["file"] == "a.py"
        assert result[0]["line"] == 5

    def test_multiple_items(self):
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Bug"},
            {"file": "b.py", "line": 20, "severity": "suggestion", "comment": "Style"},
        ]
        result = parse_review_response(json.dumps(items))
        assert len(result) == 2

    def test_empty_array(self):
        result = parse_review_response("[]")
        assert result == []

    def test_filters_out_invalid_items(self):
        text = json.dumps([
            {"file": "a.py", "line": 1, "severity": "minor", "comment": "Good"},
            {"invalid": "item"},
            {"file": "b.py", "line": 2, "severity": "major", "comment": "Also good"},
        ])
        result = parse_review_response(text)
        assert len(result) == 2

    def test_object_with_reviews_key(self):
        text = json.dumps({"reviews": [
            {"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"}
        ]})
        result = parse_review_response(text)
        assert len(result) == 1

    def test_object_with_comments_key(self):
        text = json.dumps({"comments": [
            {"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"}
        ]})
        result = parse_review_response(text)
        assert len(result) == 1

    def test_object_with_items_key(self):
        text = json.dumps({"items": [
            {"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"}
        ]})
        result = parse_review_response(text)
        assert len(result) == 1

    def test_single_object_not_array(self):
        text = json.dumps({"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"})
        result = parse_review_response(text)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# parse_review_response – markdown-wrapped JSON
# ---------------------------------------------------------------------------

class TestParseReviewResponseMarkdownWrapped:
    def test_json_fenced(self):
        raw = '```json\n[{"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"}]\n```'
        result = parse_review_response(raw)
        assert len(result) == 1
        assert result[0]["file"] == "a.py"

    def test_plain_fenced(self):
        raw = '```\n[{"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"}]\n```'
        result = parse_review_response(raw)
        assert len(result) == 1

    def test_fenced_with_extra_whitespace(self):
        raw = '  ```json\n  [{"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"}]  \n  ```  '
        result = parse_review_response(raw)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# parse_review_response – malformed / edge cases
# ---------------------------------------------------------------------------

class TestParseReviewResponseMalformed:
    def test_none_input(self):
        assert parse_review_response(None) == []

    def test_empty_string(self):
        assert parse_review_response("") == []

    def test_whitespace_only(self):
        assert parse_review_response("   \n\t  ") == []

    def test_plain_text(self):
        assert parse_review_response("This is just a plain text review with no JSON.") == []

    def test_partial_json(self):
        assert parse_review_response('[{"file": "a.py", "line": 1') == []

    def test_json_number(self):
        assert parse_review_response("42") == []

    def test_json_string(self):
        assert parse_review_response('"hello"') == []

    def test_json_boolean(self):
        assert parse_review_response("true") == []

    def test_array_of_non_objects(self):
        assert parse_review_response("[1, 2, 3]") == []

    def test_array_of_strings(self):
        assert parse_review_response('["a", "b"]') == []

    def test_mixed_valid_and_non_object(self):
        text = json.dumps([
            {"file": "a.py", "line": 1, "severity": "minor", "comment": "ok"},
            "not an object",
            42,
        ])
        result = parse_review_response(text)
        assert len(result) == 1
        assert result[0]["file"] == "a.py"


# ---------------------------------------------------------------------------
# REVIEW_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

class TestReviewSystemPrompt:
    def test_contains_json_instruction(self):
        assert "JSON array" in REVIEW_SYSTEM_PROMPT

    def test_contains_schema_fields(self):
        assert '"file"' in REVIEW_SYSTEM_PROMPT
        assert '"line"' in REVIEW_SYSTEM_PROMPT
        assert '"severity"' in REVIEW_SYSTEM_PROMPT
        assert '"comment"' in REVIEW_SYSTEM_PROMPT

    def test_forbids_markdown(self):
        assert "Do not include any markdown" in REVIEW_SYSTEM_PROMPT

    def test_mentions_empty_array(self):
        assert "[]" in REVIEW_SYSTEM_PROMPT
