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

from src.review_parser import (REVIEW_SYSTEM_PROMPT, _sanitize_suggestion,
                               _validate_review_item, parse_review_response,
                               strip_markdown_fences)

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
# _sanitize_suggestion
# ---------------------------------------------------------------------------

class TestSanitizeSuggestion:
    """Test suggestion sanitization: prose rejected, diff extracted, code preserved."""

    def test_natural_language_glob_rejected(self):
        text = (
            "Verify that all existing glob patterns and their usage throughout "
            "the codebase are compatible with glob v10. Thorough testing is recommended."
        )
        assert _sanitize_suggestion(text) is None

    def test_natural_language_eslint_rejected(self):
        text = (
            "Review the official ESLint 10 migration guide, update ESLint configurations "
            "to be compatible with the new version, and resolve any new linting errors."
        )
        assert _sanitize_suggestion(text) is None

    def test_natural_language_short_rejected(self):
        text = "Ensure all custom rules and plugins are compatible with ESLint v10."
        assert _sanitize_suggestion(text) is None

    def test_unified_diff_extracted(self):
        raw = (
            "--- a/package.json\n"
            "+++ b/package.json\n"
            "@@ -69,6 +69,7 @@\n"
            " },\n"
            '+ "overrides": {\n'
            '+ "minimatch": ">=10.2.1",\n'
            '+ "ajv": ">=6.14.0",\n'
            '+ "test-exclude": ">=8.0.0"\n'
            "+ }\n"
            "+ }"
        )
        result = _sanitize_suggestion(raw)
        assert result is not None
        assert "---" not in result
        assert "+++" not in result
        assert "@@" not in result
        assert '"ajv": ">=6.14.0"' in result
        assert result.strip().startswith('"overrides"')

    def test_single_line_diff_plus_extracted(self):
        raw = '+  "ajv": ">=6.14.0",'
        result = _sanitize_suggestion(raw)
        assert result is not None
        assert result == '  "ajv": ">=6.14.0",'

    def test_valid_code_preserved(self):
        code = "results = [x * 2 for x in items]"
        assert _sanitize_suggestion(code) == code

    def test_valid_code_multiline_preserved(self):
        code = "def foo():\n    return 42"
        assert _sanitize_suggestion(code) == code

    def test_valid_code_with_capital_letter_preserved(self):
        code = "Return True  # Python keyword as string in comment"
        assert _sanitize_suggestion(code) == code

    def test_valid_json_line_preserved(self):
        code = '"minimatch": ">=10.2.1"'
        assert _sanitize_suggestion(code) == code

    def test_empty_returns_none(self):
        assert _sanitize_suggestion("") is None
        assert _sanitize_suggestion("   \n  ") is None

    def test_none_returns_none(self):
        assert _sanitize_suggestion(None) is None  # type: ignore[arg-type]

    def test_non_string_returns_none(self):
        assert _sanitize_suggestion(123) is None  # type: ignore[arg-type]

    def test_diff_with_only_metadata_returns_none(self):
        raw = "--- a/package.json\n+++ b/package.json\n@@ -1,1 +1,1 @@"
        result = _sanitize_suggestion(raw)
        assert result is None

    def test_prose_after_diff_extraction_rejected(self):
        raw = (
            "--- a/x\n+++ b/x\n"
            "+ Verify that all existing glob patterns are compatible with glob v10."
        )
        result = _sanitize_suggestion(raw)
        assert result is None


# ---------------------------------------------------------------------------
# _validate_review_item
# ---------------------------------------------------------------------------

class TestValidateReviewItem:
    def test_valid_item(self):
        item = {"file": "main.py", "line": 10, "severity": "critical", "comment": "Bug here"}
        result = _validate_review_item(item)
        assert result == {"file": "main.py", "line": 10, "severity": "critical", "comment": "Bug here"}

    def test_missing_file(self):
        assert _validate_review_item({"line": 1, "severity": "trivial", "comment": "x"}) is None

    def test_empty_file(self):
        assert _validate_review_item({"file": "  ", "line": 1, "severity": "trivial", "comment": "x"}) is None

    def test_missing_comment(self):
        assert _validate_review_item({"file": "a.py", "line": 1, "severity": "trivial"}) is None

    def test_empty_comment(self):
        assert _validate_review_item({"file": "a.py", "line": 1, "severity": "trivial", "comment": "  "}) is None

    def test_null_line_defaults_to_zero(self):
        result = _validate_review_item({"file": "a.py", "line": None, "severity": "trivial", "comment": "ok"})
        assert result["line"] == 0

    def test_missing_line_defaults_to_zero(self):
        result = _validate_review_item({"file": "a.py", "severity": "trivial", "comment": "ok"})
        assert result["line"] == 0

    def test_string_line_coerced(self):
        result = _validate_review_item({"file": "a.py", "line": "42", "severity": "trivial", "comment": "ok"})
        assert result["line"] == 42

    def test_invalid_line_defaults_to_zero(self):
        result = _validate_review_item({"file": "a.py", "line": "abc", "severity": "trivial", "comment": "ok"})
        assert result["line"] == 0

    def test_unknown_severity_defaults_to_important(self):
        result = _validate_review_item({"file": "a.py", "line": 1, "severity": "unknown", "comment": "ok"})
        assert result["severity"] == "important"

    def test_missing_severity_defaults_to_important(self):
        result = _validate_review_item({"file": "a.py", "line": 1, "comment": "ok"})
        assert result["severity"] == "important"

    def test_severity_case_insensitive(self):
        result = _validate_review_item({"file": "a.py", "line": 1, "severity": "IMPORTANT", "comment": "ok"})
        assert result["severity"] == "important"

    def test_non_dict_returns_none(self):
        assert _validate_review_item("not a dict") is None
        assert _validate_review_item(42) is None
        assert _validate_review_item(None) is None

    def test_whitespace_trimmed(self):
        result = _validate_review_item({"file": "  a.py  ", "line": 1, "severity": " trivial ", "comment": " ok "})
        assert result == {"file": "a.py", "line": 1, "severity": "trivial", "comment": "ok"}

    def test_all_valid_severities(self):
        for severity in ("critical", "important", "trivial"):
            result = _validate_review_item({"file": "a.py", "line": 1, "severity": severity, "comment": "ok"})
            assert result["severity"] == severity


# ---------------------------------------------------------------------------
# parse_review_response – valid JSON
# ---------------------------------------------------------------------------

class TestParseReviewResponseValidJson:
    def test_single_item_array(self):
        text = json.dumps([{"file": "a.py", "line": 5, "severity": "trivial", "comment": "Looks odd"}])
        result = parse_review_response(text)
        assert len(result) == 1
        assert result[0]["file"] == "a.py"
        assert result[0]["line"] == 5

    def test_multiple_items(self):
        items = [
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "Bug"},
            {"file": "b.py", "line": 20, "severity": "trivial", "comment": "Style"},
        ]
        result = parse_review_response(json.dumps(items))
        assert len(result) == 2

    def test_empty_array(self):
        result = parse_review_response("[]")
        assert not result

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
        assert not parse_review_response(None)

    def test_empty_string(self):
        assert not parse_review_response("")

    def test_whitespace_only(self):
        assert not parse_review_response("   \n\t  ")

    def test_plain_text(self):
        assert not parse_review_response("This is just a plain text review with no JSON.")

    def test_partial_json(self):
        assert not parse_review_response('[{"file": "a.py", "line": 1')

    def test_json_number(self):
        assert not parse_review_response("42")

    def test_json_string(self):
        assert not parse_review_response('"hello"')

    def test_json_boolean(self):
        assert not parse_review_response("true")

    def test_array_of_non_objects(self):
        assert not parse_review_response("[1, 2, 3]")

    def test_array_of_strings(self):
        assert not parse_review_response('["a", "b"]')

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

    def test_suggestion_must_be_code_not_prose_or_diff(self):
        assert "exact replacement code" in REVIEW_SYSTEM_PROMPT
        assert "Never put natural language" in REVIEW_SYSTEM_PROMPT
        assert "Never use unified diff format" in REVIEW_SYSTEM_PROMPT
