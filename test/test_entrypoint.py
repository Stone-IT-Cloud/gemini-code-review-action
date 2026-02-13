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
from src.prompts import get_review_prompt
from src.review_parser import REVIEW_SYSTEM_PROMPT


class TestGetReviewPrompt:
    def test_includes_system_prompt(self):
        prompt = get_review_prompt()
        assert REVIEW_SYSTEM_PROMPT in prompt

    def test_includes_extra_prompt(self):
        prompt = get_review_prompt(extra_prompt="Focus on security")
        assert "Focus on security" in prompt
        assert REVIEW_SYSTEM_PROMPT in prompt

    def test_empty_extra_prompt_not_included(self):
        prompt = get_review_prompt(extra_prompt="  ")
        # Should not have a blank extra section but still include system prompt
        assert REVIEW_SYSTEM_PROMPT in prompt


class TestFormatReviewComment:
    def test_single_chunk_with_valid_json(self):
        items = [{"file": "a.py", "line": 10, "severity": "critical", "comment": "Bug found"}]
        chunk = json.dumps(items)
        result = format_review_comment(summarized_review="Summary", chunked_reviews=[chunk])
        assert "**[CRITICAL]**" in result
        assert "`a.py:10`" in result
        assert "Bug found" in result

    def test_single_chunk_fallback_to_raw_text(self):
        raw = "This is plain text review."
        result = format_review_comment(summarized_review="Summary", chunked_reviews=[raw])
        assert result == raw

    def test_multiple_chunks_with_valid_json(self):
        chunk1 = json.dumps([{"file": "a.py", "line": 1, "severity": "minor", "comment": "Style"}])
        chunk2 = json.dumps([{"file": "b.py", "line": 2, "severity": "major", "comment": "Logic error"}])
        result = format_review_comment(summarized_review="Summary", chunked_reviews=[chunk1, chunk2])
        assert "<details>" in result
        assert "Summary" in result
        assert "**[MINOR]**" in result
        assert "**[MAJOR]**" in result

    def test_multiple_chunks_fallback_to_raw(self):
        result = format_review_comment(
            summarized_review="Summary",
            chunked_reviews=["plain text 1", "plain text 2"],
        )
        assert "<details>" in result
        assert "plain text 1" in result
        assert "plain text 2" in result

    def test_empty_chunks_returns_summary(self):
        result = format_review_comment(summarized_review="Summary", chunked_reviews=[])
        assert result == "Summary"

    def test_file_level_comment_no_line(self):
        items = [{"file": "a.py", "line": 0, "severity": "suggestion", "comment": "Consider refactoring"}]
        chunk = json.dumps(items)
        result = format_review_comment(summarized_review="Summary", chunked_reviews=[chunk])
        assert "`a.py`" in result
        assert "a.py:0" not in result

    def test_empty_json_array_falls_back_to_summary(self):
        result = format_review_comment(summarized_review="Summary", chunked_reviews=["[]"])
        assert result == "Summary"
