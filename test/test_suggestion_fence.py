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

from src.utils import create_suggestion_fence


class TestCreateSuggestionFence:
    """Test dynamic fence sizing for suggestion blocks."""

    def test_simple_suggestion_uses_triple_backticks(self):
        """Test that simple suggestions use standard triple backticks."""
        suggestion = "x = 1"
        result = create_suggestion_fence(suggestion)
        assert result == "\n```suggestion\nx = 1\n```"

    def test_suggestion_with_single_backtick(self):
        """Test suggestion containing single backtick."""
        suggestion = "result = `command`"
        result = create_suggestion_fence(suggestion)
        assert result == "\n```suggestion\nresult = `command`\n```"

    def test_suggestion_with_double_backticks(self):
        """Test suggestion containing double backticks."""
        suggestion = "code = ``value``"
        result = create_suggestion_fence(suggestion)
        # Should use at least 3 backticks
        assert result == "\n```suggestion\ncode = ``value``\n```"

    def test_suggestion_with_triple_backticks(self):
        """Test suggestion containing triple backticks."""
        suggestion = 'markdown = "```python\\ncode\\n```"'
        result = create_suggestion_fence(suggestion)
        # Should use 4 backticks to exceed the 3 found
        assert result.startswith("\n````suggestion\n")
        assert result.endswith("\n````")
        assert 'markdown = "```python\\ncode\\n```"' in result

    def test_suggestion_with_four_backticks(self):
        """Test suggestion containing four backticks."""
        suggestion = "code = ````fence````"
        result = create_suggestion_fence(suggestion)
        # Should use 5 backticks to exceed the 4 found
        assert result.startswith("\n`````suggestion\n")
        assert result.endswith("\n`````")

    def test_suggestion_with_multiple_backtick_runs(self):
        """Test suggestion with multiple different backtick sequences."""
        suggestion = "x = `a` and y = ```b``` and z = ``c``"
        result = create_suggestion_fence(suggestion)
        # Should use 4 backticks to exceed the max of 3
        assert result.startswith("\n````suggestion\n")
        assert result.endswith("\n````")

    def test_multiline_suggestion_with_backticks(self):
        """Test multiline suggestion containing backticks."""
        suggestion = "def foo():\n    '''Docstring with ```code```'''\n    pass"
        result = create_suggestion_fence(suggestion)
        assert result.startswith("\n````suggestion\n")
        assert suggestion in result
        assert result.endswith("\n````")

    def test_suggestion_preserves_content(self):
        """Test that content is preserved exactly."""
        suggestion = "line1\nline2\nline3"
        result = create_suggestion_fence(suggestion)
        assert suggestion in result

    def test_suggestion_with_six_backticks(self):
        """Test edge case with very long backtick run."""
        suggestion = "fence = ``````long``````"
        result = create_suggestion_fence(suggestion)
        # Should use 7 backticks
        assert result.startswith("\n```````suggestion\n")
        assert result.endswith("\n```````")

    def test_empty_suggestion_still_fenced(self):
        """Test that even empty suggestion gets fenced."""
        suggestion = ""
        result = create_suggestion_fence(suggestion)
        assert result == "\n```suggestion\n\n```"

    def test_suggestion_with_leading_whitespace(self):
        """Test that leading whitespace in suggestion is preserved."""
        suggestion = "    indented_code = True"
        result = create_suggestion_fence(suggestion)
        assert "    indented_code = True" in result
        # Verify indentation is preserved
        lines = result.split("\n")
        assert lines[2] == "    indented_code = True"

    def test_suggestion_with_mixed_indentation(self):
        """Test multiline code with different indentation levels."""
        suggestion = "def foo():\n    if x:\n        return True\n    return False"
        result = create_suggestion_fence(suggestion)
        # All indentation should be preserved
        assert suggestion in result
