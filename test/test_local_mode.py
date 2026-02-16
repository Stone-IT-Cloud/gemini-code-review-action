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
"""Test local mode execution for pre-commit hooks."""
import os
import re
import subprocess
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.main import generate_diff_from_files, main, print_local_review


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class TestLocalMode:
    """Test local mode functionality."""

    # pylint: disable=unused-argument
    @patch("src.main.subprocess.run")
    @patch("src.main.genai")
    @patch("src.main.get_review")
    @patch("src.main.check_required_env_vars")
    def test_local_mode_with_no_staged_changes(
        self, mock_check_env, mock_get_review, mock_genai, mock_subprocess
    ):
        """Test that local mode exits cleanly when no staged changes."""
        # Mock subprocess to return empty diff
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)

        # Set required environment variable
        os.environ["GEMINI_API_KEY"] = "test-key"

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--local",
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command succeeded with exit code 0
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Verify get_review was NOT called (no diff to review)
            mock_get_review.assert_not_called()

        finally:
            # Cleanup
            for key in ["GEMINI_API_KEY", "LOCAL"]:
                if key in os.environ:
                    del os.environ[key]

    @patch("src.main.subprocess.run")
    @patch("src.main.genai")
    @patch("src.main.get_review")
    @patch("src.main.check_required_env_vars")
    @patch("src.main.print_local_review")
    def test_local_mode_with_critical_issues_exits_1(
        self,
        mock_print,
        mock_check_env,
        mock_get_review,
        mock_genai,
        mock_subprocess,
    ):
        """Test that local mode exits with code 1 when critical issues found."""
        # Mock subprocess to return a diff
        mock_subprocess.return_value = MagicMock(
            stdout="diff --git a/test.py b/test.py\n", returncode=0
        )

        # Mock review to return critical issue
        mock_get_review.return_value = (
            [
                '[{"file": "test.py", "line": 1, '
                '"severity": "critical", "comment": "Security issue"}]'
            ],
            "Critical security vulnerability found",
        )

        # Set required environment variable
        os.environ["GEMINI_API_KEY"] = "test-key"

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--local",
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command failed with exit code 1 (blocking commit)
            assert result.exit_code == 1, f"Expected exit code 1, got: {result.exit_code}"

            # Verify print_local_review was called
            mock_print.assert_called_once()

        finally:
            # Cleanup
            for key in ["GEMINI_API_KEY", "LOCAL"]:
                if key in os.environ:
                    del os.environ[key]

    @patch("src.main.subprocess.run")
    @patch("src.main.genai")
    @patch("src.main.get_review")
    @patch("src.main.check_required_env_vars")
    @patch("src.main.print_local_review")
    def test_local_mode_with_important_issues_exits_0(
        self,
        mock_print,
        mock_check_env,
        mock_get_review,
        mock_genai,
        mock_subprocess,
    ):
        """Test that local mode exits with code 0 when only important issues found."""
        # Mock subprocess to return a diff
        mock_subprocess.return_value = MagicMock(
            stdout="diff --git a/test.py b/test.py\n", returncode=0
        )

        # Mock review to return important issue (not critical)
        mock_get_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Logic error"}]'],
            "Some issues found",
        )

        # Set required environment variable
        os.environ["GEMINI_API_KEY"] = "test-key"

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--local",
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command succeeded with exit code 0
            assert result.exit_code == 0, f"Expected exit code 0, got: {result.exit_code}"

            # Verify print_local_review was called
            mock_print.assert_called_once()

        finally:
            # Cleanup
            for key in ["GEMINI_API_KEY", "LOCAL"]:
                if key in os.environ:
                    del os.environ[key]

    @patch("src.main.subprocess.run")
    @patch("src.main.genai")
    @patch("src.main.get_review")
    @patch("src.main.check_required_env_vars")
    @patch("src.main.print_local_review")
    def test_local_mode_with_no_issues_exits_0(
        self,
        mock_print,
        mock_check_env,
        mock_get_review,
        mock_genai,
        mock_subprocess,
    ):
        """Test that local mode exits with code 0 when no issues found."""
        # Mock subprocess to return a diff
        mock_subprocess.return_value = MagicMock(
            stdout="diff --git a/test.py b/test.py\n", returncode=0
        )

        # Mock review to return no issues
        mock_get_review.return_value = (
            ["[]"],
            "No issues found",
        )

        # Set required environment variable
        os.environ["GEMINI_API_KEY"] = "test-key"

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--local",
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command succeeded with exit code 0
            assert result.exit_code == 0, f"Expected exit code 0, got: {result.exit_code}"

            # Verify print_local_review was called
            mock_print.assert_called_once()

        finally:
            # Cleanup
            for key in ["GEMINI_API_KEY", "LOCAL"]:
                if key in os.environ:
                    del os.environ[key]

    @patch("src.main.subprocess.run")
    def test_generate_diff_from_files(self, mock_subprocess):
        """Test diff generation from file list."""
        # Mock subprocess to return diffs
        mock_subprocess.return_value = MagicMock(
            stdout="diff --git a/test.py b/test.py\n+new line\n",
            returncode=0,
        )

        files = ("test.py", "another.py")
        result = generate_diff_from_files(files)

        # Verify subprocess was called for each file
        assert mock_subprocess.call_count == 2
        assert "diff --git" in result

    @patch("src.main.subprocess.run")
    def test_generate_diff_from_files_handles_errors(self, mock_subprocess):
        """Test that diff generation handles git errors gracefully."""
        # Mock subprocess to raise CalledProcessError
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "git")

        files = ("test.py",)
        result = generate_diff_from_files(files)

        # Should return empty string on error
        assert result == ""


class TestPrintLocalReview:
    """Test the print_local_review function."""

    def test_print_local_review_no_issues(self, capsys):
        """Test output when no issues found."""
        print_local_review([], "All good!", "IMPORTANT")

        captured = capsys.readouterr()
        assert "No issues found" in captured.out
        assert "All good!" in captured.out
        assert "🤖 Gemini AI Code Review" in captured.out

    def test_print_local_review_with_critical_issue(self, capsys):
        """Test output with critical issue."""
        items = [
            {
                "file": "test.py",
                "line": 42,
                "severity": "critical",
                "comment": "SQL injection vulnerability",
                "suggestion": "Use parameterized queries",
            }
        ]

        print_local_review(items, "Security issue found", "IMPORTANT")

        captured = capsys.readouterr()
        output = strip_ansi_codes(captured.out)

        assert "CRITICAL" in output
        assert "test.py:42" in output
        assert "SQL injection" in output
        assert "Use parameterized queries" in output
        assert "Issue #1" in output

    def test_print_local_review_with_important_issue(self, capsys):
        """Test output with important issue."""
        items = [
            {
                "file": "app.py",
                "line": 10,
                "severity": "important",
                "comment": "Potential null pointer exception",
            }
        ]

        print_local_review(items, "Issues found", "TRIVIAL")

        captured = capsys.readouterr()
        output = strip_ansi_codes(captured.out)

        assert "IMPORTANT" in output
        assert "app.py:10" in output
        assert "null pointer" in output
        assert "Issue #1" in output

    def test_print_local_review_with_trivial_issue(self, capsys):
        """Test output with trivial issue."""
        items = [
            {
                "file": "utils.py",
                "line": 5,
                "severity": "trivial",
                "comment": "Missing docstring",
            }
        ]

        print_local_review(items, "", "TRIVIAL")

        captured = capsys.readouterr()
        output = strip_ansi_codes(captured.out)

        assert "TRIVIAL" in output
        assert "utils.py:5" in output
        assert "Missing docstring" in output
        assert "Issue #1" in output

    def test_print_local_review_with_multiple_issues(self, capsys):
        """Test output with multiple issues of different severities."""
        items = [
            {
                "file": "test.py",
                "line": 1,
                "severity": "critical",
                "comment": "Critical issue",
            },
            {
                "file": "test.py",
                "line": 2,
                "severity": "important",
                "comment": "Important issue",
            },
            {
                "file": "test.py",
                "line": 3,
                "severity": "trivial",
                "comment": "Trivial issue",
            },
        ]

        print_local_review(items, "Multiple issues", "TRIVIAL")

        captured = capsys.readouterr()
        assert "Found 3 issue(s)" in captured.out
        assert "1 CRITICAL" in captured.out
        assert "1 IMPORTANT" in captured.out
        assert "1 TRIVIAL" in captured.out
