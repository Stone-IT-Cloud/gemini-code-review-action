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
"""Test that review-level CLI parameter works correctly."""

import os
import tempfile
from unittest.mock import patch

from click.testing import CliRunner

from src.main import main


class TestReviewLevelCLI:
    """Test review_level CLI parameter."""

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_review_level_from_cli_parameter(self, _mock_check_env, mock_format, mock_run_review):
        """Test that --review-level CLI parameter is used."""
        # Setup mocks - use IMPORTANT instead of CRITICAL to avoid exit(1)
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Bug"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        # Set required environment variable
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        # Create a test diff file using tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            f.write("diff --git a/test.py b/test.py\n")

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--diff-file",
                    test_diff_file,
                    "--review-level",
                    "CRITICAL",
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command succeeded (no critical issues)
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Verify format_review_comment was called with CRITICAL
            mock_format.assert_called_once()
            call_kwargs = mock_format.call_args[1]
            assert call_kwargs["min_severity"] == "CRITICAL"

        finally:
            # Cleanup
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL", "REVIEW_LEVEL"]:
                if key in os.environ:
                    del os.environ[key]

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_review_level_defaults_to_env_var(self, _mock_check_env, mock_format, mock_run_review):
        """Test that environment variable is used when CLI param not provided."""
        # Setup mocks
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Issue"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        # Set environment variables
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"
        os.environ["REVIEW_LEVEL"] = "TRIVIAL"

        # Create a test diff file using tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            f.write("diff --git a/test.py b/test.py\n")

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--diff-file",
                    test_diff_file,
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command succeeded
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Verify format_review_comment was called with TRIVIAL from env
            mock_format.assert_called_once()
            call_kwargs = mock_format.call_args[1]
            assert call_kwargs["min_severity"] == "TRIVIAL"

        finally:
            # Cleanup
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL", "REVIEW_LEVEL"]:
                if key in os.environ:
                    del os.environ[key]

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_review_level_cli_overrides_env(self, _mock_check_env, mock_format, mock_run_review):
        """Test that CLI parameter takes precedence over environment variable."""
        # Setup mocks - use IMPORTANT instead of CRITICAL to avoid exit(1)
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Bug"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        # Set environment variables - env says TRIVIAL
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"
        os.environ["REVIEW_LEVEL"] = "TRIVIAL"

        # Create a test diff file using tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            f.write("diff --git a/test.py b/test.py\n")

        try:
            runner = CliRunner()
            # But CLI says CRITICAL - should win
            result = runner.invoke(
                main,
                [
                    "--diff-file",
                    test_diff_file,
                    "--review-level",
                    "CRITICAL",
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command succeeded (no critical issues)
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Verify format_review_comment was called with CRITICAL (CLI wins)
            mock_format.assert_called_once()
            call_kwargs = mock_format.call_args[1]
            assert call_kwargs["min_severity"] == "CRITICAL"

        finally:
            # Cleanup
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL", "REVIEW_LEVEL"]:
                if key in os.environ:
                    del os.environ[key]

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_review_level_defaults_to_important(self, _mock_check_env, mock_format, mock_run_review):
        """Test that default is IMPORTANT when neither CLI nor env is set."""
        # Setup mocks
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Issue"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        # Set minimal environment variables - no REVIEW_LEVEL
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        # Create a test diff file using tempfile (200 lines → medium diff → IMPORTANT)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            for i in range(200):
                f.write(f"diff line {i}\n")

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--diff-file",
                    test_diff_file,
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command succeeded
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Verify format_review_comment was called with default IMPORTANT
            mock_format.assert_called_once()
            call_kwargs = mock_format.call_args[1]
            assert call_kwargs["min_severity"] == "IMPORTANT"

        finally:
            # Cleanup
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL"]:
                if key in os.environ:
                    del os.environ[key]

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_auto_adjust_trivial_for_small_diff(self, _mock_check_env, mock_format, mock_run_review):
        """Auto-adjust to TRIVIAL when diff < 50 lines and no level specified."""
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "trivial", "comment": "Style"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            f.write("\n".join(f"diff line {i}" for i in range(10)))

        try:
            runner = CliRunner()
            result = runner.invoke(main, ["--diff-file", test_diff_file, "--model", "test-model"])
            assert result.exit_code == 0, f"Command failed: {result.output}"
            mock_format.assert_called_once()
            assert mock_format.call_args[1]["min_severity"] == "TRIVIAL"
        finally:
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL", "REVIEW_LEVEL"]:
                os.environ.pop(key, None)

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_auto_adjust_critical_for_large_diff(self, _mock_check_env, mock_format, mock_run_review):
        """Auto-adjust to CRITICAL when diff > 500 lines and no level specified.
        CRITICAL level + CRITICAL items causes exit 1 (blocks commit)."""
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "critical", "comment": "Security"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            for i in range(600):
                f.write(f"diff line {i}\n")

        try:
            runner = CliRunner()
            result = runner.invoke(main, ["--diff-file", test_diff_file, "--model", "test-model"])
            assert result.exit_code == 1, f"Expected exit 1 (critical), got {result.exit_code}"
            mock_format.assert_called_once()
            assert mock_format.call_args[1]["min_severity"] == "CRITICAL"
        finally:
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL", "REVIEW_LEVEL"]:
                os.environ.pop(key, None)

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_auto_adjust_important_for_medium_diff(self, _mock_check_env, mock_format, mock_run_review):
        """Auto-adjust to IMPORTANT when diff is 50-500 lines and no level specified."""
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Bug"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            for i in range(200):
                f.write(f"diff line {i}\n")

        try:
            runner = CliRunner()
            result = runner.invoke(main, ["--diff-file", test_diff_file, "--model", "test-model"])
            assert result.exit_code == 0, f"Command failed: {result.output}"
            mock_format.assert_called_once()
            assert mock_format.call_args[1]["min_severity"] == "IMPORTANT"
        finally:
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL", "REVIEW_LEVEL"]:
                os.environ.pop(key, None)

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_auto_adjust_respects_explicit_cli(self, _mock_check_env, mock_format, mock_run_review):
        """CLI --review-level overrides auto-adjust even for small diffs."""
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Bug"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            f.write("\n".join(f"diff line {i}" for i in range(10)))

        try:
            runner = CliRunner()
            result = runner.invoke(main, [
                "--diff-file", test_diff_file,
                "--review-level", "CRITICAL",
                "--model", "test-model",
            ])
            assert result.exit_code == 0, f"Command failed: {result.output}"
            mock_format.assert_called_once()
            assert mock_format.call_args[1]["min_severity"] == "CRITICAL"
        finally:
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL", "REVIEW_LEVEL"]:
                os.environ.pop(key, None)

    @patch("src.main.run_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_auto_adjust_respects_explicit_env(self, _mock_check_env, mock_format, mock_run_review):
        """REVIEW_LEVEL env var overrides auto-adjust even for large diffs."""
        mock_run_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "trivial", "comment": "Style"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"
        os.environ["REVIEW_LEVEL"] = "TRIVIAL"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            for i in range(600):
                f.write(f"diff line {i}\n")

        try:
            runner = CliRunner()
            result = runner.invoke(main, ["--diff-file", test_diff_file, "--model", "test-model"])
            assert result.exit_code == 0, f"Command failed: {result.output}"
            mock_format.assert_called_once()
            assert mock_format.call_args[1]["min_severity"] == "TRIVIAL"
        finally:
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL", "REVIEW_LEVEL"]:
                os.environ.pop(key, None)

    def test_invalid_review_level_rejected(self):
        """Test that Click rejects invalid --review-level values."""
        # Set required environment variables
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        # Create a test diff file using tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            test_diff_file = f.name
            f.write("diff --git a/test.py b/test.py\n")

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--diff-file",
                    test_diff_file,
                    "--review-level",
                    "INVALID",
                    "--model",
                    "test-model",
                ],
            )

            # Verify the command failed with non-zero exit code
            assert result.exit_code != 0, "Command should fail with invalid review level"

            # Verify error message contains expected text
            assert "Invalid value for '--review-level'" in result.output
            assert "INVALID" in result.output
            assert "TRIVIAL" in result.output
            assert "IMPORTANT" in result.output
            assert "CRITICAL" in result.output

        finally:
            # Cleanup
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            for key in ["GEMINI_API_KEY", "LOCAL"]:
                if key in os.environ:
                    del os.environ[key]
