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
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.main import main


class TestReviewLevelCLI:
    """Test review_level CLI parameter."""

    @patch("src.main.genai")
    @patch("src.main.get_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_review_level_from_cli_parameter(
        self, mock_check_env, mock_format, mock_get_review, mock_genai
    ):
        """Test that --review-level CLI parameter is used."""
        # Setup mocks
        mock_get_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "critical", "comment": "Bug"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        # Set required environment variable
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        # Create a test diff file
        test_diff_file = "/tmp/test_review_level.diff"
        with open(test_diff_file, "w", encoding="utf-8") as f:
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

            # Verify the command succeeded
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Verify format_review_comment was called with CRITICAL
            mock_format.assert_called_once()
            call_kwargs = mock_format.call_args[1]
            assert call_kwargs["min_severity"] == "CRITICAL"

        finally:
            # Cleanup
            if os.path.exists(test_diff_file):
                os.remove(test_diff_file)
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            if "LOCAL" in os.environ:
                del os.environ["LOCAL"]

    @patch("src.main.genai")
    @patch("src.main.get_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_review_level_defaults_to_env_var(
        self, mock_check_env, mock_format, mock_get_review, mock_genai
    ):
        """Test that environment variable is used when CLI param not provided."""
        # Setup mocks
        mock_get_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Issue"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        # Set environment variables
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"
        os.environ["REVIEW_LEVEL"] = "TRIVIAL"

        # Create a test diff file
        test_diff_file = "/tmp/test_review_level_env.diff"
        with open(test_diff_file, "w", encoding="utf-8") as f:
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

    @patch("src.main.genai")
    @patch("src.main.get_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_review_level_cli_overrides_env(
        self, mock_check_env, mock_format, mock_get_review, mock_genai
    ):
        """Test that CLI parameter takes precedence over environment variable."""
        # Setup mocks
        mock_get_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "critical", "comment": "Bug"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        # Set environment variables - env says TRIVIAL
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"
        os.environ["REVIEW_LEVEL"] = "TRIVIAL"

        # Create a test diff file
        test_diff_file = "/tmp/test_review_level_override.diff"
        with open(test_diff_file, "w", encoding="utf-8") as f:
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

            # Verify the command succeeded
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

    @patch("src.main.genai")
    @patch("src.main.get_review")
    @patch("src.main.format_review_comment")
    @patch("src.main.check_required_env_vars")
    def test_review_level_defaults_to_important(
        self, mock_check_env, mock_format, mock_get_review, mock_genai
    ):
        """Test that default is IMPORTANT when neither CLI nor env is set."""
        # Setup mocks
        mock_get_review.return_value = (
            ['[{"file": "test.py", "line": 1, "severity": "important", "comment": "Issue"}]'],
            "Summary",
        )
        mock_format.return_value = "Formatted review"

        # Set minimal environment variables - no REVIEW_LEVEL
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["LOCAL"] = "1"

        # Create a test diff file
        test_diff_file = "/tmp/test_review_level_default.diff"
        with open(test_diff_file, "w", encoding="utf-8") as f:
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
