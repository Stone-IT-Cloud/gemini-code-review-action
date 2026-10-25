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
"""Learn mode must run without a pull-request diff file."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from code_reviewer.main import main


class TestLearnModeDiffIndependence:
    def test_learn_mode_never_reads_the_diff_file(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        monkeypatch.setenv("GITHUB_REPOSITORY", "local/test")
        monkeypatch.setenv("GITHUB_PULL_REQUEST_NUMBER", "7")
        monkeypatch.setenv("GIT_COMMIT_HASH", "deadbeef")

        with (
            patch("code_reviewer.main.get_llm_client"),
            patch("code_reviewer.learner.run", return_value="ok") as learner_run,
        ):
            result = CliRunner().invoke(
                main,
                [
                    "--mode",
                    "learn",
                    # A path that does not exist: learn mode must not open it.
                    "--diff-file",
                    "/nonexistent/pr.diff",
                ],
            )

        assert result.exit_code == 0, result.output
        learner_run.assert_called_once()

    def test_review_mode_still_requires_a_diff_file(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        monkeypatch.setenv("GITHUB_REPOSITORY", "local/test")
        monkeypatch.setenv("GITHUB_PULL_REQUEST_NUMBER", "7")
        monkeypatch.setenv("GIT_COMMIT_HASH", "deadbeef")

        with patch("code_reviewer.main.get_llm_client"):
            result = CliRunner().invoke(main, ["--diff-file", ""])

        assert result.exit_code == 1
