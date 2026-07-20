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
"""Tests for src/learner.py — post-PR learning from human discussions."""

import os
import tempfile
from unittest.mock import MagicMock, patch

from src.learner import (
    _classify_decision,
    _fetch_bot_comments,
    _parse_pr_number,
    run,
)
from src.review_memory import ensure_engram, observation_count, store_decision


# ---------------------------------------------------------------------------
# _parse_pr_number
# ---------------------------------------------------------------------------


class TestParsePrNumber:
    def test_parses_valid_number(self):
        assert _parse_pr_number("42") == 42
        assert _parse_pr_number("0") == 0

    def test_parses_from_env_string(self):
        assert _parse_pr_number("123") == 123

    def test_falls_back_to_env(self):
        os.environ["GITHUB_PULL_REQUEST_NUMBER"] = "99"
        try:
            assert _parse_pr_number(None) == 99
        finally:
            del os.environ["GITHUB_PULL_REQUEST_NUMBER"]

    def test_raises_when_not_found(self):
        try:
            _parse_pr_number(None)
            assert False, "Should have raised"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# _fetch_bot_comments
# ---------------------------------------------------------------------------


class TestFetchBotComments:
    @patch("src.learner.requests.get")
    def test_returns_bot_comments(self, mock_get):
        mock_get.return_value.json.side_effect = [
            [
                {
                    "id": 1,
                    "user": {"login": "github-actions[bot]"},
                    "body": "**[CRITICAL]** Bug here",
                    "path": "src/main.py",
                    "line": 42,
                },
                {
                    "id": 10,
                    "user": {"login": "alecavallo"},
                    "body": "This is fine",
                    "path": "src/main.py",
                    "line": 42,
                    "in_reply_to_id": 1,
                },
                {
                    "id": 2,
                    "user": {"login": "human-user"},
                    "body": "Another topic",
                    "path": "src/main.py",
                    "line": 10,
                },
            ],
            [],  # second page empty → stop
        ]
        mock_get.return_value.status_code = 200

        comments = _fetch_bot_comments(
            github_token="test", repo="owner/repo", pr_number=1
        )
        assert len(comments) == 1
        assert comments[0]["id"] == 1
        assert len(comments[0]["replies"]) == 1
        assert comments[0]["replies"][0]["body"] == "This is fine"

    @patch("src.learner.requests.get")
    def test_includes_human_replies(self, mock_get):
        mock_get.return_value.json.side_effect = [
            [
                {
                    "id": 1,
                    "user": {"login": "github-actions[bot]"},
                    "body": "Use f-strings",
                    "path": "src/main.py",
                    "line": 5,
                },
                {
                    "id": 10,
                    "user": {"login": "alecavallo"},
                    "body": "No aplica, usamos formato legacy",
                    "in_reply_to_id": 1,
                },
            ],
            [],  # second page empty → stop
        ]
        mock_get.return_value.status_code = 200

        comments = _fetch_bot_comments(
            github_token="test", repo="owner/repo", pr_number=1
        )
        assert len(comments) == 1
        assert len(comments[0]["replies"]) == 1
        assert comments[0]["replies"][0]["body"] == "No aplica, usamos formato legacy"


# ---------------------------------------------------------------------------
# _classify_decision
# ---------------------------------------------------------------------------


class TestClassifyDecision:
    def test_rejected_keywords(self):
        """Rejected keywords map to 'rejected'."""
        result = _classify_decision(
            "Use f-strings",
            "No aplica, usamos formato legacy por consistencia",
        )
        assert result == "rejected"

    def test_accepted_keywords(self):
        result = _classify_decision("Add type hints", "Buen punto, lo corrijo")
        assert result == "accepted"

    def test_acknowledged_fallback(self):
        result = _classify_decision("Refactor this", "Lo voy a revisar")
        assert result == "acknowledged"

    def test_empty_reply_falls_to_acknowledged(self):
        result = _classify_decision("Fix this", "")
        assert result == "acknowledged"


# ---------------------------------------------------------------------------
# run — integration with Engram
# ---------------------------------------------------------------------------


class TestRun:
    @patch("src.learner._fetch_bot_comments")
    def test_stores_rejected_decisions_in_engram(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "id": 1,
                "user": {"login": "github-actions[bot]"},
                "body": "Use f-strings instead of percent formatting",
                "path": "src/main.py",
                "line": 5,
                "replies": [
                    {"user": {"login": "alecavallo"}, "body": "No aplica"}
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = run(
                github_token="test",
                repo="owner/repo",
                pr_number=42,
                engram_dir=tmp,
                llm_client=None,
            )
            assert result["stored"] == 1
            assert observation_count(tmp, "owner/repo") == 1

    @patch("src.learner._fetch_bot_comments")
    def test_skips_comments_without_replies(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "id": 1,
                "user": {"login": "github-actions[bot]"},
                "body": "Use f-strings",
                "path": "src/main.py",
                "line": 5,
                "replies": [],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = run(
                github_token="test",
                repo="owner/repo",
                pr_number=42,
                engram_dir=tmp,
                llm_client=None,
            )
            assert result["stored"] == 0


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestFetchBotCommentsPagination:
    @patch("src.learner.requests.get")
    def test_fetches_multiple_pages(self, mock_get):
        """Returns 150 comments across 2 pages, expects both pages fetched."""
        page1 = [{"id": i, "user": {"login": "github-actions[bot]"}, "body": f"Suggestion {i}"} for i in range(100)]
        page2 = [{"id": i + 100, "user": {"login": "github-actions[bot]"}, "body": f"Suggestion {i}"} for i in range(50)]

        # First call returns page1, second returns page2, third returns empty
        mock_get.return_value.json.side_effect = [page1, page2, []]
        mock_get.return_value.status_code = 200

        comments = _fetch_bot_comments(
            github_token="test", repo="owner/repo", pr_number=1
        )
        # Without replies, comments without replies are filtered out → empty
        assert isinstance(comments, list)
        # Verify the function fetched all pages (3rd is empty → stop)
        assert mock_get.call_count == 3, f"Expected 3 calls, got {mock_get.call_count}"

    @patch("src.learner.requests.get")
    def test_fetches_until_empty_page(self, mock_get):
        """Stops fetching when an empty page is returned."""
        mock_get.return_value.json.side_effect = [
            [{"id": 1, "user": {"login": "github-actions[bot]"}, "body": "X"}],
            [],  # empty page → stop
        ]
        mock_get.return_value.status_code = 200

        _fetch_bot_comments(github_token="test", repo="owner/repo", pr_number=1)
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------


class TestClassifyDecisionWithLLM:
    def test_llm_returns_classification(self):
        """When llm_client responds correctly, use its result."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "rejected"

        result = _classify_decision(
            "Use f-strings",
            "No aplica",
            llm_client=mock_client,
        )
        assert result == "rejected"

    def test_llm_fallback_on_invalid_response(self):
        """If LLM returns unexpected text, fall back to keywords."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "maybe"

        result = _classify_decision(
            "Use f-strings",
            "No aplica",
            llm_client=mock_client,
        )
        assert result == "rejected"  # fallback: "no aplica" → rejected

    def test_llm_fallback_on_error(self):
        """If LLM raises, fall back to keywords."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")

        result = _classify_decision(
            "Use f-strings",
            "No aplica",
            llm_client=mock_client,
        )
        assert result == "rejected"
