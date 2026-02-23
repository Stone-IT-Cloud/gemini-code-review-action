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
from unittest.mock import Mock, patch

from src.github_client import create_inline_review_comments


class TestCreateInlineReviewComments:
    """Test posting individual inline review comments."""

    @patch("src.github_client.requests.post")
    def test_posts_single_comment_successfully(self, mock_post):
        """Test posting a single inline comment."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        review_items = [{
            "file": "src/utils.py",
            "line": 42,
            "severity": "important",
            "comment": "Use list comprehension",
            "suggestion": "results = [x * 2 for x in items]"
        }]

        results = create_inline_review_comments(
            github_token="test_token",
            github_repository="owner/repo",
            pull_request_number=123,
            git_commit_hash="abc123",
            review_items=review_items
        )

        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["file"] == "src/utils.py"
        assert results[0]["line"] == 42

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "owner/repo/pulls/123/comments" in call_args[0][0]

        # Verify request body
        data = json.loads(call_args[1]["data"])
        assert data["path"] == "src/utils.py"
        assert data["line"] == 42
        assert data["commit_id"] == "abc123"
        assert data["side"] == "RIGHT"
        assert "**[IMPORTANT]**" in data["body"]
        assert "Use list comprehension" in data["body"]
        assert "```suggestion" in data["body"]
        assert "results = [x * 2 for x in items]" in data["body"]

    @patch("src.github_client.requests.post")
    def test_posts_comment_without_suggestion(self, mock_post):
        """Test posting inline comment without suggestion."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        review_items = [{
            "file": "src/config.py",
            "line": 10,
            "severity": "trivial",
            "comment": "Missing docstring"
        }]

        results = create_inline_review_comments(
            github_token="test_token",
            github_repository="owner/repo",
            pull_request_number=123,
            git_commit_hash="abc123",
            review_items=review_items
        )

        assert len(results) == 1
        assert results[0]["status"] == "success"

        # Verify request body doesn't include suggestion
        call_args = mock_post.call_args
        data = json.loads(call_args[1]["data"])
        assert "**[TRIVIAL]**" in data["body"]
        assert "Missing docstring" in data["body"]
        assert "```suggestion" not in data["body"]

    @patch("src.github_client.requests.post")
    def test_posts_multiple_comments(self, mock_post):
        """Test posting multiple inline comments."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        review_items = [
            {
                "file": "src/utils.py",
                "line": 42,
                "severity": "important",
                "comment": "Fix A",
                "suggestion": "code_a"
            },
            {
                "file": "src/auth.py",
                "line": 15,
                "severity": "critical",
                "comment": "Fix B",
                "suggestion": "code_b"
            },
            {
                "file": "src/config.py",
                "line": 8,
                "severity": "trivial",
                "comment": "Fix C"
            }
        ]

        results = create_inline_review_comments(
            github_token="test_token",
            github_repository="owner/repo",
            pull_request_number=123,
            git_commit_hash="abc123",
            review_items=review_items
        )

        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)
        assert mock_post.call_count == 3

    @patch("src.github_client.requests.post")
    def test_skips_file_level_comments(self, mock_post):
        """Test that file-level comments (line 0) are skipped."""
        review_items = [
            {
                "file": "src/utils.py",
                "line": 0,
                "severity": "trivial",
                "comment": "File-level comment"
            },
            {
                "file": "src/auth.py",
                "line": 15,
                "severity": "important",
                "comment": "Line-level comment"
            }
        ]

        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        results = create_inline_review_comments(
            github_token="test_token",
            github_repository="owner/repo",
            pull_request_number=123,
            git_commit_hash="abc123",
            review_items=review_items
        )

        assert len(results) == 2
        assert results[0]["status"] == "skipped"
        assert results[0]["line"] == 0
        assert results[1]["status"] == "success"
        assert results[1]["line"] == 15

        # Should only post one comment (the line-level one)
        assert mock_post.call_count == 1

    @patch("src.github_client.requests.post")
    def test_handles_api_failure(self, mock_post):
        """Test handling of API failure."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.text = "Validation failed"
        mock_post.return_value = mock_response

        review_items = [{
            "file": "src/utils.py",
            "line": 42,
            "severity": "important",
            "comment": "Test"
        }]

        results = create_inline_review_comments(
            github_token="test_token",
            github_repository="owner/repo",
            pull_request_number=123,
            git_commit_hash="abc123",
            review_items=review_items
        )

        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert results[0]["status_code"] == 422
        assert "Validation failed" in results[0]["error"]

    @patch("src.github_client.requests.post")
    def test_handles_exception(self, mock_post):
        """Test handling of exceptions during posting."""
        mock_post.side_effect = Exception("Network error")

        review_items = [{
            "file": "src/utils.py",
            "line": 42,
            "severity": "important",
            "comment": "Test"
        }]

        results = create_inline_review_comments(
            github_token="test_token",
            github_repository="owner/repo",
            pull_request_number=123,
            git_commit_hash="abc123",
            review_items=review_items
        )

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "Network error" in results[0]["error"]

    @patch("src.github_client.requests.post")
    def test_formats_multiline_suggestion(self, mock_post):
        """Test multiline suggestions are properly formatted."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        multiline_code = "def foo():\n    x = 1\n    return x"
        review_items = [{
            "file": "src/utils.py",
            "line": 42,
            "severity": "important",
            "comment": "Refactor function",
            "suggestion": multiline_code
        }]

        create_inline_review_comments(
            github_token="test_token",
            github_repository="owner/repo",
            pull_request_number=123,
            git_commit_hash="abc123",
            review_items=review_items
        )

        # Verify multiline code is in suggestion block
        call_args = mock_post.call_args
        data = json.loads(call_args[1]["data"])
        assert "```suggestion" in data["body"]
        assert multiline_code in data["body"]
        assert "```" in data["body"]

    @patch("src.github_client.requests.post")
    def test_empty_review_items_list(self, mock_post):
        """Test with empty review items list."""
        results = create_inline_review_comments(
            github_token="test_token",
            github_repository="owner/repo",
            pull_request_number=123,
            git_commit_hash="abc123",
            review_items=[]
        )

        assert len(results) == 0
        assert mock_post.call_count == 0
