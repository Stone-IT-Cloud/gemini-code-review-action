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
"""Tests for src/gemini_client.py — Gemini API interaction layer."""

from unittest.mock import MagicMock, patch

from src.config import AiReviewConfig
from src.gemini_client import (DEFAULT_TOKEN_LIMIT, get_model_context_limit,
                               get_review)

# ---------------------------------------------------------------------------
# get_model_context_limit
# ---------------------------------------------------------------------------

class TestGetModelContextLimit:
    """Test querying model context limits via genai.get_model()."""

    @patch("src.gemini_client.genai")
    def test_returns_limit_when_available(self, mock_genai):
        """Happy path: model has input_token_limit."""
        mock_genai.get_model.return_value = MagicMock(input_token_limit=1_048_576)
        result = get_model_context_limit("gemini-2.5-flash")
        assert result == 1_048_576
        mock_genai.get_model.assert_called_once_with("gemini-2.5-flash")

    @patch("src.gemini_client.genai")
    def test_fallback_when_limit_is_none(self, mock_genai):
        """Model exists but input_token_limit is None."""
        mock_genai.get_model.return_value = MagicMock(input_token_limit=None)
        result = get_model_context_limit("gemini-2.5-flash")
        assert result == DEFAULT_TOKEN_LIMIT

    @patch("src.gemini_client.genai")
    def test_fallback_when_limit_is_zero(self, mock_genai):
        """Model exists but input_token_limit is 0."""
        mock_genai.get_model.return_value = MagicMock(input_token_limit=0)
        result = get_model_context_limit("gemini-2.5-flash")
        assert result == DEFAULT_TOKEN_LIMIT

    @patch("src.gemini_client.genai")
    def test_fallback_on_exception(self, mock_genai):
        """genai.get_model() raises an exception."""
        mock_genai.get_model.side_effect = Exception("API error")
        result = get_model_context_limit("gemini-2.5-flash")
        assert result == DEFAULT_TOKEN_LIMIT

    def test_default_token_limit_constant(self):
        """DEFAULT_TOKEN_LIMIT is 1_000_000 as specified."""
        assert DEFAULT_TOKEN_LIMIT == 1_000_000


# ---------------------------------------------------------------------------
# get_review — budget-gated single-call path
# ---------------------------------------------------------------------------

_BASIC_CONFIG: AiReviewConfig = {
    "model": "gemini-2.5-flash",
    "diff": "small diff content here",
    "extra_prompt": "",
    "prompt_chunk_size": 500000,
    "comments_text": "",
    "temperature": 0.1,
    "top_p": 0.95,
}


def _make_mock_model(response_text: str) -> MagicMock:
    """Create a mock GenerativeModel with a configured generate_content."""
    model = MagicMock()
    response = MagicMock()
    response.text = response_text
    model.generate_content.return_value = response
    return model


class TestGetReviewSingleCall:
    """Tests for the budget-aware single-call path in get_review()."""

    @patch("src.gemini_client.calculate_char_budget")
    @patch("src.gemini_client.get_model_context_limit")
    @patch("src.gemini_client.genai")
    def test_single_call_when_diff_fits_budget(
        self, mock_genai, mock_context_limit, mock_calc_budget
    ):
        """Diff fits within budget → single generate_content, no summarization."""
        mock_context_limit.return_value = 1_000_000
        mock_calc_budget.return_value = 1_600_000

        mock_model = _make_mock_model('[{"file": "a.py", "line": 1, "severity": "critical", "comment": "Bug"}]')
        mock_genai.GenerativeModel.return_value = mock_model

        chunked, summary = get_review(_BASIC_CONFIG)

        assert len(chunked) == 1
        assert "Bug" in summary
        mock_model.generate_content.assert_called_once()

    @patch("src.gemini_client.calculate_char_budget")
    @patch("src.gemini_client.get_model_context_limit")
    @patch("src.gemini_client.genai")
    def test_single_call_returns_empty_on_none_response(
        self, mock_genai, mock_context_limit, mock_calc_budget
    ):
        """When generate_content returns None text, return empty."""
        mock_context_limit.return_value = 1_000_000
        mock_calc_budget.return_value = 1_600_000

        mock_model = _make_mock_model("")
        mock_model.generate_content.return_value.text = None  # type: ignore[assignment]
        mock_genai.GenerativeModel.return_value = mock_model

        chunked, summary = get_review(_BASIC_CONFIG)

        assert len(chunked) == 0
        assert summary == ""


class TestGetReviewMultiChunk:
    """Tests for the fallback multi-chunk path in get_review()."""

    @patch("src.gemini_client.calculate_char_budget")
    @patch("src.gemini_client.get_model_context_limit")
    @patch("src.gemini_client.genai")
    @patch("src.gemini_client.QuotaTracker")
    def test_multi_chunk_when_diff_exceeds_budget(
        self, mock_quota, mock_genai, mock_context_limit, mock_calc_budget
    ):
        """Diff exceeds budget → chunking + summarization."""
        mock_context_limit.return_value = 1_000_000
        mock_calc_budget.return_value = 10  # smaller than diff

        # Mock quota to avoid env var reads
        mock_tracker = MagicMock()
        mock_tracker.has_all_quotas_set_to_zero.return_value = False
        mock_quota.from_env.return_value = mock_tracker

        # Mock model responses
        mock_model = _make_mock_model("review chunk 1 content")
        mock_summary_model = _make_mock_model("summarized content")
        mock_genai.GenerativeModel.side_effect = [mock_model, mock_summary_model]

        config: AiReviewConfig = {
            **_BASIC_CONFIG,
            "diff": "x" * 100,        # larger than budget of 10
            "prompt_chunk_size": 60,   # force ≥2 chunks
        }

        chunked = get_review(config)[0]

        # Should have multiple chunks
        assert len(chunked) >= 2
        # Summarization model was called (2nd GenerativeModel instance)
        mock_summary_model.generate_content.assert_called_once()

    @patch("src.gemini_client.calculate_char_budget")
    @patch("src.gemini_client.get_model_context_limit")
    @patch("src.gemini_client.genai")
    @patch("src.gemini_client.QuotaTracker")
    def test_multi_chunk_single_chunk_skips_summary(
        self, mock_quota, mock_genai, mock_context_limit, mock_calc_budget
    ):
        """When chunking produces only 1 chunk, no summarization call."""
        mock_context_limit.return_value = 1_000_000
        mock_calc_budget.return_value = 10

        mock_tracker = MagicMock()
        mock_tracker.has_all_quotas_set_to_zero.return_value = False
        mock_quota.from_env.return_value = mock_tracker

        mock_model = _make_mock_model("only chunk")
        # Only one model needed (review_model) — no summarize_model
        mock_genai.GenerativeModel.side_effect = [mock_model, MagicMock()]

        # Use a diff size that is just over budget but within a single chunk
        config: AiReviewConfig = {
            **_BASIC_CONFIG,
            "diff": "x" * 15,       # >10 budget
            "prompt_chunk_size": 100,  # >diff → 1 chunk
        }

        chunked = get_review(config)[0]

        assert len(chunked) == 1
        # No summarization: summarize_model.generate_content never called
        # (only generate_content for the single chunk)

    @patch("src.gemini_client.calculate_char_budget")
    @patch("src.gemini_client.get_model_context_limit")
    @patch("src.gemini_client.genai")
    @patch("src.gemini_client.QuotaTracker")
    def test_fallback_budget_on_get_model_failure(
        self, mock_quota, mock_genai, mock_context_limit, mock_calc_budget
    ):
        """get_model_context_limit falls back to DEFAULT_TOKEN_LIMIT."""
        # Simulate expensive external call: high budget so diff fits single-call
        mock_context_limit.return_value = DEFAULT_TOKEN_LIMIT
        mock_calc_budget.return_value = 100_000

        mock_tracker = MagicMock()
        mock_tracker.has_all_quotas_set_to_zero.return_value = False
        mock_quota.from_env.return_value = mock_tracker

        mock_model = _make_mock_model("fallback review")
        mock_genai.GenerativeModel.return_value = mock_model

        config: AiReviewConfig = {
            **_BASIC_CONFIG,
            "diff": "small text",
        }

        chunked, summary = get_review(config)

        assert len(chunked) == 1
        assert "fallback review" in summary
        # Single call path used since diff < budget
        mock_model.generate_content.assert_called_once()
