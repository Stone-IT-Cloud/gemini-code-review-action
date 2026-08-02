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
"""Tests for code_reviewer/gemini_client.py — Gemini API interaction layer."""

from unittest.mock import MagicMock

import pytest
from google.genai import errors

from code_reviewer.config import AiReviewConfig
from code_reviewer.gemini_client import DEFAULT_TOKEN_LIMIT, get_model_context_limit, get_review
from code_reviewer.llm.gemini_client import GeminiClient, _handle_api_error

# ---------------------------------------------------------------------------
# get_model_context_limit
# ---------------------------------------------------------------------------


class TestGetModelContextLimit:
    """Test querying model context limits via client.models.get()."""

    def test_returns_limit_when_available(self):
        """Happy path: model has input_token_limit."""
        client = MagicMock()
        client.models.get.return_value = MagicMock(input_token_limit=1_048_576)
        result = get_model_context_limit(client, "gemini-2.5-flash")
        assert result == 1_048_576
        client.models.get.assert_called_once_with(model="gemini-2.5-flash")

    def test_fallback_when_limit_is_none(self):
        """Model exists but input_token_limit is None."""
        client = MagicMock()
        client.models.get.return_value = MagicMock(input_token_limit=None)
        result = get_model_context_limit(client, "gemini-2.5-flash")
        assert result == DEFAULT_TOKEN_LIMIT

    def test_fallback_when_limit_is_zero(self):
        """Model exists but input_token_limit is 0."""
        client = MagicMock()
        client.models.get.return_value = MagicMock(input_token_limit=0)
        result = get_model_context_limit(client, "gemini-2.5-flash")
        assert result == DEFAULT_TOKEN_LIMIT

    def test_fallback_on_exception(self):
        """client.models.get() raises an exception."""
        client = MagicMock()
        client.models.get.side_effect = Exception("API error")
        result = get_model_context_limit(client, "gemini-2.5-flash")
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


class TestGetReviewSingleCall:
    """Tests for the budget-aware single-call path in get_review()."""

    def test_single_call_when_diff_fits_budget(self, mocker):
        """Diff fits within budget → single generate_content, no summarization."""
        mock_context_limit = mocker.patch("code_reviewer.gemini_client.get_model_context_limit")
        mock_calc_budget = mocker.patch("code_reviewer.gemini_client.calculate_char_budget")
        mock_context_limit.return_value = 1_000_000
        mock_calc_budget.return_value = 1_600_000

        client = MagicMock()
        response = MagicMock()
        response.text = '[{"file": "a.py", "line": 1, "severity": "critical", "comment": "Bug"}]'
        client.models.generate_content.return_value = response

        chunked, summary = get_review(client, _BASIC_CONFIG)

        assert len(chunked) == 1
        assert "Bug" in summary
        client.models.generate_content.assert_called_once()

    def test_single_call_returns_empty_on_none_response(self, mocker):
        """When generate_content returns None text, return empty."""
        mocker.patch("code_reviewer.gemini_client.get_model_context_limit", return_value=1_000_000)
        mocker.patch("code_reviewer.gemini_client.calculate_char_budget", return_value=1_600_000)

        client = MagicMock()
        response = MagicMock()
        response.text = None
        client.models.generate_content.return_value = response

        chunked, summary = get_review(client, _BASIC_CONFIG)

        assert len(chunked) == 0
        assert summary == ""


class TestGetReviewMultiChunk:
    """Tests for the fallback multi-chunk path in get_review()."""

    def test_multi_chunk_when_diff_exceeds_budget(self, mocker):
        """Diff exceeds budget → chunking + summarization."""
        mocker.patch("code_reviewer.llm.gemini_client.calculate_char_budget", return_value=10)
        mock_tracker = MagicMock()
        mock_tracker.has_all_quotas_set_to_zero.return_value = False
        mocker.patch("code_reviewer.llm.gemini_client.QuotaTracker.from_env", return_value=mock_tracker)

        client = MagicMock()
        # wrap in GeminiClient so get_context_limit returns a high token limit
        gc = GeminiClient(client)
        mocker.patch.object(gc, "get_context_limit", return_value=1_000_000)

        def _generate_content_side_effect(*_args, **_kwargs):
            resp = MagicMock()
            resp.text = "review chunk content"
            return resp

        client.models.generate_content.side_effect = _generate_content_side_effect

        config: AiReviewConfig = {
            **_BASIC_CONFIG,
            "diff": "x" * 100,
            "prompt_chunk_size": 60,
        }

        chunked = gc.get_review(config)[0]

        assert len(chunked) >= 2

    def test_multi_chunk_single_chunk_skips_summary(self, mocker):
        """When chunking produces only 1 chunk, no separate summarization call (just returns it)."""
        mocker.patch("code_reviewer.llm.gemini_client.calculate_char_budget", return_value=10)
        mock_tracker = MagicMock()
        mock_tracker.has_all_quotas_set_to_zero.return_value = False
        mocker.patch("code_reviewer.llm.gemini_client.QuotaTracker.from_env", return_value=mock_tracker)

        client = MagicMock()
        gc = GeminiClient(client)
        mocker.patch.object(gc, "get_context_limit", return_value=1_000_000)
        response = MagicMock()
        response.text = "only chunk"
        client.models.generate_content.return_value = response

        config: AiReviewConfig = {
            **_BASIC_CONFIG,
            "diff": "x" * 15,
            "prompt_chunk_size": 100,
        }

        chunked = gc.get_review(config)[0]

        assert len(chunked) == 1

    def test_fallback_budget_on_get_model_failure(self, mocker):
        """get_context_limit falls back to DEFAULT_TOKEN_LIMIT."""
        mocker.patch("code_reviewer.llm.gemini_client.calculate_char_budget", return_value=100_000)

        client = MagicMock()
        gc = GeminiClient(client)
        mocker.patch.object(gc, "get_context_limit", return_value=DEFAULT_TOKEN_LIMIT)

        response = MagicMock()
        response.text = "fallback review"
        client.models.generate_content.return_value = response

        config: AiReviewConfig = {
            **_BASIC_CONFIG,
            "diff": "small text",
        }

        chunked, summary = gc.get_review(config)

        assert len(chunked) == 1
        assert "fallback review" in summary
        client.models.generate_content.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_api_error — new code-based error dispatch
# ---------------------------------------------------------------------------


class TestHandleApiError:
    """Tests for _handle_api_error with new errors.APIError code-based dispatch."""

    def test_429_with_quota_exhausted_fail_fast_raises(self):
        """429 + daily quota text + fail_fast → raises."""
        error = errors.APIError(
            code=429,
            response_json={"error": {"message": "requests per day exceeded"}},
        )
        with pytest.raises(errors.APIError):
            _handle_api_error(
                error,
                attempt=0,
                max_attempts=3,
                initial_wait=1.0,
                max_wait=10.0,
                fail_fast_on_no_quota=True,
            )

    def test_429_without_daily_text_retries(self):
        """429 without daily quota text → retries with backoff."""
        error = errors.APIError(
            code=429,
            response_json={"error": {"message": "Resource exhausted"}},
        )
        result = _handle_api_error(
            error,
            attempt=0,
            max_attempts=3,
            initial_wait=0.01,
            max_wait=0.1,
            fail_fast_on_no_quota=True,
        )
        assert result is True

    def test_504_retries_unless_last(self):
        """504 DeadlineExceeded → retries unless last attempt."""
        error = errors.APIError(
            code=504,
            response_json={"error": {"message": "Deadline exceeded"}},
        )

        # Not last attempt: retry
        result = _handle_api_error(
            error,
            attempt=0,
            max_attempts=3,
            initial_wait=0.01,
            max_wait=0.1,
            fail_fast_on_no_quota=False,
        )
        assert result is True

        # Last attempt: do not retry
        result = _handle_api_error(
            error,
            attempt=2,
            max_attempts=3,
            initial_wait=0.01,
            max_wait=0.1,
            fail_fast_on_no_quota=False,
        )
        assert result is False

    def test_400_no_retry(self):
        """400 InvalidArgument → never retry."""
        error = errors.APIError(
            code=400,
            response_json={"error": {"message": "Invalid argument"}},
        )
        result = _handle_api_error(
            error,
            attempt=0,
            max_attempts=3,
            initial_wait=1.0,
            max_wait=10.0,
            fail_fast_on_no_quota=False,
        )
        assert result is False

    def test_unknown_code_no_retry(self):
        """Unknown error code → do not retry."""
        error = errors.APIError(
            code=500,
            response_json={"error": {"message": "Something broke"}},
        )
        result = _handle_api_error(
            error,
            attempt=0,
            max_attempts=3,
            initial_wait=1.0,
            max_wait=10.0,
            fail_fast_on_no_quota=False,
        )
        assert result is False

    def test_non_api_error_no_retry(self):
        """Non-APIError (e.g. ValueError) → do not retry."""
        error = ValueError("not an API error")
        result = _handle_api_error(
            error,
            attempt=0,
            max_attempts=3,
            initial_wait=1.0,
            max_wait=10.0,
            fail_fast_on_no_quota=False,
        )
        assert result is False


# ---------------------------------------------------------------------------
# _process_single_chunk — chunked API call config
# ---------------------------------------------------------------------------


class TestProcessSingleChunkConfig:
    """Tests for config plumbing in _process_single_chunk and _process_chunks."""

    def test_process_single_chunk_receives_llm_config(self, mocker):
        """R3/R4: _process_single_chunk accepts and passes config to API call."""
        mock_tracker = MagicMock()
        mock_tracker.has_all_quotas_set_to_zero.return_value = False
        mocker.patch("code_reviewer.llm.gemini_client.QuotaTracker.from_env", return_value=mock_tracker)

        client = MagicMock()
        gc = GeminiClient(client)
        mocker.patch.object(gc, "get_context_limit", return_value=1_000_000)
        mocker.patch("code_reviewer.llm.gemini_client.calculate_char_budget", return_value=10)

        llm_config = MagicMock()
        llm_config.system_instruction = "You are a code reviewer."
        llm_config.temperature = 0.1
        llm_config.top_p = 0.95
        llm_config.max_output_tokens = 8192
        llm_config.model = "gemini-2.5-flash"

        response = MagicMock()
        response.text = "review output"
        client.models.generate_content.return_value = response

        # This call should work once the llm_config param is added
        result = gc._process_single_chunk(
            model="gemini-2.5-flash",
            idx=1,
            total=1,
            chunked_diff="some diff",
            comments_text="",
            max_attempts=2,
            initial_wait=0.01,
            max_wait=0.1,
            min_request_interval=0.0,
            fail_fast_on_no_quota=False,
            tracker=mock_tracker,
            llm_config=llm_config,
        )
        assert result == "review output"

        # Verify the API call included config with response_mime_type and system_instruction
        call_kwargs = client.models.generate_content.call_args.kwargs
        assert "config" in call_kwargs, "API call must include config"
        config = call_kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.system_instruction == "You are a code reviewer."

    def test_process_chunks_forwards_llm_config(self, mocker):
        """R5: _process_chunks forwards llm_config to _process_single_chunk."""
        mock_tracker = MagicMock()
        mock_tracker.has_all_quotas_set_to_zero.return_value = False
        mocker.patch("code_reviewer.llm.gemini_client.QuotaTracker.from_env", return_value=mock_tracker)
        mocker.patch("code_reviewer.llm.gemini_client.calculate_char_budget", return_value=10)

        client = MagicMock()
        gc = GeminiClient(client)
        mocker.patch.object(gc, "get_context_limit", return_value=1_000_000)

        llm_config = MagicMock()
        llm_config.system_instruction = "Review prompt"
        llm_config.temperature = 0.1
        llm_config.top_p = 0.95
        llm_config.max_output_tokens = 8192
        llm_config.model = "gemini-2.5-flash"

        response = MagicMock()
        response.text = "chunk result"
        client.models.generate_content.return_value = response

        # Spy on _process_single_chunk to verify it receives llm_config
        spy = mocker.spy(gc, "_process_single_chunk")

        gc._process_chunks(
            model="gemini-2.5-flash",
            chunked_diff_list=["chunk1", "chunk2"],
            llm_config=llm_config,
            comments_text="",
        )

        # Each call to _process_single_chunk should include llm_config
        for call_args in spy.call_args_list:
            assert "llm_config" in call_args.kwargs
            assert call_args.kwargs["llm_config"] is llm_config

    def test_generate_content_receives_config_with_system_instruction(self, mocker):
        """R3/R4: SDK generate_content receives config with response_mime_type and system_instruction."""
        mock_tracker = MagicMock()
        mock_tracker.has_all_quotas_set_to_zero.return_value = False
        mocker.patch("code_reviewer.llm.gemini_client.QuotaTracker.from_env", return_value=mock_tracker)

        client = MagicMock()
        gc = GeminiClient(client)
        mocker.patch.object(gc, "get_context_limit", return_value=1_000_000)
        mocker.patch("code_reviewer.llm.gemini_client.calculate_char_budget", return_value=10)

        llm_config = MagicMock()
        llm_config.system_instruction = "You are an expert code reviewer."

        response = MagicMock()
        response.text = "review text"
        client.models.generate_content.return_value = response

        gc._process_single_chunk(
            model="gemini-2.5-flash",
            idx=1,
            total=1,
            chunked_diff="diff content",
            comments_text="",
            max_attempts=2,
            initial_wait=0.01,
            max_wait=0.1,
            min_request_interval=0.0,
            fail_fast_on_no_quota=False,
            tracker=mock_tracker,
            llm_config=llm_config,
        )

        call_args = client.models.generate_content.call_args
        assert call_args is not None
        kwargs = call_args.kwargs
        assert "config" in kwargs
        config = kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.system_instruction == "You are an expert code reviewer."
