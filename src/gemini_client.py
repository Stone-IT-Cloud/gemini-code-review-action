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
import os
import time
from typing import List, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from loguru import logger

from src.config import AiReviewConfig
from src.prompts import get_review_prompt, get_summarize_prompt
from src.quota import NoQuotaAvailableError, QuotaTracker, _handle_api_error
from src.utils import _extract_model_text, _safe_str, chunk_string


def get_review(config: AiReviewConfig) -> tuple:
    """Get a review from Gemini for the given diff configuration."""
    model = config["model"]
    diff = config["diff"]
    extra_prompt = config["extra_prompt"]
    prompt_chunk_size = config["prompt_chunk_size"]
    comments_text = config.get("comments_text", "")
    temperature = config.get("temperature", 1)
    top_p = config.get("top_p", 0.95)
    top_k = config.get("top_k", 0)
    max_output_tokens = config.get("max_output_tokens", 8192)
    # Chunk the prompt
    review_prompt = get_review_prompt(extra_prompt=extra_prompt)
    chunked_diff_list = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)
    logger.info(f"Created {len(chunked_diff_list)} from diff")
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
    }
    genai_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=review_prompt,
    )

    # Throttling controls (defaults tuned to avoid bursty request patterns in CI).
    max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "6"))
    initial_wait = float(os.getenv("GEMINI_INITIAL_BACKOFF_SECONDS", "15"))
    max_wait = float(os.getenv("GEMINI_MAX_BACKOFF_SECONDS", "240"))
    min_request_interval = float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "6"))
    fail_fast_on_no_quota = os.getenv("GEMINI_FAIL_FAST_ON_NO_QUOTA", "1") == "1"

    tracker = QuotaTracker.from_env()
    if tracker.has_all_quotas_set_to_zero():
        raise NoQuotaAvailableError(
            "Configured quota is 0 (GEMINI_QUOTA_RPM/TPM/RPD). Refusing to start."
        )

    # Get review by chunk (1 request per chunk to reduce RPM/TPM pressure).
    chunked_reviews = []
    last_request_at = 0.0
    for idx, chunked_diff in enumerate(chunked_diff_list, start=1):
        # Enforce a minimum spacing between requests across chunks.
        since_last = time.time() - last_request_at
        if since_last < min_request_interval:
            time.sleep(min_request_interval - since_last)

        for attempt in range(max_attempts):
            try:
                prompt_parts: List[str] = [
                    review_prompt.strip(),
                    f"\n\n[Pull request diff chunk {idx}/{len(chunked_diff_list)}]\n{chunked_diff}",
                ]

                # Include PR comments only if present; this can be large, so keep it optional.
                if comments_text.strip():
                    prompt_parts.append(
                        "\n\n[Existing PR comments context]\n"
                        "Take these into consideration when performing your review.\n\n"
                        + comments_text
                    )

                prompt_parts.append(
                    "\n\nNow provide your review according to the earlier instructions."
                )

                response = genai_model.generate_content("\n".join(prompt_parts))
                now = time.time()
                tracker.note_request(now)
                last_request_at = now
                review_result = _extract_model_text(response)
                tracker.log_after_response(
                    response,
                    label=f"Gemini call success (review chunk {idx}/{len(chunked_diff_list)})",
                )
            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.DeadlineExceeded,
                google_exceptions.InvalidArgument,
                google_exceptions.GoogleAPICallError,
                google_exceptions.RetryError,
            ) as e:
                logger.error(
                    f"Chunk {idx}/{len(chunked_diff_list)} attempt {attempt + 1}/{max_attempts} failed: {_safe_str(e)}"
                )
                should_retry = _handle_api_error(
                    e,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    initial_wait=initial_wait,
                    max_wait=max_wait,
                    fail_fast_on_no_quota=fail_fast_on_no_quota,
                )
                if should_retry:
                    continue
                review_result = None

            if not review_result:
                # Don't crash the whole run; keep going so we can at least attempt other chunks.
                logger.error(
                    f"Failed to get model response for chunk {idx}/{len(chunked_diff_list)}"
                )
                break
            logger.debug(f"Response AI: {review_result}")
            chunked_reviews.append(review_result)
            break
        # Additional spacing to avoid bursts across chunks.
        time.sleep(min_request_interval)
    # If the chunked reviews are only one, return it

    if len(chunked_reviews) == 1:
        return chunked_reviews, chunked_reviews[0]

    if len(chunked_reviews) == 0:
        # Avoid another API call; also avoids guaranteed failure if we're already rate-limited.
        return (
            [],
            (
                "Unable to generate review (Gemini API rate limit/quota exceeded). "
                "Please rerun later or reduce request volume."
            ),
        )

    summarize_prompt = get_summarize_prompt()

    chunked_reviews_join = "\n".join(chunked_reviews)
    summarized_review: Optional[str] = None
    for attempt in range(max_attempts):
        try:
            since_last = time.time() - last_request_at
            if since_last < min_request_interval:
                time.sleep(min_request_interval - since_last)

            response = genai_model.generate_content(
                summarize_prompt + "\n\n" + chunked_reviews_join
            )
            now = time.time()
            tracker.note_request(now)
            last_request_at = now
            summarized_review = _extract_model_text(response)
            tracker.log_after_response(response, label="Gemini call success (summary)")
            if summarized_review:
                break
        except (
            google_exceptions.ResourceExhausted,
            google_exceptions.DeadlineExceeded,
            google_exceptions.InvalidArgument,
            google_exceptions.GoogleAPICallError,
            google_exceptions.RetryError,
        ) as e:
            logger.error(
                f"Summary attempt {attempt + 1}/{max_attempts} failed: {_safe_str(e)}"
            )
            should_retry = _handle_api_error(
                e,
                attempt=attempt,
                max_attempts=max_attempts,
                initial_wait=initial_wait,
                max_wait=max_wait,
                fail_fast_on_no_quota=fail_fast_on_no_quota,
            )
            if not should_retry:
                break

    if not summarized_review:
        summarized_review = (
            "Unable to generate summary (Gemini API rate limit/quota exceeded)."
        )
    logger.debug(f"Response AI: {summarized_review}")
    return chunked_reviews, summarized_review
