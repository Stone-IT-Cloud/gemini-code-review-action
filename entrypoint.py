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
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, TypedDict

import click
import google.generativeai as genai
import requests
from github import Auth, Github
from github.GithubException import GithubException
from google.api_core import exceptions as google_exceptions
from loguru import logger


def write_github_output(name: str, value: str) -> None:
    """Write an output for GitHub Actions.

    If the action is not running in GitHub Actions (no GITHUB_OUTPUT env var), this is a no-op.
    """
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return

    # Use a unique delimiter to safely support multiline values.
    delimiter = f"ghadelim_{int(time.time() * 1000)}"
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


class AiReviewConfig(TypedDict):
    model: str
    diff: str
    extra_prompt: str
    prompt_chunk_size: int
    comments_text: str
    temperature: float = 1
    top_p: float = 0.95
    top_k: int = 0
    max_output_tokens: int = 8192


def check_required_env_vars():
    """Check required environment variables"""
    required_env_vars = [
        "GEMINI_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_PULL_REQUEST_NUMBER",
        "GIT_COMMIT_HASH",
    ]
    # if running locally, we only check gemini api key
    if os.getenv("LOCAL") is not None:
        required_env_vars = [
            "GEMINI_API_KEY",
        ]

    for required_env_var in required_env_vars:
        if os.getenv(required_env_var) is None:
            raise ValueError(f"{required_env_var} is not set")


def get_review_prompt(extra_prompt: str = "") -> str:
    """Get a prompt template"""
    template = f"""
    {extra_prompt}.
    This is a pull request or part of a pull request if the pull request is very large.
    Suppose you review this PR as an excellent software engineer and an excellent security engineer.
    Can you tell me the issues with differences in a pull request and provide suggestions to improve it?
    You can provide a review summary and issue comments per file if any major issues are found.
    Always include the name of the file that is citing the improvement or problem.
    In the next messages I will be sending you the difference between the GitHub file codes, okay?
    """
    return template


def get_summarize_prompt() -> str:
    """Get a prompt template"""
    template = """
    Can you summarize this for me?
    It would be good to stick to highlighting pressing issues and providing code suggestions to improve the pull request.
    Here's what you need to summarize:
    """
    return template


def create_a_comment_to_pull_request(
    github_token: str,
    github_repository: str,
    pull_request_number: int,
    git_commit_hash: str,
    body: str,
):
    """Create a comment to a pull request"""
    headers = {
        "Accept": "application/vnd.github.v3.patch",
        "authorization": f"Bearer {github_token}",
    }
    data = {"body": body, "commit_id": git_commit_hash, "event": "COMMENT"}
    url = f"https://api.github.com/repos/{github_repository}/pulls/{pull_request_number}/reviews"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response


def get_all_pr_comments_text(
    github_token: str, github_repository: str, pull_request_number: int
) -> str:
    """Collect issue comments, review comments and reviews for a PR and format them.

    Returns a single text blob suitable to send as model context.
    """
    g = Github(auth=Auth.Token(github_token))
    try:
        repo = g.get_repo(github_repository)
        pr = repo.get_pull(pull_request_number)

        # Issue comments (general discussion on the PR as an issue)
        issue_comments = list(pr.as_issue().get_comments())

        # Review comments (inline file comments)
        review_comments = list(pr.get_comments())

        # Reviews (summary reviews, states like APPROVED/CHANGES_REQUESTED)
        reviews = list(pr.get_reviews())

        lines: List[str] = []
        if issue_comments:
            lines.append("[Issue comments]")
            for c in issue_comments:
                author = getattr(getattr(c, "user", None), "login", "unknown")
                created = getattr(c, "created_at", "")
                body = (getattr(c, "body", "") or "").strip()
                lines.append(f"- {author} ({created}): {body}")

        if review_comments:
            lines.append("[Review comments]")
            for c in review_comments:
                author = getattr(getattr(c, "user", None), "login", "unknown")
                created = getattr(c, "created_at", "")
                path = getattr(c, "path", "")
                line_info = (
                    getattr(c, "original_line", None) or getattr(c, "line", None) or "?"
                )
                body = (getattr(c, "body", "") or "").strip()
                lines.append(f"- {author} ({created}) {path}:{line_info}: {body}")

        if reviews:
            lines.append("[Reviews]")
            for r in reviews:
                author = getattr(getattr(r, "user", None), "login", "unknown")
                state = getattr(r, "state", "")
                submitted = getattr(r, "submitted_at", "")
                body = (getattr(r, "body", "") or "").strip()
                if body:
                    lines.append(f"- {author} ({submitted}) [{state}]: {body}")
                else:
                    lines.append(f"- {author} ({submitted}) [{state}]: <no body>")

        joined = "\n".join(lines).strip()
        logger.info(
            f"Collected PR comments payload length: {len(joined)} characters across "
            f"{len(issue_comments)} issue, {len(review_comments)} review, {len(reviews)} reviews"
        )
        return joined
    finally:
        g.close()


def chunk_string(input_string: str, chunk_size) -> List[str]:
    """Chunk a string"""
    chunked_inputs = []
    for i in range(0, len(input_string), chunk_size):
        chunked_inputs.append(input_string[i : i + chunk_size])
    return chunked_inputs


def _extract_model_text(response) -> Optional[str]:
    """Best-effort extraction of text from a Gemini SDK response."""
    if response is None:
        return None
    return getattr(response, "text", None)


def _safe_str(obj) -> str:
    """Best-effort stringify without raising."""
    try:
        return str(obj)
    except Exception:  # pylint: disable=broad-exception-caught
        return repr(obj)


def _get_usage_metadata(response) -> dict:
    """Extract usage metadata from a Gemini SDK response (best-effort)."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}

    # SDK may return an object with attributes; keep this defensive.
    prompt_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    # Some SDK versions may expose dict-like usage metadata.
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_token_count")
        output_tokens = usage.get("candidates_token_count")
        total_tokens = usage.get("total_token_count")

    out: dict = {}
    if prompt_tokens is not None:
        out["prompt_tokens"] = int(prompt_tokens)
    if output_tokens is not None:
        out["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        out["total_tokens"] = int(total_tokens)
    return out


def _looks_like_daily_quota_exhausted(message: str) -> bool:
    """Heuristic: detect daily quota exhaustion from error text."""
    msg = (message or "").lower()
    return any(
        needle in msg
        for needle in (
            "per day",
            "daily",
            "requests per day",
            "rpd",
            "day quota",
        )
    )


class NoQuotaAvailableError(RuntimeError):
    """Raised when we detect there is no quota available (fail-fast)."""


@dataclass
class QuotaTracker:
    """Track run-local usage and provide best-effort remaining-quota hints.

    Note: The Gemini API (AI Studio) does not reliably expose "remaining quota"
    counters to clients. This tracker logs:
    - actual per-request usage (when `usage_metadata` is provided)
    - run-estimated remaining RPM/TPM/RPD from *completed* requests if you
      provide limits via env vars (in-flight/pending requests are not counted)
    """

    window_seconds: int = 60
    request_timestamps: deque = field(default_factory=deque)
    token_events: deque = field(default_factory=deque)  # (timestamp, total_tokens)
    requests_total: int = 0
    tokens_total: int = 0
    last_pruned_at: float = 0.0

    quota_rpm: Optional[int] = None
    quota_tpm: Optional[int] = None
    quota_rpd: Optional[int] = None

    @staticmethod
    def from_env() -> "QuotaTracker":
        def _parse_int(name: str) -> Optional[int]:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return None
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer value for environment variable {name!r}: {raw!r}. "
                    "Please set it to a valid integer (e.g., '60') or leave it unset."
                ) from exc

        return QuotaTracker(
            quota_rpm=_parse_int("GEMINI_QUOTA_RPM"),
            quota_tpm=_parse_int("GEMINI_QUOTA_TPM"),
            quota_rpd=_parse_int("GEMINI_QUOTA_RPD"),
        )

    def _prune(self, now: float) -> None:
        # Optimization: only prune if enough time has passed since last prune
        if now - self.last_pruned_at < 1.0:
            return
        cutoff = now - self.window_seconds
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        while self.token_events and self.token_events[0][0] < cutoff:
            self.token_events.popleft()
        self.last_pruned_at = now

    def note_request(self, now: float) -> None:
        self.requests_total += 1
        self.request_timestamps.append(now)
        self._prune(now)

    def note_tokens(self, now: float, total_tokens: int) -> None:
        self.tokens_total += int(total_tokens)
        self.token_events.append((now, int(total_tokens)))
        self._prune(now)

    def recent_rpm(self, now: float) -> int:
        self._prune(now)
        return len(self.request_timestamps)

    def recent_tpm(self, now: float) -> int:
        self._prune(now)
        return sum(t for _, t in self.token_events)

    def remaining_estimate(self, now: float) -> dict:
        """Return run-estimated remaining quota (if limits are configured)."""
        rem: dict = {}
        if self.quota_rpm is not None:
            rem["rpm_remaining"] = max(0, self.quota_rpm - self.recent_rpm(now))
            rem["rpm_limit"] = self.quota_rpm
        if self.quota_tpm is not None:
            rem["tpm_remaining"] = max(0, self.quota_tpm - self.recent_tpm(now))
            rem["tpm_limit"] = self.quota_tpm
        if self.quota_rpd is not None:
            rem["rpd_remaining"] = max(0, self.quota_rpd - self.requests_total)
            rem["rpd_limit"] = self.quota_rpd
        return rem

    def log_after_response(self, response, label: str) -> None:
        now = time.time()
        usage = _get_usage_metadata(response)
        total_tokens = usage.get("total_tokens")
        if total_tokens is not None:
            self.note_tokens(now, total_tokens)

        remaining = self.remaining_estimate(now)
        usage_bits = []
        if usage:
            prompt_tokens = usage.get("prompt_tokens", "?")
            output_tokens = usage.get("output_tokens", "?")
            total_tokens_val = usage.get("total_tokens", "?")
            usage_bits.append(
                "usage_tokens="
                f"prompt={prompt_tokens},"
                f"output={output_tokens},"
                f"total={total_tokens_val}"
            )
        if remaining:
            usage_bits.append(
                "run_estimated_remaining="
                + ",".join(f"{k}={v}" for k, v in remaining.items())
            )
        if not usage_bits:
            usage_bits.append("usage_metadata=<not provided by API>")
        joined_usage_bits = ", ".join(usage_bits)
        logger.info(f"{label} {joined_usage_bits}")


def _sleep_with_jitter(seconds: float) -> None:
    """Sleep with a small random jitter to avoid synchronized retries."""
    # Jitter in [0, 1.0) seconds, capped so it never dominates the delay.
    jitter = min(1.0, random.random())
    time.sleep(max(0.0, seconds + jitter))


def _handle_api_error(
    error,
    *,
    attempt: int,
    max_attempts: int,
    initial_wait: float,
    max_wait: float,
    fail_fast_on_no_quota: bool,
) -> bool:
    """Handle API errors with exponential backoff + jitter.

    Returns True if the caller should retry (and we already waited), else False.
    """
    is_last_attempt = (attempt + 1) >= max_attempts

    if isinstance(error, google_exceptions.ResourceExhausted):
        # Rate limit / quota exceeded.
        err_text = _safe_str(error)
        logger.warning(f"Rate limit / quota exceeded details: {err_text}")

        if fail_fast_on_no_quota and _looks_like_daily_quota_exhausted(err_text):
            logger.error("Daily quota exhausted and fail-fast is enabled; aborting without retries.")
            raise error

        if is_last_attempt:
            logger.error("Rate limit hit. No retries remaining.")
            return False
        wait_time = min(max_wait, initial_wait * (2**attempt))
        logger.warning(f"Rate limit hit. Waiting {wait_time:.0f}s before retry...")
        _sleep_with_jitter(wait_time)
        return True

    if isinstance(error, google_exceptions.DeadlineExceeded):
        logger.error("API request timed out")
        return not is_last_attempt

    if isinstance(error, google_exceptions.InvalidArgument):
        logger.error(f"Invalid API request: {str(error)}")
        return False

    # Default: do not spin forever on unexpected errors.
    logger.error(f"Unexpected API error: {str(error)}")
    return False


def get_review(config: AiReviewConfig) -> List[str]:
    """Get a review"""
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
        system_instruction=extra_prompt,
    )

    # Throttling controls (defaults tuned to avoid bursty request patterns in CI).
    max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "6"))
    initial_wait = float(os.getenv("GEMINI_INITIAL_BACKOFF_SECONDS", "15"))
    max_wait = float(os.getenv("GEMINI_MAX_BACKOFF_SECONDS", "240"))
    min_request_interval = float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "6"))
    fail_fast_on_no_quota = os.getenv("GEMINI_FAIL_FAST_ON_NO_QUOTA", "1") == "1"

    tracker = QuotaTracker.from_env()
    if (
        tracker.quota_rpm is not None
        and tracker.quota_tpm is not None
        and tracker.quota_rpd is not None
        and tracker.quota_rpm == 0
        and tracker.quota_tpm == 0
        and tracker.quota_rpd == 0
    ):
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

            request_started_at = time.time()
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


def format_review_comment(summarized_review: str, chunked_reviews: List[str]) -> str:
    """Format reviews"""
    if len(chunked_reviews) == 1:
        return summarized_review
    unioned_reviews = "\n".join(chunked_reviews)
    review = f"""<details>
    <summary>{summarized_review}</summary>
    {unioned_reviews}
    </details>
    """
    return review


# pylint: disable=too-many-positional-arguments
@click.command()
@click.option(
    "--diff-file",
    type=click.STRING,
    default="/tmp/pr.diff",
    required=True,
    help="Pull request diff",
)
@click.option(
    "--diff-chunk-size",
    type=click.INT,
    required=False,
    default=3500,
    help="Pull request diff",
)
@click.option(
    "--model", type=click.STRING, required=False, default="gpt-3.5-turbo", help="Model"
)
@click.option(
    "--extra-prompt", type=click.STRING, required=False, default="", help="Extra prompt"
)
@click.option(
    "--temperature", type=click.FLOAT, required=False, default=0.1, help="Temperature"
)
@click.option("--top-p", type=click.FLOAT, required=False, default=1.0, help="Top N")
@click.option(
    "--log-level",
    type=click.STRING,
    required=False,
    default="INFO",
    help="Log level",
)
def main(
    diff_file: str,
    diff_chunk_size: int,
    model: str,
    extra_prompt: str,
    temperature: float,
    top_p: float,
    log_level: str,
):
    # Set log level
    logger.level(log_level)
    # Check if necessary environment variables are set or not
    check_required_env_vars()

    print(diff_file)
    # List the content of the /tmp folder
    tmp_files = os.popen("ls -lah .").read()
    logger.info(f"Files in curr_dir: {tmp_files}")

    with open(diff_file, "r", encoding="utf-8") as f:
        diff = f.read()

    # Set the Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    # Fetch PR comments to include as context (skip in local mode or on failure)
    comments_text = ""
    if os.getenv("LOCAL") is None:
        try:
            comments_text = get_all_pr_comments_text(
                github_token=os.getenv("GITHUB_TOKEN"),
                github_repository=os.getenv("GITHUB_REPOSITORY"),
                pull_request_number=int(os.getenv("GITHUB_PULL_REQUEST_NUMBER")),
            )
        except GithubException as exc:
            logger.warning(f"Failed to fetch PR comments: {exc}")

    # Request a code review
    review_conf: AiReviewConfig = {
        "diff": diff,
        "extra_prompt": extra_prompt,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_chunk_size": diff_chunk_size,
        "comments_text": comments_text,
    }
    chunked_reviews, summarized_review = get_review(review_conf)
    logger.debug(f"Summarized review: {summarized_review}")
    logger.debug(f"Chunked reviews: {chunked_reviews}")

    # Format reviews
    review_comment = format_review_comment(
        summarized_review=summarized_review, chunked_reviews=chunked_reviews
    )

    # Expose outputs to workflows (works for both container actions and composite wrappers).
    write_github_output("review_result", review_comment)
    # Keep this intentionally small to avoid output size limits.
    write_github_output(
        "entire_prompt_body", get_review_prompt(extra_prompt=extra_prompt)
    )

    # if it is running in a local environment don't try to create a comment
    if os.getenv("LOCAL") is not None:
        logger.info(f"Review comment: {review_comment}")
        return

    # Create a comment to a pull request
    create_a_comment_to_pull_request(
        github_token=os.getenv("GITHUB_TOKEN"),
        github_repository=os.getenv("GITHUB_REPOSITORY"),
        pull_request_number=int(os.getenv("GITHUB_PULL_REQUEST_NUMBER")),
        git_commit_hash=os.getenv("GIT_COMMIT_HASH"),
        body=review_comment,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
