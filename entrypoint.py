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
import time
from typing import List, TypedDict

import click
import google.generativeai as genai
import requests
from github import Auth, Github
from github.GithubException import GithubException
from google.api_core import exceptions as google_exceptions
from loguru import logger


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
    if os.getenv("LOCAL") is True:
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
                    getattr(c, "original_line", None)
                    or getattr(c, "line", None)
                    or "?"
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


def handle_api_error(error, attempt, max_retries=3, initial_wait=15):
    """Handle API errors with exponential backoff"""
    if isinstance(error, google_exceptions.ResourceExhausted):
        # Rate limit error
        if attempt < max_retries:
            wait_time = initial_wait * (2**attempt)  # Exponential backoff
            logger.warning(
                f"Rate limit hit. Waiting {wait_time} seconds before retry..."
            )
            time.sleep(wait_time)
            return True
    elif isinstance(error, google_exceptions.DeadlineExceeded):
        logger.error("API request timed out")
    elif isinstance(error, google_exceptions.InvalidArgument):
        logger.error(f"Invalid API request: {str(error)}")
    else:
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
    # Get summary by chunk
    chunked_reviews = []
    for chunked_diff in chunked_diff_list:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                convo = genai_model.start_chat(
                    history=[
                        {"role": "user", "parts": [review_prompt]},
                        {"role": "model", "parts": ["Ok"]},
                    ]
                )
                # 1) Send the diff chunk
                _ = convo.send_message(chunked_diff)

                # 2) Send PR comments context (if any), instructing the model to take them into account
                if comments_text.strip():
                    comments_prompt = (
                        "These are the comments already made in this PR by reviewers "
                        "and participants. Please take them into consideration when "
                        "performing your review.\n\n" + comments_text
                    )
                    _ = convo.send_message(comments_prompt)

                # 3) Request the actual output from the AI now that context is set
                final_instruction = (
                    "Now, considering the pull request diff and the PR comments above, "
                    "provide your review according to the earlier instructions."
                )
                final_response = convo.send_message(final_instruction)
                review_result = getattr(convo.last, "text", final_response.text)
            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.DeadlineExceeded,
                google_exceptions.InvalidArgument,
                google_exceptions.GoogleAPICallError,
                google_exceptions.RetryError,
            ) as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed")
                should_retry = handle_api_error(
                    e, attempt=attempt, max_retries=max_attempts
                )
                if not should_retry or attempt == max_attempts - 1:
                    review_result = None
                    break
                continue

            if review_result is None:
                raise ValueError("Failed to get model response")
            logger.debug(f"Response AI: {review_result}")
            chunked_reviews.append(review_result)
            break
        time.sleep(10)  # wating for 10 seconds to avoid rate limit
    # If the chunked reviews are only one, return it

    if len(chunked_reviews) == 1:
        return chunked_reviews, chunked_reviews[0]

    if len(chunked_reviews) == 0:
        summarize_prompt = (
            "Say that you didn't find any relevant changes to comment on any file"
        )
    else:
        summarize_prompt = get_summarize_prompt()

    chunked_reviews_join = "\n".join(chunked_reviews)
    summary_response = None
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            convo = genai_model.start_chat(history=[])
            summary_response = convo.send_message(
                summarize_prompt + "\n\n" + chunked_reviews_join
            )
        except (
            google_exceptions.ResourceExhausted,
            google_exceptions.DeadlineExceeded,
            google_exceptions.InvalidArgument,
            google_exceptions.GoogleAPICallError,
            google_exceptions.RetryError,
        ) as e:
            logger.error(f"Attempt {attempt + 1}/{max_attempts} failed")
            should_retry = handle_api_error(
                e, attempt=attempt, max_retries=max_attempts
            )
            if not should_retry or attempt == max_attempts - 1:
                summary_response = None
                break
            continue
    summarized_review = getattr(convo.last, "text", summary_response.text)
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
