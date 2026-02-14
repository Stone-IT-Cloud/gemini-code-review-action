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

import click
import google.generativeai as genai
from github.GithubException import GithubException
from loguru import logger

from src.config import AiReviewConfig, check_required_env_vars
from src.gemini_client import get_review
from src.github_client import (
    create_a_comment_to_pull_request,
    get_all_pr_comments_text,
    write_github_output,
)
from src.prompts import get_review_prompt
from src.review_formatter import format_review_comment


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
@click.option(
    "--review-level",
    type=click.STRING,
    required=False,
    default=None,
    help="Minimum severity level to comment on (TRIVIAL, IMPORTANT, CRITICAL)",
)
def main(
    diff_file: str,
    diff_chunk_size: int,
    model: str,
    extra_prompt: str,
    temperature: float,
    top_p: float,
    log_level: str,
    review_level: str,
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

    # Format reviews with severity filtering
    # Priority: CLI argument > environment variable > default
    min_severity = review_level or os.getenv("REVIEW_LEVEL", "IMPORTANT")
    review_comment = format_review_comment(
        summarized_review=summarized_review,
        chunked_reviews=chunked_reviews,
        min_severity=min_severity,
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
    response = create_a_comment_to_pull_request(
        github_token=os.getenv("GITHUB_TOKEN"),
        github_repository=os.getenv("GITHUB_REPOSITORY"),
        pull_request_number=int(os.getenv("GITHUB_PULL_REQUEST_NUMBER")),
        git_commit_hash=os.getenv("GIT_COMMIT_HASH"),
        body=review_comment,
    )
    if response.status_code >= 400:
        logger.error(
            f"Failed to post PR review comment: HTTP {response.status_code} - {response.text}"
        )
        raise RuntimeError(
            f"GitHub API returned {response.status_code} when posting review comment"
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
