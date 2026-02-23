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
import subprocess
import sys

import click
import google.generativeai as genai
from github.GithubException import GithubException
from loguru import logger

from src.config import AiReviewConfig, check_required_env_vars
from src.gemini_client import get_review
from src.github_client import (create_a_comment_to_pull_request,
                               create_inline_review_comments,
                               get_all_pr_comments_text, write_github_output)
from src.prompts import get_review_prompt
from src.review_formatter import filter_by_severity, format_review_comment
from src.review_parser import parse_review_response

# ANSI color codes for local review output (module-level to reduce locals in print_local_review)
_ANSI = {
    "RED": "\033[91m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "MAGENTA": "\033[95m",
    "GRAY": "\033[90m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RESET": "\033[0m",
    "BG_RED": "\033[101m",
    "BG_YELLOW": "\033[103m",
    "BG_BLUE": "\033[104m",
}


def generate_diff_from_files(files: tuple) -> str:
    """Generate a unified diff from a list of files."""
    all_diffs = []
    for file_path in files:
        try:
            # Get the staged diff for this file
            result = subprocess.run(
                ["git", "diff", "--cached", file_path],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                all_diffs.append(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to get diff for {file_path}: {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Unexpected error getting diff for {file_path}: {e}")

    return "\n".join(all_diffs)


def print_local_review(filtered_items: list, summarized_review: str, min_severity: str):
    """Print review results in human-readable format for local mode."""
    a = _ANSI
    dash_line = "─" * 74
    print("\n" + "=" * 80)
    print(f"{a["BOLD"]}{a["CYAN"]}🤖 Gemini AI Code Review{a["RESET"]}")
    print("=" * 80 + "\n")

    if not filtered_items:
        print(f"{a["GREEN"]}{a["BOLD"]}✓ No issues found at {min_severity} level or above.{a["RESET"]}\n")
        if summarized_review:
            print(f"{a["BOLD"]}{a["CYAN"]}Summary:{a["RESET"]}")
            print(f"{a["DIM"]}{summarized_review}{a["RESET"]}")
        return

    # Group by severity
    critical_items = [
        item for item in filtered_items
        if item.get("severity", "").lower() == "critical"
    ]
    important_items = [
        item for item in filtered_items
        if item.get("severity", "").lower() == "important"
    ]
    trivial_items = [
        item for item in filtered_items
        if item.get("severity", "").lower() == "trivial"
    ]

    # Print summary with counts
    total = len(filtered_items)
    print(f"{a["BOLD"]}Found {total} issue(s):{a["RESET"]}")
    if critical_items:
        print(f"  {a["RED"]}{a["BOLD"]}● {len(critical_items)} CRITICAL{a["RESET"]} {a["GRAY"]}(blocking){a["RESET"]}")
    if important_items:
        print(f"  {a["YELLOW"]}{a["BOLD"]}● {len(important_items)} IMPORTANT{a["RESET"]}")
    if trivial_items:
        print(f"  {a["BLUE"]}{a["BOLD"]}● {len(trivial_items)} TRIVIAL{a["RESET"]}")
    print()

    # Print each item with enhanced formatting
    for i, item in enumerate(filtered_items, 1):
        severity = item.get("severity", "unknown").upper()
        file_name = item.get("file", "unknown")
        line_num = item.get("line", "?")
        comment = item.get("comment", "")
        suggestion = item.get("suggestion", "")

        # Choose color and styling based on severity
        if severity == "CRITICAL":
            bg_color = a["BG_RED"]
            icon = "🔴"
            label = "CRITICAL"
        elif severity == "IMPORTANT":
            bg_color = a["BG_YELLOW"]
            icon = "🟡"
            label = "IMPORTANT"
        else:
            bg_color = a["BG_BLUE"]
            icon = "🔵"
            label = "TRIVIAL"

        # Header with severity badge
        print(f"{icon} {a["BOLD"]}Issue #{i}{a["RESET"]} {bg_color}{a["BOLD"]} {label} {a["RESET"]}")

        # File and line info with syntax highlighting
        print(f"   {a["CYAN"]}📄 {file_name}{a["RESET"]}{a["GRAY"]}:{line_num}{a["RESET"]}")

        # Comment with word wrapping and indentation
        print(f"   {a["MAGENTA"]}💬 Comment:{a["RESET"]}")
        for comment_line in comment.split("\n"):
            # Wrap long lines
            if len(comment_line) > 70:
                words = comment_line.split()
                current_line = "      "
                for word in words:
                    if len(current_line) + len(word) + 1 > 76:
                        print(current_line)
                        current_line = "      " + word
                    else:
                        current_line += " " + word if current_line.strip() else word
                if current_line.strip():
                    print(current_line)
            else:
                print(f"      {comment_line}")

        # Code suggestion with syntax highlighting
        if suggestion:
            print(f"   {a["GREEN"]}💡 Suggested Fix:{a["RESET"]}")
            print(f"   {a["GRAY"]}{dash_line}{a["RESET"]}")

            # Colorize code lines (basic syntax highlighting)
            for line in suggestion.split("\n"):
                if line.strip().startswith("-"):
                    print(f"   {a["RED"]}{line}{a["RESET"]}")
                elif line.strip().startswith("+"):
                    print(f"   {a["GREEN"]}{line}{a["RESET"]}")
                elif line.strip().startswith("@@"):
                    print(f"   {a["CYAN"]}{line}{a["RESET"]}")
                elif any(kw in line for kw in ["def ", "class ", "import ", "from "]):
                    print(f"   {a["MAGENTA"]}{line}{a["RESET"]}")
                else:
                    print(f"   {a["DIM"]}{line}{a["RESET"]}")

            print(f"   {a["GRAY"]}{dash_line}{a["RESET"]}")

        print()

    # Overall summary with styling
    if summarized_review:
        print("=" * 80)
        print(f"{a["BOLD"]}{a["CYAN"]}📋 Overall Summary:{a["RESET"]}")
        print(f"{a["DIM"]}{summarized_review}{a["RESET"]}")
        print("=" * 80)
    print()


# pylint: disable=too-many-positional-arguments,broad-exception-caught
@click.command()
@click.option(
    "--diff-file",
    type=click.STRING,
    default="/tmp/pr.diff",
    required=False,
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
    type=click.Choice(["TRIVIAL", "IMPORTANT", "CRITICAL"], case_sensitive=False),
    required=False,
    default=None,
    help="Minimum severity level to comment on",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Run in local mode (for pre-commit hooks)",
)
@click.argument("files", nargs=-1, type=click.Path(exists=True))
def main(
    diff_file: str,
    diff_chunk_size: int,
    model: str,
    extra_prompt: str,
    temperature: float,
    top_p: float,
    log_level: str,
    review_level: str,
    local: bool,
    files: tuple,
):
    # Set log level
    logger.level(log_level)

    # Set LOCAL environment variable if --local flag is set
    if local:
        os.environ["LOCAL"] = "1"

    # Check if necessary environment variables are set or not
    check_required_env_vars()

    # In local mode, generate diff from staged files
    if local and files:
        logger.info(f"Running in local mode with {len(files)} files")
        # Generate diff for the staged files
        diff = generate_diff_from_files(files)
        if not diff.strip():
            logger.info("No changes detected in staged files.")
            sys.exit(0)
    elif local:
        # No files provided in local mode - get staged files from git
        logger.info("Running in local mode, getting staged files from git")
        try:
            result = subprocess.run(
                ["git", "diff", "--cached"],
                capture_output=True,
                text=True,
                check=True,
            )
            diff = result.stdout
            if not diff.strip():
                logger.info("No staged changes detected.")
                sys.exit(0)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get git diff: {e}")
            sys.exit(1)
    else:
        # CI mode - read from diff file
        if not diff_file:
            logger.error("--diff-file is required when not in local mode")
            sys.exit(1)

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

    # Parse all review items from chunked responses
    all_review_items = []
    for chunk_text in chunked_reviews:
        parsed_items = parse_review_response(chunk_text)
        all_review_items.extend(parsed_items)

    # Apply severity filtering
    filtered_items = filter_by_severity(all_review_items, min_severity)

    # Format for output (backward compatibility)
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

    # if it is running in a local environment, print human-readable output and exit
    if os.getenv("LOCAL") is not None:
        print_local_review(filtered_items, summarized_review, min_severity)

        # Exit with non-zero code if critical issues are found
        critical_items = [
            item for item in filtered_items
            if item.get("severity", "").lower() == "critical"
        ]

        if critical_items:
            logger.error(f"Found {len(critical_items)} CRITICAL issue(s). Blocking commit.")
            sys.exit(1)

        logger.info("Review complete. No critical issues found.")
        sys.exit(0)

    # Post individual inline comments for items with file/line info
    if filtered_items:
        logger.info(
            f"Posting {len(filtered_items)} individual inline review comments"
        )
        results = create_inline_review_comments(
            github_token=os.getenv("GITHUB_TOKEN"),
            github_repository=os.getenv("GITHUB_REPOSITORY"),
            pull_request_number=int(os.getenv("GITHUB_PULL_REQUEST_NUMBER")),
            git_commit_hash=os.getenv("GIT_COMMIT_HASH"),
            review_items=filtered_items,
        )

        # Check if any failed (ignore skipped file-level comments)
        failed = [r for r in results if r.get("status") in ("failed", "error")]
        if failed:
            logger.warning(
                f"{len(failed)} inline comments failed to post. "
                "Falling back to single review comment."
            )
            # Fall back to posting as single review if inline comments fail
            response = create_a_comment_to_pull_request(
                github_token=os.getenv("GITHUB_TOKEN"),
                github_repository=os.getenv("GITHUB_REPOSITORY"),
                pull_request_number=int(
                    os.getenv("GITHUB_PULL_REQUEST_NUMBER")
                ),
                git_commit_hash=os.getenv("GIT_COMMIT_HASH"),
                body=review_comment,
            )
            if response.status_code >= 400:
                logger.error(
                    f"Failed to post PR review comment: HTTP "
                    f"{response.status_code} - {response.text}"
                )
                raise RuntimeError(
                    f"GitHub API returned {response.status_code} when "
                    "posting review comment"
                )
    else:
        # No review items, post summary as single comment
        logger.info("No review items found, posting summary only")
        response = create_a_comment_to_pull_request(
            github_token=os.getenv("GITHUB_TOKEN"),
            github_repository=os.getenv("GITHUB_REPOSITORY"),
            pull_request_number=int(os.getenv("GITHUB_PULL_REQUEST_NUMBER")),
            git_commit_hash=os.getenv("GIT_COMMIT_HASH"),
            body=review_comment,
        )
        if response.status_code >= 400:
            logger.error(
                f"Failed to post PR review comment: HTTP "
                f"{response.status_code} - {response.text}"
            )
            raise RuntimeError(
                f"GitHub API returned {response.status_code} when "
                "posting review comment"
            )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
