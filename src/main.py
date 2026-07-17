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
"""CLI entry point for Gemini Code Review — local and CI modes."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
from github.GithubException import GithubException
from google import genai
from loguru import logger

from src.config import AiReviewConfig, check_required_env_vars
from src.gemini_client import get_review
from src.github_client import (
    create_a_comment_to_pull_request,
    create_inline_review_comments,
    get_all_pr_comments_text,
    write_github_output,
)
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


def _print_review_title(c: SimpleNamespace) -> None:
    """Print the review title banner."""
    print("\n" + "=" * 80)
    print(f"{c.bold}{c.cyan}🤖 Gemini AI Code Review{c.reset}")
    print("=" * 80 + "\n")


def _print_severity_summary(filtered_items: list, min_severity: str, c: SimpleNamespace) -> None:
    """Group items by severity and print counts."""
    critical_items = [item for item in filtered_items if item.get("severity", "").lower() == "critical"]
    important_items = [item for item in filtered_items if item.get("severity", "").lower() == "important"]
    trivial_items = [item for item in filtered_items if item.get("severity", "").lower() == "trivial"]

    total = len(filtered_items)
    print(f"{c.bold}Found {total} issue(s):{c.reset}")
    if critical_items:
        print(f"  {c.red}{c.bold}● {len(critical_items)} CRITICAL{c.reset} {c.gray}(blocking){c.reset}")
    if important_items:
        print(f"  {c.yellow}{c.bold}● {len(important_items)} IMPORTANT{c.reset}")
    if trivial_items:
        print(f"  {c.blue}{c.bold}● {len(trivial_items)} TRIVIAL{c.reset}")
    print()


def _severity_display(severity: str, c: SimpleNamespace) -> tuple:
    """Get display colors and labels for a severity level.

    Returns:
        Tuple of (background_color, icon, label).
    """
    if severity == "CRITICAL":
        return c.bg_red, "🔴", "CRITICAL"
    if severity == "IMPORTANT":
        return c.bg_yellow, "🟡", "IMPORTANT"
    return c.bg_blue, "🔵", "TRIVIAL"


def _render_suggestion_block(suggestion: str, dash_line: str, c: SimpleNamespace) -> None:
    """Print a suggestion with basic syntax highlighting."""
    print(f"   {c.green}💡 Suggested Fix:{c.reset}")
    print(f"   {c.gray}{dash_line}{c.reset}")

    for line in suggestion.split("\n"):
        if line.strip().startswith("-"):
            print(f"   {c.red}{line}{c.reset}")
        elif line.strip().startswith("+"):
            print(f"   {c.green}{line}{c.reset}")
        elif line.strip().startswith("@@"):
            print(f"   {c.cyan}{line}{c.reset}")
        elif any(kw in line for kw in ["def ", "class ", "import ", "from "]):
            print(f"   {c.magenta}{line}{c.reset}")
        else:
            print(f"   {c.dim}{line}{c.reset}")

    print(f"   {c.gray}{dash_line}{c.reset}")


def _render_comment_block(comment: str) -> None:
    """Print a comment with word wrapping and indentation."""
    for comment_line in comment.split("\n"):
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


def _render_item(item: dict, index: int, dash_line: str, c: SimpleNamespace) -> None:
    """Print a single review item with full formatting."""
    severity = item.get("severity", "unknown").upper()
    file_name = item.get("file", "unknown")
    line_num = item.get("line", "?")
    comment = item.get("comment", "")
    suggestion = item.get("suggestion", "")

    bg_color, icon, label = _severity_display(severity, c)

    # Header with severity badge
    print(f"{icon} {c.bold}Issue #{index}{c.reset} {bg_color}{c.bold} {label} {c.reset}")

    # File and line info
    print(f"   {c.cyan}📄 {file_name}{c.reset}{c.gray}:{line_num}{c.reset}")

    # Comment with word wrapping
    print(f"   {c.magenta}💬 Comment:{c.reset}")
    _render_comment_block(comment)

    # Code suggestion with syntax highlighting
    if suggestion:
        _render_suggestion_block(suggestion, dash_line, c)

    print()


def _print_overall_summary(summarized_review: str, c: SimpleNamespace) -> None:
    """Print the overall summary footer."""
    print("=" * 80)
    print(f"{c.bold}{c.cyan}📋 Overall Summary:{c.reset}")
    print(f"{c.dim}{summarized_review}{c.reset}")
    print("=" * 80)


def print_local_review(filtered_items: list, summarized_review: str, min_severity: str):
    """Print review results in human-readable format for local mode."""
    c = SimpleNamespace(**{k.lower(): v for k, v in _ANSI.items()})
    dash_line = "─" * 74

    _print_review_title(c)

    if not filtered_items:
        print(f"{c.green}{c.bold}✓ No issues found at {min_severity} level or above.{c.reset}\n")
        if summarized_review:
            print(f"{c.bold}{c.cyan}Summary:{c.reset}")
            print(f"{c.dim}{summarized_review}{c.reset}")
        return

    _print_severity_summary(filtered_items, min_severity, c)

    for i, item in enumerate(filtered_items, 1):
        _render_item(item, i, dash_line, c)

    if summarized_review:
        _print_overall_summary(summarized_review, c)

    print()


def _resolve_local_diff_source(files: tuple) -> str:
    """Resolve diff from local staged files or full git diff.

    Exits with code 0 if no changes detected, code 1 on git errors.
    """
    if files:
        logger.info(f"Running in local mode with {len(files)} files")
        diff = generate_diff_from_files(files)
        if not diff.strip():
            logger.info("No changes detected in staged files.")
            sys.exit(0)
        return diff

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
        return diff
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get git diff: {e}")
        sys.exit(1)


def _resolve_ci_diff_source(diff_file: str | None) -> str:
    """Read the diff from a CI diff file path.

    Exits with code 1 if no diff_file is provided.
    """
    if not diff_file:
        logger.error("--diff-file is required when not in local mode")
        sys.exit(1)

    curr_files = [str(p) for p in Path(".").iterdir() if p.is_file()]
    logger.info(f"Files in curr_dir: {curr_files}")
    with open(diff_file, encoding="utf-8") as f:
        return f.read()


def _resolve_diff_source(local: bool, files: tuple, diff_file: str | None) -> str:
    """Resolve the diff content from local staged files or CI diff file."""
    if local:
        return _resolve_local_diff_source(files)
    return _resolve_ci_diff_source(diff_file)


def _resolve_ci_env_vars() -> tuple:
    """Resolve CI-only environment variables and PR comments.

    Returns:
        Tuple of (github_token, github_repo, pr_number, commit_hash, comments_text).
    """
    github_token = os.environ["GITHUB_TOKEN"]
    github_repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = int(os.environ["GITHUB_PULL_REQUEST_NUMBER"])
    commit_hash = os.environ["GIT_COMMIT_HASH"]

    try:
        comments_text = get_all_pr_comments_text(
            github_token=github_token,
            github_repository=github_repo,
            pull_request_number=pr_number,
        )
    except GithubException as exc:
        logger.warning(f"Failed to fetch PR comments: {exc}")
        comments_text = ""

    return github_token, github_repo, pr_number, commit_hash, comments_text


def _dispatch_local_output(filtered_items: list, summarized_review: str, min_severity: str) -> None:
    """Print local review output and determine exit code.

    Exits with code 1 if critical issues are found, code 0 otherwise.
    """
    print_local_review(filtered_items, summarized_review, min_severity)

    critical_items = [item for item in filtered_items if item.get("severity", "").lower() == "critical"]

    if critical_items:
        logger.error(f"Found {len(critical_items)} CRITICAL issue(s). Blocking commit.")
        sys.exit(1)

    logger.info("Review complete. No critical issues found.")
    sys.exit(0)


def _post_single_review_comment(
    body: str,
    github_token: str,
    github_repository: str,
    pull_request_number: int,
    git_commit_hash: str,
) -> None:
    """Post a single review comment to the PR.

    Raises RuntimeError if the GitHub API returns an error status.
    """
    response = create_a_comment_to_pull_request(
        github_token=github_token,
        github_repository=github_repository,
        pull_request_number=pull_request_number,
        git_commit_hash=git_commit_hash,
        body=body,
    )
    if response.status_code >= 400:
        logger.error(f"Failed to post PR review comment: HTTP {response.status_code} - {response.text}")
        raise RuntimeError(f"GitHub API returned {response.status_code} when posting review comment")


def _dispatch_ci_output(
    filtered_items: list,
    review_comment: str,
    github_token: str,
    github_repository: str,
    pull_request_number: int,
    git_commit_hash: str,
) -> None:
    """Post inline comments or a single PR comment in CI mode.

    Attempts inline comments first; falls back to a single review comment
    if inline posting fails.
    """
    if filtered_items:
        logger.info(f"Posting {len(filtered_items)} individual inline review comments")
        results = create_inline_review_comments(
            github_token=github_token,
            github_repository=github_repository,
            pull_request_number=pull_request_number,
            git_commit_hash=git_commit_hash,
            review_items=filtered_items,
        )

        failed = [r for r in results if r.get("status") in ("failed", "error")]
        if failed:
            logger.warning(f"{len(failed)} inline comments failed to post. " "Falling back to single review comment.")
            _post_single_review_comment(
                body=review_comment,
                github_token=github_token,
                github_repository=github_repository,
                pull_request_number=pull_request_number,
                git_commit_hash=git_commit_hash,
            )
    else:
        logger.info("No review items found, posting summary only")
        _post_single_review_comment(
            body=review_comment,
            github_token=github_token,
            github_repository=github_repository,
            pull_request_number=pull_request_number,
            git_commit_hash=git_commit_hash,
        )


# pylint: disable=too-many-positional-arguments,broad-exception-caught
@click.command()
@click.option(
    "--diff-file",
    type=click.STRING,
    default=None,
    required=False,
    help="Pull request diff path (required in CI mode)",
)
@click.option(
    "--diff-chunk-size",
    type=click.INT,
    required=False,
    default=500000,
    help="Pull request diff",
)
@click.option(
    "--model",
    type=click.STRING,
    required=False,
    default="gemini-2.5-flash",
    help="Gemini model name (e.g. gemini-2.5-flash, gemini-2.5-pro)",
)
@click.option("--extra-prompt", type=click.STRING, required=False, default="", help="Extra prompt")
@click.option("--temperature", type=click.FLOAT, required=False, default=0.1, help="Temperature")
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
    diff_file: str | None,
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
    """Run the code review CLI — local or CI mode."""
    logger.level(log_level)

    if local:
        os.environ["LOCAL"] = "1"

    check_required_env_vars()

    # Resolve diff from the appropriate source
    diff = _resolve_diff_source(local, files, diff_file)

    # Set up Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # Resolve CI-only env vars (only when not in local mode)
    comments_text = ""
    _github_token = ""
    _github_repo = ""
    _pr_number = 0
    _commit_hash = ""
    if os.getenv("LOCAL") is None:
        _github_token, _github_repo, _pr_number, _commit_hash, comments_text = _resolve_ci_env_vars()

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
    chunked_reviews, summarized_review = get_review(client, review_conf)
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

    # Expose outputs to workflows
    write_github_output("review_result", review_comment)
    write_github_output("entire_prompt_body", get_review_prompt(extra_prompt=extra_prompt))

    # Dispatch output based on mode (local vs CI)
    if os.getenv("LOCAL") is not None:
        _dispatch_local_output(filtered_items, summarized_review, min_severity)
        return  # _dispatch_local_output calls sys.exit, but this keeps the linter happy

    _dispatch_ci_output(
        filtered_items=filtered_items,
        review_comment=review_comment,
        github_token=_github_token,
        github_repository=_github_repo,
        pull_request_number=_pr_number,
        git_commit_hash=_commit_hash,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
