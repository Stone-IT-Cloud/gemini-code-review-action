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
from typing import List

import requests
from github import Auth, Github
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


def create_a_comment_to_pull_request(
    github_token: str,
    github_repository: str,
    pull_request_number: int,
    git_commit_hash: str,
    body: str,
):
    """Create a comment to a pull request."""
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
