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
"""Post-PR learning — analyse human discussions and persist decisions to Engram.

When a PR is merged or closed, this module fetches all review comments
posted by the bot, identifies human replies, classifies the outcome
(rejected / accepted / acknowledged), and stores each decision in
Engram so future reviews can avoid repeating rejected suggestions.
"""

from __future__ import annotations

import os
import re
from typing import Any

import requests
from loguru import logger

from src.review_memory import ensure_engram, store_decisions_batch

# ── Keywords used when no LLM is available (fallback) ──────────────────────

_REJECTED_KEYWORDS = [
    "no aplica",
    "falso",
    "falso positivo",
    "no corresponde",
    "wont fix",
    "won't fix",
    "not applicable",
    "not relevant",
    "irrelevant",
    "false positive",
    "no es necesario",
    "no hace falta",
    "cerrando como no aplica",
    "ya se hace eso",
]

_ACCEPTED_KEYWORDS = [
    "good catch",
    "nice catch",
    "fixed",
    "done",
    "corregido",
    "implementado",
    "gracias",
    "thanks",
    "aplicado",
    "lo corrijo",
    "buen punto",
    "tienes razón",
    "tenés razón",
    "bien visto",
]

# ── Public API ─────────────────────────────────────────────────────────────


def run(
    github_token: str,
    repo: str,
    pr_number: int,
    engram_dir: str,
    llm_client: Any | None = None,
) -> dict[str, int]:
    """Analyse a closed PR and persist human decisions to Engram.

    Args:
        github_token: GitHub API token.
        repo: Repository in ``owner/repo`` format.
        pr_number: Pull request number.
        engram_dir: Path to the ``.engram`` directory (cached by GHA).
        llm_client: Optional LLM client for classification. If ``None``,
                    keyword-based fallback is used.

    Returns:
        Dict with ``stored`` (count of decisions persisted).
    """
    ensure_engram(engram_dir)
    logger.info(f"Learning from PR #{pr_number} in {repo}")

    comments = _fetch_bot_comments(github_token, repo, pr_number)
    logger.info(f"Found {len(comments)} bot comments with human replies")

    decisions: list[dict[str, str]] = []
    for comment in comments:
        suggestion = _clean_suggestion(comment.get("body", ""))
        replies = comment.get("replies", [])
        if not replies:
            continue

        # Use the first human reply for classification
        reply = replies[0].get("body", "")
        decision = _classify_decision(suggestion, reply, llm_client)

        decisions.append({
            "suggestion": suggestion,
            "file_pattern": comment.get("path", "*"),
            "decision": decision,
            "reason": reply[:200],
        })

    if decisions:
        stored = store_decisions_batch(
            engram_dir=engram_dir,
            repo=repo,
            pr_number=pr_number,
            decisions=decisions,
            session_id=f"learn-pr-{pr_number}",
        )
        logger.info(f"Stored {stored} decisions from PR #{pr_number}")
        return {"stored": stored}

    logger.info(f"No decisions to store from PR #{pr_number}")
    return {"stored": 0}


# ── GitHub API ─────────────────────────────────────────────────────────────


def _fetch_bot_comments(
    github_token: str,
    repo: str,
    pr_number: int,
) -> list[dict[str, Any]]:
    """Fetch all review comments by ``github-actions[bot]`` with their replies.

    Uses the GitHub Pull Request Review Comments API to fetch inline comments,
    then filters to those authored by the bot and checks for reply threads.

    Returns:
        List of bot comment dicts, each with ``replies`` key containing
        human responses.
    """
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    all_comments = response.json()

    # Build a lookup of parent→child replies
    replies_by_parent: dict[int, list[dict]] = {}
    for comment in all_comments:
        parent_id = comment.get("in_reply_to_id")
        if parent_id:
            replies_by_parent.setdefault(parent_id, []).append(comment)

    # Collect bot comments that have human replies
    bot_comments: list[dict[str, Any]] = []
    for comment in all_comments:
        if comment.get("user", {}).get("login") != "github-actions[bot]":
            continue
        replies = replies_by_parent.get(comment["id"], [])
        if not replies:
            continue  # skip bot comments nobody replied to

        bot_comments.append({
            "id": comment["id"],
            "user": comment["user"],
            "body": comment.get("body", ""),
            "path": comment.get("path", ""),
            "line": comment.get("line", 0),
            "replies": [
                {
                    "user": r["user"],
                    "body": r.get("body", ""),
                }
                for r in replies
            ],
        })

    return bot_comments


# ── Classification ─────────────────────────────────────────────────────────


def _classify_decision(
    suggestion: str,
    reply: str,
    llm_client: Any | None = None,
) -> str:
    """Classify the human decision on a bot suggestion.

    Args:
        suggestion: The bot's original suggestion text.
        reply: The human's reply text.
        llm_client: Optional LLM client. If None, keyword fallback is used.

    Returns:
        One of ``"rejected"``, ``"accepted"``, ``"acknowledged"``.
    """
    # If an LLM client is available, use it (TODO: future enhancement)
    if llm_client is not None:
        return _classify_with_llm(suggestion, reply, llm_client)

    return _classify_keywords(reply)


def _classify_keywords(reply: str) -> str:
    """Keyword-based classification fallback."""
    reply_lower = reply.strip().lower()

    for kw in _REJECTED_KEYWORDS:
        if kw in reply_lower:
            return "rejected"

    for kw in _ACCEPTED_KEYWORDS:
        if kw in reply_lower:
            return "accepted"

    return "acknowledged"


def _classify_with_llm(
    suggestion: str,
    reply: str,
    llm_client: Any,
) -> str:
    """Classify using an LLM."""
    # This is a placeholder for future LLM-based classification
    return _classify_keywords(reply)


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse_pr_number(value: str | None) -> int:
    """Parse PR number from a CLI arg or fall back to env var."""
    if value is not None:
        return int(value)
    env_val = os.getenv("GITHUB_PULL_REQUEST_NUMBER")
    if env_val:
        return int(env_val)
    raise ValueError(
        "PR number not provided and GITHUB_PULL_REQUEST_NUMBER env var not set"
    )


def _clean_suggestion(body: str) -> str:
    """Strip markdown formatting from a bot comment body.

    Handles: ``**[SEVERITY]** `path:line`: comment`` → ``comment``
    """
    # Remove severity tag: **[IMPORTANT]** or **[CRITICAL]**
    cleaned = re.sub(r"\*\*\[[A-Z]+\]\*\*\s*", "", body)
    # Remove code location: `path:line` or `path`
    cleaned = re.sub(r"`[^`]+`:\s*", "", cleaned)
    # Remove lone backtick references
    cleaned = re.sub(r"`[^`]+`\s*", "", cleaned)
    return cleaned.strip()
