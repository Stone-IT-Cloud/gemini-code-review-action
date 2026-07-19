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
"""Cross-PR review memory backed by Engram SQLite DB.

Works directly with ``.engram/engram.db`` using Python's built-in ``sqlite3``,
so the Docker image needs no extra dependency.

Lifecycle in CI (GitHub Actions):
  1. actions/cache restores ``.engram/`` (DB + chunks).
  2. ``ensure_engram()`` verifies / initialises the DB.
  3. Before review → ``query_similar()`` finds past decisions via FTS.
  4. After review  → ``store_decision()`` persists the outcome.
  5. actions/cache saves the updated ``.engram/``.

Lifecycle locally:
  - Run ``engram sync --import`` in the cloned repo to pull CI decisions
    into your local Engram, or run ``engram sync --export`` to push local
    decisions into the repo's ``.engram/`` for the next CI run.

Schema — ``observations`` table::

    id, sync_id, session_id, type, title, content, project, scope,
    topic_key, normalized_hash, revision_count, duplicate_count,
    created_at, updated_at, deleted_at, embedding
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

_SUGGESTION_HASH_PREFIX = "review:"

# ---------------------------------------------------------------------------
# 1.  Ensure Engram DB exists and is ready
# ---------------------------------------------------------------------------


def ensure_engram(engram_dir: str) -> Path:
    """Return the path to the Engram DB, creating it if necessary.

    Creates a minimal ``.engram/engram.db`` with the required schema when no
    Engram installation is present.  The schema mirrors what ``engram init``
    would create so the DB is compatible with ``engram sync``.
    """
    root = Path(engram_dir)
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "engram.db"

    if db_path.exists():
        return db_path

    logger.info(f"Initialising Engram DB at {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()
    return db_path


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    directory   TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at    TEXT,
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sync_id          TEXT,
    session_id       TEXT NOT NULL,
    type             TEXT NOT NULL,
    title            TEXT NOT NULL,
    content          TEXT NOT NULL,
    tool_name        TEXT,
    project          TEXT,
    scope            TEXT NOT NULL DEFAULT 'project',
    topic_key        TEXT,
    normalized_hash  TEXT,
    revision_count   INTEGER NOT NULL DEFAULT 1,
    duplicate_count  INTEGER NOT NULL DEFAULT 1,
    last_seen_at     TEXT,
    pinned           BOOLEAN NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
    deleted_at       TEXT,
    review_after     TEXT,
    expires_at       TEXT,
    embedding        BLOB,
    embedding_model  TEXT,
    embedding_created_at TEXT
);

CREATE TABLE IF NOT EXISTS sync_chunks (
    target_key  TEXT NOT NULL DEFAULT 'local',
    chunk_id    TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sync_enrolled_projects (
    project     TEXT,
    enrolled_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sync_mutations (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    target_key  TEXT NOT NULL,
    entity      TEXT NOT NULL,
    entity_key  TEXT NOT NULL,
    op          TEXT NOT NULL,
    payload     TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'local',
    occurred_at TEXT NOT NULL DEFAULT (datetime('now')),
    acked_at    TEXT,
    project     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sync_state (
    target_key          TEXT PRIMARY KEY,
    lifecycle           TEXT NOT NULL DEFAULT 'idle',
    last_enqueued_seq   INTEGER NOT NULL DEFAULT 0,
    last_acked_seq      INTEGER NOT NULL DEFAULT 0,
    last_pulled_seq     INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    backoff_until       TEXT,
    lease_owner         TEXT,
    lease_until         TEXT,
    last_error          TEXT,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    reason_code         TEXT,
    reason_message      TEXT
);

-- Full-text search virtual table
CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts
USING fts5(title, content, tool_name, type, project, topic_key,
           content=observations, content_rowid=id);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
    INSERT INTO observations_fts(rowid, title, content, tool_name, type, project, topic_key)
    VALUES (new.id, new.title, new.content, new.tool_name, new.type, new.project, new.topic_key);
END;

CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, content, tool_name, type, project, topic_key)
    VALUES ('delete', old.id, old.title, old.content, old.tool_name, old.type, old.project, old.topic_key);
END;

CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, title, content, tool_name, type, project, topic_key)
    VALUES ('delete', old.id, old.title, old.content, old.tool_name, old.type, old.project, old.topic_key);
    INSERT INTO observations_fts(rowid, title, content, tool_name, type, project, topic_key)
    VALUES (new.id, new.title, new.content, new.tool_name, new.type, new.project, new.topic_key);
END;
"""

# ---------------------------------------------------------------------------
# 2.  Load existing decisions from Engram DB
# ---------------------------------------------------------------------------


def _db(engram_dir: str) -> sqlite3.Connection:
    return sqlite3.connect(str(Path(engram_dir) / "engram.db"))


def load_decisions(engram_dir: str, repo: str, limit: int = 50) -> list[dict[str, Any]]:
    """Load the most recent review decisions for *repo* from Engram.

    Args:
        engram_dir: Path to the ``.engram`` directory.
        repo: Repository identifier (e.g. ``owner/repo``).
        limit: Max decisions to return (newest first).  Keeps token cost low.

    Returns:
        List of observation dicts with keys ``id``, ``title``, ``content``,
        ``decision``, ``reason``, ``pr``, ``topic_key``.
    """
    ensure_engram(engram_dir)
    conn = _db(engram_dir)
    try:
        rows = conn.execute(
            """
            SELECT id, sync_id, title, content, topic_key, created_at
            FROM observations
            WHERE project = ? AND type = 'review-decision' AND deleted_at IS NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (repo, limit),
        ).fetchall()

        decisions = []
        for row in rows:
            d = {
                "id": row[0],
                "sync_id": row[1],
                "title": row[2],
                "content": row[3],
                "topic_key": row[4],
                "created_at": row[5],
            }
            # Parse structured data from topic_key:  review:decision:reason:pr
            parts = (d["topic_key"] or "").split(":")
            d["decision"] = parts[1] if len(parts) > 1 else "unknown"
            d["reason"] = parts[2] if len(parts) > 2 else ""
            d["pr"] = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            decisions.append(d)
        return decisions
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3.  Query similar decisions via FTS5  (token-efficient)
# ---------------------------------------------------------------------------


def query_similar(
    engram_dir: str,
    repo: str,
    suggestion: str,
    diff_files: list[str] | None = None,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Search for past decisions relevant to *suggestion* via FTS5.

    Uses Engram's full-text search to find semantically similar observations.
    This is far more token-efficient than loading all decisions:  only the
    N most relevant results are returned, and they're compressed into a short
    summary for the prompt.

    Args:
        engram_dir: Path to the ``.engram`` directory.
        repo: Repository identifier.
        suggestion: The suggestion text to search for.
        diff_files: Optional list of file paths to further narrow results.
        max_results: Max similar decisions to return.

    Returns:
        List of matching decision dicts.
    """
    ensure_engram(engram_dir)
    conn = _db(engram_dir)
    try:
        # Build query terms from the suggestion — strip common words,
        # use the key nouns and verbs
        terms = _fts_terms(suggestion)
        if not terms:
            return []

        query = f"""
            SELECT o.id, o.title, o.content, o.topic_key, o.created_at,
                   rank
            FROM observations_fts
            JOIN observations o ON o.id = observations_fts.rowid
            WHERE observations_fts MATCH ?
              AND o.project = ?
              AND o.type = 'review-decision'
              AND o.deleted_at IS NULL
            ORDER BY rank
            LIMIT ?
        """
        rows = conn.execute(query, (terms, repo, max_results)).fetchall()

        results = []
        for row in rows:
            d = {
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "topic_key": row[3],
                "created_at": row[4],
                "rank": row[5],
            }
            parts = (d["topic_key"] or "").split(":")
            d["decision"] = parts[1] if len(parts) > 1 else "unknown"
            d["reason"] = parts[2] if len(parts) > 2 else ""
            d["pr"] = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            results.append(d)
        return results
    finally:
        conn.close()


def _fts_terms(text: str) -> str:
    """Extract meaningful FTS5 query terms from natural language.

    Strips punctuation and very short words, returns an AND query.
    """
    import re

    # Remove common boilerplate
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because",
        "but", "and", "or", "if", "while", "about", "up",
        "this", "that", "these", "those",
        "use", "used", "using", "get", "got", "make", "made",
        "need", "needs", "please", "consider", "should", "must",
        "also", "already", "always", "never", "ever",
        "code", "function", "class", "method", "variable", "file",
        "line", "change", "changes", "added", "removed", "updated",
    }

    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = [t for t in clean.split() if len(t) > 2 and t not in stop_words]
    # Keep at most 8 terms for the query
    return " AND ".join(tokens[:8])


# ---------------------------------------------------------------------------
# 4.  Build token-efficient prompt context
# ---------------------------------------------------------------------------


def build_context(
    decisions: list[dict[str, Any]],
    similar: list[dict[str, Any]] | None = None,
    max_chars: int = 1500,
) -> str:
    """Build a concise context string for the LLM prompt.

    Token-minimised:  only rejected/wont-fix decisions are included,
    truncated to fit within *max_chars*.
    """
    rejected = [
        d for d in decisions if d.get("decision") in ("rejected", "wont-fix")
    ]
    # Deduplicate by suggestion hash
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for d in reversed(rejected[-30:]):
        h = _content_hash(d.get("content", ""))
        if h not in seen:
            seen.add(h)
            unique.append(d)

    if not unique:
        return ""

    lines = ["\n[Past review decisions — DO NOT repeat rejected suggestions]"]
    char_count = len(lines[0])

    for d in unique:
        suggestion = (d.get("content") or d.get("title", ""))[:100]
        reason = (d.get("reason") or "")[:150]
        pr = d.get("pr", "?")
        entry = f"\n- REJECTED in PR #{pr}: \"{suggestion}\""
        if reason:
            entry += f" — {reason}"
        if char_count + len(entry) > max_chars:
            break
        lines.append(entry)
        char_count += len(entry)

    lines.append(
        "\nThe suggestions above were already reviewed and rejected. "
        "Do NOT repeat them."
    )
    return "".join(lines)


# ---------------------------------------------------------------------------
# 5.  Store a decision in Engram DB
# ---------------------------------------------------------------------------


def store_decision(
    engram_dir: str,
    repo: str,
    suggestion: str,
    file_pattern: str,
    decision: str,
    reason: str,
    pr_number: int,
    session_id: str | None = None,
) -> int:
    """Persist a review decision into the Engram DB.

    ``decision`` must be one of: ``accepted``, ``rejected``, ``wont-fix``,
    ``out-of-scope``, ``pending``.

    Returns the observation ``id``.
    """
    ensure_engram(engram_dir)
    conn = _db(engram_dir)
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        _session_id = session_id or f"pr-{pr_number}"
        _sync_id = f"obs-{uuid4().hex[:16]}"

        # topic_key encodes structured info:  review:decision:reason:pr
        _reason = reason.replace(":", " ")[:200]
        topic_key = (
            f"{_SUGGESTION_HASH_PREFIX}{decision}:{_reason}:{pr_number}"
        )

        conn.execute(
            """
            INSERT INTO sessions (id, project, directory, started_at)
            VALUES (?, ?, '', ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (_session_id, repo, now),
        )

        conn.execute(
            """
            INSERT INTO observations
                (sync_id, session_id, type, title, content, project,
                 scope, topic_key, normalized_hash, last_seen_at,
                 created_at, updated_at)
            VALUES (?, ?, 'review-decision', ?, ?, ?,
                    'project', ?, ?, ?,
                    ?, ?)
            """,
            (
                _sync_id,
                _session_id,
                f"Review decision: {decision} — {suggestion[:80]}",
                suggestion,
                repo,
                topic_key,
                _content_hash(suggestion),
                now,
                now,
                now,
            ),
        )

        conn.commit()
        obs_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.info(
            f"Stored review decision #{obs_id}: {decision} — "
            f"{suggestion[:60]}"
        )
        return obs_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 6.  Batch-store multiple decisions (token-efficient: single transaction)
# ---------------------------------------------------------------------------


def store_decisions_batch(
    engram_dir: str,
    repo: str,
    pr_number: int,
    decisions: list[dict[str, str]],
    session_id: str | None = None,
) -> int:
    """Persist multiple review decisions in a single transaction.

    *decisions* is a list of ``{"suggestion", "file_pattern", "decision", "reason"}``.

    Returns the count of stored observations.
    """
    ensure_engram(engram_dir)
    conn = _db(engram_dir)
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        _session_id = session_id or f"pr-{pr_number}"

        conn.execute(
            """
            INSERT INTO sessions (id, project, directory, started_at)
            VALUES (?, ?, '', ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (_session_id, repo, now),
        )

        count = 0
        for dec in decisions:
            suggestion = dec.get("suggestion", "")
            decision = dec.get("decision", "pending")
            reason = dec.get("reason", "")
            topic_key = (
                f"{_SUGGESTION_HASH_PREFIX}{decision}:"
                f"{reason.replace(':', ' ')[:200]}:{pr_number}"
            )
            conn.execute(
                """
                INSERT INTO observations
                    (sync_id, session_id, type, title, content, project,
                     scope, topic_key, normalized_hash, last_seen_at,
                     created_at, updated_at)
                VALUES (?, ?, 'review-decision', ?, ?, ?,
                        'project', ?, ?, ?,
                        ?, ?)
                """,
                (
                    f"obs-{uuid4().hex[:16]}",
                    _session_id,
                    f"Review decision: {decision} — {suggestion[:80]}",
                    suggestion,
                    repo,
                    topic_key,
                    _content_hash(suggestion),
                    now,
                    now,
                    now,
                ),
            )
            count += 1

        conn.commit()
        logger.info(f"Stored {count} review decisions for PR #{pr_number}")
        return count
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().lower().encode()).hexdigest()[:16]


def observation_count(engram_dir: str, repo: str) -> int:
    """Return the number of review decisions stored for *repo*."""
    ensure_engram(engram_dir)
    conn = _db(engram_dir)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM observations "
            "WHERE project = ? AND type = 'review-decision' AND deleted_at IS NULL",
            (repo,),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()
