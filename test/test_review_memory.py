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
"""Tests for src/review_memory.py — Engram-backed cross-PR review memory."""

import json
import os
import tempfile

from src.review_memory import (
    _content_hash,
    _fts_terms,
    build_context,
    ensure_engram,
    load_decisions,
    observation_count,
    query_similar,
    store_decision,
    store_decisions_batch,
)

# ---------------------------------------------------------------------------
# _content_hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_consistent_hash(self):
        """Same input always produces the same hash."""
        assert _content_hash("Use f-strings") == _content_hash("Use f-strings")

    def test_case_insensitive(self):
        assert _content_hash("Use F-STRINGS") == _content_hash("use f-strings")

    def test_length(self):
        """Hash is always 16 hex characters."""
        assert len(_content_hash("any text")) == 16

    def test_different_inputs_different_hashes(self):
        assert _content_hash("foo") != _content_hash("bar")


# ---------------------------------------------------------------------------
# _fts_terms
# ---------------------------------------------------------------------------


class TestFtsTerms:
    def test_extracts_keywords(self):
        terms = _fts_terms("This function uses a loop that could be optimised")
        assert "loop" in terms
        assert "optimised" in terms
        assert "this" not in terms  # stop word removed

    def test_removes_stop_words(self):
        terms = _fts_terms("the and of in for with at by")
        assert terms == ""

    def test_short_tokens_removed(self):
        terms = _fts_terms("a an to of in it")
        assert terms == ""

    def test_limits_to_8_terms(self):
        long = "one two three four five six seven eight nine ten"
        terms = _fts_terms(long)
        assert len(terms.split(" AND ")) <= 8

    def test_removes_punctuation(self):
        terms = _fts_terms("use f-strings, not %-formatting!")
        assert "," not in terms
        assert "!" not in terms


# ---------------------------------------------------------------------------
# ensure_engram
# ---------------------------------------------------------------------------


class TestEnsureEngram:
    def test_creates_db_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = ensure_engram(tmp)
            assert db_path.exists()
            assert db_path.name == "engram.db"

    def test_creates_observations_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            ensure_engram(tmp)
            import sqlite3

            conn = sqlite3.connect(os.path.join(tmp, "engram.db"))
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            conn.close()
            assert "observations" in tables
            assert "sessions" in tables
            assert "observations_fts" in tables

    def test_reuses_existing_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = ensure_engram(tmp)
            p2 = ensure_engram(tmp)
            assert p1 == p2
            assert p1.exists()

    def test_engram_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "sub", "dir")
            db_path = ensure_engram(nested)
            assert db_path.exists()


# ---------------------------------------------------------------------------
# store_decision
# ---------------------------------------------------------------------------


class TestStoreDecision:
    def test_stores_and_returns_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs_id = store_decision(
                engram_dir=tmp,
                repo="owner/repo",
                suggestion="Use f-strings instead of % formatting",
                file_pattern="*.py",
                decision="rejected",
                reason="Project uses legacy format for consistency",
                pr_number=42,
            )
            assert isinstance(obs_id, int)
            assert obs_id > 0

    def test_stored_in_correct_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_decision(
                engram_dir=tmp,
                repo="owner/repo",
                suggestion="Add type hints",
                file_pattern="*.py",
                decision="accepted",
                reason="Good practice",
                pr_number=42,
            )
            count = observation_count(tmp, "owner/repo")
            assert count == 1

    def test_isolates_by_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_decision(
                engram_dir=tmp,
                repo="owner/repo",
                suggestion="Use f-strings",
                file_pattern="*.py",
                decision="rejected",
                reason="Consistency",
                pr_number=1,
            )
            count_other = observation_count(tmp, "other/repo")
            assert count_other == 0


# ---------------------------------------------------------------------------
# load_decisions
# ---------------------------------------------------------------------------


class TestLoadDecisions:
    def test_returns_empty_when_no_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ensure_engram(tmp)
            decisions = load_decisions(tmp, "owner/repo")
            assert decisions == []

    def test_returns_stored_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_decision(
                engram_dir=tmp,
                repo="owner/repo",
                suggestion="Add type hints",
                file_pattern="*.py",
                decision="accepted",
                reason="Good",
                pr_number=1,
            )
            decisions = load_decisions(tmp, "owner/repo", limit=10)
            assert len(decisions) == 1
            assert decisions[0]["decision"] == "accepted"

    def test_parses_topic_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_decision(
                engram_dir=tmp,
                repo="owner/repo",
                suggestion="Bad pattern",
                file_pattern="*.go",
                decision="rejected",
                reason="Not idiomatic Go",
                pr_number=7,
            )
            decisions = load_decisions(tmp, "owner/repo")
            assert decisions[0]["pr"] == 7
            assert decisions[0]["reason"] == "Not idiomatic Go"
            assert decisions[0]["decision"] == "rejected"

    def test_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                store_decision(
                    engram_dir=tmp,
                    repo="owner/repo",
                    suggestion=f"Suggestion {i}",
                    file_pattern="*.py",
                    decision="rejected",
                    reason="test",
                    pr_number=i,
                )
            decisions = load_decisions(tmp, "owner/repo", limit=3)
            assert len(decisions) <= 3


# ---------------------------------------------------------------------------
# query_similar (FTS5)
# ---------------------------------------------------------------------------


class TestQuerySimilar:
    def test_returns_empty_when_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_decision(
                engram_dir=tmp,
                repo="owner/repo",
                suggestion="Use f-strings for string formatting",
                file_pattern="*.py",
                decision="rejected",
                reason="Consistency",
                pr_number=1,
            )
            results = query_similar(
                tmp, "owner/repo", "unrelated topic about databases"
            )
            assert results == []

    def test_finds_similar_suggestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_decision(
                engram_dir=tmp,
                repo="owner/repo",
                suggestion="Use f-strings instead of percent formatting",
                file_pattern="*.py",
                decision="rejected",
                reason="Consistency",
                pr_number=1,
            )
            results = query_similar(
                tmp, "owner/repo", "using percent formatting"
            )
            assert len(results) >= 1
            assert results[0]["decision"] == "rejected"

    def test_respects_project_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_decision(
                engram_dir=tmp,
                repo="owner/repo",
                suggestion="Use f-strings",
                file_pattern="*.py",
                decision="rejected",
                reason="Consistency",
                pr_number=1,
            )
            results = query_similar(tmp, "other/repo", "use f strings")
            assert results == []


# ---------------------------------------------------------------------------
# store_decisions_batch
# ---------------------------------------------------------------------------


class TestStoreDecisionsBatch:
    def test_stores_multiple_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            decisions_list = [
                {"suggestion": "Use f-strings", "file_pattern": "*.py", "decision": "rejected", "reason": "Style"},
                {"suggestion": "Add type hints", "file_pattern": "*.py", "decision": "accepted", "reason": "Good"},
            ]
            count = store_decisions_batch(
                engram_dir=tmp,
                repo="owner/repo",
                pr_number=42,
                decisions=decisions_list,
            )
            assert count == 2
            assert observation_count(tmp, "owner/repo") == 2


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_empty_when_no_decisions(self):
        assert build_context([]) == ""

    def test_includes_rejected_decisions(self):
        decisions = [
            {"decision": "rejected", "content": "Use f-strings", "reason": "Style", "pr": 1},
        ]
        ctx = build_context(decisions)
        assert "REJECTED" in ctx
        assert "Use f-strings" in ctx
        assert "Style" in ctx
        assert "PR #1" in ctx

    def test_ignores_accepted_decisions(self):
        decisions = [
            {"decision": "accepted", "content": "Add types", "reason": "", "pr": 1},
        ]
        assert build_context(decisions) == ""

    def test_deduplicates_same_suggestion(self):
        decisions = [
            {"decision": "rejected", "content": "Use f-strings", "reason": "A", "pr": 1},
            {"decision": "rejected", "content": "Use f-strings", "reason": "B", "pr": 2},
        ]
        ctx = build_context(decisions)
        # Only one mention of f-strings in rejected
        assert ctx.count("f-strings") <= 1

    def test_respects_max_chars(self):
        decisions = [
            {"decision": "rejected", "content": "A" * 500, "reason": "B" * 500, "pr": 1},
            {"decision": "rejected", "content": "C" * 500, "reason": "D" * 500, "pr": 2},
        ]
        ctx = build_context(decisions, max_chars=200)
        assert len(ctx) <= 200


# ---------------------------------------------------------------------------
# REVIEW_SYSTEM_PROMPT — new sections
# ---------------------------------------------------------------------------


class TestReviewSystemPromptNewSections:
    from src.review_parser import REVIEW_SYSTEM_PROMPT

    def test_mentions_all_context(self):
        assert "Consider ALL available context" in self.REVIEW_SYSTEM_PROMPT

    def test_mentions_other_people_comments(self):
        assert "PR comments and reviews from other people" in self.REVIEW_SYSTEM_PROMPT

    def test_mentions_automated_analysis(self):
        assert "Bugbot" in self.REVIEW_SYSTEM_PROMPT
        assert "linters" in self.REVIEW_SYSTEM_PROMPT

    def test_mentions_test_results(self):
        assert "Test results and coverage reports" in self.REVIEW_SYSTEM_PROMPT
        assert "coverage" in self.REVIEW_SYSTEM_PROMPT

    def test_mentions_engram_memory(self):
        assert "Engram memory" in self.REVIEW_SYSTEM_PROMPT

    def test_mentions_issue_references(self):
        assert "Issue references" in self.REVIEW_SYSTEM_PROMPT
