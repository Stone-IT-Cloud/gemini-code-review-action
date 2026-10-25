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
"""Tests for the short-lived model health store."""

from __future__ import annotations

import json

import pytest

from code_reviewer.model_health import ModelHealthStore


class TestModelHealthStore:
    """TTL, persistence and corruption tolerance."""

    def test_block_and_report(self, tmp_path):
        store = ModelHealthStore(path=str(tmp_path / "health.json"), ttl_seconds=3600)

        assert store.blocked_models() == set()

        store.block("gemini:gemini-3.8-flash", code=503, reason="503 UNAVAILABLE")

        assert store.is_blocked("gemini:gemini-3.8-flash")
        assert store.blocked_models() == {"gemini:gemini-3.8-flash"}

    def test_blocked_entry_expires_after_ttl(self, tmp_path):
        now = [1000.0]
        store = ModelHealthStore(
            path=str(tmp_path / "health.json"),
            ttl_seconds=3600,
            clock=lambda: now[0],
        )
        store.block("opaque:model", code=503)

        now[0] += 3599
        assert store.is_blocked("opaque:model")

        now[0] += 2
        assert not store.is_blocked("opaque:model")

    def test_state_is_shared_across_instances(self, tmp_path):
        path = str(tmp_path / "health.json")
        first = ModelHealthStore(path=path, ttl_seconds=3600, clock=lambda: 1000.0)
        first.block("gemini:gemini-3.8-flash", code=503, reason="503 UNAVAILABLE")

        second = ModelHealthStore(path=path, ttl_seconds=3600, clock=lambda: 1000.0)
        assert second.blocked_models() == {"gemini:gemini-3.8-flash"}

    def test_corrupt_state_file_is_ignored(self, tmp_path):
        path = tmp_path / "health.json"
        path.write_text("{not json", encoding="utf-8")

        store = ModelHealthStore(path=str(path), ttl_seconds=3600)

        assert store.blocked_models() == set()

    @pytest.mark.parametrize(
        "payload",
        [
            '{"version":1,"models":{"m":{"blocked_until":null}}}',
            '{"version":1,"models":{"m":{"blocked_until":"soon"}}}',
            '{"version":1,"models":{"m":{"blocked_until":[1]}}}',
        ],
    )
    def test_semantically_corrupt_entry_never_raises(self, tmp_path, payload):
        path = tmp_path / "health.json"
        path.write_text(payload, encoding="utf-8")

        store = ModelHealthStore(path=str(path), ttl_seconds=3600, clock=lambda: 1000.0)

        assert store.blocked_models() == set()
        assert store.snapshot() == {}

    def test_unwritable_path_degrades_to_memory(self, tmp_path):
        # A directory cannot be opened as a file: persistence must fail softly.
        store = ModelHealthStore(path=str(tmp_path), ttl_seconds=3600)

        store.block("gemini:gemini-3.8-flash", code=503)

        assert store.is_blocked("gemini:gemini-3.8-flash")

    def test_dirty_only_when_state_changed(self, tmp_path):
        store = ModelHealthStore(path=str(tmp_path / "health.json"), ttl_seconds=3600)
        store.blocked_models()
        assert not store.dirty

        store.block("gemini:gemini-3.6-flash", code=503)
        assert store.dirty

    def test_expired_entries_are_pruned_from_disk(self, tmp_path):
        path = tmp_path / "health.json"
        now = [1000.0]
        writer = ModelHealthStore(path=str(path), ttl_seconds=3600, clock=lambda: now[0])
        writer.block("gemini:gemini-3.8-flash", code=503)

        now[0] += 4000
        reader = ModelHealthStore(path=str(path), ttl_seconds=3600, clock=lambda: now[0])
        assert reader.blocked_models() == set()

        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["models"] == {}

    def test_env_configuration(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_MODEL_HEALTH_FILE", str(tmp_path / "health.json"))
        monkeypatch.setenv("LLM_MODEL_HEALTH_TTL_SECONDS", "60")

        store = ModelHealthStore.from_env()

        assert store.ttl_seconds == 60
        assert store.path == str(tmp_path / "health.json")

    def test_invalid_ttl_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_HEALTH_TTL_SECONDS", "not-a-number")

        store = ModelHealthStore.from_env()

        assert store.ttl_seconds == 3600
