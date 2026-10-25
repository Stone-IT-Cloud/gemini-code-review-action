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
"""Short-lived health flags for LLM models.

A model that answers ``503 UNAVAILABLE`` is saturated at the provider, not at
the caller. Marking it here lets later chunks -- and later workflow runs -- skip
it instead of burning their retry budget again.

The store is deliberately best-effort: a missing, corrupt, or unwritable state
file degrades to in-memory state and never fails the review.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable

from loguru import logger

DEFAULT_TTL_SECONDS = 3600
_STATE_VERSION = 1


def _parse_entries(models: dict) -> dict[str, dict]:
    """Keep only the well-formed entries of a loaded state file."""
    parsed: dict[str, dict] = {}
    for key, entry in models.items():
        if not isinstance(entry, dict):
            continue
        try:
            blocked_until = float(entry.get("blocked_until", 0))
        except (TypeError, ValueError):
            # Syntactically valid JSON with a semantically broken entry must not
            # take the whole review down.
            continue
        code = entry.get("code")
        parsed[str(key)] = {
            "blocked_until": blocked_until,
            "code": code if isinstance(code, int) else None,
            "reason": str(entry.get("reason", ""))[:500],
        }
    return parsed


class ModelHealthStore:
    """JSON-backed map of ``provider:model -> blocked_until`` with a TTL.

    Args:
        path: State file location. ``None`` keeps everything in memory.
        ttl_seconds: How long a flagged model stays blocked.
        clock: Injectable time source, used by tests.
    """

    def __init__(
        self,
        path: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Open the store; nothing is read from disk until it is first used."""
        self.path = path
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._clock = clock
        self._entries: dict[str, dict] = {}
        self._loaded = False
        self._persistent = path is not None
        self._dirty = False

    @classmethod
    def from_env(cls) -> ModelHealthStore:
        """Build a store from ``LLM_MODEL_HEALTH_FILE``/``..._TTL_SECONDS``."""
        ttl_seconds = DEFAULT_TTL_SECONDS
        raw_ttl = os.getenv("LLM_MODEL_HEALTH_TTL_SECONDS")
        if raw_ttl is not None and str(raw_ttl).strip():
            try:
                ttl_seconds = int(str(raw_ttl).strip())
            except ValueError:
                logger.warning(f"Invalid LLM_MODEL_HEALTH_TTL_SECONDS={raw_ttl!r}; using {DEFAULT_TTL_SECONDS}s")
        path = os.getenv("LLM_MODEL_HEALTH_FILE") or None
        return cls(path=path, ttl_seconds=ttl_seconds)

    @property
    def dirty(self) -> bool:
        """True when this run changed the state (worth persisting)."""
        return self._dirty

    def blocked_models(self) -> set[str]:
        """Return the keys still blocked, pruning expired entries."""
        now = self._clock()
        self._load()
        expired = [key for key, entry in self._entries.items() if float(entry.get("blocked_until", 0)) <= now]
        for key in expired:
            del self._entries[key]
            self._dirty = True
        if expired:
            expired_label = ", ".join(sorted(expired))
            logger.info(f"Model health flags expired: {expired_label}")
            self._persist()
        return set(self._entries)

    def is_blocked(self, key: str) -> bool:
        """Whether *key* is currently flagged as saturated."""
        return key in self.blocked_models()

    def block(
        self,
        key: str,
        *,
        code: int | None = None,
        reason: str = "",
        ttl_seconds: int | None = None,
    ) -> None:
        """Flag *key* as saturated for the configured TTL."""
        ttl = self.ttl_seconds if ttl_seconds is None else max(0, int(ttl_seconds))
        self._load()
        self._entries[key] = {
            "blocked_until": self._clock() + ttl,
            "code": code,
            "reason": reason[:500],
        }
        self._dirty = True
        logger.warning(f"Flagging model {key} as saturated for {ttl}s")
        self._persist()

    def unblock(self, key: str) -> None:
        """Remove a flag, e.g. after a successful probe."""
        self._load()
        if self._entries.pop(key, None) is not None:
            self._dirty = True
            self._persist()

    def snapshot(self) -> dict[str, dict]:
        """Copy of the raw state, for logging and tests."""
        self._load()
        return {key: dict(entry) for key, entry in self._entries.items()}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._persistent or not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning(f"Ignoring unreadable model health state {self.path}: {exc}")
            return
        if not isinstance(payload, dict):
            logger.warning(f"Ignoring model health state with unexpected shape: {payload!r}")
            return
        models = payload.get("models")
        if not isinstance(models, dict):
            return
        self._entries = _parse_entries(models)
        if self._entries:
            loaded_label = ", ".join(sorted(self._entries))
            logger.info(f"Loaded model health state: {loaded_label}")

    def _persist(self) -> None:
        if not self._persistent or not self.path:
            return
        directory = os.path.dirname(self.path) or "."
        tmp_path = f"{self.path}.tmp-{os.getpid()}"
        payload = {"version": _STATE_VERSION, "models": self._entries}
        try:
            os.makedirs(directory, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, self.path)
        except OSError as exc:
            logger.warning(f"Could not persist model health state to {self.path} ({exc}); keeping it in memory")
            self._persistent = False
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
