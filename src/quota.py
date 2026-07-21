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
"""Quota-aware request throttling with exponential backoff and jitter.

Provider-agnostic: the ``QuotaTracker`` works with any LLM provider.
Per-provider error handling (``_handle_api_error``) lives in the
respective provider module (e.g. ``src/llm/gemini_client.py``).
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field

from src.utils import _get_usage_metadata, _safe_str


@dataclass
class QuotaTracker:
    """Track run-local usage and provide best-effort remaining-quota hints.

    The tracker logs:
    - actual per-request usage (when ``usage_metadata`` is provided)
    - run-estimated remaining RPM/TPM/RPD from *completed* requests if you
      provide limits via env vars (in-flight/pending requests are not counted)
    """

    window_seconds: int = 60
    request_timestamps: deque = field(default_factory=deque)
    token_events: deque = field(default_factory=deque)  # (timestamp, total_tokens)
    requests_total: int = 0
    tokens_total: int = 0
    last_pruned_at: float = 0.0
    prune_interval_seconds: float = 1.0

    quota_rpm: int | None = None
    quota_tpm: int | None = None
    quota_rpd: int | None = None

    @staticmethod
    def from_env(prefix: str = "GEMINI") -> QuotaTracker:
        """Create a QuotaTracker from environment variable config.

        Args:
            prefix: Env var prefix for quota limits (e.g. ``"GEMINI"``
                    reads ``GEMINI_QUOTA_RPM``, ``GEMINI_QUOTA_TPM``, etc.).

        Returns:
            A configured ``QuotaTracker`` instance.
        """

        def _parse_int(name: str) -> int | None:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return None
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer value for environment variable {name!r}: {raw!r}. "
                    "Please set it to a valid integer (e.g., '60') or leave it unset."
                ) from exc
            if value < 0:
                raise ValueError(
                    f"Invalid non-negative value for environment variable {name!r}: {value!r}. "
                    "Please set it to a non-negative integer or leave it unset."
                )
            return value

        return QuotaTracker(
            quota_rpm=_parse_int(f"{prefix}_QUOTA_RPM"),
            quota_tpm=_parse_int(f"{prefix}_QUOTA_TPM"),
            quota_rpd=_parse_int(f"{prefix}_QUOTA_RPD"),
        )

    def _prune(self, now: float) -> None:
        if now - self.last_pruned_at < self.prune_interval_seconds:
            return
        cutoff = now - self.window_seconds
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        while self.token_events and self.token_events[0][0] < cutoff:
            self.token_events.popleft()
        self.last_pruned_at = now

    def note_request(self, now: float) -> None:
        """Record a request at the given timestamp."""
        self.requests_total += 1
        self.request_timestamps.append(now)
        self._prune(now)

    def note_tokens(self, now: float, total_tokens: int) -> None:
        """Record token usage at the given timestamp."""
        self.tokens_total += int(total_tokens)
        self.token_events.append((now, int(total_tokens)))
        self._prune(now)

    def recent_rpm(self, now: float) -> int:
        """Return recent requests per minute."""
        self._prune(now)
        return len(self.request_timestamps)

    def recent_tpm(self, now: float) -> int:
        """Return recent tokens per minute."""
        self._prune(now)
        return sum(t for _, t in self.token_events)

    def remaining_estimate(self, now: float) -> dict:
        """Return run-estimated remaining quota (if limits are configured)."""
        rem: dict = {}
        if self.quota_rpm is not None:
            rem["rpm_remaining"] = max(0, self.quota_rpm - self.recent_rpm(now))
            rem["rpm_limit"] = self.quota_rpm
        if self.quota_tpm is not None:
            rem["tpm_remaining"] = max(0, self.quota_tpm - self.recent_tpm(now))
            rem["tpm_limit"] = self.quota_tpm
        if self.quota_rpd is not None:
            rem["rpd_remaining"] = max(0, self.quota_rpd - self.requests_total)
            rem["rpd_limit"] = self.quota_rpd
        return rem

    def has_all_quotas_set_to_zero(self) -> bool:
        """Check if all quota limits are explicitly configured and set to zero."""
        return (
            self.quota_rpm is not None
            and self.quota_tpm is not None
            and self.quota_rpd is not None
            and self.quota_rpm == 0
            and self.quota_tpm == 0
            and self.quota_rpd == 0
        )

    def log_after_response(self, response, label: str, prefix: str = "GEMINI") -> None:
        """Log usage metadata and remaining quota estimate after a response."""
        now = time.time()
        usage = _get_usage_metadata(response)
        total_tokens = usage.get("total_tokens")
        if total_tokens is not None:
            self.note_tokens(now, total_tokens)

        remaining = self.remaining_estimate(now)
        usage_bits = []
        if usage:
            prompt_tokens = usage.get("prompt_tokens", "?")
            output_tokens = usage.get("output_tokens", "?")
            total_tokens_val = usage.get("total_tokens", "?")
            usage_bits.append(
                "usage_tokens="
                f"prompt={prompt_tokens},"
                f"output={output_tokens},"
                f"total={total_tokens_val}"
            )
        if remaining:
            usage_bits.append("run_estimated_remaining=" + ",".join(f"{k}={v}" for k, v in remaining.items()))
        if not usage_bits:
            usage_bits.append("usage_metadata=<not provided by API>")
        joined_usage_bits = ", ".join(usage_bits)

        # Local import to avoid circular dependency at module level
        from loguru import logger  # pylint: disable=import-outside-toplevel

        logger.info(f"{label} {joined_usage_bits}")
