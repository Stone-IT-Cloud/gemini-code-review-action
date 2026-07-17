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
"""Tests for src/utils.py — utility functions."""

from src.utils import CHAR_PER_TOKEN_ESTIMATE, calculate_char_budget

# ---------------------------------------------------------------------------
# calculate_char_budget
# ---------------------------------------------------------------------------

class TestCalculateCharBudget:
    """Test the calculate_char_budget pure function."""

    def test_typical_limit(self):
        """Standard case: 1_000_000 tokens, 20% overhead."""
        # token_limit * (1 - 0.2) * 2.0 = 1_000_000 * 0.8 * 2.0 = 1_600_000
        assert calculate_char_budget(1_000_000, 0.2) == 1_600_000

    def test_zero_overhead(self):
        """No overhead means 100% of tokens used."""
        assert calculate_char_budget(1_000_000, 0.0) == 2_000_000

    def test_full_overhead(self):
        """100% overhead means zero budget."""
        assert calculate_char_budget(1_000_000, 1.0) == 0

    def test_small_token_limit(self):
        """Very small token limit yields small budget."""
        assert calculate_char_budget(100, 0.2) == 160

    def test_zero_token_limit(self):
        """Zero token limit yields zero budget."""
        assert calculate_char_budget(0, 0.2) == 0

    def test_overhead_50_percent(self):
        """50% overhead reserves half the tokens."""
        assert calculate_char_budget(1_000_000, 0.5) == 1_000_000

    def test_constant_value(self):
        """CHAR_PER_TOKEN_ESTIMATE is 2.0 as specified in the design."""
        assert CHAR_PER_TOKEN_ESTIMATE == 2.0
