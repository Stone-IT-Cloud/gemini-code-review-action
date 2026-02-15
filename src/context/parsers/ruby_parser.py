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
"""Ruby configuration file parser."""

import re
from typing import Any, Dict

from src.context.parsers.base_parser import BaseParser


class RubyParser(BaseParser):
    """Parser for Gemfile files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse Gemfile content."""
        result = {"type": "Gemfile"}

        # Look for Rails
        rails_match = re.search(r"gem\s+['\"]rails['\"],?\s+['\"]([^'\"]+)['\"]", content)
        if rails_match:
            result["rails_version"] = rails_match.group(1)

        # Extract gem dependencies
        gems = re.findall(r"gem\s+['\"]([^'\"]+)['\"]", content)
        if gems:
            result["gems"] = gems[:30]

        return result
