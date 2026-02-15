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
"""Rust configuration file parser."""

import re
from typing import Any, Dict, Optional

from src.context.parsers.base_parser import BaseParser


def _parse_toml_section(content: str, section: str) -> Optional[str]:
    """Extract a TOML section from content."""
    try:
        pattern = rf'\[{re.escape(section)}\](.*?)(?:\n\[|$)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    except re.error:
        return None


class RustParser(BaseParser):
    """Parser for Cargo.toml files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse Cargo.toml content."""
        result = {"type": "Cargo.toml"}

        # Extract package name
        name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
        if name_match:
            result["name"] = name_match.group(1)

        # Extract edition
        edition_match = re.search(r'edition\s*=\s*"([^"]+)"', content)
        if edition_match:
            result["edition"] = edition_match.group(1)

        # Extract dependencies section
        deps_section = _parse_toml_section(content, "dependencies")
        if deps_section:
            result["dependencies"] = deps_section[:500]

        return result
