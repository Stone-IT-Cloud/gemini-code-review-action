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
"""Golang configuration file parser."""

import re
from typing import Any

from code_reviewer.context.parsers.base_parser import BaseParser


def _parse_require_block(content: str) -> list[str]:
    """Extract dependencies from parenthesized require blocks."""
    deps: list[str] = []
    block = re.search(r"require\s*\((.*?)\)", content, re.DOTALL)
    if not block:
        return deps

    for line in block.group(1).split("\n"):
        line = line.strip()
        if line and not line.startswith("//"):
            parts = line.split()
            if len(parts) >= 2:
                deps.append(f"{parts[0]} {parts[1]}")
    return deps


def _parse_require_single(content: str) -> list[str]:
    """Extract dependencies from single-line require statements."""
    deps: list[str] = []
    for match in re.finditer(r"^require\s+(?!\()(\S+)\s+(\S+)", content, re.MULTILINE):
        module_name, version = match.groups()
        deps.append(f"{module_name} {version}")
    return deps


class GolangParser(BaseParser):
    """Parser for go.mod files."""

    def parse(self, content: str) -> dict[str, Any]:
        """Parse go.mod content."""
        result: dict[str, Any] = {"type": "go.mod"}

        module_match = re.search(r"^module\s+(\S+)", content, re.MULTILINE)
        if module_match:
            result["module"] = module_match.group(1)

        go_match = re.search(r"^go\s+(\S+)", content, re.MULTILINE)
        if go_match:
            result["go_version"] = go_match.group(1)

        deps = _parse_require_block(content) + _parse_require_single(content)
        if deps:
            result["dependencies"] = deps[:30]

        return result
