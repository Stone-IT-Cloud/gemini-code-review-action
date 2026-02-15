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
from typing import Any, Dict

from src.context.parsers.base_parser import BaseParser


class GolangParser(BaseParser):
    """Parser for go.mod files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse go.mod content."""
        result = {"type": "go.mod"}

        # Extract module name
        module_match = re.search(r'^module\s+(\S+)', content, re.MULTILINE)
        if module_match:
            result["module"] = module_match.group(1)

        # Extract Go version
        go_match = re.search(r'^go\s+(\S+)', content, re.MULTILINE)
        if go_match:
            result["go_version"] = go_match.group(1)

        # Extract direct dependencies
        deps = []

        # Handle parenthesized require blocks: require ( ... )
        require_block = re.search(r'require\s*\((.*?)\)', content, re.DOTALL)
        if require_block:
            for line in require_block.group(1).split('\n'):
                line = line.strip()
                if line and not line.startswith('//'):
                    parts = line.split()
                    if len(parts) >= 2:
                        deps.append(f"{parts[0]} {parts[1]}")

        # Handle single-line require statements: require module version
        for match in re.finditer(r'^require\s+(?!\()(\S+)\s+(\S+)', content, re.MULTILINE):
            module_name, version = match.groups()
            deps.append(f"{module_name} {version}")

        if deps:
            result["dependencies"] = deps[:30]

        return result
