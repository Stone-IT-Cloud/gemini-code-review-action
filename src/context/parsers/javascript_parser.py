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
"""JavaScript/TypeScript configuration file parser."""

import json
from typing import Any, Dict, Optional

from src.context.parsers.base_parser import BaseParser


def _parse_json_safe(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON content safely."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None


class JavaScriptParser(BaseParser):
    """Parser for package.json files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse package.json content."""
        data = _parse_json_safe(content)
        if not data:
            return {"type": "package.json", "error": "Invalid JSON"}

        result = {"type": "package.json"}

        if "name" in data:
            result["name"] = data["name"]
        if "dependencies" in data:
            result["dependencies"] = data["dependencies"]
        if "devDependencies" in data:
            result["devDependencies"] = data["devDependencies"]
        if "scripts" in data:
            result["scripts"] = data["scripts"]

        return result
