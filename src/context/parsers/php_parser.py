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
"""PHP configuration file parser."""

import json
from typing import Any, Dict, Optional

from src.context.parsers.base_parser import BaseParser


def _parse_json_safe(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON content safely."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None


class PHPParser(BaseParser):
    """Parser for composer.json files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse composer.json content."""
        data = _parse_json_safe(content)
        if not data:
            return {"type": "composer.json", "error": "Invalid JSON"}

        result = {"type": "composer.json"}

        if "require" in data:
            result["require"] = data["require"]
        if "require-dev" in data:
            result["require-dev"] = data["require-dev"]

        # Detect framework
        if "require" in data:
            if "laravel/framework" in data["require"]:
                result["framework"] = "Laravel"
            elif "symfony/symfony" in data["require"] or "symfony/framework-bundle" in data["require"]:
                result["framework"] = "Symfony"

        return result
