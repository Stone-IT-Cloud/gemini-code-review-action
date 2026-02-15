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
"""Helm configuration file parser."""

import re
from typing import Any, Dict

from src.context.parsers.base_parser import BaseParser


class HelmParser(BaseParser):
    """Parser for Chart.yaml files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse Chart.yaml content."""
        result = {"type": "Chart.yaml"}

        # Extract chart name
        name_match = re.search(r'name:\s*(.+)', content)
        if name_match:
            result["name"] = name_match.group(1).strip()

        # Extract version
        version_match = re.search(r'version:\s*(.+)', content)
        if version_match:
            result["version"] = version_match.group(1).strip()

        # Extract appVersion
        app_version_match = re.search(r'appVersion:\s*(.+)', content)
        if app_version_match:
            result["appVersion"] = app_version_match.group(1).strip()

        return result
