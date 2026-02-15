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
"""Terraform configuration file parser."""

import re
from typing import Any, Dict

from src.context.parsers.base_parser import BaseParser


class TerraformParser(BaseParser):
    """Parser for .tf files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse .tf content."""
        result = {"type": "terraform"}

        # Extract required providers
        providers = re.findall(r"required_providers\s*\{([^}]+)\}", content, re.DOTALL)
        if providers:
            providers_block = providers[0]
            # Keep a short preview snippet of the required_providers block
            result["required_providers_snippet"] = providers_block[:300]
            # Parse provider identifiers (e.g., aws, random) from the block
            provider_names = re.findall(r"([A-Za-z0-9_-]+)\s*=\s*\{", providers_block)
            if provider_names:
                # Return unique provider names; keep order stable by using dict.fromkeys
                result["required_providers"] = list(dict.fromkeys(provider_names))

        # Extract terraform version constraint
        version_match = re.search(r"required_version\s*=\s*\"([^\"]+)\"", content)
        if version_match:
            result["required_version"] = version_match.group(1)

        # Extract resource types
        resources = re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"', content)
        if resources:
            result["resources"] = [f"{rtype} {rname}" for rtype, rname in resources[:20]]

        return result
