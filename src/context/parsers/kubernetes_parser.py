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
"""Kubernetes configuration file parser."""

import re
from typing import Any, Dict

from src.context.parsers.base_parser import BaseParser


class KubernetesParser(BaseParser):
    """Parser for Kubernetes YAML files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse Kubernetes YAML content."""
        result = {"type": "kubernetes"}

        # Extract metadata.name (prefer name inside metadata: block)
        metadata_block_match = re.search(
            r"^metadata:\s*\n((?:[ \t].*\n?)*)", content, re.MULTILINE
        )
        name_match = None
        if metadata_block_match:
            metadata_block = metadata_block_match.group(1)
            name_match = re.search(r"^\s*name:\s*(.+)$", metadata_block, re.MULTILINE)
        if not name_match:
            # Fall back to a global search if metadata block not found
            name_match = re.search(r"^\s*name:\s*(.+)$", content, re.MULTILINE)
        if name_match:
            result["name"] = name_match.group(1).strip()

        # Extract kind
        kind_match = re.search(r"^kind:\s*(.+)$", content, re.MULTILINE)
        if kind_match:
            result["kind"] = kind_match.group(1).strip()

        # Extract container images
        images = re.findall(r"image:\s*(.+)", content)
        if images:
            result["container_images"] = [img.strip() for img in images[:5]]

        return result
