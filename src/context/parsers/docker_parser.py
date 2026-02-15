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
"""Docker configuration file parsers."""

import re
from typing import Any, Dict

from src.context.parsers.base_parser import BaseParser


class DockerParser(BaseParser):
    """Parser for Dockerfile files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse Dockerfile content."""
        result = {"type": "Dockerfile"}

        # Extract base images
        from_statements = re.findall(r'^FROM\s+(.+?)(?:\s+as\s+.+)?$', content, re.MULTILINE | re.IGNORECASE)
        if from_statements:
            result["base_images"] = from_statements

        # Extract exposed ports
        expose_statements = re.findall(r'^EXPOSE\s+(.+)$', content, re.MULTILINE | re.IGNORECASE)
        if expose_statements:
            result["exposed_ports"] = expose_statements

        return result


class DockerComposeParser(BaseParser):
    """Parser for docker-compose.yml files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse docker-compose.yml content."""
        result = {"type": "docker-compose.yml"}

        # Extract service names
        services_match = re.search(r'services:(.*?)(?:\n\S|\Z)', content, re.DOTALL)
        if services_match:
            service_lines = services_match.group(1).split('\n')
            services = []
            for line in service_lines:
                service_match = re.match(r'\s+([A-Za-z0-9_-]+)\s*:', line)
                if service_match:
                    services.append(service_match.group(1))
            if services:
                result["services"] = services

        return result
