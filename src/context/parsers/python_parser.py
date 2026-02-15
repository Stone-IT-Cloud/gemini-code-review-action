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
"""Python configuration file parsers."""

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


class PythonRequirementsParser(BaseParser):
    """Parser for requirements.txt files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse requirements.txt content."""
        if not content.strip():
            return {"type": "requirements.txt", "packages": []}

        lines = [line.strip() for line in content.split('\n')]
        packages = [
            line for line in lines
            if line and not line.startswith('#') and not line.startswith('-')
        ]
        return {"type": "requirements.txt", "packages": packages[:20]}


class PythonPipfileParser(BaseParser):
    """Parser for Pipfile files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse Pipfile content."""
        result = {"type": "Pipfile"}

        packages_section = _parse_toml_section(content, "packages")
        if packages_section:
            result["packages"] = packages_section

        dev_packages_section = _parse_toml_section(content, "dev-packages")
        if dev_packages_section:
            result["dev_packages"] = dev_packages_section

        return result


class PythonPyprojectParser(BaseParser):
    """Parser for pyproject.toml files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse pyproject.toml content."""
        result = {"type": "pyproject.toml"}

        # Parse Poetry dependencies from [tool.poetry.dependencies]
        poetry_deps = _parse_toml_section(content, "tool.poetry.dependencies")
        if poetry_deps:
            result["poetry_dependencies"] = poetry_deps[:500]

        # Parse PEP 621 dependencies from [project] table
        # dependencies is a key (array) inside [project], not a separate table
        project_section = _parse_toml_section(content, "project.dependencies")
        if project_section:
            result["project_dependencies"] = project_section[:500]
        else:
            # Try parsing dependencies array directly from [project] section
            project_match = re.search(r'\[project\](.*?)(?=\n\[|\Z)', content, re.DOTALL)
            if project_match:
                project_content = project_match.group(1)
                deps_match = re.search(
                    r'dependencies\s*=\s*\[(.*?)\]', project_content, re.DOTALL
                )
                if deps_match:
                    deps_str = deps_match.group(1)
                    # Parse array items (simple quoted strings)
                    deps = re.findall(r'["\']([^"\']+)["\']', deps_str)
                    if deps:
                        result["project_dependencies"] = deps[:500]

        return result
