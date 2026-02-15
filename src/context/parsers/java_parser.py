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
"""Java configuration file parsers."""

import re
from typing import Any, Dict

from src.context.parsers.base_parser import BaseParser


class MavenParser(BaseParser):
    """Parser for pom.xml files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse pom.xml content."""
        result = {"type": "pom.xml"}

        # Extract groupId
        group_match = re.search(r'<groupId>(.*?)</groupId>', content)
        if group_match:
            result["groupId"] = group_match.group(1)

        # Extract artifactId
        artifact_match = re.search(r'<artifactId>(.*?)</artifactId>', content)
        if artifact_match:
            result["artifactId"] = artifact_match.group(1)

        # Extract Java version property
        java_version_match = re.search(r'<java\.version>(.*?)</java\.version>', content)
        if java_version_match:
            result["java_version"] = java_version_match.group(1)

        # Extract Maven compiler source
        maven_compiler_match = re.search(r'<maven\.compiler\.source>(.*?)</maven\.compiler\.source>', content)
        if maven_compiler_match:
            result["compiler_source"] = maven_compiler_match.group(1)

        return result


class GradleParser(BaseParser):
    """Parser for build.gradle files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse build.gradle content."""
        result = {"type": "build.gradle"}

        # Look for sourceCompatibility (capture without quotes)
        source_compat_match = re.search(r'sourceCompatibility\s*=\s*["\']?([^"\'\s]+)["\']?', content)
        if source_compat_match:
            result["sourceCompatibility"] = source_compat_match.group(1)

        # Look for dependencies block
        deps_match = re.search(r'dependencies\s*\{(.*?)\}', content, re.DOTALL)
        if deps_match:
            result["dependencies_snippet"] = deps_match.group(1)[:500]

        return result
