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
""".NET configuration file parser."""

import re
from typing import Any, Dict

from src.context.parsers.base_parser import BaseParser


class DotNetParser(BaseParser):
    """Parser for .csproj/.fsproj/.vbproj files."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse .NET project file content."""
        result = {"type": "dotnet_project"}

        # Extract target framework
        framework_match = re.search(r'<TargetFramework>(.*?)</TargetFramework>', content)
        if framework_match:
            result["target_framework"] = framework_match.group(1)

        # Extract package references
        packages = re.findall(r'<PackageReference\s+Include="([^"]+)"\s+Version="([^"]+)"', content)
        if packages:
            result["packages"] = [f"{name} {version}" for name, version in packages[:30]]

        return result
