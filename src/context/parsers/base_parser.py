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
"""Base parser interface for configuration files."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseParser(ABC):
    """Abstract base class for configuration file parsers."""

    @abstractmethod
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse configuration file content.

        Args:
            content: File content as string

        Returns:
            Dictionary containing parsed data
        """
