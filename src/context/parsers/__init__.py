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
"""Parsers package for configuration file parsing."""

from src.context.parsers.docker_parser import DockerComposeParser, DockerParser
from src.context.parsers.dotnet_parser import DotNetParser
from src.context.parsers.golang_parser import GolangParser
from src.context.parsers.helm_parser import HelmParser
from src.context.parsers.java_parser import GradleParser, MavenParser
from src.context.parsers.javascript_parser import JavaScriptParser
from src.context.parsers.kubernetes_parser import KubernetesParser
from src.context.parsers.php_parser import PHPParser
from src.context.parsers.python_parser import (PythonPipfileParser,
                                               PythonPyprojectParser,
                                               PythonRequirementsParser)
from src.context.parsers.ruby_parser import RubyParser
from src.context.parsers.rust_parser import RustParser
from src.context.parsers.terraform_parser import TerraformParser

__all__ = [
    "DockerParser",
    "DockerComposeParser",
    "DotNetParser",
    "GolangParser",
    "GradleParser",
    "HelmParser",
    "JavaScriptParser",
    "KubernetesParser",
    "MavenParser",
    "PHPParser",
    "PythonPipfileParser",
    "PythonPyprojectParser",
    "PythonRequirementsParser",
    "RubyParser",
    "RustParser",
    "TerraformParser",
]
