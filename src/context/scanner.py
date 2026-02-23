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
"""Context scanner for polyglot project analysis."""

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from src.context.parsers import (DockerComposeParser, DockerParser,
                                 DotNetParser, GolangParser, GradleParser,
                                 HelmParser, JavaScriptParser,
                                 KubernetesParser, MavenParser, PHPParser,
                                 PythonPipfileParser, PythonPyprojectParser,
                                 PythonRequirementsParser, RubyParser,
                                 RustParser, TerraformParser)

# Token budget constraints
MAX_BYTES_PER_FILE = 2048  # 2KB
MAX_LINES_PER_FILE = 50

# Exceptions that parsers may raise (used to avoid broad Exception)
PARSER_EXCEPTIONS = (ValueError, KeyError, TypeError, AttributeError, IndexError, OSError)


class ContextScanner:
    """Scanner for detecting and parsing project configuration files."""

    def __init__(self, repo_root: str):
        """Initialize the context scanner.

        Args:
            repo_root: Path to the repository root directory
        """
        self.repo_root = Path(repo_root)
        self.context: Dict[str, Any] = {}

    def read_file_limited(self, filepath: Path) -> Optional[str]:
        """Read file content with token budget limits.

        Limits content to 2KB or 50 lines, whichever comes first.
        Uses best-effort approach: returns None on any error.

        Args:
            filepath: Path to the file to read

        Returns:
            File content as string, or None if reading fails
        """
        try:
            if not filepath.exists() or not filepath.is_file():
                return None

            # Read with byte limit (read in binary mode then decode)
            with open(filepath, "rb") as file:
                raw_bytes = file.read(MAX_BYTES_PER_FILE)

            # Decode with best-effort error handling
            content = raw_bytes.decode("utf-8", errors="ignore")

            # Apply line limit
            lines = content.split("\n")
            if len(lines) > MAX_LINES_PER_FILE:
                content = "\n".join(lines[:MAX_LINES_PER_FILE])

            return content
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug(f"Failed to read {filepath}: {exc}")
            return None

    def _scan_python(self) -> None:
        """Scan for Python configuration files."""
        # requirements.txt
        req_path = self.repo_root / "requirements.txt"
        content = self.read_file_limited(req_path)
        if content is not None:
            try:
                parser = PythonRequirementsParser()
                self.context["python_requirements"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse requirements.txt: {exc}")

        # Pipfile
        pipfile_path = self.repo_root / "Pipfile"
        content = self.read_file_limited(pipfile_path)
        if content is not None:
            try:
                parser = PythonPipfileParser()
                self.context["python_pipfile"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse Pipfile: {exc}")

        # pyproject.toml
        pyproject_path = self.repo_root / "pyproject.toml"
        content = self.read_file_limited(pyproject_path)
        if content is not None:
            try:
                parser = PythonPyprojectParser()
                self.context["python_pyproject"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse pyproject.toml: {exc}")

    def _scan_php(self) -> None:
        """Scan for PHP configuration files."""
        composer_path = self.repo_root / "composer.json"
        content = self.read_file_limited(composer_path)
        if content is not None:
            try:
                parser = PHPParser()
                self.context["php_composer"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse composer.json: {exc}")

    def _scan_javascript(self) -> None:
        """Scan for JavaScript/TypeScript configuration files."""
        package_path = self.repo_root / "package.json"
        content = self.read_file_limited(package_path)
        if content is not None:
            try:
                parser = JavaScriptParser()
                self.context["javascript_package"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse package.json: {exc}")

    def _scan_golang(self) -> None:
        """Scan for Golang configuration files."""
        gomod_path = self.repo_root / "go.mod"
        content = self.read_file_limited(gomod_path)
        if content is not None:
            try:
                parser = GolangParser()
                self.context["golang_mod"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse go.mod: {exc}")

    def _scan_ruby(self) -> None:
        """Scan for Ruby configuration files."""
        gemfile_path = self.repo_root / "Gemfile"
        content = self.read_file_limited(gemfile_path)
        if content is not None:
            try:
                parser = RubyParser()
                self.context["ruby_gemfile"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse Gemfile: {exc}")

    def _scan_java(self) -> None:
        """Scan for Java configuration files."""
        # Maven pom.xml
        pom_path = self.repo_root / "pom.xml"
        content = self.read_file_limited(pom_path)
        if content is not None:
            try:
                parser = MavenParser()
                self.context["java_pom"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse pom.xml: {exc}")

        # Gradle build.gradle
        gradle_path = self.repo_root / "build.gradle"
        content = self.read_file_limited(gradle_path)
        if content is not None:
            try:
                parser = GradleParser()
                self.context["java_gradle"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse build.gradle: {exc}")

    def _scan_dotnet(self) -> None:
        """Scan for .NET configuration files."""
        # Look for .csproj, .fsproj, .vbproj files
        for pattern in ["*.csproj", "*.fsproj", "*.vbproj"]:
            for proj_file in self.repo_root.glob(pattern):
                content = self.read_file_limited(proj_file)
                if content is not None:
                    try:
                        parser = DotNetParser()
                        key = f"dotnet_{proj_file.stem}"
                        self.context[key] = parser.parse(content)
                        return  # Exit function after parsing first project file
                    except PARSER_EXCEPTIONS as exc:
                        logger.debug(f"Failed to parse {proj_file}: {exc}")

    def _scan_rust(self) -> None:
        """Scan for Rust configuration files."""
        cargo_path = self.repo_root / "Cargo.toml"
        content = self.read_file_limited(cargo_path)
        if content is not None:
            try:
                parser = RustParser()
                self.context["rust_cargo"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse Cargo.toml: {exc}")

    def _scan_docker(self) -> None:
        """Scan for Docker configuration files."""
        # Dockerfile
        dockerfile_path = self.repo_root / "Dockerfile"
        content = self.read_file_limited(dockerfile_path)
        if content is not None:
            try:
                parser = DockerParser()
                self.context["docker_dockerfile"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse Dockerfile: {exc}")

        # docker-compose files (v1 and v2 naming conventions)
        compose_files = [
            "docker-compose.yml",
            "docker-compose.yaml",
            "compose.yml",
            "compose.yaml"
        ]
        for compose_name in compose_files:
            compose_path = self.repo_root / compose_name
            content = self.read_file_limited(compose_path)
            if content is not None:
                try:
                    parser = DockerComposeParser()
                    self.context["docker_compose"] = parser.parse(content)
                    break
                except PARSER_EXCEPTIONS as exc:
                    logger.debug(f"Failed to parse {compose_name}: {exc}")

    def _scan_kubernetes(self) -> None:
        """Scan for Kubernetes configuration files."""
        # Scan YAML files across the entire repo, focusing on specific Kubernetes kinds
        target_kinds = {"Deployment", "StatefulSet", "Service"}
        k8s_resources = []
        max_resources = 10
        max_files_scanned = 200
        files_scanned = 0

        # Walk the repository and look for *.yaml / *.yml files
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix not in (".yaml", ".yml"):
                continue

            files_scanned += 1
            if files_scanned > max_files_scanned or len(k8s_resources) >= max_resources:
                break

            content = self.read_file_limited(path)
            if content is None:
                continue

            try:
                parser = KubernetesParser()
                parsed = parser.parse(content)
                kind = parsed.get("kind")
                if kind in target_kinds:
                    parsed["filename"] = path.name
                    k8s_resources.append(parsed)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse {path}: {exc}")

        if k8s_resources:
            self.context["kubernetes_resources"] = {
                "type": "kubernetes",
                "resources": k8s_resources[:max_resources]
            }

    def _scan_helm(self) -> None:
        """Scan for Helm chart files."""
        # Look for Chart.yaml
        chart_path = self.repo_root / "Chart.yaml"
        content = self.read_file_limited(chart_path)
        if content is not None:
            try:
                parser = HelmParser()
                self.context["helm_chart"] = parser.parse(content)
            except PARSER_EXCEPTIONS as exc:
                logger.debug(f"Failed to parse Chart.yaml: {exc}")

        # Look for values.yaml and extract first 50 lines for global config
        values_path = self.repo_root / "values.yaml"
        content = self.read_file_limited(values_path)
        if content is not None:
            # Store first 50 lines as they often contain global config
            lines = content.split("\n")[:50]  # First 50 lines as preview
            self.context["helm_values"] = {
                "type": "values.yaml",
                "preview": "\n".join(lines)
            }

    def _scan_terraform(self) -> None:
        """Scan for Terraform configuration files."""
        # Find .tf files
        tf_files = list(self.repo_root.glob("*.tf"))
        if tf_files:
            # Prefer main.tf, then provider.tf, then fall back to the first file
            selected_tf = next((f for f in tf_files if f.name == "main.tf"), None)
            if selected_tf is None:
                selected_tf = next((f for f in tf_files if f.name == "provider.tf"), None)
            if selected_tf is None:
                selected_tf = tf_files[0]

            content = self.read_file_limited(selected_tf)
            if content is not None:
                try:
                    parser = TerraformParser()
                    self.context["terraform"] = parser.parse(content)
                    self.context["terraform"]["files_found"] = [f.name for f in tf_files[:10]]
                except PARSER_EXCEPTIONS as exc:
                    logger.debug(f"Failed to parse {selected_tf}: {exc}")

    def _scan_documentation(self) -> None:
        """Scan for critical Markdown documentation."""
        docs = {}

        # Scan root directory for common documentation files
        doc_files = ["README.md", "CONTRIBUTING.md", "ARCHITECTURE.md", "SECURITY.md"]
        for doc_file in doc_files:
            doc_path = self.repo_root / doc_file
            content = self.read_file_limited(doc_path)
            if content:
                # Extract first ~2000 chars for better context
                if len(content) > 2000:
                    content = content[:2000] + "..."
                docs[doc_file] = content

        # Scan docs/ directory for additional documentation (top 2 files)
        docs_dir = self.repo_root / "docs"
        if docs_dir.exists() and docs_dir.is_dir():
            md_files = list(docs_dir.glob("*.md"))[:2]  # Limit to first 2 files
            for doc_file in md_files:
                content = self.read_file_limited(doc_file)
                if content:
                    # Extract first ~2000 chars for better context
                    if len(content) > 2000:
                        content = content[:2000] + "..."
                    docs[f"docs/{doc_file.name}"] = content

        if docs:
            self.context["documentation"] = {
                "type": "markdown_docs",
                "files": docs
            }

    def scan(self) -> Dict[str, Any]:
        """Scan the repository for configuration files.

        Returns:
            Dictionary containing parsed context from all detected files
        """
        # Clear context to ensure deterministic results per scan
        self.context = {}

        logger.info(f"Scanning repository at {self.repo_root}")

        # Scan all supported technologies
        self._scan_python()
        self._scan_php()
        self._scan_javascript()
        self._scan_golang()
        self._scan_ruby()
        self._scan_java()
        self._scan_dotnet()
        self._scan_rust()
        self._scan_docker()
        self._scan_kubernetes()
        self._scan_helm()
        self._scan_terraform()
        self._scan_documentation()

        logger.info(f"Found {len(self.context)} context items")
        return self.context

    def get_context_summary(self) -> str:
        """Generate a formatted summary of the scanned context.

        Returns:
            Human-readable context summary string
        """
        if not self.context:
            return "No project context detected."

        lines = ["## Project Context"]

        # Language/Framework detection
        detected_techs = []
        if ("python_requirements" in self.context or
                "python_pipfile" in self.context or
                "python_pyproject" in self.context):
            detected_techs.append("Python")
        if "php_composer" in self.context:
            detected_techs.append("PHP")
        if "javascript_package" in self.context:
            detected_techs.append("JavaScript/TypeScript")
        if "golang_mod" in self.context:
            detected_techs.append("Go")
        if "ruby_gemfile" in self.context:
            detected_techs.append("Ruby")
        if "java_pom" in self.context or "java_gradle" in self.context:
            detected_techs.append("Java")
        if any(k.startswith("dotnet_") for k in self.context):
            detected_techs.append(".NET")
        if "rust_cargo" in self.context:
            detected_techs.append("Rust")

        if detected_techs:
            tech_list = ", ".join(detected_techs)
            lines.append(f"**Technologies:** {tech_list}")

        # Infrastructure
        infra_techs = []
        if "docker_dockerfile" in self.context or "docker_compose" in self.context:
            infra_techs.append("Docker")
        if "kubernetes_resources" in self.context:
            infra_techs.append("Kubernetes")
        if "helm_chart" in self.context:
            infra_techs.append("Helm")
        if "terraform" in self.context:
            infra_techs.append("Terraform")

        if infra_techs:
            infra_list = ", ".join(infra_techs)
            lines.append(f"**Infrastructure:** {infra_list}")

        # Framework-specific details
        if "php_composer" in self.context:
            composer = self.context["php_composer"]
            if "framework" in composer:
                fw = composer["framework"]
                lines.append(f"**PHP Framework:** {fw}")

        if "ruby_gemfile" in self.context:
            gemfile = self.context["ruby_gemfile"]
            if "rails_version" in gemfile:
                rv = gemfile["rails_version"]
                lines.append(f"**Rails Version:** {rv}")

        if "golang_mod" in self.context:
            gomod = self.context["golang_mod"]
            if "go_version" in gomod:
                gv = gomod["go_version"]
                lines.append(f"**Go Version:** {gv}")

        return "\n".join(lines)
