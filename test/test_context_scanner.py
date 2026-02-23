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
"""Tests for the context scanner module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.context.scanner import (MAX_BYTES_PER_FILE, MAX_LINES_PER_FILE,
                                 ContextScanner)


class TestContextScannerFileReading:
    """Test file reading with token budget limits."""

    def test_read_file_within_limits(self):
        """Test reading a file within both byte and line limits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("line1\nline2\nline3")

            scanner = ContextScanner(tmpdir)
            content = scanner.read_file_limited(test_file)

            assert content == "line1\nline2\nline3"

    def test_read_file_exceeds_line_limit(self):
        """Test reading a file that exceeds the line limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            # Create file with more than MAX_LINES_PER_FILE lines
            lines = [f"line{i}" for i in range(MAX_LINES_PER_FILE + 10)]
            test_file.write_text("\n".join(lines))

            scanner = ContextScanner(tmpdir)
            content = scanner.read_file_limited(test_file)

            # Should only have MAX_LINES_PER_FILE lines
            assert content is not None
            assert content.count("\n") == MAX_LINES_PER_FILE - 1

    def test_read_file_exceeds_byte_limit(self):
        """Test reading a file that exceeds the byte limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            # Create file larger than MAX_BYTES_PER_FILE
            large_content = "x" * (MAX_BYTES_PER_FILE + 1000)
            test_file.write_text(large_content)

            scanner = ContextScanner(tmpdir)
            content = scanner.read_file_limited(test_file)

            # Should be limited to MAX_BYTES_PER_FILE
            assert content is not None
            assert len(content) <= MAX_BYTES_PER_FILE

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = ContextScanner(tmpdir)
            content = scanner.read_file_limited(Path(tmpdir) / "nonexistent.txt")

            assert content is None

    def test_read_directory_instead_of_file(self):
        """Test attempting to read a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = ContextScanner(tmpdir)
            content = scanner.read_file_limited(Path(tmpdir))

            assert content is None


class TestPythonScanning:
    """Test Python configuration file scanning."""

    def test_scan_requirements_txt(self):
        """Test scanning requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("requests==2.28.0\nflask>=2.0.0\ndjango")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "python_requirements" in context
            assert context["python_requirements"]["type"] == "requirements.txt"
            assert "requests==2.28.0" in context["python_requirements"]["packages"]

    def test_scan_pipfile(self):
        """Test scanning Pipfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipfile = Path(tmpdir) / "Pipfile"
            pipfile.write_text("""
[packages]
requests = "*"
flask = ">=2.0"

[dev-packages]
pytest = "*"
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "python_pipfile" in context
            assert context["python_pipfile"]["type"] == "Pipfile"

    def test_scan_pyproject_toml(self):
        """Test scanning pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("""
[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28.0"

[project]
dependencies = [
    "flask>=2.0",
]
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "python_pyproject" in context
            assert context["python_pyproject"]["type"] == "pyproject.toml"

    def test_scan_malformed_requirements(self):
        """Test that malformed requirements don't crash the scanner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            # Empty file
            req_file.write_text("")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            # Should still have entry but with empty packages
            assert "python_requirements" in context


class TestPHPScanning:
    """Test PHP configuration file scanning."""

    def test_scan_composer_json_laravel(self):
        """Test scanning composer.json with Laravel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            composer = Path(tmpdir) / "composer.json"
            composer.write_text(json.dumps({
                "require": {
                    "php": "^8.0",
                    "laravel/framework": "^9.0"
                },
                "require-dev": {
                    "phpunit/phpunit": "^9.5"
                }
            }))

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "php_composer" in context
            assert context["php_composer"]["framework"] == "Laravel"
            assert "laravel/framework" in context["php_composer"]["require"]

    def test_scan_composer_json_symfony(self):
        """Test scanning composer.json with Symfony."""
        with tempfile.TemporaryDirectory() as tmpdir:
            composer = Path(tmpdir) / "composer.json"
            composer.write_text(json.dumps({
                "require": {
                    "symfony/framework-bundle": "^6.0"
                }
            }))

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "php_composer" in context
            assert context["php_composer"]["framework"] == "Symfony"

    def test_scan_malformed_composer_json(self):
        """Test that malformed composer.json doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            composer = Path(tmpdir) / "composer.json"
            composer.write_text("{ invalid json")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "php_composer" in context
            assert "error" in context["php_composer"]


class TestJavaScriptScanning:
    """Test JavaScript/TypeScript configuration file scanning."""

    def test_scan_package_json(self):
        """Test scanning package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            package = Path(tmpdir) / "package.json"
            package.write_text(json.dumps({
                "name": "my-app",
                "dependencies": {
                    "react": "^18.0.0",
                    "express": "^4.18.0"
                },
                "devDependencies": {
                    "jest": "^29.0.0"
                },
                "scripts": {
                    "test": "jest",
                    "build": "webpack"
                }
            }))

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "javascript_package" in context
            assert context["javascript_package"]["name"] == "my-app"
            assert "react" in context["javascript_package"]["dependencies"]
            assert "scripts" in context["javascript_package"]

    def test_scan_malformed_package_json(self):
        """Test that malformed package.json doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            package = Path(tmpdir) / "package.json"
            package.write_text("not valid json {")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "javascript_package" in context
            assert "error" in context["javascript_package"]


class TestGolangScanning:
    """Test Golang configuration file scanning."""

    def test_scan_go_mod(self):
        """Test scanning go.mod."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gomod = Path(tmpdir) / "go.mod"
            gomod.write_text("""module github.com/user/myapp

go 1.19

require (
    github.com/gin-gonic/gin v1.8.1
    github.com/spf13/cobra v1.6.0
)
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "golang_mod" in context
            assert context["golang_mod"]["module"] == "github.com/user/myapp"
            assert context["golang_mod"]["go_version"] == "1.19"
            assert "dependencies" in context["golang_mod"]

    def test_scan_go_mod_minimal(self):
        """Test scanning minimal go.mod."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gomod = Path(tmpdir) / "go.mod"
            gomod.write_text("module myapp\n\ngo 1.20")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "golang_mod" in context
            assert context["golang_mod"]["module"] == "myapp"


class TestRubyScanning:
    """Test Ruby configuration file scanning."""

    def test_scan_gemfile_with_rails(self):
        """Test scanning Gemfile with Rails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gemfile = Path(tmpdir) / "Gemfile"
            gemfile.write_text("""
source 'https://rubygems.org'

gem 'rails', '~> 7.0.4'
gem 'pg', '~> 1.1'
gem 'puma', '~> 5.0'
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "ruby_gemfile" in context
            assert context["ruby_gemfile"]["rails_version"] == "~> 7.0.4"
            assert "rails" in context["ruby_gemfile"]["gems"]

    def test_scan_gemfile_without_rails(self):
        """Test scanning Gemfile without Rails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gemfile = Path(tmpdir) / "Gemfile"
            gemfile.write_text("""
gem 'sinatra'
gem 'rack'
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "ruby_gemfile" in context
            assert "rails_version" not in context["ruby_gemfile"]


class TestJavaScanning:
    """Test Java configuration file scanning."""

    def test_scan_pom_xml(self):
        """Test scanning pom.xml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pom = Path(tmpdir) / "pom.xml"
            pom.write_text("""<?xml version="1.0"?>
<project>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <properties>
        <java.version>17</java.version>
        <maven.compiler.source>17</maven.compiler.source>
    </properties>
</project>
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "java_pom" in context
            assert context["java_pom"]["groupId"] == "com.example"
            assert context["java_pom"]["artifactId"] == "my-app"
            assert context["java_pom"]["java_version"] == "17"

    def test_scan_build_gradle(self):
        """Test scanning build.gradle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gradle = Path(tmpdir) / "build.gradle"
            gradle.write_text("""
sourceCompatibility = '17'

dependencies {
    implementation 'org.springframework.boot:spring-boot-starter'
    testImplementation 'junit:junit:4.13'
}
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "java_gradle" in context
            assert context["java_gradle"]["sourceCompatibility"] == "17"


class TestDotNetScanning:
    """Test .NET configuration file scanning."""

    def test_scan_csproj(self):
        """Test scanning .csproj file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csproj = Path(tmpdir) / "MyApp.csproj"
            csproj.write_text("""<Project Sdk="Microsoft.NET.Sdk">
    <PropertyGroup>
        <TargetFramework>net7.0</TargetFramework>
    </PropertyGroup>
    <ItemGroup>
        <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
        <PackageReference Include="Serilog" Version="2.12.0" />
    </ItemGroup>
</Project>
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "dotnet_MyApp" in context
            assert context["dotnet_MyApp"]["target_framework"] == "net7.0"
            assert "Newtonsoft.Json 13.0.1" in context["dotnet_MyApp"]["packages"]

    def test_scan_fsproj(self):
        """Test scanning .fsproj file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fsproj = Path(tmpdir) / "MyApp.fsproj"
            fsproj.write_text("""<Project Sdk="Microsoft.NET.Sdk">
    <PropertyGroup>
        <TargetFramework>net6.0</TargetFramework>
    </PropertyGroup>
</Project>
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "dotnet_MyApp" in context


class TestRustScanning:
    """Test Rust configuration file scanning."""

    def test_scan_cargo_toml(self):
        """Test scanning Cargo.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cargo = Path(tmpdir) / "Cargo.toml"
            cargo.write_text("""[package]
name = "my-rust-app"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "rust_cargo" in context
            assert context["rust_cargo"]["name"] == "my-rust-app"
            assert context["rust_cargo"]["edition"] == "2021"


class TestDockerScanning:
    """Test Docker configuration file scanning."""

    def test_scan_dockerfile(self):
        """Test scanning Dockerfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("""FROM python:3.11-slim AS base
FROM base AS builder
EXPOSE 8000
EXPOSE 8080
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "docker_dockerfile" in context
            assert "python:3.11-slim" in context["docker_dockerfile"]["base_images"]
            assert len(context["docker_dockerfile"]["exposed_ports"]) == 2

    def test_scan_docker_compose_yml(self):
        """Test scanning docker-compose.yml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compose = Path(tmpdir) / "docker-compose.yml"
            compose.write_text("""version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
  db:
    image: postgres:14
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "docker_compose" in context
            assert "web" in context["docker_compose"]["services"]
            assert "db" in context["docker_compose"]["services"]

    def test_scan_docker_compose_yaml(self):
        """Test scanning docker-compose.yaml (alternative extension)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compose = Path(tmpdir) / "docker-compose.yaml"
            compose.write_text("""services:
  api:
    build: .
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "docker_compose" in context

    def test_scan_compose_yml_v2(self):
        """Test scanning compose.yml (Docker Compose V2 naming)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compose = Path(tmpdir) / "compose.yml"
            compose.write_text("""services:
  web:
    image: nginx
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "docker_compose" in context
            assert "web" in context["docker_compose"]["services"]

    def test_scan_compose_yaml_v2(self):
        """Test scanning compose.yaml (Docker Compose V2 naming)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compose = Path(tmpdir) / "compose.yaml"
            compose.write_text("""services:
  app:
    build: .
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "docker_compose" in context
            assert "app" in context["docker_compose"]["services"]


class TestKubernetesScanning:
    """Test Kubernetes configuration file scanning."""

    def test_scan_kubernetes_files(self):
        """Test scanning Kubernetes YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            k8s_dir = Path(tmpdir) / "k8s"
            k8s_dir.mkdir()

            (k8s_dir / "deployment.yaml").write_text("""apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      containers:
      - name: app
        image: nginx:1.19
""")
            (k8s_dir / "service.yml").write_text("""apiVersion: v1
kind: Service
metadata:
  name: my-service
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "kubernetes_resources" in context
            resources = context["kubernetes_resources"]["resources"]
            assert len(resources) == 2
            # Check that metadata was extracted
            names = [r.get("name") for r in resources]
            assert "my-app" in names or "my-service" in names

    def test_scan_no_kubernetes_files(self):
        """Test when no Kubernetes files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "kubernetes_resources" not in context


class TestHelmScanning:
    """Test Helm chart file scanning."""

    def test_scan_helm_chart(self):
        """Test scanning Chart.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chart = Path(tmpdir) / "Chart.yaml"
            chart.write_text("""apiVersion: v2
name: my-chart
version: 1.2.3
appVersion: "2.0"
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "helm_chart" in context
            assert context["helm_chart"]["name"] == "my-chart"
            assert context["helm_chart"]["version"] == "1.2.3"
            assert context["helm_chart"]["appVersion"] == '"2.0"'

    def test_scan_helm_values(self):
        """Test detecting and extracting values.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            values = Path(tmpdir) / "values.yaml"
            values.write_text("replicaCount: 3\nimage:\n  repository: nginx")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "helm_values" in context
            assert "preview" in context["helm_values"]
            assert "replicaCount" in context["helm_values"]["preview"]


class TestTerraformScanning:
    """Test Terraform configuration file scanning."""

    def test_scan_terraform_files(self):
        """Test scanning .tf files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_tf = Path(tmpdir) / "main.tf"
            main_tf.write_text("""terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

resource "aws_instance" "web" {
  ami = "ami-123456"
}

resource "aws_s3_bucket" "data" {
  bucket = "my-bucket"
}
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "terraform" in context
            assert context["terraform"]["required_version"] == ">= 1.0"
            assert "aws_instance web" in context["terraform"]["resources"]
            assert "main.tf" in context["terraform"]["files_found"]


class TestDocumentationScanning:
    """Test Markdown documentation scanning."""

    def test_scan_readme(self):
        """Test scanning README.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme = Path(tmpdir) / "README.md"
            readme.write_text("""# My Project

This is a great project that does amazing things.

## Installation

Run npm install.
""")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "documentation" in context
            assert "README.md" in context["documentation"]["files"]
            assert "My Project" in context["documentation"]["files"]["README.md"]

    def test_scan_multiple_docs(self):
        """Test scanning multiple documentation files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "README.md").write_text("# README")
            (Path(tmpdir) / "CONTRIBUTING.md").write_text("# Contributing Guide")
            (Path(tmpdir) / "SECURITY.md").write_text("# Security Policy")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "documentation" in context
            assert len(context["documentation"]["files"]) == 3

    def test_scan_long_readme(self):
        """Test that long README is truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme = Path(tmpdir) / "README.md"
            # Create a very long first paragraph (3000 characters)
            long_text = "x" * 3000
            readme.write_text(long_text)

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            # Should be truncated to 2000 chars + "..." = 2003
            assert len(context["documentation"]["files"]["README.md"]) <= 2003


class TestContextSummary:
    """Test context summary generation."""

    def test_summary_with_python(self):
        """Test summary generation with Python project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "requirements.txt").write_text("flask")

            scanner = ContextScanner(tmpdir)
            scanner.scan()
            summary = scanner.get_context_summary()

            assert "Python" in summary

    def test_summary_with_multiple_languages(self):
        """Test summary with multiple languages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "requirements.txt").write_text("flask")
            (Path(tmpdir) / "package.json").write_text('{"name": "app"}')
            (Path(tmpdir) / "go.mod").write_text("module app\n\ngo 1.19")

            scanner = ContextScanner(tmpdir)
            scanner.scan()
            summary = scanner.get_context_summary()

            assert "Python" in summary
            assert "JavaScript/TypeScript" in summary
            assert "Go" in summary

    def test_summary_with_infrastructure(self):
        """Test summary with infrastructure tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Dockerfile").write_text("FROM node:16")
            (Path(tmpdir) / "main.tf").write_text("terraform {}")

            scanner = ContextScanner(tmpdir)
            scanner.scan()
            summary = scanner.get_context_summary()

            assert "Docker" in summary
            assert "Terraform" in summary

    def test_summary_empty_project(self):
        """Test summary with no detected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = ContextScanner(tmpdir)
            scanner.scan()
            summary = scanner.get_context_summary()

            assert "No project context detected" in summary

    def test_summary_with_frameworks(self):
        """Test summary includes framework details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            composer = Path(tmpdir) / "composer.json"
            composer.write_text(json.dumps({
                "require": {"laravel/framework": "^9.0"}
            }))

            scanner = ContextScanner(tmpdir)
            scanner.scan()
            summary = scanner.get_context_summary()

            assert "Laravel" in summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_scanner_with_nonexistent_directory(self):
        """Test scanner with non-existent directory."""
        scanner = ContextScanner("/nonexistent/path/12345")
        context = scanner.scan()

        # Should return empty context, not crash
        assert isinstance(context, dict)

    def test_scanner_best_effort_on_parse_errors(self):
        """Test that scanner continues on parse errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid and invalid files
            (Path(tmpdir) / "requirements.txt").write_text("valid-package")
            (Path(tmpdir) / "package.json").write_text("invalid json {{{")
            (Path(tmpdir) / "go.mod").write_text("module app\ngo 1.19")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            # Should have parsed requirements.txt and go.mod
            assert "python_requirements" in context
            assert "golang_mod" in context
            # package.json should be present but with error
            assert "javascript_package" in context

    def test_unicode_content_handling(self):
        """Test handling of Unicode content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("# Comment with émoji 🎉\nrequests")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            assert "python_requirements" in context

    def test_binary_file_handling(self):
        """Test handling of binary files (should be ignored)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_file = Path(tmpdir) / "package.json"
            binary_file.write_bytes(b"\x00\x01\x02\x03")

            scanner = ContextScanner(tmpdir)
            # Should not crash
            context = scanner.scan()

            # May or may not parse, but shouldn't crash
            assert isinstance(context, dict)

    def test_empty_files(self):
        """Test handling of empty configuration files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "requirements.txt").write_text("")
            (Path(tmpdir) / "Dockerfile").write_text("")

            scanner = ContextScanner(tmpdir)
            context = scanner.scan()

            # Should handle gracefully
            assert isinstance(context, dict)

    def test_symlink_handling(self):
        """Test that symlinks are handled safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_file = Path(tmpdir) / "real.txt"
            real_file.write_text("content")

            symlink = Path(tmpdir) / "requirements.txt"
            try:
                symlink.symlink_to(real_file)

                scanner = ContextScanner(tmpdir)
                context = scanner.scan()

                # Should follow symlink and read content
                assert isinstance(context, dict)
            except OSError:
                # Symlinks might not be supported on all systems
                pytest.skip("Symlinks not supported")
