"""Rename integrity guardrail for the `src` -> `code_reviewer` package rename.

These tests are structural regression guards:

* no residual ``src`` package references (imports, mock patch strings,
  dynamic ``__import__`` calls, config files) anywhere in the repo;
* the Dockerfile entrypoint keeps the ``-P`` (safe-path) flag;
* the entrypoint actually resolves the action's module when CWD contains a
  decoy ``src/`` + ``code_reviewer/`` shadow (the original defect).

grep exit-code semantics used throughout: 0 = matches found (FAIL),
1 = no matches (PASS), 2 = grep error (FAIL, e.g. missing path).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories whose Python files must never reference the old `src` package.
CODE_DIRS = [ROOT / "code_reviewer", ROOT / "test"]

# Config files that must contain no `src` token after the rename.
CONFIG_FILES = [
    ROOT / "pyproject.toml",
    ROOT / "Dockerfile",
    ROOT / "Makefile",
    ROOT / ".pre-commit-config.yaml",
    ROOT / "test" / "run-local.sh",
]

# Patterns referencing the old package in imports, mocks, and dynamic loads.
# Plain BRE literals: `\(` or `\)` would be an unmatched group on BSD grep
# (exit 2 = error, not no-match), and `\b` is unsupported on BSD grep.
SRC_PATTERNS = [
    r"from src\.",  # from src.llm.base import ...
    "import src",  # import src / import src.main
    '@patch("src',  # unittest.mock patch targets
    'mocker.patch("src',  # pytest-mock patch targets
    '__import__("src',  # dynamic imports
]


def _python_files(paths: list[Path]) -> list[Path]:
    """Explicit .py file lists — never recursive grep over directories.

    Recursive grep would also scan __pycache__/*.pyc, and the guardrail's own
    compiled bytecode contains these pattern strings as literals (false hit).
    """
    files: list[Path] = []
    for directory in paths:
        files.extend(p for p in directory.rglob("*.py") if p.name != "test_rename_integrity.py")
    return files


def _run_grep(pattern: str, paths: list[Path], extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run portable grep over the given paths (no -r: paths are explicit files)."""
    cmd = ["grep", "-n", *(extra_args or []), pattern, *(str(p) for p in paths)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _grep_must_not_match(pattern: str, paths: list[Path], what: str, extra_args: list[str] | None = None) -> None:
    """Assert grep exits 1 (no match). Exit 0 (match) or 2 (error) is a failure."""
    result = _run_grep(pattern, paths, extra_args)
    assert result.returncode == 1, (
        f"grep exit {result.returncode} (expected 1 = no match) for {what} "
        f"pattern {pattern!r}:\n{result.stdout}"
    )


def test_no_src_package_references_in_python() -> None:
    """No `src` package imports, patch strings, or dynamic loads in code or tests."""
    # Explicit file list: the guardrail's own bytecode in __pycache__ would
    # otherwise match these patterns as string literals.
    python_files = _python_files(CODE_DIRS)
    for pattern in SRC_PATTERNS:
        _grep_must_not_match(pattern, python_files, what="python")


def test_no_src_token_in_config_files() -> None:
    """Config files reference the renamed package only (no `src` token)."""
    _grep_must_not_match("src", CONFIG_FILES, what="config", extra_args=["-w"])


def test_dockerfile_keeps_safe_path_entrypoint() -> None:
    """Dockerfile must copy the renamed package and keep `-P` + PYTHONPATH."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY code_reviewer/ ./code_reviewer/" in dockerfile, (
        "Dockerfile must COPY the renamed package"
    )
    assert "ENV PYTHONPATH=/app" in dockerfile, (
        "Dockerfile must keep PYTHONPATH=/app so intra-package imports resolve"
    )
    assert '"python", "-P", "-m", "code_reviewer.main"' in dockerfile, (
        "Dockerfile ENTRYPOINT must be [\"python\", \"-P\", \"-m\", \"code_reviewer.main\"]; "
        "dropping -P regresses the src-layout shadowing defect"
    )


def _make_decoy_repo(tmp_path: Path) -> Path:
    """Create a reviewed-repo lookalike with a decoy src/main.py and code_reviewer/ shadow.

    Both decoy packages get an ``__init__.py``: a directory without one is only a
    namespace-package portion, and CPython prefers the real regular package on
    PYTHONPATH over it — the decoy must be a REGULAR package to shadow (this is
    also how the original linia-ai/linia defect worked: its src/ had __init__.py).
    """
    for pkg in ("src", "code_reviewer"):
        pkg_dir = tmp_path / pkg
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "main.py").write_text(
        'print("DECOY_SRC_MAIN_EXECUTED")\n', encoding="utf-8"
    )
    (tmp_path / "code_reviewer" / "main.py").write_text(
        'print("DECOY_SHADOW_EXECUTED")\n', encoding="utf-8"
    )
    return tmp_path


def _run_entrypoint(cwd: Path, safe_path: bool) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable]
    if safe_path:
        cmd.append("-P")
    cmd += ["-m", "code_reviewer.main", "--help"]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


def test_safe_path_entrypoint_defeats_decoy_shadow(tmp_path) -> None:
    """`python -P -m code_reviewer.main --help` must run the action's module
    even when CWD holds a decoy src/ and a code_reviewer/ shadow."""
    decoy_dir = _make_decoy_repo(tmp_path)
    result = _run_entrypoint(decoy_dir, safe_path=True)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"exit {result.returncode}:\n{combined}"
    assert "DECOY" not in combined, f"decoy module shadowed the action:\n{combined}"
    assert "Usage:" in combined, f"expected click CLI help, got:\n{combined}"


def test_without_safe_path_flag_decoy_shadows(tmp_path) -> None:
    """Without -P, CWD lands on sys.path and the decoy shadows the action —
    proving the safe-path test above exercises the real defect."""
    decoy_dir = _make_decoy_repo(tmp_path)
    result = _run_entrypoint(decoy_dir, safe_path=False)
    assert "DECOY_SHADOW_EXECUTED" in result.stdout, (
        "expected decoy code_reviewer/ to shadow the action when -P is missing:\n"
        f"{result.stdout}\n{result.stderr}"
    )
