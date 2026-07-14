from __future__ import annotations

import ast
from pathlib import Path

import bayesprobe


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "src" / "bayesprobe_terminal_bench"
FORBIDDEN_FILES = {"loop.py", "core.py", "evidence.py", "updater.py"}


def test_nested_project_is_materialized() -> None:
    assert (PACKAGE_DIR / "__init__.py").is_file()
    assert (PROJECT_DIR / "uv.lock").is_file()


def test_required_runtime_types_are_public_root_exports() -> None:
    required = {
        "AutonomousQuestionRunner",
        "BayesProbeCore",
        "ExternalSignal",
        "ProbeExecutor",
        "ProbeToolGateway",
    }
    assert required.issubset(set(bayesprobe.__all__))


def test_benchmark_has_no_shadow_kernel_modules() -> None:
    assert {path.name for path in PACKAGE_DIR.glob("*.py")}.isdisjoint(FORBIDDEN_FILES)


def test_production_source_uses_only_bayesprobe_root_imports() -> None:
    violations: list[str] = []
    for source_path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "bayesprobe" or alias.name.startswith("bayesprobe."):
                        violations.append(f"{source_path.name}:{node.lineno}: import {alias.name}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("bayesprobe."):
                    violations.append(f"{source_path.name}:{node.lineno}: from {module}")
    assert violations == []
