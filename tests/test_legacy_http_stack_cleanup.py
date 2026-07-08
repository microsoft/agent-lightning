# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path


def _iter_code_files(root: Path):
    for path in root.rglob("*.py"):
        if path.name == "__pycache__":
            continue
        yield path


def test_core_runtime_does_not_import_legacy_client_server_stack() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "agentlightning"

    violations: list[tuple[Path, str]] = []

    for path in _iter_code_files(package_root):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names = {alias.name for alias in node.names}

                if module == "agentlightning.client" and (
                    "AgentLightningClient" in imported_names or "DevTaskLoader" in imported_names
                ):
                    violations.append((path, f"imports {', '.join(sorted(imported_names))} from {module}"))

                if module == "agentlightning.server" and "AgentLightningServer" in imported_names:
                    violations.append((path, f"imports {', '.join(sorted(imported_names))} from {module}"))

                if (
                    module == "agentlightning"
                    and ("AgentLightningClient" in imported_names or "AgentLightningServer" in imported_names)
                ):
                    violations.append((path, f"imports {', '.join(sorted(imported_names))} from {module}"))

    if violations:
        lines = "\n".join(f"{path}: {msg}" for path, msg in violations)
        raise AssertionError("Core runtime should not import legacy HTTP stack directly:\n" + lines)


def test_legacy_client_server_stack_files_are_removed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    removed_paths = [
        repo_root / "agentlightning" / "client.py",
        repo_root / "agentlightning" / "server.py",
        repo_root / "agentlightning" / "trainer" / "legacy.py",
        repo_root / "tests" / "test_client.py",
    ]

    existing = [path for path in removed_paths if path.exists()]
    assert existing == []


def test_examples_and_workflows_do_not_reference_removed_legacy_examples() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    roots = [
        repo_root / "examples",
        repo_root / ".github" / "workflows",
    ]
    removed_names = {
        "legacy_apo_client.py",
        "legacy_apo_server.py",
        "legacy_calc_agent.py",
        "legacy_calc_agent_debug.py",
        "legacy_train.sh",
    }
    violations: list[tuple[Path, str]] = []

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".py", ".yml", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for removed_name in removed_names:
                if removed_name in text:
                    violations.append((path, removed_name))

    if violations:
        lines = "\n".join(f"{path}: references {removed_name}" for path, removed_name in violations)
        raise AssertionError(f"Removed legacy examples are still referenced:\n{lines}")
