# Copyright (c) Microsoft. All rights reserved.

"""CPU-only readiness regression coverage for the example launchers."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
BASH = shutil.which("bash")
LINUX_ONLY = pytest.mark.skipif(os.name == "nt" or BASH is None, reason="requires Linux bash")


@dataclass(frozen=True)
class Launcher:
    path: str
    cwd: str


LAUNCHERS = [
    Launcher("examples/calc_x/run_local.sh", "examples/calc_x"),
    Launcher("examples/calc_x/run_minikube.sh", "examples/calc_x"),
    Launcher("examples/gsm8k/run_local.sh", "examples/gsm8k"),
    Launcher("examples/science_world/run_local.sh", "."),
    Launcher("examples/search_r1/run.sh", "."),
    Launcher("examples/llm-in-sandbox/run.sh", "."),
]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _copy_launcher(tmp_path: Path, launcher: Launcher) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    target = root / launcher.path
    target.parent.mkdir(parents=True)
    target.write_text((REPO_ROOT / launcher.path).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    target.chmod(0o755)
    cwd = root if launcher.cwd == "." else root / launcher.cwd
    cwd.mkdir(parents=True, exist_ok=True)
    return target, cwd


def _mock_path(tmp_path: Path) -> Path:
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("date", "dirname", "env", "seq"):
        executable = shutil.which(name)
        assert executable is not None
        (tools / name).symlink_to(executable)

    command = """#!/bin/sh
line="${0##*/}"
for argument in "$@"; do
    line="$line <$argument>"
done
printf '%s\n' "$line" >> "$MOCK_COMMAND_LOG"
exit 0
"""
    for name in ("agl-controller", "agl-server", "minikube", "pkill", "python", "ray", "sleep"):
        _write_executable(tools / name, command)

    _write_executable(
        tools / "curl",
        """#!/bin/sh
count=0
if [ -f "$MOCK_CURL_COUNT" ]; then
    read -r count < "$MOCK_CURL_COUNT"
fi
count=$((count + 1))
printf '%s\n' "$count" > "$MOCK_CURL_COUNT"
printf '%s\n' "$*" >> "$MOCK_CURL_LOG"
if [ "$MOCK_CURL_SUCCEED_AT" -gt 0 ] && [ "$count" -ge "$MOCK_CURL_SUCCEED_AT" ]; then
    exit 0
fi
exit 22
""",
    )
    return tools


def _run(
    tmp_path: Path, launcher: Launcher, *, succeed_at: int
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    assert BASH is not None
    script, cwd = _copy_launcher(tmp_path, launcher)
    command_log = tmp_path / "commands.log"
    curl_log = tmp_path / "curl.log"
    env = {
        **os.environ,
        "PATH": str(_mock_path(tmp_path)),
        "MOCK_COMMAND_LOG": str(command_log),
        "MOCK_CURL_COUNT": str(tmp_path / "curl-count"),
        "MOCK_CURL_LOG": str(curl_log),
        "MOCK_CURL_SUCCEED_AT": str(succeed_at),
    }
    result = subprocess.run(
        [BASH, str(script), "--marker", "value"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    commands = command_log.read_text(encoding="utf-8").splitlines()
    curls = curl_log.read_text(encoding="utf-8").splitlines()
    return result, commands, curls


@LINUX_ONLY
@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda launcher: launcher.path)
def test_launcher_stops_when_server_never_becomes_ready(tmp_path: Path, launcher: Launcher) -> None:
    result, commands, curls = _run(tmp_path, launcher, succeed_at=0)

    assert result.returncode != 0
    assert "Agent Lightning server did not become ready" in result.stderr
    assert not any(command.startswith("agl-controller") for command in commands)
    assert not any(command.startswith("python ") for command in commands)
    assert (tmp_path / "curl-count").read_text(encoding="utf-8").strip() == "60"
    assert all("--max-time 1" in call for call in curls)


@LINUX_ONLY
@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda launcher: launcher.path)
def test_launcher_continues_after_later_success_and_preserves_arguments(tmp_path: Path, launcher: Launcher) -> None:
    result, commands, curls = _run(tmp_path, launcher, succeed_at=3)

    assert result.returncode == 0, result.stderr
    assert any(command.startswith("agl-controller") for command in commands)
    trainer = next(command for command in commands if command.startswith("python "))
    assert trainer.endswith(" <--marker> <value>")
    assert (tmp_path / "curl-count").read_text(encoding="utf-8").strip() == "3"
    assert all("--max-time 1" in call for call in curls)


@LINUX_ONLY
def test_launcher_continues_after_first_success(tmp_path: Path) -> None:
    result, commands, curls = _run(tmp_path, LAUNCHERS[0], succeed_at=1)

    assert result.returncode == 0, result.stderr
    assert any(command.startswith("agl-controller") for command in commands)
    assert any(command.startswith("python ") for command in commands)
    assert (tmp_path / "curl-count").read_text(encoding="utf-8").strip() == "1"
    assert all("--max-time 1" in call for call in curls)
