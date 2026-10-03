# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
BASH = shutil.which("bash")
LINUX_ONLY = pytest.mark.skipif(os.name == "nt" or BASH is None, reason="requires Linux bash")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _prepare_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    example = root / "examples" / "calc_x"
    data = example / "data"
    data.mkdir(parents=True)

    script = example / "run_local.sh"
    script.write_text(
        (REPO_ROOT / "examples/calc_x/run_local.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    script.chmod(0o755)
    (example / "train_calc_agent.py").touch()
    (data / "train.parquet").touch()
    (data / "test.parquet").touch()
    return root, script


def _mock_path(tmp_path: Path) -> Path:
    tools = tmp_path / "bin"
    tools.mkdir(parents=True)

    for name in ("date", "dirname", "env", "seq"):
        executable = shutil.which(name)
        assert executable is not None
        (tools / name).symlink_to(executable)

    no_op = "#!/bin/sh\nexit 0\n"
    for name in ("agl-controller", "agl-server", "curl", "pkill", "ray", "sleep"):
        _write_executable(tools / name, no_op)

    _write_executable(
        tools / "python",
        """#!/bin/sh
if [ "$PWD" != "$MOCK_EXPECTED_CWD" ]; then
    printf 'unexpected cwd: %s\n' "$PWD" >&2
    exit 91
fi
if [ "$1" != "train_calc_agent.py" ] || [ ! -f "$1" ]; then
    printf 'trainer path is not resolvable: %s\n' "$1" >&2
    exit 92
fi
if [ ! -f data/train.parquet ] || [ ! -f data/test.parquet ]; then
    printf 'default data paths are not resolvable\n' >&2
    exit 93
fi
printf '%s\n' "$@" > "$MOCK_PYTHON_ARGS"
""",
    )
    return tools


@LINUX_ONLY
@pytest.mark.parametrize("start_directory", ["repo", "example", "unrelated"])
def test_calc_x_launcher_uses_example_directory(tmp_path: Path, start_directory: str) -> None:
    assert BASH is not None
    root, script = _prepare_repo(tmp_path)
    example = script.parent
    directories = {
        "repo": root,
        "example": example,
        "unrelated": tmp_path / "unrelated directory",
    }
    cwd = directories[start_directory]
    cwd.mkdir(parents=True, exist_ok=True)
    args_file = tmp_path / f"{start_directory}-args"
    env = {
        **os.environ,
        "PATH": str(_mock_path(tmp_path / start_directory)),
        "MOCK_EXPECTED_CWD": str(example),
        "MOCK_PYTHON_ARGS": str(args_file),
    }

    result = subprocess.run(
        [BASH, str(script), "--label", "value with spaces"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "train_calc_agent.py",
        "--agl-base-url",
        "http://localhost:8181",
        "--agl-key",
        "dummy",
        "--run-name",
        "local",
        "--label",
        "value with spaces",
    ]
