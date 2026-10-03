# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for the Search-R1 data preparation cache."""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "examples" / "search_r1" / "data_process.sh"
BASH = shutil.which("bash")
LINUX_ONLY = pytest.mark.skipif(os.name == "nt" or BASH is None, reason="requires Linux bash")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _tool_path(tmp_path: Path, downloader: str) -> Path:
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("awk", "dirname", "grep", "gzip", "mkdir", "mktemp", "mv", "rm"):
        target = shutil.which(name)
        assert target is not None
        (tools / name).symlink_to(target)

    cat = shutil.which("cat")
    assert cat is not None
    _write_executable(
        tools / "cat",
        f"""#!/bin/sh
if [ "${{MOCK_CAT_MODE:-success}}" = "fail-index" ] && [ "$#" -eq 2 ] && \
   [ "${{1##*/}}" = "part_aa" ] && [ "${{2##*/}}" = "part_ab" ]; then
    {cat} "$1"
    exit 23
fi
exec {cat} "$@"
""",
    )

    _write_executable(
        tools / "conda",
        """#!/bin/sh
if [ "$1 $2 $3" = "shell.bash hook" ]; then
    printf ':\\n'
elif [ "$1 $2" = "env list" ]; then
    printf 'retriever /mock/retriever\\n'
fi
exit 0
""",
    )
    _write_executable(
        tools / downloader,
        """#!/bin/sh
set -eu
url=''
output=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o|-O) output="$2"; shift 2 ;;
        http*) url="$1"; shift ;;
        *) shift ;;
    esac
done
printf '%s\\n' "$url" >> "$MOCK_DOWNLOAD_LOG"
if [ "$MOCK_DOWNLOAD_MODE" = fail ]; then
    printf partial > "$output"
    exit 22
fi
cp "$MOCK_FIXTURE_DIR/${url##*/}" "$output"
""",
    )
    cp = shutil.which("cp")
    assert cp is not None
    (tools / "cp").symlink_to(cp)
    return tools


def _fixtures(tmp_path: Path) -> Path:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    with gzip.open(fixtures / "wiki-18.jsonl.gz", "wb") as output:
        output.write(b'{"id": 1}\n')
    (fixtures / "part_aa").write_bytes(b"AA")
    (fixtures / "part_ab").write_bytes(b"AB")
    (fixtures / "train.parquet").write_bytes(b"TRAIN")
    (fixtures / "test.parquet").write_bytes(b"TEST")
    return fixtures


def _run(script_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    # Git may materialize shell files with CRLF in the Windows checkout used to
    # drive WSL tests. Execute an LF copy of the exact source text.
    script = Path(script_env["SEARCH_R1_DATA_DIR"]).parent / "data_process.sh"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    return subprocess.run(
        [BASH, str(script)],
        env=script_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _environment(tmp_path: Path, downloader: str) -> tuple[dict[str, str], Path, Path]:
    data = tmp_path / "data"
    log = tmp_path / "downloads.log"
    tools = _tool_path(tmp_path, downloader)
    env = {
        **os.environ,
        "PATH": str(tools),
        "SEARCH_R1_DATA_DIR": str(data),
        "SEARCH_R1_SKIP_RETRIEVER_INSTALL": "1",
        "MOCK_DOWNLOAD_MODE": "success",
        "MOCK_DOWNLOAD_LOG": str(log),
        "MOCK_FIXTURE_DIR": str(_fixtures(tmp_path)),
        "MOCK_CAT_MODE": "success",
    }
    return env, data, log


@LINUX_ONLY
@pytest.mark.parametrize("downloader", ["curl", "wget"])
def test_failed_download_does_not_poison_cache_and_can_retry(
    tmp_path: Path,
    downloader: str,
) -> None:
    env, data, log = _environment(tmp_path, downloader)
    env["MOCK_DOWNLOAD_MODE"] = "fail"

    failed = _run(env)

    assert failed.returncode != 0
    assert not (data / "wiki-18.jsonl.gz").exists()
    assert list(data.glob("*.tmp.*")) == []

    env["MOCK_DOWNLOAD_MODE"] = "success"
    retried = _run(env)
    assert retried.returncode == 0, retried.stderr
    assert gzip.decompress((data / "wiki-18.jsonl.gz").read_bytes()) == b'{"id": 1}\n'
    assert (data / "wiki-18.jsonl").read_bytes() == b'{"id": 1}\n'

    calls_after_success = log.read_text(encoding="utf-8").splitlines()
    env["MOCK_DOWNLOAD_MODE"] = "fail"
    cached = _run(env)
    assert cached.returncode == 0, cached.stderr
    assert log.read_text(encoding="utf-8").splitlines() == calls_after_success


@LINUX_ONLY
def test_index_ignores_unrelated_part_files(tmp_path: Path) -> None:
    env, data, _ = _environment(tmp_path, "curl")
    data.mkdir()
    (data / "part_backup").write_bytes(b"STALE")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert (data / "e5_Flat.index").read_bytes() == b"AAAB"


@LINUX_ONLY
def test_failed_index_concatenation_does_not_poison_cache_and_can_retry(tmp_path: Path) -> None:
    env, data, _ = _environment(tmp_path, "curl")
    env["MOCK_CAT_MODE"] = "fail-index"

    failed = _run(env)

    assert failed.returncode != 0
    assert not (data / "e5_Flat.index").exists()
    assert list(data.glob("e5_Flat.index.tmp.*")) == []

    env["MOCK_CAT_MODE"] = "success"
    retried = _run(env)
    assert retried.returncode == 0, retried.stderr
    assert (data / "e5_Flat.index").read_bytes() == b"AAAB"
