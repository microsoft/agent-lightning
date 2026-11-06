# Copyright (c) Microsoft. All rights reserved.

"""Timed-out shell actions must not keep running during later agent turns."""

from __future__ import annotations

import os
import shlex
import signal
import sys
import time

import pytest

from examples.swe_smith.agents import smith_agent

pytestmark = pytest.mark.skipif(os.name != "posix", reason="The SWE-smith harness runs in POSIX containers")


@pytest.mark.parametrize("shell_exits_first", [False, True], ids=["foreground-command", "inherited-output-pipes"])
def test_timeout_stops_child_and_grandchild(monkeypatch, tmp_path, shell_exits_first):
    script = tmp_path / "command.py"
    script.write_text(
        "import os, pathlib, subprocess, sys, time\n"
        "root = pathlib.Path(__file__).parent\n"
        "role = sys.argv[1]\n"
        "(root / (role + '.ready')).write_text(str(os.getpid()))\n"
        "child = subprocess.Popen([sys.executable, __file__, 'grandchild']) if role == 'child' else None\n"
        "while not (root / 'release').exists():\n"
        "    time.sleep(0.01)\n"
        "(root / (role + '.late')).write_text('changed after timeout')\n"
        "if child is not None:\n"
        "    child.wait()\n"
    )
    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path))
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} child"
    # Keep a foreground shell alive, or let it exit while descendants hold the
    # captured pipes open. Both cases must terminate the entire command group.
    command += " &" if shell_exits_first else " && echo done"

    try:
        start = time.monotonic()
        output, returncode = smith_agent._run(command, timeout=1)
        elapsed = time.monotonic() - start
        assert (output, returncode) == ("[timed out after 1s]", 124)
        assert elapsed < 4
        assert (tmp_path / "child.ready").exists()
        assert (tmp_path / "grandchild.ready").exists()

        # A surviving descendant can now mutate the worktree in the next turn.
        (tmp_path / "release").touch()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not list(tmp_path.glob("*.late")):
            time.sleep(0.01)
        assert list(tmp_path.glob("*.late")) == []
    finally:
        # Let the original implementation's surviving processes exit as well,
        # so running the regression before the fix leaves no waiting children.
        (tmp_path / "release").touch()


def test_timeout_tolerates_process_group_exit_race(monkeypatch, tmp_path):
    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path))
    killpg = os.killpg

    def already_exited(process_group: int, sig: int) -> None:
        assert sig == signal.SIGKILL
        killpg(process_group, sig)
        raise ProcessLookupError("group exited before signal delivery")

    monkeypatch.setattr(os, "killpg", already_exited)
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(30)'"

    assert smith_agent._run(command, timeout=1) == ("[timed out after 1s]", 124)


def test_normal_command_preserves_output_and_exit_status(monkeypatch, tmp_path):
    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path))

    assert smith_agent._run("printf stdout; printf stderr >&2; exit 7", timeout=5) == ("stdoutstderr", 7)


def test_timeout_discards_partial_output_without_decoding_it(monkeypatch, tmp_path):
    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path))

    assert smith_agent._run("printf '\\377'; sleep 30", timeout=1) == ("[timed out after 1s]", 124)


def test_command_start_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path / "missing"))

    output, returncode = smith_agent._run("true", timeout=5)

    assert returncode == 1
    assert output.startswith("[failed to start:")
