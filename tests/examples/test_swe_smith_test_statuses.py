# Copyright (c) Microsoft. All rights reserved.

"""Match pytest reports to the full test IDs used for SWE-smith rewards."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from examples.swe_smith.agents import smith_agent


@pytest.mark.parametrize("include_failure", [False, True])
def test_evaluate_parametrized_tests_with_real_pytest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, include_failure: bool
) -> None:
    (tmp_path / "test_example.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', [1, 2, 3], ids=['hello world', 'a - b', 'nested [two  words]'])\n"
        "def test_pass(value):\n"
        "    assert value > 0\n"
        "def test_fail():\n"
        "    assert False, 'failure - with [details]'\n"
    )
    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path))
    monkeypatch.setattr(smith_agent, "restore_f2p_tests", lambda *_args: None)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "")
    run = smith_agent._run

    def run_without_xdist(command: str, timeout: int) -> tuple[str, int]:
        if command == "python -c 'import xdist'":
            return "", 1
        return run(command, timeout)

    monkeypatch.setattr(smith_agent, "_run", run_without_xdist)
    fail_to_pass = ["test_example.py::test_pass[hello world]", "test_example.py::test_pass[a - b]"]
    if include_failure:
        fail_to_pass.append("test_example.py::test_fail")
    reward, resolved, reason, timed_out = smith_agent.evaluate(
        {
            "instance_id": "test-instance",
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": ["test_example.py::test_pass[nested [two  words]]"],
        },
        timeout=30,
    )

    assert timed_out is False
    assert reward == (0.0 if include_failure else 1.0)
    assert resolved is (not include_failure)
    assert f"FAIL_TO_PASS 2/{len(fail_to_pass)} passed" in reason
    assert "PASS_TO_PASS 1/1 ok" in reason


@pytest.mark.parametrize(
    "parameter", ["hello world", "two  spaces", "a - b", "nested [two words]", "a.+? (b|c)", "status PASSED word"]
)
@pytest.mark.parametrize(
    ("status", "line"),
    [
        ("PASSED", "PASSED {node}"),
        ("FAILED", "FAILED {node} - AssertionError: reason - with [details]"),
        ("XFAIL", "{node} XFAIL (expected failure) [ 50%]"),
        ("PASSED", "[gw2] [100%] \x1b[32mPASSED\x1b[0m {node}"),
    ],
)
def test_statuses_preserve_full_node_ids(parameter: str, status: str, line: str) -> None:
    node = f"tests/test_example.py::TestExample::test_case[{parameter}]"
    assert smith_agent.parse_test_statuses(line.format(node=node), [node]) == {node: status}


def test_known_node_ids_do_not_match_prefixes_or_failure_reasons() -> None:
    node = "tests/test_example.py::test_case"
    longer_node = node + "[two words]"
    output = "\n".join(
        [
            f"{longer_node} PASSED [ 50%]",
            f"FAILED {node} - AssertionError: report includes [{longer_node}]",
            f"PASSED {node}_other",
            "PASSED tests/unrelated.py::test_case",
        ]
    )
    assert smith_agent.parse_test_statuses(output, [node, longer_node]) == {node: "FAILED", longer_node: "PASSED"}


def test_no_expected_tests_returns_no_statuses() -> None:
    assert smith_agent.parse_test_statuses("PASSED tests/test_example.py::test_case", []) == {}
