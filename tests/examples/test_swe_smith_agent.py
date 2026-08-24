# Copyright (c) Microsoft. All rights reserved.

"""Coverage for the SWE-smith agent loop's failure handling.

Focused on the context-overflow case: once a multi-turn rollout's prompt grows
past ``max_model_len``, vLLM rejects the request with HTTP 400. The loop must end
the episode cleanly (no empty assistant turn appended) so the trajectory stays a
clean token-level prefix chain for the GRPO bridge. A *generic* error, by
contrast, must keep today's behavior (burn the turn, keep going).
"""

from __future__ import annotations

import subprocess
from typing import Any

import httpx
from openai import BadRequestError, InternalServerError

from examples.swe_smith.agents import smith_agent

_OVERFLOW_MESSAGE = (
    "'max_tokens' is too large: 2048. This model's maximum context length is "
    "32768 tokens and your request has 30888 input tokens "
    "(2048 > 32768 - 30888)."
)


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "http://vllm/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body=None)


def _server_error(message: str) -> InternalServerError:
    request = httpx.Request("POST", "http://vllm/v1/chat/completions")
    response = httpx.Response(500, request=request)
    return InternalServerError(message, response=response, body=None)


class _RaisingClient:
    """OpenAI-shaped client whose completion call always raises ``exc``."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

        outer = self

        class _Completions:
            def create(self, **_kwargs: Any) -> Any:
                outer.calls += 1
                raise outer._exc

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_context_overflow_400_ends_loop_without_empty_turn() -> None:
    client = _RaisingClient(_bad_request(_OVERFLOW_MESSAGE))

    submitted, turns_used, max_prompt_tokens = smith_agent.run_agent_loop(
        client, "fix the bug", max_turns=40, cmd_timeout=1, obs_cap=100, max_tokens=2048
    )

    # Episode ends unresolved: no submit was ever issued.
    assert submitted is False
    # The doomed request fires exactly once, then we stop — no spin to max_turns.
    assert client.calls == 1
    # Overflow raises before a completion is recorded, so no turn is consumed.
    assert turns_used == 0
    # No successful request, so no prompt-token measurement.
    assert max_prompt_tokens == 0


def test_non_400_with_overflow_text_is_not_treated_as_overflow() -> None:
    # A 5xx that merely echoes the overflow phrasing must NOT end the episode:
    # only a real 400 means the request is structurally too large. This one
    # takes the generic path (burn the turn, keep going).
    client = _RaisingClient(_server_error("maximum context length exceeded"))

    submitted, turns_used, _max_prompt_tokens = smith_agent.run_agent_loop(
        client, "fix the bug", max_turns=3, cmd_timeout=1, obs_cap=100, max_tokens=2048
    )

    assert submitted is False
    assert client.calls == 3
    # Each turn burns a completion, so the loop runs to the cap.
    assert turns_used == 3


def test_generic_error_burns_turn_and_continues() -> None:
    client = _RaisingClient(RuntimeError("transient hiccup"))

    submitted, turns_used, _max_prompt_tokens = smith_agent.run_agent_loop(
        client, "fix the bug", max_turns=3, cmd_timeout=1, obs_cap=100, max_tokens=2048
    )

    # A non-overflow error keeps today's behavior: each turn appends an empty
    # assistant message and the loop spins to the cap.
    assert submitted is False
    assert client.calls == 3
    assert turns_used == 3


def test_git_retry_removes_stale_index_lock_after_nonzero(monkeypatch, tmp_path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock = git_dir / "index.lock"
    calls = 0

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        nonlocal calls
        calls += 1
        assert not lock.exists()
        if calls == 1:
            lock.write_text("stale")
            return subprocess.CompletedProcess(
                command,
                128,
                "",
                "fatal: Unable to create '/testbed/.git/index.lock': File exists.",
            )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path))
    monkeypatch.setattr(smith_agent.subprocess, "run", fake_run)
    monkeypatch.setattr(smith_agent.time, "sleep", lambda _delay: None)

    proc = smith_agent._git_retry(["status"], timeout=1, attempts=2)

    assert calls == 2
    assert proc is not None
    assert proc.returncode == 0
    assert not lock.exists()


def test_git_retry_removes_stale_index_lock_after_timeout(monkeypatch, tmp_path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock = git_dir / "index.lock"
    calls = 0

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        nonlocal calls
        calls += 1
        assert not lock.exists()
        if calls == 1:
            lock.write_text("stale")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path))
    monkeypatch.setattr(smith_agent.subprocess, "run", fake_run)
    monkeypatch.setattr(smith_agent.time, "sleep", lambda _delay: None)

    proc = smith_agent._git_retry(["checkout", "branch"], timeout=1, attempts=2)

    assert calls == 2
    assert proc is not None
    assert proc.returncode == 0
    assert not lock.exists()


def test_git_retry_sleeps_before_each_checkout(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(smith_agent, "TESTBED", str(tmp_path))
    monkeypatch.setattr(smith_agent.random, "uniform", lambda low, high: 17.5)
    monkeypatch.setattr(smith_agent.time, "sleep", sleeps.append)
    monkeypatch.setattr(smith_agent.subprocess, "run", fake_run)

    proc = smith_agent._git_retry(["checkout", "repo.instance"], timeout=1, attempts=1)

    assert proc is not None
    assert proc.returncode == 0
    assert calls == [["git", "checkout", "repo.instance"]]
    assert sleeps == [17.5]


def test_checkout_bug_commit_uses_swesmith_official_checkout(monkeypatch) -> None:
    calls: list[tuple[list[str], int, int]] = []

    def fake_git_retry(args: list[str], *, timeout: int = 120, attempts: int = 3) -> subprocess.CompletedProcess:
        calls.append((args, timeout, attempts))
        return subprocess.CompletedProcess(["git", *args], 0, "ok", "")

    monkeypatch.setattr(smith_agent, "_git_retry", fake_git_retry)

    smith_agent.checkout_bug_commit("repo.instance")

    assert calls == [(["checkout", "repo.instance"], 120, 3)]


def test_forbidden_action_blocks_git_usage(monkeypatch) -> None:
    monkeypatch.setattr(smith_agent, "_HIDDEN_GIT_DIR", "/opt/agl_tmp")
    fa = smith_agent._forbidden_action
    for cmd in (
        "git log --oneline -10",
        "cd /testbed && git checkout 614b134 -- pkg/x.py",
        "git -C /testbed diff HEAD",
        "FOO=bar git status",
        "result=$(git rev-parse HEAD)",
        "/usr/bin/git show HEAD~1",
        "git-checkout main -- a.py",
    ):
        reason = fa(cmd)
        assert reason is not None and reason.startswith("git is disabled"), cmd


def test_forbidden_action_blocks_git_metadata_access(monkeypatch) -> None:
    monkeypatch.setattr(smith_agent, "_HIDDEN_GIT_DIR", "/opt/agl_tmp")
    fa = smith_agent._forbidden_action
    for cmd in (
        "cat /testbed/.git/config",
        "find / -name .git -type d",
        "cat /opt/agl_tmp/HEAD",
        "grep x --git-dir=/opt/agl_tmp",
    ):
        assert fa(cmd) is not None, cmd


def test_forbidden_action_allows_legit_commands(monkeypatch) -> None:
    monkeypatch.setattr(smith_agent, "_HIDDEN_GIT_DIR", "/opt/agl_tmp")
    fa = smith_agent._forbidden_action
    for cmd in (
        "cat /testbed/pkg/utils.py",
        "python -m pytest tests/ -q",
        "echo 'use git to debug'",
        "cat .gitignore",
        "grep -rn 'legit digital' src/",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    ):
        assert fa(cmd) is None, cmd


def test_forbidden_action_blocks_network_install_and_tamper(monkeypatch) -> None:
    # Network/package-install/test-harness-tamper routes are reward-hacking
    # channels (fetch the upstream fix, pull the target package, force PASS).
    # They must be blocked so the agent solves from local /testbed source only.
    monkeypatch.setattr(smith_agent, "_HIDDEN_GIT_DIR", "/opt/agl_tmp")
    fa = smith_agent._forbidden_action
    for cmd in (
        "pip install gitpython",
        "pip install requests==2.0",
        "python -m pip install foo",
        "conda install numpy",
        "curl -sL https://raw.githubusercontent.com/x/y/main/z.py -o /testbed/z.py",
        "wget http://example.com/patch.py",
        'python -c "import urllib.request; urllib.request.urlopen(u)"',
        "echo 'x' > /testbed/conftest.py",
        "tee /testbed/pytest.ini",
    ):
        assert fa(cmd) is not None, cmd


def test_agent_env_hides_relocated_git_dir(monkeypatch) -> None:
    monkeypatch.setattr(smith_agent, "_HIDDEN_GIT_DIR", "/opt/agl_tmp")
    monkeypatch.setenv("SMITH_HIDDEN_GIT_DIR", "/opt/agl_tmp")
    monkeypatch.setenv("SOME_PATH", "/opt/agl_tmp/objects")  # value leaks the path
    monkeypatch.setenv("SMITH_MAX_TURNS", "40")  # unrelated SMITH_* var stays

    env = smith_agent._agent_env()

    assert "SMITH_HIDDEN_GIT_DIR" not in env
    assert "SOME_PATH" not in env
    assert not any("/opt/agl_tmp" in v for v in env.values())
    assert env.get("SMITH_MAX_TURNS") == "40"
    assert env["PAGER"] == "cat"  # _CMD_ENV overrides still applied


def test_relocate_git_moves_worktree_git_and_routes_harness(monkeypatch, tmp_path) -> None:
    testbed = tmp_path / "testbed"
    (testbed / ".git").mkdir(parents=True)
    (testbed / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    hidden = tmp_path / "hidden_git"

    monkeypatch.setattr(smith_agent, "TESTBED", str(testbed))
    monkeypatch.setattr(smith_agent, "_HIDDEN_GIT_DIR", str(hidden))

    # Before relocation: in-tree .git, plain git base.
    assert smith_agent._git_base() == ["git"]
    assert smith_agent._git_dir() == str(testbed / ".git")

    smith_agent.relocate_git()

    # .git moved out of the worktree and harness now targets it explicitly.
    assert not (testbed / ".git").exists()
    assert (hidden / "HEAD").read_text() == "ref: refs/heads/main\n"
    assert smith_agent._git_dir() == str(hidden)
    assert smith_agent._git_base() == [
        "git",
        "--git-dir",
        str(hidden),
        "--work-tree",
        str(testbed),
        "-c",
        "safe.directory=*",
    ]


def test_relocate_git_missing_worktree_git_is_noop(monkeypatch, tmp_path) -> None:
    testbed = tmp_path / "testbed"
    testbed.mkdir()
    hidden = tmp_path / "hidden_git"
    monkeypatch.setattr(smith_agent, "TESTBED", str(testbed))
    monkeypatch.setattr(smith_agent, "_HIDDEN_GIT_DIR", str(hidden))

    smith_agent.relocate_git()  # must not raise

    assert not hidden.exists()


def test_length_penalty_only_applies_to_solved() -> None:
    # Unsolved (reward==0) trajectories are never penalized, no matter how long,
    # so hard rollouts keep full exploration room.
    assert smith_agent.length_penalized_reward(0.0, 100, 100, t0=55, lam=0.2, is_train=True) == 0.0
    assert smith_agent.length_penalized_reward(0.0, 40, 100, t0=55, lam=0.2, is_train=True) == 0.0


def test_length_penalty_skips_validation() -> None:
    # Validation reward (is_train=False) must NEVER be reshaped — it is the true
    # metric used for checkpoint selection. Even a long solved rollout keeps 1.0.
    assert smith_agent.length_penalized_reward(1.0, 100, 100, t0=55, lam=0.2, is_train=False) == 1.0
    assert smith_agent.length_penalized_reward(1.0, 90, 100, t0=55, lam=0.2, is_train=False) == 1.0
    # And unsolved val is untouched too.
    assert smith_agent.length_penalized_reward(0.0, 100, 100, t0=55, lam=0.2, is_train=False) == 0.0


def test_length_penalty_solved_within_budget_keeps_full_reward() -> None:
    # A train solve at or below the T0 turn budget keeps reward 1.0.
    assert smith_agent.length_penalized_reward(1.0, 55, 100, t0=55, lam=0.2, is_train=True) == 1.0
    assert smith_agent.length_penalized_reward(1.0, 30, 100, t0=55, lam=0.2, is_train=True) == 1.0


def test_length_penalty_solved_ramps_to_cap() -> None:
    # Linear ramp between T0 and max_turns; runs-to-cap gets exactly 1 - lam.
    # Values mirror the report's simulation table (T0=55, lam=0.2, max=100).
    assert smith_agent.length_penalized_reward(1.0, 100, 100, t0=55, lam=0.2, is_train=True) == 0.8
    r69 = smith_agent.length_penalized_reward(1.0, 69, 100, t0=55, lam=0.2, is_train=True)
    assert abs(r69 - 0.9377777777777778) < 1e-9
    mid = smith_agent.length_penalized_reward(1.0, 90, 100, t0=55, lam=0.2, is_train=True)
    assert abs(mid - 0.8444444444444444) < 1e-9


def test_length_penalty_guards_degenerate_span() -> None:
    # If T0 >= max_turns the span guard prevents division by zero and saturates.
    assert smith_agent.length_penalized_reward(1.0, 100, 100, t0=100, lam=0.2, is_train=True) == 1.0
    assert smith_agent.length_penalized_reward(1.0, 120, 100, t0=100, lam=0.2, is_train=True) == 0.8


def _plp(reward, mpt, is_train=True, solved=True):
    return smith_agent.prompt_length_penalty(
        reward,
        mpt,
        soft_start=50000,
        hard_cap=64000,
        max_pen=0.1,
        is_train=is_train,
        solved=solved,
    )


def test_prompt_length_penalty_only_applies_to_solved() -> None:
    # Unsolved rollouts are never penalized, regardless of prompt size.
    assert _plp(0.0, 64000, solved=False) == 0.0
    assert _plp(0.0, 80000, solved=False) == 0.0


def test_prompt_length_penalty_skips_validation() -> None:
    # Validation reward must never be reshaped, even for a long solved rollout.
    assert _plp(1.0, 64000, is_train=False) == 1.0
    assert _plp(1.0, 100000, is_train=False) == 1.0


def test_prompt_length_penalty_below_soft_start_keeps_reward() -> None:
    # A solve whose longest prompt fits under soft_start keeps its full reward.
    assert _plp(1.0, 50000) == 1.0
    assert _plp(1.0, 30000) == 1.0


def test_prompt_length_penalty_ramps_linearly() -> None:
    # Linear ramp between soft_start (50K) and hard_cap (64K), 0 -> max_pen (0.1).
    # 57000 is the midpoint -> half the max penalty.
    assert abs(_plp(1.0, 57000) - 0.95) < 1e-9
    # 53500 is a quarter of the way -> 0.025 penalty.
    assert abs(_plp(1.0, 53500) - 0.975) < 1e-9


def test_prompt_length_penalty_saturates_at_hard_cap() -> None:
    # At or above hard_cap the full max_pen is applied.
    assert abs(_plp(1.0, 64000) - 0.9) < 1e-9
    assert abs(_plp(1.0, 80000) - 0.9) < 1e-9


def test_prompt_length_penalty_stacks_with_turn_penalty() -> None:
    # plan A then plan B compose: a train solve that runs to the turn cap AND uses a
    # long prompt loses both. Turn penalty: 1 - 0.2 = 0.8. Then prompt penalty at
    # hard_cap subtracts 0.1 -> 0.7.
    after_turns = smith_agent.length_penalized_reward(1.0, 100, 100, t0=55, lam=0.2, is_train=True)
    assert after_turns == 0.8
    assert abs(_plp(after_turns, 64000) - 0.7) < 1e-9
