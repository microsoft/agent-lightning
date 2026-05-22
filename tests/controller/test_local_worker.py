"""Tests for the local_worker process — runs the worker in-process via main()."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from agl_lite.controller import local_worker


@pytest.fixture
def agl_env(tmp_path: Path) -> Iterator[Path]:
    """Set the env vars the worker reads; restore them on exit."""
    out_path = tmp_path / "agent_out.json"
    old: dict[str, str | None] = {}
    keys = ["AGL_LOCAL_AGENT_CLASS", "AGL_TASK_INPUT", "AGL_FAKE_AGENT_OUT"]
    for k in keys:
        old[k] = os.environ.get(k)
    try:
        os.environ["AGL_FAKE_AGENT_OUT"] = str(out_path)
        yield out_path
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_sync_agent_runs_and_exits_zero(agl_env: Path) -> None:
    os.environ["AGL_LOCAL_AGENT_CLASS"] = "tests.fixtures.local_agent:FakeAgent"
    os.environ["AGL_TASK_INPUT"] = json.dumps({"q": "2+2"})

    rc = local_worker.main()
    assert rc == 0
    assert json.loads(agl_env.read_text()) == {"task": {"q": "2+2"}}


def test_async_agent_runs_and_exits_zero(agl_env: Path) -> None:
    os.environ["AGL_LOCAL_AGENT_CLASS"] = "tests.fixtures.local_agent:FakeAsyncAgent"
    os.environ["AGL_TASK_INPUT"] = json.dumps({"q": "async"})

    rc = local_worker.main()
    assert rc == 0
    payload = json.loads(agl_env.read_text())
    assert payload["task"] == {"q": "async"}
    assert payload["async"] is True


def test_raising_agent_exits_nonzero(agl_env: Path) -> None:
    os.environ["AGL_LOCAL_AGENT_CLASS"] = "tests.fixtures.local_agent:FakeRaisingAgent"
    os.environ["AGL_TASK_INPUT"] = json.dumps({})

    rc = local_worker.main()
    assert rc == 1


def test_missing_module_exits_nonzero(agl_env: Path) -> None:
    os.environ["AGL_LOCAL_AGENT_CLASS"] = "tests.fixtures.does_not_exist:Whatever"
    os.environ["AGL_TASK_INPUT"] = json.dumps({})

    rc = local_worker.main()
    assert rc == 1


def test_target_not_a_class_exits_nonzero(agl_env: Path) -> None:
    os.environ["AGL_LOCAL_AGENT_CLASS"] = "tests.fixtures.local_agent:NOT_A_CLASS"
    os.environ["AGL_TASK_INPUT"] = json.dumps({})

    rc = local_worker.main()
    assert rc == 1


def test_dot_syntax_class_path(agl_env: Path) -> None:
    os.environ["AGL_LOCAL_AGENT_CLASS"] = "tests.fixtures.local_agent.FakeAgent"
    os.environ["AGL_TASK_INPUT"] = json.dumps({"q": "dot"})

    rc = local_worker.main()
    assert rc == 0
    assert json.loads(agl_env.read_text()) == {"task": {"q": "dot"}}


# ---------------------------------------------------------------------------
# settings validator + cli dispatch coverage
# ---------------------------------------------------------------------------


def test_settings_local_missing_pool_size_raises() -> None:
    from pydantic import ValidationError

    from agl_lite.controller.config import ControllerSettings, RunnerType

    with pytest.raises(ValidationError):
        ControllerSettings(
            base_url="http://x",
            namespace="default",
            runner_type=RunnerType.LOCAL,
            local_agent_class="tests.fixtures.local_agent:FakeAgent",
        )


def test_settings_local_missing_agent_class_raises() -> None:
    from pydantic import ValidationError

    from agl_lite.controller.config import ControllerSettings, RunnerType

    with pytest.raises(ValidationError):
        ControllerSettings(
            base_url="http://x",
            namespace="default",
            runner_type=RunnerType.LOCAL,
            local_pool_size=2,
        )


def test_settings_local_negative_pool_size_raises() -> None:
    from pydantic import ValidationError

    from agl_lite.controller.config import ControllerSettings, RunnerType

    with pytest.raises(ValidationError):
        ControllerSettings(
            base_url="http://x",
            namespace="default",
            runner_type=RunnerType.LOCAL,
            local_pool_size=0,
            local_agent_class="tests.fixtures.local_agent:FakeAgent",
        )


def test_settings_k8s_default_does_not_require_local_fields() -> None:
    from agl_lite.controller.config import ControllerSettings, RunnerType

    s = ControllerSettings(
        base_url="http://x",
        namespace="default",
        job_manifest_template="path.yaml",
    )
    assert s.runner_type == RunnerType.K8S
    assert s.local_pool_size is None
