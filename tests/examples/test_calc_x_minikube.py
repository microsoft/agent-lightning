# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest
from omegaconf import DictConfig, OmegaConf

from agentlightning.controller.k8s_reconciler import build_job_spec
from agentlightning.schemas import Rollout, RolloutConfig, RolloutK8sConfig, RolloutLifecycleStatus

REPO_ROOT = Path(__file__).parents[2]
BASH = shutil.which("bash")
LINUX_ONLY = pytest.mark.skipif(os.name == "nt" or BASH is None, reason="requires Linux bash")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _mock_path(tmp_path: Path) -> Path:
    tools = tmp_path / "bin"
    tools.mkdir()

    for name in ("date", "env", "seq"):
        executable = shutil.which(name)
        assert executable is not None
        (tools / name).symlink_to(executable)

    no_op = "#!/bin/sh\nexit 0\n"
    for name in ("agl-server", "curl", "minikube", "pkill", "ray", "sleep"):
        _write_executable(tools / name, no_op)

    _write_executable(
        tools / "agl-controller",
        """#!/bin/sh
temporary="${MOCK_CONTROLLER_ARGS}.tmp"
printf '%s\n' "$@" > "$temporary"
/bin/mv "$temporary" "$MOCK_CONTROLLER_ARGS"
: > "$MOCK_CONTROLLER_READY"
""",
    )
    _write_executable(
        tools / "python",
        """#!/bin/sh
attempt=0
while [ ! -f "$MOCK_CONTROLLER_READY" ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 100 ]; then
        printf 'controller arguments were not captured\n' >&2
        exit 94
    fi
    /usr/bin/sleep 0.01
done
printf '%s\n' "$@" > "$MOCK_TRAINER_ARGS"
""",
    )
    return tools


def _run_launcher(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    assert BASH is not None
    example = tmp_path / "calc_x"
    example.mkdir()
    script = example / "run_minikube.sh"
    script.write_text(
        (REPO_ROOT / "examples/calc_x/run_minikube.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    script.chmod(0o755)

    controller_args = tmp_path / "controller-args"
    controller_ready = tmp_path / "controller-ready"
    trainer_args = tmp_path / "trainer-args"
    env = {
        **os.environ,
        "PATH": str(_mock_path(tmp_path)),
        "MOCK_CONTROLLER_ARGS": str(controller_args),
        "MOCK_CONTROLLER_READY": str(controller_ready),
        "MOCK_TRAINER_ARGS": str(trainer_args),
    }
    result = subprocess.run(
        [BASH, str(script), "--marker", "value with spaces"],
        cwd=example,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return (
        result,
        controller_args.read_text(encoding="utf-8").splitlines(),
        trainer_args.read_text(encoding="utf-8").splitlines(),
    )


@LINUX_ONLY
def test_minikube_launcher_uses_separate_host_and_agent_urls(tmp_path: Path) -> None:
    result, controller_args, trainer_args = _run_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert controller_args == [
        "runner_type=k8s",
        "agl_server.url=http://localhost:8181",
        "agl_server.agent_url=http://host.minikube.internal:8181",
        "agl_server.key=dummy",
        "k8s_runner.ttl_after_finished=600",
    ]
    assert trainer_args[-2:] == ["--marker", "value with spaces"]

    config = cast(
        DictConfig,
        OmegaConf.merge(
            OmegaConf.load(REPO_ROOT / "agentlightning/config/controller.yaml"),
            OmegaConf.from_dotlist(controller_args),
        ),
    )
    assert config.runner_type == "k8s"
    assert config.agl_server.url == "http://localhost:8181"
    assert config.agl_server.agent_url == "http://host.minikube.internal:8181"
    assert config.agl_server.key == "dummy"
    assert config.k8s_runner.ttl_after_finished == 600

    rollout = Rollout(
        rollout_id="minikube-addresses",
        input={"question": "1 + 1", "result": 2},
        config=RolloutConfig(
            k8s=RolloutK8sConfig(
                job_template=(REPO_ROOT / "examples/calc_x/job-template.yaml").read_text(encoding="utf-8")
            )
        ),
        status=RolloutLifecycleStatus(created_at=1.0, updated_at=1.0),
    )
    manifest = build_job_spec(rollout, config)
    container_env = {
        item["name"]: item["value"] for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert container_env["AGL_OPENAI_BASE_URL"].startswith("http://host.minikube.internal:8181/proxy/")
    assert container_env["AGL_EVENT_URL"].startswith("http://host.minikube.internal:8181/api/rollouts/")
    assert container_env["AGL_KEY"] == "dummy"
