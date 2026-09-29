# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from agentlightning.verl import entrypoint


def test_run_ppo_preflight_happens_before_ray_init(monkeypatch) -> None:
    ray_init_called = False

    def fail_preflight(*_args, **_kwargs):
        raise RuntimeError("readiness unavailable")

    def ray_init(**_kwargs):
        nonlocal ray_init_called
        ray_init_called = True

    monkeypatch.setattr(entrypoint, "prepare_datasets", fail_preflight)
    monkeypatch.setattr(entrypoint.ray, "init", ray_init)
    monkeypatch.setattr(entrypoint.ray, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="readiness unavailable"):
        entrypoint.run_ppo(
            OmegaConf.create({}),
            [{"instance_id": "train"}],
            [{"instance_id": "val"}],
        )

    assert ray_init_called is False
