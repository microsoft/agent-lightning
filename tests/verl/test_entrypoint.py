# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

from omegaconf import OmegaConf

from agentlightning.verl import entrypoint


class RayStartupReached(Exception):
    """Stop at the Ray boundary without starting distributed workers."""


@pytest.fixture
def ray_startup(monkeypatch: pytest.MonkeyPatch) -> Mock:
    boundary = Mock(side_effect=RayStartupReached)
    monkeypatch.setattr(entrypoint.ray, "is_initialized", boundary)
    return boundary


@pytest.mark.parametrize("val_batch_size", [1, 32, 0, False, pytest.param("None", id="string-None")])
def test_run_ppo_rejects_non_null_validation_batch_size(val_batch_size: int | bool | str, ray_startup: Mock) -> None:
    config = OmegaConf.create({"data": {"val_batch_size": val_batch_size}})

    with pytest.raises(AssertionError, match=r"data\.val_batch_size=null"):
        entrypoint.run_ppo(config, train_dataset=[{}], val_dataset=[{}])

    ray_startup.assert_not_called()


def test_run_ppo_accepts_full_validation_batch(ray_startup: Mock) -> None:
    config = OmegaConf.create({"data": {"val_batch_size": None}})

    with pytest.raises(RayStartupReached):
        entrypoint.run_ppo(config, train_dataset=[{}], val_dataset=[{}])

    ray_startup.assert_called_once_with()
