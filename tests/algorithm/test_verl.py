# Copyright (c) Microsoft. All rights reserved.

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

verl_interface = pytest.importorskip(
    "agentlightning.algorithm.verl.interface", reason="VERL optional dependencies are not installed"
)


def test_run_ppo_rejects_missing_store() -> None:
    """run_ppo must reject missing store before starting Ray."""
    verl_entrypoint = pytest.importorskip(
        "agentlightning.verl.entrypoint", reason="VERL optional dependencies are not installed"
    )
    with pytest.raises(ValueError, match="does not support v0 fallback mode"):
        verl_entrypoint.run_ppo(
            config=SimpleNamespace(),
            train_dataset=None,
            val_dataset=None,
            store=None,
            llm_proxy=None,
            adapter=None,
            trainer_cls=cast(type[Any], MagicMock()),
            daemon_cls=cast(type[Any], MagicMock()),
        )
