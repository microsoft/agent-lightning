# Copyright (c) Microsoft. All rights reserved.

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from agentlightning.execution.events import ThreadingEvent
from agentlightning.store.base import LightningStore
from agentlightning.types import AlgorithmContext

verl_interface = pytest.importorskip(
    "agentlightning.algorithm.verl.interface", reason="VERL optional dependencies are not installed"
)


def _context_without_store() -> AlgorithmContext:
    return AlgorithmContext(
        store=cast(LightningStore, None),
        event=ThreadingEvent(),
        train_dataset=None,
        val_dataset=None,
    )


def test_verl_run_rejects_missing_store() -> None:
    """VERL.run must reject missing store before importing the optional runtime."""
    algorithm = object.__new__(verl_interface.VERL)

    with pytest.raises(ValueError, match="does not support v0 fallback mode"):
        algorithm.run(_context_without_store())


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
