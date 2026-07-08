# Copyright (c) Microsoft. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock
from typing import Any, cast

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


def test_verl_run_rejects_missing_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """VERL.run must reject missing store and avoid calling run_ppo."""
    mocked_run_ppo = MagicMock()
    monkeypatch.setattr(verl_interface, "run_ppo", mocked_run_ppo)

    algorithm = verl_interface.VERL(config={})

    with pytest.raises(ValueError, match="does not support v0 fallback mode"):
        algorithm.run(_context_without_store())

    mocked_run_ppo.assert_not_called()


def test_run_ppo_rejects_missing_store() -> None:
    """run_ppo must reject missing store before starting Ray."""
    with pytest.raises(ValueError, match="does not support v0 fallback mode"):
        verl_interface.run_ppo(
            config=SimpleNamespace(),
            train_dataset=None,
            val_dataset=None,
            store=None,
            llm_proxy=None,
            adapter=None,
            trainer_cls=cast(type[Any], MagicMock()),
            daemon_cls=cast(type[Any], MagicMock()),
        )
