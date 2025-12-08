# Copyright (c) Microsoft. All rights reserved.

"""Tests for Trainer.dev requirements."""

from __future__ import annotations

from typing import Any

import pytest

from agentlightning.algorithm import Algorithm, Baseline
from agentlightning.execution.base import ExecutionStrategy
from agentlightning.litagent import LitAgent
from agentlightning.trainer import Trainer


class DummyStrategy(ExecutionStrategy):
    """Execution strategy that only records invocation."""

    def __init__(self) -> None:
        self.called = False

    def execute(self, algorithm_bundle, runner_bundle, store) -> None:  # type: ignore[override]
        self.called = True


class DummyAgent(LitAgent[Any]):
    """Minimal agent for exercising Trainer.dev."""


class SlowAlgorithm(Algorithm):
    """Algorithm that does not qualify as FastAlgorithm."""

    def run(self, train_dataset=None, val_dataset=None):  # type: ignore[override]
        return None


def test_dev_requires_fast_algorithm() -> None:
    trainer = Trainer(strategy=DummyStrategy(), algorithm=SlowAlgorithm())
    agent = DummyAgent()

    with pytest.raises(TypeError):
        trainer.dev(agent)


def test_dev_allows_fast_algorithm() -> None:
    strategy = DummyStrategy()
    trainer = Trainer(strategy=strategy, algorithm=Baseline())
    agent = DummyAgent()

    trainer.dev(agent)

    assert strategy.called is True
