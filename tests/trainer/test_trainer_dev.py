# Copyright (c) Microsoft. All rights reserved.

"""Tests for Trainer.dev requirements."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentlightning.algorithm import Algorithm, Baseline
from agentlightning.execution.base import AlgorithmBundle, ExecutionStrategy, RunnerBundle
from agentlightning.execution.events import ThreadingEvent
from agentlightning.litagent import LitAgent
from agentlightning.store import LightningStore
from agentlightning.trainer import Trainer
from agentlightning.types import AlgorithmContext


class DummyStrategy(ExecutionStrategy):
    """Execution strategy that only records invocation."""

    def __init__(self) -> None:
        self.called = False

    def execute(
        self,
        algorithm: AlgorithmBundle,
        runner: RunnerBundle,
        store: LightningStore,
    ) -> None:
        self.called = True


class DummyAgent(LitAgent[Any]):
    """Minimal agent for exercising Trainer.dev."""


class SlowAlgorithm(Algorithm):
    """Algorithm that does not qualify as FastAlgorithm."""

    def run(self, context: AlgorithmContext) -> None:
        return None


class AlgorithmOnlyStrategy(ExecutionStrategy):
    """Execute only the algorithm role to inspect its runtime context."""

    def execute(
        self,
        algorithm: AlgorithmBundle,
        runner: RunnerBundle,
        store: LightningStore,
    ) -> None:
        asyncio.run(algorithm(store, ThreadingEvent()))


class RecordingAlgorithm(Algorithm):
    """Record the context supplied by Trainer."""

    def __init__(self) -> None:
        self.context: AlgorithmContext | None = None

    def run(self, context: AlgorithmContext) -> None:
        self.context = context


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


def test_fit_passes_runtime_dependencies_through_algorithm_context() -> None:
    algorithm = RecordingAlgorithm()
    strategy = AlgorithmOnlyStrategy()
    train_dataset = [{"input": "train"}]
    val_dataset = [{"input": "val"}]
    trainer = Trainer(
        strategy=strategy,
        algorithm=algorithm,
        initial_resources={},
    )

    trainer.fit(DummyAgent(), train_dataset, val_dataset=val_dataset)

    assert algorithm.context is not None
    assert algorithm.context.store is trainer.store
    assert algorithm.context.adapter is trainer.adapter
    assert algorithm.context.initial_resources is trainer.initial_resources
    assert algorithm.context.train_dataset is train_dataset
    assert algorithm.context.val_dataset is val_dataset
