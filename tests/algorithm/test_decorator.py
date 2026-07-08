# Copyright (c) Microsoft. All rights reserved.

"""Test that @algo decorator preserves function executability."""

import inspect
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from agentlightning.algorithm.decorator import FunctionalAlgorithm, algo
from agentlightning.execution.events import ThreadingEvent
from agentlightning.store.base import LightningStore
from agentlightning.types import AlgorithmContext, Dataset


def _context(
    *,
    train_dataset: Optional[Dataset[Any]] = None,
    val_dataset: Optional[Dataset[Any]] = None,
) -> AlgorithmContext:
    return AlgorithmContext(
        store=MagicMock(spec=LightningStore),
        event=ThreadingEvent(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


@algo
def sample_algorithm_func(context: AlgorithmContext) -> None:
    """A test function with algorithm decorator."""
    sample_algorithm_func.last_context = context  # type: ignore[attr-defined]


def test_algorithm_preserves_executability():
    """Test that @algo decorated functions remain executable."""
    test_train = ["train1", "train2"]
    test_val = ["val1"]
    context = _context(train_dataset=test_train, val_dataset=test_val)

    # Function should be callable
    assert callable(sample_algorithm_func)

    # Function should execute with a context
    sample_algorithm_func(context)

    # Verify it was called with context
    assert sample_algorithm_func.last_context is context  # type: ignore[attr-defined]
    assert sample_algorithm_func.last_context.train_dataset == test_train  # type: ignore[attr-defined]
    assert sample_algorithm_func.last_context.val_dataset == test_val  # type: ignore[attr-defined]


def test_algorithm_preserves_metadata():
    """Test that @algo preserves function metadata."""
    assert sample_algorithm_func.__name__ == "sample_algorithm_func"  # type: ignore[attr-defined]
    assert sample_algorithm_func.__doc__ == "A test function with algorithm decorator."  # type: ignore[attr-defined]


def test_algorithm_returns_functional_algorithm_instance():
    """Test that @algo returns a FunctionalAlgorithm instance."""
    assert isinstance(sample_algorithm_func, FunctionalAlgorithm)
    assert hasattr(sample_algorithm_func, "run")
    assert hasattr(sample_algorithm_func, "get_store")


def test_algorithm_preserves_signature():
    """Test that @algo preserves function signature."""
    params = list(inspect.signature(sample_algorithm_func).parameters.keys())
    assert params == ["context"]


def test_algorithm_run_method():
    """Test that the run method works correctly."""

    @algo
    def test_algo(context: AlgorithmContext) -> None:
        """Test algorithm."""
        test_algo.executed = True  # type: ignore[attr-defined]
        test_algo.train = context.train_dataset  # type: ignore[attr-defined]
        test_algo.val = context.val_dataset  # type: ignore[attr-defined]

    test_algo.executed = False  # type: ignore[attr-defined]

    context = _context(train_dataset=["item1", "item2"], val_dataset=["val1"])
    test_algo.run(context)

    assert test_algo.executed  # type: ignore[attr-defined]
    assert test_algo.train == ["item1", "item2"]  # type: ignore[attr-defined]
    assert test_algo.val == ["val1"]  # type: ignore[attr-defined]


def test_algorithm_callable_shortcut():
    """Test that calling the instance directly works."""

    @algo
    def test_algo(context: AlgorithmContext) -> None:
        """Test algorithm."""
        test_algo.called = True  # type: ignore[attr-defined]

    test_algo.called = False  # type: ignore[attr-defined]
    test_algo(_context())

    assert test_algo.called  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_async_function_with_algorithm():
    """Test that async functions work with @algo decorator."""

    @algo
    async def async_algo(context: AlgorithmContext) -> None:
        """An async test function."""
        async_algo.executed = True  # type: ignore[attr-defined]
        async_algo.train = context.train_dataset  # type: ignore[attr-defined]

    async_algo.executed = False  # type: ignore[attr-defined]
    test_data = ["async-test"]

    await async_algo(_context(train_dataset=test_data))

    assert async_algo.executed  # type: ignore[attr-defined]
    assert async_algo.train == test_data  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_async_algorithm_run_method():
    """Test that async algorithms work with the run method."""

    @algo
    async def async_algo(context: AlgorithmContext) -> None:
        """An async algorithm."""
        async_algo.run_executed = True  # type: ignore[attr-defined]
        async_algo.run_train = context.train_dataset  # type: ignore[attr-defined]
        async_algo.run_val = context.val_dataset  # type: ignore[attr-defined]

    async_algo.run_executed = False  # type: ignore[attr-defined]

    result = async_algo.run(_context(train_dataset=["async-train"], val_dataset=["async-val"]))
    assert inspect.iscoroutine(result)
    await result

    assert async_algo.run_executed  # type: ignore[attr-defined]
    assert async_algo.run_train == ["async-train"]  # type: ignore[attr-defined]
    assert async_algo.run_val == ["async-val"]  # type: ignore[attr-defined]


def test_algorithm_with_none_datasets():
    """Test that algorithm works with no datasets."""

    @algo
    def nullable_algo(context: AlgorithmContext) -> None:
        """Algorithm with no datasets."""
        nullable_algo.called_with_none = (
            context.train_dataset is None and context.val_dataset is None
        )  # type: ignore[attr-defined]

    nullable_algo.run(_context())
    assert nullable_algo.called_with_none  # type: ignore[attr-defined]


def test_multiple_algorithm_instances():
    """Test that multiple decorated functions work independently."""

    @algo
    def algo1(context: AlgorithmContext) -> None:
        """First algorithm."""
        algo1.count = getattr(algo1, "count", 0) + 1  # type: ignore[attr-defined]

    @algo
    def algo2(context: AlgorithmContext) -> None:
        """Second algorithm."""
        algo2.count = getattr(algo2, "count", 0) + 1  # type: ignore[attr-defined]

    algo1.count = 0  # type: ignore[attr-defined]
    algo2.count = 0  # type: ignore[attr-defined]

    context = _context()
    algo1(context)
    algo1(context)
    algo2(context)

    assert algo1.count == 2  # type: ignore[attr-defined]
    assert algo2.count == 1  # type: ignore[attr-defined]


def test_algorithm_base_algorithm_methods():
    """Test that Algorithm methods are available."""

    @algo
    def test_algo(context: AlgorithmContext) -> None:
        """Test algorithm."""
        pass

    assert hasattr(test_algo, "set_llm_proxy")
    assert hasattr(test_algo, "get_llm_proxy")
    assert hasattr(test_algo, "set_adapter")
    assert hasattr(test_algo, "get_adapter")
    assert hasattr(test_algo, "set_store")
    assert hasattr(test_algo, "get_store")
    assert hasattr(test_algo, "get_initial_resources")
    assert hasattr(test_algo, "set_initial_resources")


def test_algorithm_no_longer_exposes_trainer_accessors() -> None:
    """Algorithm no longer exposes legacy trainer accessors."""

    @algo
    def test_algo(context: AlgorithmContext) -> None:
        """Test algorithm."""
        pass

    assert not hasattr(test_algo, "set_trainer")
    assert not hasattr(test_algo, "get_trainer")


def test_algorithm_no_longer_exposes_get_client() -> None:
    """Algorithm no longer exposes legacy client accessors."""

    @algo
    def test_algo(context: AlgorithmContext) -> None:
        """Test algorithm."""
        pass

    assert not hasattr(test_algo, "get_client")
