# Copyright (c) Microsoft. All rights reserved.

"""Test that @algorithm decorator preserves function executability."""

import inspect
from typing import Any, Optional, cast

import pytest

from agentlightning.algorithm.base import FunctionalAlgorithm, algorithm
from agentlightning.types.core import Dataset


@algorithm
def sample_algorithm_func(
    train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None
) -> None:
    """A test function with algorithm decorator."""
    # Store the datasets in a way we can verify
    sample_algorithm_func.last_train = train_dataset  # type: ignore
    sample_algorithm_func.last_val = val_dataset  # type: ignore


def test_algorithm_preserves_executability():
    """Test that @algorithm decorated functions remain executable."""
    test_train = ["train1", "train2"]
    test_val = ["val1"]

    # Function should be callable
    assert callable(sample_algorithm_func)

    # Function should execute
    sample_algorithm_func(test_train, test_val)

    # Verify it was called with the right arguments
    assert sample_algorithm_func.last_train == test_train  # type: ignore
    assert sample_algorithm_func.last_val == test_val  # type: ignore


def test_algorithm_preserves_metadata():
    """Test that @algorithm preserves function metadata."""
    # Function name should be preserved
    assert sample_algorithm_func.__name__ == "sample_algorithm_func"  # type: ignore

    # Docstring should be preserved
    assert sample_algorithm_func.__doc__ == "A test function with algorithm decorator."


def test_algorithm_returns_functional_algorithm_instance():
    """Test that @algorithm returns a FunctionalAlgorithm instance."""
    assert isinstance(sample_algorithm_func, FunctionalAlgorithm)

    # Should have algorithm methods
    assert hasattr(sample_algorithm_func, "run")
    assert hasattr(sample_algorithm_func, "get_store")
    assert hasattr(sample_algorithm_func, "set_trainer")


def test_algorithm_preserves_signature():
    """Test that @algorithm preserves function signature."""
    sig = inspect.signature(sample_algorithm_func)
    params = list(sig.parameters.keys())

    # Should have the expected parameters
    assert params == ["train_dataset", "val_dataset"]


def test_algorithm_run_method():
    """Test that the run method works correctly."""

    @algorithm
    def test_algo(train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None) -> None:
        """Test algorithm."""
        test_algo.executed = True  # type: ignore
        test_algo.train = train_dataset  # type: ignore
        test_algo.val = val_dataset  # type: ignore

    test_algo.executed = False  # type: ignore

    train_data = ["item1", "item2"]
    val_data = ["val1"]

    # Call run method
    test_algo.run(cast(Dataset[Any], train_data), cast(Dataset[Any], val_data))

    # Verify execution
    assert test_algo.executed  # type: ignore
    assert test_algo.train == train_data  # type: ignore
    assert test_algo.val == val_data  # type: ignore


def test_algorithm_callable_shortcut():
    """Test that calling the instance directly works."""

    @algorithm
    def test_algo(train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None) -> None:
        """Test algorithm."""
        test_algo.called = True  # type: ignore

    test_algo.called = False  # type: ignore

    # Direct call should work
    test_algo(None, None)

    assert test_algo.called  # type: ignore


@pytest.mark.asyncio
async def test_async_function_with_algorithm():
    """Test that async functions work with @algorithm decorator."""

    @algorithm
    async def async_algo(
        train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None
    ) -> None:
        """An async test function."""
        async_algo.executed = True  # type: ignore
        async_algo.train = train_dataset  # type: ignore

    async_algo.executed = False  # type: ignore

    # Should be callable
    assert callable(async_algo)

    # Should preserve async nature when called directly
    test_data = ["async-test"]
    await async_algo(test_data, None)

    assert async_algo.executed  # type: ignore
    assert async_algo.train == test_data  # type: ignore


@pytest.mark.asyncio
async def test_async_algorithm_run_method():
    """Test that async algorithms work with the run method."""

    @algorithm
    async def async_algo(
        train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None
    ) -> None:
        """An async algorithm."""
        async_algo.run_executed = True  # type: ignore
        async_algo.run_train = train_dataset  # type: ignore
        async_algo.run_val = val_dataset  # type: ignore

    async_algo.run_executed = False  # type: ignore

    train_data = ["async-train"]
    val_data = ["async-val"]

    # Run method should return an awaitable
    result = async_algo.run(cast(Dataset[Any], train_data), cast(Dataset[Any], val_data))
    assert inspect.iscoroutine(result)

    # Await the result
    await result

    assert async_algo.run_executed  # type: ignore
    assert async_algo.run_train == train_data  # type: ignore
    assert async_algo.run_val == val_data  # type: ignore


def test_algorithm_with_none_datasets():
    """Test that algorithm works with None datasets."""

    @algorithm
    def nullable_algo(train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None) -> None:
        """Algorithm that accepts None."""
        nullable_algo.called_with_none = train_dataset is None and val_dataset is None  # type: ignore

    nullable_algo(None, None)
    assert nullable_algo.called_with_none  # type: ignore

    # Also test via run method
    nullable_algo.called_with_none = False  # type: ignore
    nullable_algo.run()
    assert nullable_algo.called_with_none  # type: ignore


def test_multiple_algorithm_instances():
    """Test that multiple decorated functions work independently."""

    @algorithm
    def algo1(train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None) -> None:
        """First algorithm."""
        algo1.count = getattr(algo1, "count", 0) + 1  # type: ignore

    @algorithm
    def algo2(train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None) -> None:
        """Second algorithm."""
        algo2.count = getattr(algo2, "count", 0) + 1  # type: ignore

    algo1.count = 0  # type: ignore
    algo2.count = 0  # type: ignore

    algo1(None, None)
    algo1(None, None)
    algo2(None, None)

    assert algo1.count == 2  # type: ignore
    assert algo2.count == 1  # type: ignore


def test_algorithm_base_algorithm_methods():
    """Test that BaseAlgorithm methods are available."""

    @algorithm
    def test_algo(train_dataset: Optional[Dataset[Any]] = None, val_dataset: Optional[Dataset[Any]] = None) -> None:
        """Test algorithm."""
        pass

    # Should have all BaseAlgorithm methods
    assert hasattr(test_algo, "set_trainer")
    assert hasattr(test_algo, "get_trainer")
    assert hasattr(test_algo, "set_llm_proxy")
    assert hasattr(test_algo, "get_llm_proxy")
    assert hasattr(test_algo, "set_adapter")
    assert hasattr(test_algo, "get_adapter")
    assert hasattr(test_algo, "set_store")
    assert hasattr(test_algo, "get_store")
    assert hasattr(test_algo, "get_initial_resources")
    assert hasattr(test_algo, "set_initial_resources")
