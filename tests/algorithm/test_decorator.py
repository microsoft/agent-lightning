# Copyright (c) Microsoft. All rights reserved.

"""Test that @algo decorator preserves function executability."""

import inspect
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from agentlightning.algorithm.decorator import FunctionalAlgorithm, algo
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset


@algo
def sample_algorithm_func(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
    """A test function with algorithm decorator."""
    # Store the datasets in a way we can verify
    sample_algorithm_func.last_train = train_dataset  # type: ignore
    sample_algorithm_func.last_val = val_dataset  # type: ignore


def test_algorithm_preserves_executability():
    """Test that @algo decorated functions remain executable."""
    test_train = ["train1", "train2"]
    test_val = ["val1"]

    # Function should be callable
    assert callable(sample_algorithm_func)

    # Function should execute with keyword arguments
    sample_algorithm_func(train_dataset=test_train, val_dataset=test_val)

    # Verify it was called with the right arguments
    assert sample_algorithm_func.last_train == test_train  # type: ignore
    assert sample_algorithm_func.last_val == test_val  # type: ignore


def test_algorithm_preserves_metadata():
    """Test that @algo preserves function metadata."""
    # Function name should be preserved
    assert sample_algorithm_func.__name__ == "sample_algorithm_func"  # type: ignore

    # Docstring should be preserved
    assert sample_algorithm_func.__doc__ == "A test function with algorithm decorator."


def test_algorithm_returns_functional_algorithm_instance():
    """Test that @algo returns a FunctionalAlgorithm instance."""
    assert isinstance(sample_algorithm_func, FunctionalAlgorithm)

    # Should have algorithm methods
    assert hasattr(sample_algorithm_func, "run")
    assert hasattr(sample_algorithm_func, "get_store")
    assert hasattr(sample_algorithm_func, "set_trainer")


def test_algorithm_preserves_signature():
    """Test that @algo preserves function signature."""
    sig = inspect.signature(sample_algorithm_func)
    params = list(sig.parameters.keys())

    # Should have the expected parameters
    assert params == ["train_dataset", "val_dataset"]


def test_algorithm_run_method():
    """Test that the run method works correctly."""

    @algo
    def test_algo(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
        """Test algorithm."""
        test_algo.executed = True  # type: ignore
        test_algo.train = train_dataset  # type: ignore
        test_algo.val = val_dataset  # type: ignore

    test_algo.executed = False  # type: ignore

    train_data = ["item1", "item2"]
    val_data = ["val1"]

    # Call run method
    test_algo.run(train_data, val_data)

    # Verify execution
    assert test_algo.executed  # type: ignore
    assert test_algo.train == train_data  # type: ignore
    assert test_algo.val == val_data  # type: ignore


def test_algorithm_callable_shortcut():
    """Test that calling the instance directly works."""

    @algo
    def test_algo(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
        """Test algorithm."""
        test_algo.called = True  # type: ignore

    test_algo.called = False  # type: ignore

    # Direct call should work with keyword arguments
    test_algo(train_dataset=None, val_dataset=None)

    assert test_algo.called  # type: ignore


@pytest.mark.asyncio
async def test_async_function_with_algorithm():
    """Test that async functions work with @algo decorator."""

    @algo
    async def async_algo(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
        """An async test function."""
        async_algo.executed = True  # type: ignore
        async_algo.train = train_dataset  # type: ignore

    async_algo.executed = False  # type: ignore

    # Should be callable
    assert callable(async_algo)

    # Should preserve async nature when called directly with keyword arguments
    test_data = ["async-test"]
    await async_algo(train_dataset=test_data, val_dataset=None)

    assert async_algo.executed  # type: ignore
    assert async_algo.train == test_data  # type: ignore


@pytest.mark.asyncio
async def test_async_algorithm_run_method():
    """Test that async algorithms work with the run method."""

    @algo
    async def async_algo(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
        """An async algorithm."""
        async_algo.run_executed = True  # type: ignore
        async_algo.run_train = train_dataset  # type: ignore
        async_algo.run_val = val_dataset  # type: ignore

    async_algo.run_executed = False  # type: ignore

    train_data = ["async-train"]
    val_data = ["async-val"]

    # Run method should return an awaitable
    assert async_algo.is_async()
    result = async_algo.run(train_data, val_data)
    assert inspect.iscoroutine(result)

    # Await the result
    await result

    assert async_algo.run_executed  # type: ignore
    assert async_algo.run_train == train_data  # type: ignore
    assert async_algo.run_val == val_data  # type: ignore


def test_algorithm_with_none_datasets():
    """Test that algorithm works with None datasets."""

    @algo
    def nullable_algo(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
        """Algorithm that accepts None."""
        nullable_algo.called_with_none = train_dataset is None and val_dataset is None  # type: ignore

    nullable_algo(train_dataset=None, val_dataset=None)
    assert nullable_algo.called_with_none  # type: ignore

    # Also test via run method
    nullable_algo.called_with_none = False  # type: ignore
    nullable_algo.run()
    assert nullable_algo.called_with_none  # type: ignore


def test_multiple_algorithm_instances():
    """Test that multiple decorated functions work independently."""

    @algo
    def algo1(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
        """First algorithm."""
        algo1.count = getattr(algo1, "count", 0) + 1  # type: ignore

    @algo
    def algo2(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
        """Second algorithm."""
        algo2.count = getattr(algo2, "count", 0) + 1  # type: ignore

    algo1.count = 0  # type: ignore
    algo2.count = 0  # type: ignore

    algo1(train_dataset=None, val_dataset=None)
    algo1(train_dataset=None, val_dataset=None)
    algo2(train_dataset=None, val_dataset=None)

    assert algo1.count == 2  # type: ignore
    assert algo2.count == 1  # type: ignore


def test_algorithm_base_algorithm_methods():
    """Test that Algorithm methods are available."""

    @algo
    def test_algo(*, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None:
        """Test algorithm."""
        pass

    # Should have all Algorithm methods
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


# New tests for parameter injection and error handling


def test_algorithm_without_datasets():
    """Test that algorithms can be defined without train_dataset/val_dataset parameters."""

    @algo
    def no_dataset_algo(*, store: LightningStore) -> None:
        """Algorithm that doesn't use datasets."""
        no_dataset_algo.store_passed = store  # type: ignore
        no_dataset_algo.executed = True  # type: ignore

    no_dataset_algo.executed = False  # type: ignore

    # Set up the store
    mock_store = MagicMock(spec=LightningStore)
    no_dataset_algo.set_store(mock_store)

    # Call run method without datasets
    no_dataset_algo.run()

    assert no_dataset_algo.executed  # type: ignore
    assert no_dataset_algo.store_passed == mock_store  # type: ignore


def test_algorithm_raises_error_on_unsupported_train_dataset():
    """Test that TypeError is raised when train_dataset is provided but not supported."""

    @algo
    def no_train_algo(*, val_dataset: Optional[Dataset[Any]]) -> None:
        """Algorithm that only accepts val_dataset."""
        pass

    # Providing train_dataset should raise TypeError
    with pytest.raises(TypeError, match="train_dataset is provided but not supported"):
        no_train_algo.run(train_dataset=["data"], val_dataset=None)


def test_algorithm_raises_error_on_unsupported_val_dataset():
    """Test that TypeError is raised when val_dataset is provided but not supported."""

    @algo
    def no_val_algo(*, train_dataset: Optional[Dataset[Any]]) -> None:
        """Algorithm that only accepts train_dataset."""
        pass

    # Providing val_dataset should raise TypeError
    with pytest.raises(TypeError, match="val_dataset is provided but not supported"):
        no_val_algo.run(train_dataset=None, val_dataset=["data"])


def test_algorithm_with_all_injected_parameters():
    """Test that all injectable parameters (store, adapter, llm_proxy, initial_resources) work."""

    @algo
    def full_algo(
        *,
        store: LightningStore,
        adapter: Any,
        llm_proxy: Optional[Any] = None,
        initial_resources: Optional[Any] = None,
        train_dataset: Optional[Dataset[Any]],
        val_dataset: Optional[Dataset[Any]],
    ) -> None:
        """Algorithm with all injectable parameters."""
        full_algo.store = store  # type: ignore
        full_algo.adapter = adapter  # type: ignore
        full_algo.llm_proxy = llm_proxy  # type: ignore
        full_algo.initial_resources = initial_resources  # type: ignore
        full_algo.train = train_dataset  # type: ignore
        full_algo.val = val_dataset  # type: ignore

    # Set up all dependencies
    mock_store = MagicMock(spec=LightningStore)
    mock_adapter = MagicMock()
    mock_llm_proxy = MagicMock()
    mock_resources = MagicMock()

    full_algo.set_store(mock_store)
    full_algo.set_adapter(mock_adapter)
    full_algo.set_llm_proxy(mock_llm_proxy)
    full_algo.set_initial_resources(mock_resources)

    train_data = ["train"]
    val_data = ["val"]

    # Run the algorithm
    full_algo.run(train_data, val_data)

    # Verify all parameters were injected correctly
    assert full_algo.store == mock_store  # type: ignore
    assert full_algo.adapter == mock_adapter  # type: ignore
    assert full_algo.llm_proxy == mock_llm_proxy  # type: ignore
    assert full_algo.initial_resources == mock_resources  # type: ignore
    assert full_algo.train == train_data  # type: ignore
    assert full_algo.val == val_data  # type: ignore


def test_algorithm_with_only_store():
    """Test algorithm that only uses the store parameter."""

    @algo
    def store_only_algo(*, store: LightningStore) -> None:
        """Algorithm that only needs store."""
        store_only_algo.got_store = True  # type: ignore
        store_only_algo.store_value = store  # type: ignore

    store_only_algo.got_store = False  # type: ignore

    mock_store = MagicMock(spec=LightningStore)
    store_only_algo.set_store(mock_store)

    # Should work without any datasets
    store_only_algo.run()

    assert store_only_algo.got_store  # type: ignore
    assert store_only_algo.store_value == mock_store  # type: ignore


@pytest.mark.asyncio
async def test_async_algorithm_with_injected_parameters():
    """Test that async algorithms also support parameter injection."""

    @algo
    async def async_full_algo(
        *,
        store: LightningStore,
        train_dataset: Optional[Dataset[Any]],
    ) -> None:
        """Async algorithm with injected parameters."""
        async_full_algo.store = store  # type: ignore
        async_full_algo.train = train_dataset  # type: ignore

    mock_store = MagicMock(spec=LightningStore)
    async_full_algo.set_store(mock_store)  # type: ignore

    train_data = ["async-train"]
    await async_full_algo.run(train_data)  # type: ignore

    assert async_full_algo.store == mock_store  # type: ignore
    assert async_full_algo.train == train_data  # type: ignore
