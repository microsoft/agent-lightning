# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Dict,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    overload,
)

from agentlightning.adapter import TraceAdapter
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset, NamedResources

if TYPE_CHECKING:
    from agentlightning.llm_proxy import LLMProxy

from .base import Algorithm

# Algorithm function signature types
# We've missed a lot of combinations here.
# Let's add them in future.


class AlgorithmFuncSyncFull(Protocol):
    def __call__(
        self,
        *,
        store: LightningStore,
        train_dataset: Optional[Dataset[Any]],
        val_dataset: Optional[Dataset[Any]],
        llm_proxy: Optional[LLMProxy],
        adapter: Optional[TraceAdapter[Any]],
        initial_resources: Optional[NamedResources],
    ) -> None: ...


class AlgorithmFuncSyncOnlyStore(Protocol):
    def __call__(self, *, store: LightningStore) -> None: ...


class AlgorithmFuncSyncOnlyDataset(Protocol):
    def __call__(self, *, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]) -> None: ...


class AlgorithmFuncAsyncFull(Protocol):
    def __call__(
        self,
        *,
        store: LightningStore,
        train_dataset: Optional[Dataset[Any]],
        val_dataset: Optional[Dataset[Any]],
        llm_proxy: Optional[LLMProxy],
        adapter: Optional[TraceAdapter[Any]],
        initial_resources: Optional[NamedResources],
    ) -> Awaitable[None]: ...


class AlgorithmFuncAsyncOnlyStore(Protocol):
    def __call__(self, *, store: LightningStore) -> Awaitable[None]: ...


class AlgorithmFuncAsyncOnlyDataset(Protocol):
    def __call__(
        self, *, train_dataset: Optional[Dataset[Any]], val_dataset: Optional[Dataset[Any]]
    ) -> Awaitable[None]: ...


AlgorithmFuncAsync = Union[AlgorithmFuncAsyncOnlyStore, AlgorithmFuncAsyncOnlyDataset, AlgorithmFuncAsyncFull]

AlgorithmFuncSync = Union[AlgorithmFuncSyncOnlyStore, AlgorithmFuncSyncOnlyDataset, AlgorithmFuncSyncFull]


class AlgorithmFuncSyncFallback(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class AlgorithmFuncAsyncFallback(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


AlgorithmFuncSyncLike = Union[AlgorithmFuncSync, AlgorithmFuncSyncFallback]
AlgorithmFuncAsyncLike = Union[AlgorithmFuncAsync, AlgorithmFuncAsyncFallback]

AlgorithmFunc = Union[AlgorithmFuncSyncLike, AlgorithmFuncAsyncLike]


AsyncFlag = Literal[True, False]
AF = TypeVar("AF", bound=AsyncFlag)


class FunctionalAlgorithm(Algorithm, Generic[AF]):
    """A Algorithm that wraps a function-based algorithm implementation.

    This class allows users to define algorithm behavior using a simple function
    that takes train_dataset and val_dataset parameters, rather than implementing
    a full Algorithm subclass.
    """

    @overload
    def __init__(self: "FunctionalAlgorithm[Literal[False]]", algorithm_func: AlgorithmFuncSyncLike) -> None: ...

    @overload
    def __init__(self: "FunctionalAlgorithm[Literal[True]]", algorithm_func: AlgorithmFuncAsyncLike) -> None: ...

    def __init__(self, algorithm_func: Union[AlgorithmFuncSyncLike, AlgorithmFuncAsyncLike]) -> None:
        """
        Initialize the FunctionalAlgorithm with an algorithm function.

        Args:
            algorithm_func: A function that defines the algorithm's behavior.
                           Can be sync or async with signature:
                           (train_dataset, val_dataset) -> None
        """
        super().__init__()
        self._algorithm_func = algorithm_func
        self._sig = inspect.signature(algorithm_func)
        self._is_async = inspect.iscoroutinefunction(algorithm_func)

        # Copy function metadata to preserve type hints and other attributes
        functools.update_wrapper(self, algorithm_func)  # type: ignore

    def is_async(self) -> bool:
        return self._is_async

    @overload
    def run(
        self: "FunctionalAlgorithm[Literal[False]]",
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> None: ...

    @overload
    def run(
        self: "FunctionalAlgorithm[Literal[True]]",
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> Awaitable[None]: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._algorithm_func(*args, **kwargs)  # type: ignore

    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> Union[None, Awaitable[None]]:
        """Execute the algorithm using the wrapped function.

        Args:
            train_dataset: The dataset to train on.
            val_dataset: The dataset to validate on.

        Returns:
            None or Awaitable[None] if the function is async.
        """
        kwargs: Dict[str, Any] = {}
        if "store" in self._sig.parameters:
            kwargs["store"] = self.get_store()
        if "adapter" in self._sig.parameters:
            kwargs["adapter"] = self.get_adapter()
        if "llm_proxy" in self._sig.parameters:
            kwargs["llm_proxy"] = self.get_llm_proxy()
        if "initial_resources" in self._sig.parameters:
            kwargs["initial_resources"] = self.get_initial_resources()
        if "train_dataset" in self._sig.parameters:
            kwargs["train_dataset"] = train_dataset
        elif train_dataset is not None:
            raise TypeError(
                f"train_dataset is provided but not supported by the algorithm function: {self._algorithm_func}"
            )
        if "val_dataset" in self._sig.parameters:
            kwargs["val_dataset"] = val_dataset
        elif val_dataset is not None:
            raise TypeError(
                f"val_dataset is provided but not supported by the algorithm function: {self._algorithm_func}"
            )
        # both sync and async functions can be called with the same signature
        result = self._algorithm_func(**kwargs)  # type: ignore[misc]
        if self._is_async:
            return cast(Awaitable[None], result)
        return None


@overload
def algo(func: AlgorithmFuncAsync) -> FunctionalAlgorithm[Literal[True]]: ...


@overload
def algo(func: AlgorithmFuncAsyncFallback) -> FunctionalAlgorithm[Any]: ...


@overload
def algo(func: AlgorithmFuncSync) -> FunctionalAlgorithm[Literal[False]]: ...


@overload
def algo(func: AlgorithmFuncSyncFallback) -> FunctionalAlgorithm[Any]: ...


def algo(
    func: Union[
        AlgorithmFuncSync,
        AlgorithmFuncAsync,
        AlgorithmFuncSyncFallback,
        AlgorithmFuncAsyncFallback,
    ],
) -> Union[FunctionalAlgorithm[Literal[False]], FunctionalAlgorithm[Literal[True]]]:
    """Create a Algorithm from a function.

    This decorator allows you to define an algorithm using a simple function
    instead of creating a full Algorithm subclass. The returned FunctionalAlgorithm
    instance is callable, preserving the original function's behavior.

    Args:
        func: A function that defines the algorithm's behavior with signature:
              (train_dataset, val_dataset) -> None
              Can be sync or async.

    Returns:
        A callable FunctionalAlgorithm instance that preserves the original function's
        type hints and behavior while providing all algorithm functionality.

    Example:
        @algo
        def my_algorithm(train_dataset, val_dataset):
            # Algorithm logic here
            for task in train_dataset:
                # Process training tasks
                pass

        @algo
        async def my_async_algorithm(train_dataset, val_dataset):
            # Async algorithm logic here
            async for task in train_dataset:
                # Process training tasks asynchronously
                pass

        # Function is still callable with original behavior
        my_algorithm(train_data, val_data)

        # Algorithm methods are also available
        my_algorithm.run(train_data, val_data)
    """
    return FunctionalAlgorithm(func)
