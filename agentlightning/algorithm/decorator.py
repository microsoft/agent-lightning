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
    """An algorithm wrapper built from a callable implementation.

    Functional algorithms let you provide an ordinary function instead of
    subclassing [`Algorithm`][agentlightning.Algorithm]. The wrapper inspects
    the callable signature to supply optional dependencies
    such as the store, adapter, and LLM proxy.
    """

    @overload
    def __init__(self: "FunctionalAlgorithm[Literal[False]]", algorithm_func: AlgorithmFuncSyncLike) -> None: ...

    @overload
    def __init__(self: "FunctionalAlgorithm[Literal[True]]", algorithm_func: AlgorithmFuncAsyncLike) -> None: ...

    def __init__(self, algorithm_func: Union[AlgorithmFuncSyncLike, AlgorithmFuncAsyncLike]) -> None:
        """Wrap a function that implements algorithm behaviour.

        Args:
            algorithm_func: Sync or async callable implementing the algorithm
                contract. Arguments are detected automatically based on the
                function signature.
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
        """Execute the wrapped function with injected dependencies.

        Args:
            train_dataset: Optional training dataset passed through when the
                callable declares a `train_dataset` parameter.
            val_dataset: Optional validation dataset passed through when the
                callable declares a `val_dataset` parameter.

        Returns:
            None for sync callables or an awaitable when the callable is async.

        Raises:
            TypeError: If a dataset is provided but the function signature does
                not accept the corresponding argument.
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
    """Convert a callable into a [`FunctionalAlgorithm`][agentlightning.algorithm.decorator.FunctionalAlgorithm].

    The decorator inspects the callable signature to decide which dependencies
    to inject at runtime, enabling concise algorithm definitions that still
    leverage the full training runtime.

    Args:
        func: Function implementing the algorithm logic. May be synchronous or
            asynchronous. The function can expect all of, or a subset of the following parameters:

            - `store`: [`LightningStore`][agentlightning.store.base.LightningStore],
            - `train_dataset`: [`Dataset`][agentlightning.Dataset],
            - `val_dataset`: [`Dataset`][agentlightning.Dataset],
            - `llm_proxy`: [`LLMProxy`][agentlightning.LLMProxy],
            - `adapter`: [`TraceAdapter`][agentlightning.TraceAdapter],
            - `initial_resources`: [`NamedResources`][agentlightning.NamedResources],

            If the function does not expect a parameter, the wrapper will not inject it into the call.
            Using `*args` and `**kwargs` will not work and no parameters will be injected.

    Returns:
        FunctionalAlgorithm that proxies the callable while exposing the
        `Algorithm` interface.

    Examples:
        ```python
        from agentlightning.algorithm.decorator import algo

        @algo
        def batching_algorithm(*, store, train_dataset, val_dataset):
            for sample in train_dataset:
                store.enqueue_rollout(input=sample, mode="train")

        @algo
        async def async_algorithm(*, store, train_dataset=None, val_dataset=None):
            await store.enqueue_rollout(input={"prompt": "hello"}, mode="train")
        ```
    """
    return FunctionalAlgorithm(func)
