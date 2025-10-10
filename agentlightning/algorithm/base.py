# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import inspect
import weakref
from typing import TYPE_CHECKING, Any, Awaitable, Dict, Generic, Literal, Optional, Protocol, TypeVar, Union, overload

from agentlightning.adapter import TraceAdapter
from agentlightning.client import AgentLightningClient
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset, NamedResources

if TYPE_CHECKING:
    from agentlightning.llm_proxy import LLMProxy
    from agentlightning.trainer import Trainer


class BaseAlgorithm:
    """Algorithm is the strategy, or tuner to train the agent."""

    _trainer_ref: weakref.ReferenceType[Trainer] | None = None
    _llm_proxy_ref: weakref.ReferenceType["LLMProxy"] | None = None
    _store: LightningStore | None = None
    _initial_resources: NamedResources | None = None
    _adapter_ref: weakref.ReferenceType[TraceAdapter[Any]] | None = None

    def set_trainer(self, trainer: Trainer) -> None:
        """
        Set the trainer for this algorithm.

        Args:
            trainer: The Trainer instance that will handle training and validation.
        """
        self._trainer_ref = weakref.ref(trainer)

    def get_trainer(self) -> Trainer:
        """
        Get the trainer for this algorithm.

        Returns:
            The Trainer instance associated with this agent.
        """
        if self._trainer_ref is None:
            raise ValueError("Trainer has not been set for this agent.")
        trainer = self._trainer_ref()
        if trainer is None:
            raise ValueError("Trainer reference is no longer valid (object has been garbage collected).")
        return trainer

    def set_llm_proxy(self, llm_proxy: LLMProxy | None) -> None:
        """
        Set the LLM proxy for this algorithm to reuse when available.

        Args:
            llm_proxy: The LLMProxy instance configured by the trainer, if any.
        """
        self._llm_proxy_ref = weakref.ref(llm_proxy) if llm_proxy is not None else None

    def get_llm_proxy(self) -> Optional[LLMProxy]:
        """
        Retrieve the configured LLM proxy instance, if one has been set.

        Returns:
            The active LLMProxy instance or None when not configured.
        """
        if self._llm_proxy_ref is None:
            return None

        llm_proxy = self._llm_proxy_ref()
        if llm_proxy is None:
            raise ValueError("LLM proxy reference is no longer valid (object has been garbage collected).")

        return llm_proxy

    def set_adapter(self, adapter: TraceAdapter[Any]) -> None:
        """
        Set the adapter for this algorithm to collect and convert traces.
        """
        self._adapter_ref = weakref.ref(adapter)

    def get_adapter(self) -> TraceAdapter[Any]:
        """
        Retrieve the adapter for this algorithm to communicate with the runners.
        """
        if self._adapter_ref is None:
            raise ValueError("Adapter has not been set for this algorithm.")
        adapter = self._adapter_ref()
        if adapter is None:
            raise ValueError("Adapter reference is no longer valid (object has been garbage collected).")
        return adapter

    def set_store(self, store: LightningStore) -> None:
        """
        Set the store for this algorithm to communicate with the runners.

        Store is set directly instead of using weakref because its copy is meant to be
        maintained throughout the algorithm's lifecycle.
        """
        self._store = store

    def get_store(self) -> LightningStore:
        """
        Retrieve the store for this algorithm to communicate with the runners.
        """
        if self._store is None:
            raise ValueError("Store has not been set for this algorithm.")
        return self._store

    def get_initial_resources(self) -> Optional[NamedResources]:
        """
        Get the initial resources for this algorithm.
        """
        return self._initial_resources

    def set_initial_resources(self, resources: NamedResources) -> None:
        """
        Set the initial resources for this algorithm.
        """
        self._initial_resources = resources

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> Union[None, Awaitable[None]]:
        """Subclasses should implement this method to implement the algorithm.

        Args:
            train_dataset: The dataset to train on. Not all algorithms require a training dataset.
            val_dataset: The dataset to validate on. Not all algorithms require a validation dataset.

        Returns:
            Algorithm should refrain from returning anything. It should just run the algorithm.
        """
        raise NotImplementedError("Subclasses must implement run().")

    def get_client(self) -> AgentLightningClient:
        """Get the client to communicate with the algorithm.

        If the algorithm does not require a server-client communication, it can also create a mock client
        that never communicates with itself.

        Deprecated and will be removed in a future version.

        Returns:
            The AgentLightningClient instance associated with this algorithm.
        """
        raise NotImplementedError("Subclasses must implement get_client().")


class FastAlgorithm(BaseAlgorithm):
    """Algorithm that can run fast and qualify for dev mode.

    Fast algorithms enable agent developers to quickly iterate on agent development
    without waiting for a long training to complete.
    """


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

AlgorithmFunc = Union[AlgorithmFuncSync, AlgorithmFuncAsync]


AsyncFlag = Literal[True, False]
AF = TypeVar("AF", bound=AsyncFlag)


class FunctionalAlgorithm(BaseAlgorithm, Generic[AF]):
    """A BaseAlgorithm that wraps a function-based algorithm implementation.

    This class allows users to define algorithm behavior using a simple function
    that takes train_dataset and val_dataset parameters, rather than implementing
    a full BaseAlgorithm subclass.
    """

    @overload
    def __init__(self: "FunctionalAlgorithm[Literal[False]]", algorithm_func: AlgorithmFuncSync) -> None: ...

    @overload
    def __init__(self: "FunctionalAlgorithm[Literal[True]]", algorithm_func: AlgorithmFuncAsync) -> None: ...

    def __init__(self, algorithm_func: Union[AlgorithmFuncSync, AlgorithmFuncAsync]) -> None:
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
        return self._algorithm_func(**kwargs)  # type: ignore


@overload
def algo(func: AlgorithmFuncSync) -> FunctionalAlgorithm[Literal[False]]: ...


@overload
def algo(func: AlgorithmFuncAsync) -> FunctionalAlgorithm[Literal[True]]: ...


def algo(func: AlgorithmFunc) -> Union[FunctionalAlgorithm[Literal[False]], FunctionalAlgorithm[Literal[True]]]:
    """Create a BaseAlgorithm from a function.

    This decorator allows you to define an algorithm using a simple function
    instead of creating a full BaseAlgorithm subclass. The returned FunctionalAlgorithm
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
