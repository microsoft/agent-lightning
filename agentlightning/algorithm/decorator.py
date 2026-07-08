# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import inspect
from typing import Any, Awaitable, Generic, Literal, Protocol, TypeVar, Union, overload

from agentlightning.types import AlgorithmContext

from .base import Algorithm


class AlgorithmFuncSync(Protocol):
    def __call__(self, context: AlgorithmContext) -> None: ...


class AlgorithmFuncAsync(Protocol):
    def __call__(self, context: AlgorithmContext) -> Awaitable[None]: ...


AlgorithmFunc = Union[AlgorithmFuncSync, AlgorithmFuncAsync]


AsyncFlag = Literal[True, False]
AF = TypeVar("AF", bound=AsyncFlag)


class FunctionalAlgorithm(Algorithm, Generic[AF]):
    """An algorithm wrapper built from a callable implementation.

    Functional algorithms let you provide a plain function instead of subclassing
    [`Algorithm`][agentlightning.Algorithm], while still receiving a fully
    initialized [`AlgorithmContext`][agentlightning.AlgorithmContext].
    """

    @overload
    def __init__(self: "FunctionalAlgorithm[Literal[False]]", algorithm_func: AlgorithmFuncSync) -> None: ...

    @overload
    def __init__(self: "FunctionalAlgorithm[Literal[True]]", algorithm_func: AlgorithmFuncAsync) -> None: ...

    def __init__(self, algorithm_func: AlgorithmFunc) -> None:
        """Wrap a function that implements algorithm behaviour.

        Args:
            algorithm_func: Sync or async callable implementing the context-based
                algorithm contract.
        """
        super().__init__()
        self._algorithm_func = algorithm_func
        self._is_async = inspect.iscoroutinefunction(algorithm_func)

        # Copy function metadata to preserve behavior expected by callers.
        functools.update_wrapper(self, algorithm_func)  # type: ignore

    def is_async(self) -> bool:
        return self._is_async

    @overload
    def run(self: "FunctionalAlgorithm[Literal[False]]", context: AlgorithmContext) -> None: ...

    @overload
    def run(self: "FunctionalAlgorithm[Literal[True]]", context: AlgorithmContext) -> Awaitable[None]: ...

    def run(self, context: AlgorithmContext) -> Union[None, Awaitable[None]]:
        """Execute the wrapped function with the provided context."""
        result = self._algorithm_func(context)  # type: ignore[arg-type]
        if self._is_async:
            return result
        return None

    def __call__(self, context: AlgorithmContext) -> Union[None, Awaitable[None]]:
        return self.run(context)


@overload
def algo(func: AlgorithmFuncAsync) -> FunctionalAlgorithm[Literal[True]]: ...


@overload
def algo(func: AlgorithmFuncSync) -> FunctionalAlgorithm[Literal[False]]: ...


def algo(
    func: AlgorithmFunc,
) -> Union[FunctionalAlgorithm[Literal[False]], FunctionalAlgorithm[Literal[True]]]:
    """Convert a callable into a [`FunctionalAlgorithm`][agentlightning.algorithm.decorator.FunctionalAlgorithm].

    The decorated callable must accept one argument named `context`.

    ```python
    from agentlightning.algorithm.decorator import algo

    @algo
    def batching_algorithm(context):
        for sample in context.train_dataset or []:
            context.store.enqueue_rollout(input=sample, mode=\"train\")

    @algo
    async def async_algorithm(context):
        await context.store.enqueue_rollout(input={\"prompt\": \"hello\"}, mode=\"train\")
    ```
    """
    return FunctionalAlgorithm(func)
