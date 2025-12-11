# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import logging
import random
from types import CoroutineType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Iterator,
    List,
    Literal,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    overload,
)

from agentlightning.types import Dataset

if TYPE_CHECKING:
    from agentlightning.llm_proxy import LLMProxy
    from agentlightning.store.base import LightningStore

    from .base import Algorithm

T_task = TypeVar("T_task")
T_algo = TypeVar("T_algo", bound="Algorithm")

P = ParamSpec("P")
R = TypeVar("R")


logger = logging.getLogger(__name__)


def batch_iter_over_dataset(dataset: Dataset[T_task], batch_size: int) -> Iterator[Sequence[T_task]]:
    """
    Create an infinite iterator that yields batches from the dataset.

    When batch_size >= dataset size, yields the entire shuffled dataset repeatedly.
    When batch_size < dataset size, yields batches of the specified size, reshuffling
    after each complete pass through the dataset.

    Args:
        dataset: The dataset to iterate over.
        batch_size: The desired batch size.

    Yields:
        Sequences of tasks from the dataset. Each task appears at most once per epoch.
    """
    if batch_size >= len(dataset):
        while True:
            dataset_copy = [dataset[i] for i in range(len(dataset))]
            random.shuffle(dataset_copy)
            yield dataset_copy

    else:
        current_batch: List[int] = []
        while True:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            for index in indices:
                if index in current_batch:
                    continue
                current_batch.append(index)
                if len(current_batch) == batch_size:
                    yield [dataset[index] for index in current_batch]
                    current_batch = []


def with_store(
    func: Callable[Concatenate[T_algo, LightningStore, P], CoroutineType[Any, Any, R]],
) -> Callable[Concatenate[T_algo, P], CoroutineType[Any, Any, R]]:
    """Inject the algorithm's `LightningStore` into coroutine methods.

    The decorator calls `Algorithm.get_store()` once per invocation and passes the
    resulting store as an explicit argument to the wrapped coroutine. Decorated
    methods therefore receive the resolved store even when invoked by helper
    utilities rather than directly by the algorithm.

    Args:
        func: The coroutine that expects `(self, store, *args, **kwargs)`.

    Returns:
        A coroutine wrapper that automatically retrieves the store and forwards it
        to `func`.
    """

    @functools.wraps(func)
    async def wrapper(self: T_algo, *args: P.args, **kwargs: P.kwargs) -> R:
        store = self.get_store()
        return await func(self, store, *args, **kwargs)

    return wrapper


@overload
def with_llm_proxy(
    required: Literal[False] = False,
    auto_start: bool = True,
) -> Callable[
    [Callable[Concatenate[T_algo, Optional[LLMProxy], P], CoroutineType[Any, Any, R]]],
    Callable[Concatenate[T_algo, P], CoroutineType[Any, Any, R]],
]: ...


@overload
def with_llm_proxy(
    required: Literal[True],
    auto_start: bool = True,
) -> Callable[
    [Callable[Concatenate[T_algo, LLMProxy, P], CoroutineType[Any, Any, R]]],
    Callable[Concatenate[T_algo, P], CoroutineType[Any, Any, R]],
]: ...


def with_llm_proxy(
    required: bool = False,
    auto_start: bool = True,
) -> Callable[
    [Callable[..., CoroutineType[Any, Any, Any]]],
    Callable[..., CoroutineType[Any, Any, Any]],
]:
    """Resolve and optionally lifecycle-manage the configured LLM proxy.

    Args:
        required: When True, raises `ValueError` if the algorithm does not have an
            [`LLMProxy`][agentlightning.LLMProxy] set. When False, the wrapped coroutine receives
            `None` if no proxy is available.
        auto_start: When True, [`LLMProxy.start()`][agentlightning.LLMProxy.start] is invoked if the proxy is not
            already running before calling `func` and [`LLMProxy.stop()`][agentlightning.LLMProxy.stop] is
            called afterwards.

    Returns:
        A decorator that injects the [`LLMProxy`][agentlightning.LLMProxy] (or `None`) as the first
        argument after `self` and manages automatic startup/shutdown when requested.
    """

    def decorator(
        func: Callable[..., CoroutineType[Any, Any, Any]],
    ) -> Callable[..., CoroutineType[Any, Any, Any]]:
        @functools.wraps(func)
        async def wrapper(self: Algorithm, *args: Any, **kwargs: Any) -> Any:
            llm_proxy = self.get_llm_proxy()

            if required and llm_proxy is None:
                raise ValueError("LLM proxy not found")

            auto_started = False
            if auto_start and llm_proxy is not None:
                if llm_proxy.is_running():
                    logger.info("Proxy is already running, skipping start")
                else:
                    logger.info("Starting proxy, managed by the algorithm")
                    await llm_proxy.start()
                    auto_started = True

            try:
                # At type level, overloads guarantee that if `required=True`
                # then `func` expects a non-optional LLMProxy.
                return await func(self, llm_proxy, *args, **kwargs)
            finally:
                if auto_started and llm_proxy is not None:
                    logger.info("Stopping proxy, managed by the algorithm")
                    await llm_proxy.stop()

        return wrapper

    return decorator
