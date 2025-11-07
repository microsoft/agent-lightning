# Copyright (c) Microsoft. All rights reserved.

"""This file contains a configurable async retry decorator based on exception type."""

from __future__ import annotations

import functools
import importlib
import logging
import random
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, Type, TypeVar

from tenacity import AsyncRetrying, RetryCallState, retry_if_exception

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logger = logging.getLogger("async_retry")
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------
# Type alias for async callable
# ----------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


# ----------------------------------------------------------------------
# Dataclass definition for retry configuration
# ----------------------------------------------------------------------
@dataclass
class RetryStrategy:
    """Configuration schema for retry behavior of a specific exception type.
    The wait time before $n$-th retry is calculated as ($n$ starts from 1):
        wait_time = wait_seconds * (backoff ** (n - 1)) * (1 + jitter * U(-1, 1))
    where U(-1, 1) is a uniform random variable between -1 and 1.
    Attributes:
        max_attempts: Maximum number of attempts before giving up. Default is 1 (no retry). None means infinite retries.
        max_retry_delay: Optional maximum delay between retries in seconds. Default is None (no limit).
        wait_seconds: Base wait time in seconds before the first retry. Default is 0.0.
        max_wait_seconds: Maximum wait time in seconds between retries. Default is None (no limit).
        backoff: Exponential backoff multiplier. Default is 1.0 (no backoff).
        jitter: Fractional (relative) jitter to apply to wait time. Default is 0.0 (no jitter).
        log: Whether to log each retry attempt. Default is False.
    """

    max_attempts: Optional[int] = 1
    max_retry_delay: Optional[float] = None
    wait_seconds: float = 0.0
    max_wait_seconds: Optional[float] = None
    backoff: float = 1.0
    jitter: float = 0.0
    log: bool = False

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1 or None for infinite retries")
        if self.wait_seconds < 0.0:
            raise ValueError("wait_seconds must be non-negative")
        if self.backoff < 1.0:
            raise ValueError("backoff must be at least 1.0")
        if not (0.0 <= self.jitter <= 1.0):
            raise ValueError("jitter must be between 0.0 and 1.0")

    def _get_wait_time(self, attempt_number: int) -> float:
        """Calculate the wait time before the given attempt number."""
        base_wait = self.wait_seconds * (self.backoff ** (attempt_number - 1))
        if self.jitter > 0:
            delta = base_wait * self.jitter
            wait_time = random.uniform(base_wait - delta, base_wait + delta)
        else:
            wait_time = base_wait
        wait_time = max(wait_time, 0.0)
        if self.max_wait_seconds is not None:
            wait_time = min(wait_time, self.max_wait_seconds)
        return wait_time

    def wait_func(self, retry_state: RetryCallState) -> float:
        """Tenacity wait function based on the given strategy."""
        return self._get_wait_time(retry_state.attempt_number)

    def stop_func(self, retry_state: RetryCallState) -> bool:
        """Tenacity stop function based on the given strategy."""
        if self.max_attempts is not None:
            if retry_state.attempt_number >= self.max_attempts:
                return True
        if self.max_retry_delay is not None:
            time_since_start = retry_state.seconds_since_start
            if time_since_start is None:
                logger.warning("Cannot determine time since start for retry stop condition.")
                return False
            if time_since_start >= self.max_retry_delay:
                return True
        return False

    async def before_sleep(self, retry_state: RetryCallState):
        """Tenacity before_sleep callback to log retry attempts."""
        if self.log:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            next_wait = self.wait_func(retry_state)
            logger.warning(
                f"[Retry] {exc.__class__.__name__}: attempt={retry_state.attempt_number}, "
                f"next_wait={next_wait:.2f}s, message={exc}"
            )


# ----------------------------------------------------------------------
# Exception Registry — shared, reusable, and extensible
# ----------------------------------------------------------------------
class ExceptionRegistry:
    """
    Global registry for mapping string keys to Exception classes.
    Supports dynamic registration and fallback to importlib.
    """

    _registry: Dict[str, Type[BaseException]] = {}

    @classmethod
    def register(cls, name: str, exc_type: Type[BaseException] | None = None) -> None:
        """Register an exception type under a given name."""
        if name in cls._registry:
            logger.warning(f"Overwriting existing exception registration for name '{name}'.")
        if exc_type is None:
            # Try to dynamically import the exception class
            try:
                module_name, class_name = name.rsplit(".", 1)
                module = importlib.import_module(module_name)
                exc_type = getattr(module, class_name)
                if exc_type is None:
                    raise TypeError(f"{name} is not an Exception type.")
            except (ImportError, AttributeError, ValueError, TypeError) as e:
                raise ValueError(f"Cannot resolve exception type for name '{name}': {e}")
        cls._registry[name] = exc_type

    @classmethod
    def all_registered(cls) -> Dict[str, Type[BaseException]]:
        """Return the current registry mapping."""
        return dict(cls._registry)

    @classmethod
    def clear(cls):
        """Clear all registered exception mappings."""
        cls._registry.clear()


# ----------------------------------------------------------------------
# Async Retry Decorator
# ----------------------------------------------------------------------
class AsyncTypeBasedRetry:
    """
    A configurable async retry decorator based on exception type.

    - Takes configuration as a Dict[str, RetryStrategy].
    - Provides `from_json()` for quick loading.
    - Uses a global ExceptionRegistry to resolve exception names.
    """

    def __init__(self, strategies: Dict[str, RetryStrategy], default_strategy: RetryStrategy | None = None):
        self.exception_map = self._build_exception_map(strategies)
        self.default_strategy = default_strategy or RetryStrategy()

    # ------------------------------------------------------------------
    # Build exception map
    # ------------------------------------------------------------------
    def _build_exception_map(self, strategies: Dict[str, RetryStrategy]) -> Dict[Type[BaseException], RetryStrategy]:
        mapping: Dict[Type[BaseException], RetryStrategy] = {}
        all_registered = ExceptionRegistry.all_registered()
        for name, strat in strategies.items():
            if name in all_registered:
                exc_type = all_registered[name]
            else:
                raise ValueError(f"Exception type '{name}' is not registered in ExceptionRegistry.")
            mapping[exc_type] = strat
        return mapping

    # ------------------------------------------------------------------
    # Retry core logic
    # ------------------------------------------------------------------
    def get_exception(self, retry_state: RetryCallState) -> Optional[BaseException]:
        """Get the exception from the given retry state, if any."""
        return retry_state.outcome.exception() if retry_state.outcome else None

    def get_strategy(self, retry_state: RetryCallState) -> Optional[RetryStrategy]:
        """Get the RetryStrategy for the exception in the given retry state.
        IF no matching exception type is found, return the default strategy.
        IF no exception is found, return None.
        """
        exc = self.get_exception(retry_state)
        if exc is None:
            return None
        for exc_type, strat in self.exception_map.items():
            if isinstance(exc, exc_type):
                return strat
        return self.default_strategy

    def should_retry(self, exc: BaseException) -> bool:
        return any(isinstance(exc, t) for t in self.exception_map.keys())

    def wait_func(self, retry_state: RetryCallState) -> float:
        strat = self.get_strategy(retry_state)
        if strat is None:
            return 0.0
        return strat.wait_func(retry_state)

    def stop_func(self, retry_state: RetryCallState) -> bool:
        strat = self.get_strategy(retry_state)
        if strat is None:
            return False
        return strat.stop_func(retry_state)

    async def before_sleep(self, retry_state: RetryCallState):
        strat = self.get_strategy(retry_state)
        if strat is None:
            return
        await strat.before_sleep(retry_state)

    # ------------------------------------------------------------------
    # Decorator entry point
    # ------------------------------------------------------------------
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(lambda e: self.should_retry(e)),
                wait=self.wait_func,
                stop=self.stop_func,
                before_sleep=self.before_sleep,
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)

        return wrapper  # type: ignore


# ----------------------------------------------------------------------
# A configurable async retrier for any code block
# ----------------------------------------------------------------------


class AsyncRetryBlock:
    """
    Async retry helper for a single exception type and strategy.

    Usage:
        async with AsyncRetryBlock(strategy):
            await some_async_function()
    """

    def __init__(self, strategy: RetryStrategy, **retry_kwargs):  # type: ignore
        self.strategy = strategy
        self._retryer = AsyncRetrying(
            wait=self._wait_func,
            stop=self._stop_func,
            before_sleep=self._before_sleep,
            **retry_kwargs,  # type: ignore
        )

    async def run(self, coro: Callable[..., Awaitable[Any]]) -> Any:
        """Run the given coroutine with retries according to the strategy.
        For example:
            async def my_coro():
                ...
            retry_block = AsyncRetryBlock(strategy)
            result = await retry_block.run(my_coro)
        """
        async for attempt in self._retryer:
            with attempt:
                return await coro()

    # ------------------------------------------------------------------
    # Core: async iterator interface
    # ------------------------------------------------------------------
    def __aiter__(self) -> AsyncIterator[Any]:
        """Return an async iterator that yields retry attempts.
        Usage:
            async for attempt in retry_block:
                with attempt:
                    await some_async_function()
        """
        return self._retryer.__aiter__()

    # ------------------------------------------------------------------
    # Context manager entry
    # ------------------------------------------------------------------
    async def __aenter__(self):
        self._aiter = self._retryer.__aiter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        # Consume the retry iterator
        try:
            # If exception occurred, let the retryer handle it
            async for attempt in self._aiter:
                with attempt:
                    if exc_val:
                        raise exc_val
        except Exception:
            # Allow exception to propagate if retries exhausted
            pass
        return False

    # ------------------------------------------------------------------
    # Strategy function
    # ------------------------------------------------------------------
    def _wait_func(self, retry_state: RetryCallState) -> float:
        return self.strategy.wait_func(retry_state)

    def _stop_func(self, retry_state: RetryCallState) -> bool:
        return self.strategy.stop_func(retry_state)

    async def _before_sleep(self, retry_state: RetryCallState):
        await self.strategy.before_sleep(retry_state)
