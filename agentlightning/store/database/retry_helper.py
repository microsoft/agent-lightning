# Copyright (c) Microsoft. All rights reserved.
"""This file contains a configurable async retry decorator based on exception type.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import functools
import importlib
from dataclasses import dataclass
from typing import Dict, Type, Any, TypeVar, Callable, Awaitable
from tenacity import AsyncRetrying, retry_if_exception, RetryCallState

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
        max_attempts: Maximum number of attempts before giving up. Default is 1 (no retry).
        wait_seconds: Base wait time in seconds before the first retry. Default is 0.0.
        backoff: Exponential backoff multiplier. Default is 1.0 (no backoff).
        jitter: Fractional (relative) jitter to apply to wait time. Default is 0.0 (no jitter).
        log: Whether to log each retry attempt. Default is False.
    """
    max_attempts: int = 1
    wait_seconds: float = 0.0
    backoff: float = 1.0
    jitter: float = 0.0
    log: bool = False

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.wait_seconds < 0.0:
            raise ValueError("wait_seconds must be non-negative")
        if self.backoff < 1.0:
            raise ValueError("backoff must be at least 1.0")
        if not (0.0 <= self.jitter <= 1.0):
            raise ValueError("jitter must be between 0.0 and 1.0")

    def get_wait_time(self, attempt_number: int) -> float:
        """Calculate the wait time before the given attempt number."""
        base_wait = self.wait_seconds * (self.backoff ** (attempt_number - 1))
        if self.jitter > 0:
            delta = base_wait * self.jitter
            wait_time = random.uniform(base_wait - delta, base_wait + delta)
        else:
            wait_time = base_wait
        return max(wait_time, 0.0)

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
    def register(cls, name: str, exc_type: Type[BaseException]|None = None) -> None:
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
                # Try to dynamically import the exception class
                try:
                    module_name, class_name = name.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    exc_type = getattr(module, class_name)
                    if not issubclass(exc_type, BaseException):
                        raise TypeError(f"{name} is not an Exception type.")
                except (ImportError, AttributeError, ValueError, TypeError) as e:
                    raise ValueError(f"Cannot resolve exception type for name '{name}': {e}")
            mapping[exc_type] = strat
        return mapping

    # ------------------------------------------------------------------
    # Retry core logic
    # ------------------------------------------------------------------
    def get_strategy(self, exc: BaseException) -> RetryStrategy:
        for exc_type, strat in self.exception_map.items():
            if isinstance(exc, exc_type):
                return strat
        return self.default_strategy

    def should_retry(self, exc: BaseException) -> bool:
        return any(isinstance(exc, t) for t in self.exception_map.keys())

    def wait_func(self, retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        if outcome is None or outcome.failed is False:
            return 0.0
        exc = outcome.exception()
        if exc is None:
            return 0.0
        strat = self.get_strategy(exc)
        return strat.get_wait_time(retry_state.attempt_number)

    def stop_func(self, retry_state: RetryCallState) -> bool:
        outcome = retry_state.outcome
        if outcome is None:
            return False
        exc = outcome.exception()
        if exc is None:
            return False
        strat = self.get_strategy(exc)
        return retry_state.attempt_number >= strat.max_attempts

    async def before_sleep(self, retry_state: RetryCallState):
        outcome = retry_state.outcome
        if outcome is None or outcome.failed is False:
            return
        exc = outcome.exception()
        if exc is None:
            return
        strat = self.get_strategy(exc)
        if strat.log:
            next_wait = self.wait_func(retry_state)
            logger.warning(
                f"[Retry] {exc.__class__.__name__}: attempt={retry_state.attempt_number}, "
                f"next_wait={next_wait:.2f}s, message={exc}"
            )

    # ------------------------------------------------------------------
    # Decorator entry point
    # ------------------------------------------------------------------
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs): # type: ignore
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
