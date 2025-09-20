# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import warnings
from typing import Any, Callable, Literal, Optional, TypedDict, TypeVar

from agentops.sdk.decorators import operation


class RewardSpanData(TypedDict):
    type: Literal["reward"]
    value: Optional[float]


FnType = TypeVar("FnType", bound=Callable[..., Any])


def reward(fn: FnType) -> FnType:
    """
    A decorator to wrap a function that computes rewards.
    It will automatically handle the input and output of the function.
    """

    def wrap_result(result: Optional[float]) -> RewardSpanData:
        """
        Wrap the result of the function in a dict.
        """
        if result is None:
            return {"type": "reward", "value": None}
        if not isinstance(result, (float, int)):  # type: ignore
            warnings.warn(f"Reward is ignored because it is not a number: {result}")
            return {"type": "reward", "value": None}
        return {"type": "reward", "value": float(result)}

    # Check if the function is async
    is_async = asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn)

    if is_async:

        async def wrapper_async(*args: Any, **kwargs: Any) -> Any:
            result: Optional[float] = None

            @operation
            async def agentops_reward_operation() -> RewardSpanData:
                # The reward function we are interested in tracing
                # It takes zero inputs and return a formatted dict
                nonlocal result
                result = await fn(*args, **kwargs)
                return wrap_result(result)

            await agentops_reward_operation()
            return result

        return wrapper_async  # type: ignore

    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result: Optional[float] = None

            @operation
            def agentops_reward_operation() -> RewardSpanData:
                nonlocal result
                result = fn(*args, **kwargs)
                return wrap_result(result)

            agentops_reward_operation()
            return result

        return wrapper  # type: ignore
