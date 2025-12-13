# Copyright (c) Microsoft. All rights reserved.

"""Convenient helpers for creating spans / traces.

All emitters operate in two modes, switchable via the `propagate` parameter.
The emitters first [`SpanCreationRequest`][agentlightning.SpanCreationRequest] object, then:

1. When `propagate` is True, this creation request will be propagated to the active tracer
   and a [`Span`][agentlightning.Span] instance will be created (possibly deferred).
   Nothing will be returned.
2. When `propagate` is False, the creation request will be returned directly. Useful for cases
   when you don't have a tracer but you want to create a creation request for later use.
"""

from .annotation import emit_annotation, operation
from .exception import emit_exception
from .message import emit_message, get_message_value
from .object import emit_object, get_object_value
from .reward import (
    emit_reward,
    find_final_reward,
    find_reward_spans,
    get_reward_value,
    get_rewards_from_span,
    is_reward_span,
    reward,
)

__all__ = [
    "reward",
    "operation",
    "emit_reward",
    "get_reward_value",
    "get_rewards_from_span",
    "is_reward_span",
    "find_reward_spans",
    "find_final_reward",
    "emit_message",
    "emit_object",
    "emit_exception",
    "emit_annotation",
    "get_message_value",
    "get_object_value",
]
