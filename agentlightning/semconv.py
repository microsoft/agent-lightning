# Copyright (c) Microsoft. All rights reserved.

"""Semantic conventions for Agent-lightning spans.

Conventions in this file are added on demand. We generally DO NOT add
new semantic conventions unless it's absolutely needed for certain algorithms or scenarios.
"""

from enum import Enum

AGL_ANNOTATION = "agentlightning.reward"
"""Agent-lightning's standard span name for annotations.

Annotations are minimal span units for rewards, tags, and metadatas.
They are used to "annotate" a specific event or a part of rollout.
"""

AGL_MESSAGE = "agentlightning.message"
"""Agent-lightning's standard span name for messages and logs."""

AGL_OBJECT = "agentlightning.object"
"""Agent-lightning's standard span name for customized objects."""

AGL_EXCEPTION = "agentlightning.exception"
"""Agent-lightning's standard span name for exceptions.

Used by the exception emitter to record exception details.
"""

AGL_VIRTUAL = "agentlightning.virtual"
"""Agent-lightning's standard span name for virtual operations.

Mostly used in adapter when needing to represent the root or intermediate operations.
"""


class LightningResourceAttributes(Enum):
    """Resource attribute names used in Agent-lightning spans."""

    ROLLOUT_ID = "agentlightning.rollout_id"
    """Resource name for rollout ID in Agent-lightning spans."""

    ATTEMPT_ID = "agentlightning.attempt_id"
    """Resource name for attempt ID in Agent-lightning spans."""

    SPAN_SEQUENCE_ID = "agentlightning.span_sequence_id"
    """Resource name for span sequence ID in Agent-lightning spans."""


class LightningSpanAttributes(Enum):
    """Attribute names that commonly appear in Agent-lightning spans.

    Exception types can't be found here because they are defined in OpenTelemetry's official semantic conventions.
    """

    REWARD = "agentlightning.reward"
    """Attribute prefix for rewards-related data in reward spans.

    It should be used as a prefix. For example, "agentlightning.reward.0.value" can
    be used to track a specific metric. See [agentlightning.semconv.RewardAttributes].
    """

    LINK = "agentlightning.link"
    """Attribute name for linking the current span to another span or other objects like requests/responses."""

    MESSAGE_TEXT = "agentlightning.message"
    """Attribute name for message text in message spans.

    It should be a list, so the real attributes should look like `agentlightning.message.0`, `agentlightning.message.1`, etc.
    """

    OBJECT_TYPE = "agentlightning.object.type"
    """Attribute name for object type (full qualified name) in object spans.

    I think builtin types like str, int, bool, list, dict are self-explanatory and
    should also be qualified to use here.
    """

    OBJECT_LITERAL = "agentlightning.object.literal"
    """Attribute name for object literal value in object spans (for str, int, bool, ...)."""

    OBJECT_JSON = "agentlightning.object.json"
    """Attribute name for object serialized value (JSON) in object spans."""


class RewardAttributes(Enum):
    """Multi-dimensional reward attributes will look like:

    ```json
    {"agentlightning.reward.0.name": "efficiency", "agentlightning.reward.0.value": 0.75}
    ```

    The first reward in the reward list will automatically be the primary reward.
    If the reward list has greater than 1, it shall be a multi-dimensional case.
    """

    REWARD_NAME = "name"
    """Key for each dimension in multi-dimensional reward spans."""

    REWARD_VALUE = "value"
    """Value for each dimension in multi-dimensional reward spans."""


class LinkAttributes(Enum):
    """Standard link types used in Agent-lightning spans.

    The link is more powerful than [OpenTelemetry link](https://opentelemetry.io/docs/specs/otel/trace/api/#link)
    in that it supports linking to a queryset of spans.
    It can even link to span object that hasn't been emitted yet.
    """

    KEY_MATCH = "key_match"
    """Linking to spans with matching attribute keys (or `span_id`).

    For example, it can be `gen_ai.response.id` if intended to be link to a chat completion response span.
    Or it can be `span_id` to link to a specific span by its ID.
    """

    VALUE_MATCH = "value_match"
    """Linking to spans with corresponding attribute values on those keys."""


class ObjectAttributes(Enum):
    """Standard object types used in Agent-lightning object spans."""

    TYPE_STR = "str"
    TYPE_INT = "int"
    TYPE_BOOL = "bool"
    TYPE_FLOAT = "float"
    TYPE_LIST = "list"
    TYPE_DICT = "dict"
    TYPE_SET = "set"
    TYPE_TUPLE = "tuple"
    TYPE_BYTES = "bytes"
