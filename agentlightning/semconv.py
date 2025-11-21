from enum import Enum

AGL_REWARD = "agentlightning.reward"
"""Agent-lightning's standard span name for reward emissions.

One reward span could contain multiple reward values in its attributes.
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

    AGL_ROLLOUT_ID = "agentlightning.rollout_id"
    """Resource name for rollout ID in Agent-lightning spans."""

    AGL_ATTEMPT_ID = "agentlightning.attempt_id"
    """Resource name for attempt ID in Agent-lightning spans."""

    AGL_SPAN_SEQUENCE_ID = "agentlightning.span_sequence_id"
    """Resource name for span sequence ID in Agent-lightning spans."""


class LightningSpanAttributes(Enum):
    """Attribute names that commonly appear in Agent-lightning spans.

    Exception types can't be found here because they are defined in OpenTelemetry's official semantic conventions.
    """

    REWARD_VALUE = "agentlightning.reward.primary"
    """Attribute name for primary reward value in reward spans."""

    REWARD_PREFIX = "agentlightning.reward"
    """Attribute prefix for rewards-related data in reward spans.

    It should be used as a prefix. For example, "agentlightning.reward.efficiency" can
    be used to track a cost-sensitive metric.
    """

    REWARD_METADATA_PREFIX = "agentlightning.reward.metadata"
    """Attribute prefix for reward metadata in reward spans."""

    REWARD_ASSOCIATED_GEN_AI_RESPONSE_ID = "agentlightning.reward.metadata.associated.gen_ai_response_id"

    MESSAGE_TEXT = "agentlightning.message.text"
    """Attribute name for message text in message spans."""

    OBJECT_TYPE = "agentlightning.object.type"
    """Attribute name for object type (full qualified name) in object spans.

    I think builtin types like str, int, bool, list, dict are self-explanatory and
    should also be qualified to use here.
    """

    OBJECT_VALUE = "agentlightning.object.value"
    """Attribute name for object serialized value (JSON) in object spans."""


class LightningRewardSourceValues(Enum):
    """Enumerated values for where the reward comes from."""

    HUMAN = "human"
    """Reward provided by a human input (feedback)."""

    ENVIRONMENT = "environment"
    """Reward provided by the environment (simulator or real world)."""

    MODEL = "model"
    """Reward provided by another model (e.g., reward model)."""
