# Copyright (c) Microsoft. All rights reserved.

"""Semantic conventions for open emergence span attributes."""

from enum import Enum


class EmergenceSpanAttributes(Enum):
    """Attributes for open emergence monitoring.

    These extend LightningSpanAttributes without modifying it.
    """

    ENTROPY_SHAPE = "agentlightning.emergence.entropy.shape"
    ENTROPY_TOOL = "agentlightning.emergence.entropy.tool"
    ENTROPY_REWARD = "agentlightning.emergence.entropy.reward"
    COLLAPSE_SEVERITY = "agentlightning.emergence.collapse.severity"
    COLLAPSE_DESCRIPTION = "agentlightning.emergence.collapse.description"
    NOVELTY_SCORE = "agentlightning.emergence.novelty.score"
    NOVELTY_CLASSIFICATION = "agentlightning.emergence.novelty.classification"
    NOVELTY_NEAREST_SHAPE = "agentlightning.emergence.novelty.nearest_shape"
