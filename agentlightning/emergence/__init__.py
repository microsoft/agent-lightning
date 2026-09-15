# Copyright (c) Microsoft. All rights reserved.

"""Open Emergence: an opposing worldview for agent training.

This module sits alongside agent-lightning's optimization loop, holding
optimization in tension with exploration rather than replacing it.  Every
component is additive and best-effort -- removing the entire module leaves
the core training loop unchanged.

Reference: docs/open-emergence-design.md
"""

from agentlightning.emergence.novelty import NoveltyAwareAdapter, NoveltyDetector, NoveltyScore, TrajectoryShape
from agentlightning.emergence.entropy import TrajectoryEntropy
from agentlightning.emergence.monitoring import CollapseSignal, EntropySnapshot, ExplorationDecayMonitor
from agentlightning.emergence.pareto import ParetoClassification, ParetoPoint, ParetoTracker
from agentlightning.emergence.reward_audit import RewardAuditAdapter, RewardStalenessAuditor
from agentlightning.emergence.dissolution import DissolutionEngine, DissolutionMetadata, DissolutionPolicy, ValidityCondition
from agentlightning.emergence.semconv import EmergenceSpanAttributes

__all__ = [
    # Gap 5: Novelty Detection
    "NoveltyDetector",
    "NoveltyAwareAdapter",
    "NoveltyScore",
    "TrajectoryShape",
    # Gap 1: Entropy Monitoring
    "TrajectoryEntropy",
    "ExplorationDecayMonitor",
    "EntropySnapshot",
    "CollapseSignal",
    # Gap 3: Pareto Tension
    "ParetoTracker",
    "ParetoPoint",
    "ParetoClassification",
    # Gap 2: Reward Staleness
    "RewardStalenessAuditor",
    "RewardAuditAdapter",
    # Gap 4: Dissolution
    "DissolutionEngine",
    "DissolutionMetadata",
    "DissolutionPolicy",
    "ValidityCondition",
    # Semantic Conventions
    "EmergenceSpanAttributes",
]
