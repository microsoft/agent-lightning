# Copyright (c) Microsoft. All rights reserved.

"""Gap 1: Trajectory entropy computation.

Computes behavioral diversity metrics from trace trees. Entropy is computed
over the distribution of trajectory *shapes* (tool call sequences, branching
patterns) rather than trajectory *content* (specific tokens).

Dissolution condition: when agent-lightning's native metrics include behavioral
diversity metrics, making external entropy computation redundant.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from agentlightning.types import Triplet

from .novelty import TrajectoryShape

if TYPE_CHECKING:
    from agentlightning.adapter.triplet import TraceTree

logger = logging.getLogger(__name__)


class TrajectoryEntropy:
    """Compute behavioral diversity metrics from trace trees.

    Entropy is computed over the distribution of trajectory *shapes*
    (tool call sequences, branching patterns) rather than trajectory
    *content* (specific tokens). This distinguishes structural diversity
    from surface-level variation.
    """

    def compute_shape_entropy(
        self,
        trees: Sequence["TraceTree"],
        window_size: int = 50,
    ) -> float:
        """H(shape distribution) over recent trajectories.

        Shape = tuple of (span.name, depth) for each node in traversal order.
        High entropy: agents take structurally different paths.
        Low entropy: agents follow the same structural pattern.
        """
        recent = list(trees)[-window_size:]
        if not recent:
            return 0.0

        shapes: List[str] = []
        for tree in recent:
            try:
                shape = TrajectoryShape.from_trace_tree(tree)
                shapes.append(shape.fingerprint)
            except Exception:
                logger.debug("Shape extraction failed for tree; skipping.", exc_info=True)
                continue

        return _shannon_entropy(shapes)

    def compute_tool_entropy(
        self,
        trees: Sequence["TraceTree"],
        window_size: int = 50,
        llm_call_match: str = r"openai\.chat\.completion",
    ) -> float:
        """H(tool call distribution) over recent trajectories.

        Measures whether agents use diverse tools or converge on
        a narrow subset. Computed from tool call spans matched via
        the same llm_call_match regex used by TracerTraceToTriplet.
        """
        recent = list(trees)[-window_size:]
        if not recent:
            return 0.0

        tool_names: List[str] = []
        for tree in recent:
            try:
                for node in tree.traverse():
                    if re.search(llm_call_match, node.span.name):
                        tool_names.append(node.span.name)
            except Exception:
                logger.debug("Tool extraction failed; skipping.", exc_info=True)
                continue

        return _shannon_entropy(tool_names)

    def compute_reward_entropy(
        self,
        triplets: Sequence[Triplet],
        n_bins: int = 20,
        window_size: int = 50,
    ) -> float:
        """H(reward distribution) over recent triplets.

        High reward entropy: outcomes are spread across the reward range.
        Low reward entropy: outcomes cluster at one value (convergence).
        Note: low entropy + high mean reward could be genuine mastery
        OR premature convergence. This metric alone cannot distinguish.
        """
        recent = [t for t in list(triplets)[-window_size:] if t.reward is not None]
        if not recent:
            return 0.0

        rewards = [t.reward for t in recent]
        min_r = min(rewards)  # type: ignore[type-var]
        max_r = max(rewards)  # type: ignore[type-var]

        if max_r == min_r:
            return 0.0

        # Bin rewards into n_bins buckets
        bin_width = (max_r - min_r) / n_bins  # type: ignore[operator]
        bins: List[str] = []
        for r in rewards:
            bin_idx = min(int((r - min_r) / bin_width), n_bins - 1)  # type: ignore[operator]
            bins.append(str(bin_idx))

        return _shannon_entropy(bins)


def _shannon_entropy(items: List[str]) -> float:
    """Compute Shannon entropy of a list of categorical items.

    Returns entropy in nats (natural log), normalized to [0, 1] range
    where 1 means maximum diversity.
    """
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    n_categories = len(counts)
    if n_categories <= 1:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)

    # Normalize by max possible entropy
    max_entropy = math.log(n_categories)
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy
