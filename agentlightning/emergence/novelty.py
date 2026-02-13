# Copyright (c) Microsoft. All rights reserved.

"""Gap 5: Novel vs. routine behavior distinction.

Detects whether a trajectory represents novel or routine behavior by
comparing its structural *shape* (tool call sequences, branching patterns)
against a running codebook of known shapes.

Dissolution condition: when agent-lightning's span system includes native
trajectory fingerprinting, making external shape computation redundant.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict

from agentlightning.adapter.base import TraceAdapter
from agentlightning.types import Span, Triplet

if TYPE_CHECKING:
    from agentlightning.adapter.triplet import TracerTraceToTriplet, TraceTree

logger = logging.getLogger(__name__)


class TrajectoryShape(BaseModel):
    """Structural skeleton of a trajectory, ignoring token content."""

    nodes: Tuple[Tuple[str, int, int], ...]
    """Sequence of (span_name, depth, child_count) tuples."""
    fingerprint: str
    """Hex digest uniquely identifying this shape."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_trace_tree(cls, tree: "TraceTree") -> "TrajectoryShape":
        """Extract shape from a trace tree via depth-first traversal."""
        nodes: List[Tuple[str, int, int]] = []

        def _visit(node: "TraceTree", depth: int) -> None:
            nodes.append((node.span.name, depth, len(node.children)))
            for child in node.children:
                _visit(child, depth + 1)

        _visit(tree, 0)
        nodes_tuple = tuple(nodes)
        fp = hashlib.sha256(str(nodes_tuple).encode()).hexdigest()[:16]
        return cls(nodes=nodes_tuple, fingerprint=fp)


class ShapeEntry(BaseModel):
    """Codebook entry tracking a known trajectory shape."""

    shape: TrajectoryShape
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    novelty_score: float = 1.0


class NoveltyScore(BaseModel):
    """Result of scoring a trajectory's novelty."""

    score: float
    """0.0 (completely routine) to 1.0 (never seen)."""
    nearest_shape: Optional[str] = None
    """Fingerprint of the most similar known shape."""
    similarity_to_nearest: float = 0.0
    first_seen: bool = False
    classification: str = "novel"
    """'novel' | 'familiar' | 'routine'"""


def _shape_similarity(a: TrajectoryShape, b: TrajectoryShape) -> float:
    """Compute structural similarity between two shapes.

    Uses set overlap of (name, depth) pairs as a fast proxy for
    structural similarity. Returns 0.0-1.0.
    """
    set_a = set((name, depth) for name, depth, _ in a.nodes)
    set_b = set((name, depth) for name, depth, _ in b.nodes)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class NoveltyDetector:
    """Detect whether a trajectory represents novel or routine behavior.

    Novelty is defined structurally: a trajectory is novel if its shape
    (sequence of tool calls, branching pattern, response structure) has
    not been seen before. Token-level variation within a known shape is
    NOT novelty -- it's exploration noise.

    The detector maintains a running codebook of known trajectory shapes.
    New shapes start with high novelty scores that decay as they're seen
    more frequently.

    Dissolution condition: when neural novelty detection (learned embeddings
    of trajectory space) becomes efficient enough for online use.
    """

    def __init__(
        self,
        shape_similarity_threshold: float = 0.85,
        novelty_decay_rate: float = 0.95,
        max_codebook_size: int = 1000,
    ):
        self._similarity_threshold = shape_similarity_threshold
        self._decay_rate = novelty_decay_rate
        self._max_codebook_size = max_codebook_size
        self._codebook: Dict[str, ShapeEntry] = {}
        self._recent_classifications: deque[str] = deque(maxlen=1000)

    def compute_shape(self, tree: "TraceTree") -> TrajectoryShape:
        """Extract structural shape from a trace tree."""
        return TrajectoryShape.from_trace_tree(tree)

    def _find_nearest(self, shape: TrajectoryShape) -> Tuple[Optional[str], float]:
        """Find the most similar shape in the codebook."""
        best_fp: Optional[str] = None
        best_sim = 0.0
        for fp, entry in self._codebook.items():
            sim = _shape_similarity(shape, entry.shape)
            if sim > best_sim:
                best_sim = sim
                best_fp = fp
        return best_fp, best_sim

    def score_novelty(self, tree: "TraceTree") -> NoveltyScore:
        """Score a trajectory's novelty against the codebook.

        Updates the codebook as a side effect.
        """
        shape = self.compute_shape(tree)
        now = time.time()

        # Exact match
        if shape.fingerprint in self._codebook:
            entry = self._codebook[shape.fingerprint]
            entry.count += 1
            entry.last_seen = now
            entry.novelty_score *= self._decay_rate
            score = entry.novelty_score
            classification = "routine" if score < 0.3 else "familiar"
            self._recent_classifications.append(classification)
            return NoveltyScore(
                score=score,
                nearest_shape=shape.fingerprint,
                similarity_to_nearest=1.0,
                first_seen=False,
                classification=classification,
            )

        # Check for similar shapes
        nearest_fp, nearest_sim = self._find_nearest(shape)

        if nearest_sim >= self._similarity_threshold and nearest_fp is not None:
            # Similar enough to count as a variant of a known shape
            entry = self._codebook[nearest_fp]
            entry.count += 1
            entry.last_seen = now
            entry.novelty_score *= self._decay_rate
            score = max(entry.novelty_score, 1.0 - nearest_sim)
            classification = "familiar"
            self._recent_classifications.append(classification)
            return NoveltyScore(
                score=score,
                nearest_shape=nearest_fp,
                similarity_to_nearest=nearest_sim,
                first_seen=False,
                classification=classification,
            )

        # Genuinely novel shape
        if len(self._codebook) >= self._max_codebook_size:
            # Evict least recently seen entry
            oldest_fp = min(self._codebook, key=lambda fp: self._codebook[fp].last_seen)
            del self._codebook[oldest_fp]

        self._codebook[shape.fingerprint] = ShapeEntry(
            shape=shape,
            count=1,
            first_seen=now,
            last_seen=now,
            novelty_score=1.0,
        )
        self._recent_classifications.append("novel")
        return NoveltyScore(
            score=1.0,
            nearest_shape=nearest_fp,
            similarity_to_nearest=nearest_sim,
            first_seen=True,
            classification="novel",
        )

    def get_discovery_rate(self, window_size: int = 100) -> float:
        """Fraction of recent trajectories classified as 'novel'.

        Declining discovery rate is a leading indicator of exploration
        collapse -- the system is no longer finding new behavioral patterns.
        """
        if not self._recent_classifications:
            return 0.0
        recent = list(self._recent_classifications)[-window_size:]
        return sum(1 for c in recent if c == "novel") / len(recent)

    def get_codebook_summary(self) -> str:
        """Summarize known trajectory shapes with 'could be' language."""
        if not self._codebook:
            return "Codebook: empty (no trajectories observed yet)."

        total_observations = sum(e.count for e in self._codebook.values())
        sorted_entries = sorted(self._codebook.values(), key=lambda e: e.count, reverse=True)

        lines = [f"Codebook: {len(self._codebook)} known shapes."]
        top_n = min(5, len(sorted_entries))
        lines.append(f"Top {top_n} by frequency:")
        cumulative_pct = 0.0
        for i, entry in enumerate(sorted_entries[:top_n]):
            pct = (entry.count / total_observations * 100) if total_observations > 0 else 0
            cumulative_pct += pct
            # Represent shape as its first few span names
            shape_repr = " -> ".join(n for n, _, _ in entry.shape.nodes[:4])
            if len(entry.shape.nodes) > 4:
                shape_repr += " -> ..."
            lines.append(f"  {i + 1}. [{shape_repr}] (seen {entry.count}x, ~{pct:.0f}%)")

        discovery_rate = self.get_discovery_rate()
        lines.append(f"Discovery rate (last 100): {discovery_rate:.2f}.")
        if discovery_rate < 0.05 and len(self._codebook) > 10:
            lines.append(
                f"Could indicate behavioral convergence -- "
                f"{top_n} shapes account for ~{cumulative_pct:.0f}% of all trajectories."
            )

        return "\n".join(lines)


class NoveltyAwareAdapter(TraceAdapter[List[Triplet]]):
    """Wraps TracerTraceToTriplet to weight novel trajectories.

    Does NOT replace the base adapter. Produces the same Triplet format
    with additional metadata and optional sampling weights.

    Novel high-reward trajectories get higher sampling weight.
    Routine high-reward trajectories get standard weight.
    Novel low-reward trajectories get standard weight (exploration is
    not unconditionally good -- it needs reward context).

    Dissolution condition: when the base TracerTraceToTriplet adapter supports
    configurable weighting functions, making this wrapper unnecessary.
    """

    def __init__(
        self,
        base_adapter: "TracerTraceToTriplet",
        novelty_detector: NoveltyDetector,
        novelty_weight_multiplier: float = 2.0,
    ):
        self._base = base_adapter
        self._detector = novelty_detector
        self._weight_multiplier = novelty_weight_multiplier

    def adapt(self, source: Sequence[Span], /) -> List[Triplet]:
        """Adapt with novelty annotation.

        Each Triplet.metadata gets:
        - 'novelty_score': float
        - 'novelty_classification': str
        - 'sampling_weight': float (1.0 for routine, multiplier for novel+rewarded)
        """
        from agentlightning.adapter.triplet import TraceTree

        triplets = self._base.adapt(source)
        if not triplets:
            return triplets

        # Build the trace tree to score novelty
        try:
            source_normalized = list(source)
            tree = TraceTree.from_spans(source_normalized)
            if self._base.repair_hierarchy:
                tree.repair_hierarchy()
            novelty = self._detector.score_novelty(tree)
        except Exception:
            logger.debug("Novelty scoring failed; proceeding without enrichment.", exc_info=True)
            return triplets

        enriched: List[Triplet] = []
        for triplet in triplets:
            weight = 1.0
            if novelty.classification == "novel" and triplet.reward is not None and triplet.reward > 0:
                weight = self._weight_multiplier

            updated_metadata = {
                **triplet.metadata,
                "novelty_score": novelty.score,
                "novelty_classification": novelty.classification,
                "sampling_weight": weight,
            }
            enriched.append(triplet.model_copy(update={"metadata": updated_metadata}))

        return enriched
