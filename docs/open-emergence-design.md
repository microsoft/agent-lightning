# Open Emergence: An Opposing Worldview for Agent Training

> Design document for [Issue #490: Exploration Collapse](https://github.com/microsoft/agent-lightning/issues/490)
>
> Cross-reference: Dissolution conditions for every structural decision in this document
> are collected in the final section.

---

## The Problem This Document Addresses

Agent-lightning implements a **closed optimization loop**: runners emit spans, adapters
compress them into triplets, algorithms consume triplets to produce better resources,
and those resources flow back to runners. Every component exists to tighten this cycle.

This is the system's greatest strength. It is also the architectural root of exploration
collapse.

The tighter the loop, the faster policies converge. The faster policies converge, the
narrower the behavioral distribution. The narrower the distribution, the less capacity
the system retains for discovering behaviors outside the reward function's scope. Agents
that are better by any metric are also more predictable.

This document proposes five engineering interventions — not to replace optimization, but
to hold it in tension with open-ended exploration. The goal is not a system that explores
*instead of* optimizing, but one that can do both simultaneously without collapsing either.

---

## The Core Tension (Hold It, Don't Resolve It)

Optimization asks: "How do I get more of what I already know is good?"
Exploration asks: "What exists that I don't yet know how to value?"

These are not complementary. They are structurally contradictory. Optimization narrows;
exploration widens. Any system that claims to do both is either oscillating between them
(which works but wastes cycles) or has found a way to hold both simultaneously (which is
what this document attempts).

The architectural pattern: **every intervention below creates a tension that the system
cannot automatically resolve.** Resolution requires judgment — human or agent — applied
in context. The interventions surface the *need* for judgment; they do not substitute
for it.

---

## Design Principles

### 1. Non-Invasive Integration

Every intervention plugs into agent-lightning's existing architecture through documented
interfaces. No core classes are modified. The system should work identically when emergence
modules are absent — they are additive, not replacement.

**Concretely:** New adapters implement `TraceAdapter[T_to]`. New span attributes extend
`LightningSpanAttributes`. New store behaviors wrap `LightningStore` via delegation. New
reward policies extend `RewardMatchPolicy`. No monkey-patching, no subclass overrides of
internal methods.

### 2. "Could Be" Language

All interpretive output from emergence modules uses hedged language: "could indicate,"
"~X%," "at current rates." This prevents the emergence layer from becoming another
optimization signal that the system collapses into certainty.

### 3. Best-Effort Enrichment

All emergence computations are best-effort. If entropy calculation fails, the system
proceeds without it. If novelty detection times out, the span ships without a novelty
score. No emergence feature blocks the core training loop.

**Pattern:** `Promise.allSettled()` (or Python equivalent: `asyncio.gather(*tasks, return_exceptions=True)`) wrapping all enrichment calls.

### 4. Dissolution Conditions

Every structural decision in this document carries an explicit condition under which it
should be removed. See the final section.

---

## Gap 1: Exploration Decay Monitoring

### What's Missing

Agent-lightning has no metrics to distinguish genuine behavioral improvement from policy
narrowing. A reward curve that goes up and to the right looks identical whether the agent
is discovering diverse high-quality strategies or converging to a single high-reward
behavior repeated with minor variations.

### Where It Lands

**Primary integration point:** The `TraceTree` class in `agentlightning/adapter/triplet.py`.

The trace tree already builds a hierarchical representation of agent behavior via
`TraceTree.from_spans()`. It traverses depth-first via `traverse()`, detects LLM calls
via `find_llm_calls()`, and converts to trajectories via `to_trajectory()`. This is the
natural place to compute behavioral entropy — the tree structure is already there.

**Secondary integration point:** New span attributes in `agentlightning/semconv.py`.

### Proposed Architecture

```
agentlightning/
  emergence/
    entropy.py          # Trajectory entropy computation
    monitoring.py       # Sliding window decay detection
```

#### `entropy.py`: Trajectory Entropy

```python
class TrajectoryEntropy:
    """Compute behavioral diversity metrics from trace trees.

    Entropy is computed over the distribution of trajectory *shapes*
    (tool call sequences, branching patterns) rather than trajectory
    *content* (specific tokens). This distinguishes structural diversity
    from surface-level variation.
    """

    def compute_shape_entropy(
        self,
        trees: Sequence[TraceTree],
        window_size: int = 50,
    ) -> float:
        """H(shape distribution) over recent trajectories.

        Shape = tuple of (span.name, depth) for each node in traversal order.
        High entropy: agents take structurally different paths.
        Low entropy: agents follow the same structural pattern.
        """

    def compute_tool_entropy(
        self,
        trees: Sequence[TraceTree],
        window_size: int = 50,
    ) -> float:
        """H(tool call distribution) over recent trajectories.

        Measures whether agents use diverse tools or converge on
        a narrow subset. Computed from tool call spans matched via
        the same llm_call_match regex used by TracerTraceToTriplet.
        """

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
```

#### `monitoring.py`: Decay Detection

```python
class ExplorationDecayMonitor:
    """Track entropy over sliding windows and detect collapse.

    Collapse detection: when entropy drops below threshold AND reward
    is still improving, the system may be narrowing rather than improving.

    This is a signal, not a conclusion. The monitor surfaces the tension;
    it does not resolve it.
    """

    def __init__(
        self,
        window_size: int = 50,
        alert_threshold: float = 0.3,  # Entropy below this triggers alert
        trend_window: int = 5,          # Number of windows to compute trend
    ):
        self._history: deque[EntropySnapshot] = deque(maxlen=1000)

    def record(self, trees: Sequence[TraceTree], triplets: Sequence[Triplet]) -> EntropySnapshot:
        """Compute and store entropy snapshot for current window."""

    def detect_collapse(self) -> Optional[CollapseSignal]:
        """Check if entropy is declining while reward is stable/improving.

        Returns CollapseSignal with:
        - entropy_trend: slope of entropy over trend_window
        - reward_trend: slope of reward over trend_window
        - severity: "low" | "medium" | "high"
        - description: hedged interpretation ("could indicate...")

        Returns None if no collapse pattern detected.
        """

    def summary(self) -> str:
        """Human-readable summary with 'could be' language.

        Example:
        'Trajectory entropy: 0.42 (↓ from 0.71 over 5 windows).
         Reward: 0.85 (↑ from 0.72). Could indicate policy narrowing —
         high reward may reflect convergence to a single strategy rather
         than genuine improvement across diverse approaches.'
        """
```

#### Span Attribute Extensions

New semantic conventions in `semconv.py`:

```python
class EmergenceSpanAttributes(Enum):
    """Attributes for open emergence monitoring."""

    ENTROPY_SHAPE = "agentlightning.emergence.entropy.shape"
    ENTROPY_TOOL = "agentlightning.emergence.entropy.tool"
    ENTROPY_REWARD = "agentlightning.emergence.entropy.reward"
    COLLAPSE_SEVERITY = "agentlightning.emergence.collapse.severity"
    COLLAPSE_DESCRIPTION = "agentlightning.emergence.collapse.description"
```

### Integration with Existing Components

The `ExplorationDecayMonitor` integrates at two points:

1. **Algorithm.run()**: After each evaluation round, the algorithm can optionally
   query the monitor. This is advisory — the algorithm decides whether to act on
   collapse signals. For APO, this could trigger beam width expansion. For VERL,
   this could increase the KL divergence penalty weight.

2. **Dashboard**: Entropy metrics surface as time-series alongside reward curves,
   giving operators visual detection of narrowing.

The monitor does **not** automatically modify training behavior. It surfaces information.
The decision to act is human (or algorithm-specific logic that the developer writes).

---

## Gap 2: Reward Function Staleness Detection

### What's Missing

Agent-lightning's reward system (`emit_reward()` in `emitter/reward.py`) assumes that
once a reward function is defined, it remains valid. There is no mechanism to detect when
the relationship between reward signal and actual task success has drifted — due to
environment changes, API updates, distribution shifts, or reward hacking.

### Where It Lands

**Primary integration point:** Between `emit_reward()` and the store. The staleness
detector wraps reward emission to accumulate comparison data.

**Secondary integration point:** A new `RewardAuditAdapter` that sits alongside
`TracerTraceToTriplet` and `TraceToMessages`, implementing the same `TraceAdapter`
interface but producing audit reports instead of training data.

### Proposed Architecture

```
agentlightning/
  emergence/
    reward_audit.py     # Staleness detection and audit reporting
```

#### `reward_audit.py`: Staleness Detection

```python
class RewardStalenessAuditor:
    """Detect drift between reward signals and independent success metrics.

    The auditor maintains two parallel streams:
    1. Reward values from emit_reward() — the optimization signal
    2. Independent success measurements — ground truth checks

    When these diverge beyond a threshold, the reward function may be stale.

    This requires the developer to define independent_check() — a function
    that evaluates task success without using the reward function. The auditor
    cannot detect staleness without a reference signal.
    """

    def __init__(
        self,
        audit_frequency: int = 50,          # Audit every N rollouts
        divergence_threshold: float = 0.2,  # Spearman rank correlation drop
        window_size: int = 100,
    ):
        self._reward_history: deque[float] = deque(maxlen=window_size)
        self._success_history: deque[float] = deque(maxlen=window_size)

    def record_reward(self, reward: float, rollout_id: str) -> None:
        """Record emitted reward for audit comparison."""

    def record_independent_check(self, success: float, rollout_id: str) -> None:
        """Record independent success measurement.

        This is the critical input that the developer must provide.
        Without it, staleness cannot be detected — only reward distribution
        changes (which could be genuine improvement).
        """

    def audit(self) -> Optional[StalenessReport]:
        """Run staleness check if enough data has accumulated.

        Computes rank correlation between reward and independent success.
        If correlation drops below threshold, returns StalenessReport.

        Returns None if insufficient data or no staleness detected.
        """

    def get_distribution_shift(self) -> Optional[DistributionShiftReport]:
        """Detect reward distribution changes even without independent checks.

        Uses KL divergence between recent and historical reward distributions.
        This is weaker than correlation-based staleness detection — distribution
        shift could be genuine improvement — but requires no independent signal.

        Output uses 'could be' language:
        'Reward distribution has shifted (KL divergence: 0.34). Could indicate
         reward hacking, environment change, or genuine capability improvement.
         Independent validation recommended.'
        """
```

#### `RewardAuditAdapter`: Audit-Oriented Trace Processing

```python
class RewardAuditAdapter(TraceAdapter[List[AuditRecord]]):
    """Adapter that processes traces for reward audit rather than training.

    Implements the same TraceAdapter[T_to] interface as TracerTraceToTriplet,
    but produces audit records instead of training triplets.

    Can run alongside the training adapter without interference.
    """

    def adapt(self, source: Sequence[Span]) -> List[AuditRecord]:
        """Extract reward spans and pair with task metadata for audit.

        Returns AuditRecord containing:
        - rollout_id, attempt_id
        - emitted reward value
        - task input hash (for grouping by task type)
        - span timestamp (for temporal analysis)
        - reward dimension breakdown (if multi-dimensional)
        """
```

### Integration Pattern

The auditor is **opt-in** and runs as a side-channel:

```python
trainer = Trainer(
    algorithm=APO(...),
    # ... standard config ...
)

# Opt-in: attach reward auditor
auditor = RewardStalenessAuditor(audit_frequency=50)
trainer.attach_auditor(auditor)  # New method on Trainer

# Developer provides independent check
@auditor.independent_check
async def check_task_success(rollout: AttemptedRollout) -> float:
    """Developer-defined ground truth evaluation."""
    # e.g., actually verify the agent's output against known-good answers
    ...
```

The auditor **never modifies** the training loop. It produces reports that surface in
the dashboard and optionally trigger alerts.

---

## Gap 3: Multi-Objective Tension Support

### What's Missing

Agent-lightning forces competing objectives into single scalar values. The `Triplet`
model (`agentlightning/types/core.py`) has `reward: Optional[float]` — a single number.
The reward matching policies (`FIRST_OCCURRENCE`, `FIRST_SIBLING`) each produce one
float per LLM call. Multi-dimensional rewards exist in emission (`emit_reward()` accepts
`Dict[str, float]`) but are collapsed to a primary key before reaching the algorithm.

This collapse destroys the Pareto structure of multi-objective problems. When "speed"
and "thoroughness" compete, averaging them produces mediocre agents that are neither
fast nor thorough. The trade-off is invisible to the algorithm.

### Where It Lands

**Primary integration point:** The `Triplet` class and `TraceTree.to_trajectory()`.

**Secondary integration point:** A new `RewardMatchPolicy.PARETO_FRONT` that preserves
dimensional tension instead of collapsing to scalar.

### Proposed Architecture

```
agentlightning/
  emergence/
    pareto.py           # Pareto front tracking and tension preservation
```

#### Extending the Triplet Model

The existing `Triplet` already has `metadata: Dict[str, Any]`. Rather than modifying
the core class (which would break existing algorithms), dimensional rewards flow through
metadata:

```python
# In Triplet.metadata:
{
    "reward_dimensions": {
        "speed": 0.9,
        "thoroughness": 0.3,
        "novelty": 0.7,
    },
    "pareto_rank": 2,           # 0 = Pareto-optimal
    "dominated_by": ["rollout_abc"],
    "tension": "speed vs thoroughness (ρ = -0.72)",
}
```

#### `pareto.py`: Pareto Front Tracking

```python
class ParetoTracker:
    """Track Pareto fronts across reward dimensions.

    Instead of collapsing multi-dimensional rewards to a scalar,
    maintains the full Pareto surface and surfaces the trade-offs
    that optimization would otherwise hide.
    """

    def __init__(
        self,
        dimensions: List[str],          # e.g., ["speed", "thoroughness", "novelty"]
        primary_key: Optional[str] = None,  # For backward compatibility
    ):
        self._front: List[ParetoPoint] = []
        self._history: deque[ParetoPoint] = deque(maxlen=5000)

    def add_point(
        self,
        rollout_id: str,
        values: Dict[str, float],
    ) -> ParetoClassification:
        """Classify a new point against the current front.

        Returns:
        - rank: 0 if Pareto-optimal, N if dominated by N front layers
        - dominated_by: list of rollout_ids that dominate this point
        - dominates: list of rollout_ids this point displaces from front
        - tension_report: which dimensions are in trade-off
        """

    def get_front(self, rank: int = 0) -> List[ParetoPoint]:
        """Get the Nth Pareto front layer."""

    def get_tension_map(self) -> Dict[Tuple[str, str], float]:
        """Pairwise correlation between dimensions across all points.

        Negative correlation = structural trade-off (tension).
        Positive correlation = aligned objectives (no tension).
        Near-zero = independent objectives.

        Example output:
        {
            ("speed", "thoroughness"): -0.72,  # Strong tension
            ("speed", "novelty"): 0.15,         # Independent
            ("thoroughness", "novelty"): 0.41,  # Mildly aligned
        }
        """

    def summary(self) -> str:
        """Human-readable tension summary.

        Example:
        'Pareto front: 12 non-dominated solutions across 3 dimensions.
         Primary tension: speed vs thoroughness (ρ = -0.72).
         Current front favors speed — thoroughness ceiling could indicate
         unexplored strategies that sacrifice speed for depth.'
        """
```

#### New Reward Match Policy

Extend `RewardMatchPolicy` enum:

```python
class RewardMatchPolicy(str, Enum):
    FIRST_SIBLING = "first_sibling"
    FIRST_OCCURRENCE = "first_occurrence"

    # New: preserve dimensional structure
    DIMENSIONAL = "dimensional"
    """Preserve all reward dimensions in Triplet.metadata instead of
    collapsing to primary_key scalar. The scalar Triplet.reward field
    gets the primary_key value for backward compatibility; full dimensions
    are in metadata['reward_dimensions'].

    Algorithms that understand multi-objective can read metadata.
    Algorithms that don't can ignore it and use the scalar as before.
    """
```

### Integration with Algorithms

**APO**: The textual gradient computation in `textual_gradient_and_apply_edit()` could
receive a `tension_report` alongside the rollout results, allowing the LLM-based critic
to reason about trade-offs explicitly: "This prompt improved speed but sacrificed
thoroughness. The Pareto front suggests this trade-off is steep — consider prompts that
maintain thoroughness while recovering speed through structural efficiency."

**VERL**: The PPO reward can optionally receive Pareto rank as a supplementary signal,
penalizing solutions that are dominated on multiple fronts rather than just low on the
scalar reward.

Both integrations are **opt-in** and backward-compatible. Existing algorithms that read
only `Triplet.reward` work unchanged.

---

## Gap 4: Policy Dissolution Mechanism

### What's Missing

Agent-lightning's resource versioning (`ResourcesUpdate` in `types/resources.py`) is
monotonically increasing. Version 5 strictly replaces version 4. Resources accumulate
permanently — there is no unlearning, no expiry, no re-validation.

This means trained behaviors persist even when:
- The environment has changed (API updates, new tool availability)
- The reward function has drifted (Gap 2)
- The behavior was optimal for a specific context that no longer applies
- The behavior was a local optimum that blocks discovery of better strategies

### Where It Lands

**Primary integration point:** `ResourcesUpdate` in `types/resources.py` and the store's
`add_resources()` / `get_latest_resources()` / `update_resources()` methods.

**Secondary integration point:** The `Algorithm.run()` loop, which currently fetches
the latest resource version unconditionally.

### Proposed Architecture

```
agentlightning/
  emergence/
    dissolution.py      # TTL metadata, validity conditions, re-validation
```

#### Resource Dissolution Metadata

Extend `ResourcesUpdate` metadata (not the class itself — use the existing
`metadata` pattern):

```python
class DissolutionMetadata(BaseModel):
    """Metadata attached to resource versions for dissolution tracking.

    Stored in ResourcesUpdate's metadata dict under the key
    'agentlightning.emergence.dissolution'.
    """

    # Temporal dissolution
    ttl_seconds: Optional[int] = None
    """Time-to-live. After this duration, the resource should be
    re-validated before use. None = no temporal expiry."""

    created_at: float
    """Timestamp when this resource version was created."""

    # Conditional dissolution
    validity_conditions: List[ValidityCondition] = []
    """Conditions that must remain true for this resource to be valid.
    When any condition fails, the resource should be re-validated."""

    # Audit trail
    validation_history: List[ValidationRecord] = []
    """Record of re-validation attempts and results."""

    # Dissolution policy
    on_dissolution: DissolutionPolicy = DissolutionPolicy.REVALIDATE
    """What to do when dissolution triggers:
    REVALIDATE: re-run validation, keep if still good
    REGRESS: fall back to previous version
    EXPLORE: switch to exploration mode (no resource pinning)
    """

class ValidityCondition(BaseModel):
    """A condition that must remain true for a resource to be valid."""

    name: str
    """Human-readable condition name."""

    description: str
    """What this condition checks."""

    check_type: Literal["reward_threshold", "entropy_threshold", "custom"]
    """Type of validity check."""

    parameters: Dict[str, Any] = {}
    """Parameters for the check (threshold values, etc.)."""

class DissolutionPolicy(str, Enum):
    REVALIDATE = "revalidate"
    REGRESS = "regress"
    EXPLORE = "explore"
```

#### `dissolution.py`: Dissolution Engine

```python
class DissolutionEngine:
    """Manages resource lifecycle with TTL, validity conditions, and re-validation.

    Wraps a LightningStore to intercept resource retrieval and check
    dissolution conditions before returning resources.
    """

    def __init__(
        self,
        store: LightningStore,
        default_ttl: Optional[int] = None,
        check_interval: int = 10,  # Check every N rollouts
    ):
        self._store = store
        self._dissolution_cache: Dict[str, DissolutionMetadata] = {}

    async def get_resources_with_dissolution_check(
        self,
        resources_id: Optional[str] = None,
    ) -> Tuple[ResourcesUpdate, Optional[DissolutionSignal]]:
        """Fetch resources, checking dissolution conditions.

        Returns the resources AND any dissolution signal. The caller
        decides what to do — the engine does not block resource access.

        DissolutionSignal contains:
        - trigger: which condition fired ("ttl_expired", "reward_below_threshold", ...)
        - severity: "advisory" | "warning" | "critical"
        - recommendation: hedged text ("Resource version 5 has been active for 48h.
          Could indicate the environment has changed since training. Consider
          re-validation.")
        """

    async def attach_dissolution_metadata(
        self,
        resources_id: str,
        ttl_seconds: Optional[int] = None,
        validity_conditions: Optional[List[ValidityCondition]] = None,
        policy: DissolutionPolicy = DissolutionPolicy.REVALIDATE,
    ) -> None:
        """Attach dissolution metadata to a resource version."""

    async def check_conditions(
        self,
        resources_id: str,
    ) -> List[ConditionResult]:
        """Evaluate all validity conditions for a resource version.

        Returns per-condition results. Failed conditions are signals,
        not automatic actions.
        """

    async def dissolve(
        self,
        resources_id: str,
        trigger: str,
    ) -> DissolutionAction:
        """Execute dissolution policy for a resource version.

        REVALIDATE: re-run validation rollouts, keep or discard
        REGRESS: find previous version, mark current as dissolved
        EXPLORE: clear resource pinning, let runners use no resource

        Returns DissolutionAction describing what was done.
        """
```

### Integration with Training Loop

The dissolution engine sits between the algorithm and the store:

```python
# In Algorithm.run() or a custom algorithm:
engine = DissolutionEngine(store=self.get_store(), default_ttl=3600)

# Before evaluation round:
resources, signal = await engine.get_resources_with_dissolution_check()
if signal and signal.severity == "critical":
    logger.warning(f"Dissolution signal: {signal.recommendation}")
    # Developer decides: re-validate, regress, or continue
```

The engine **never** automatically removes resources. It surfaces dissolution signals.
The algorithm (or human operator) decides whether to act.

---

## Gap 5: Novel vs. Routine Behavior Distinction

### What's Missing

Agent-lightning's span system captures what happened but not whether it's new. A span
with `name="openai.chat.completion"` and high reward could represent:

- A genuinely novel strategy the agent has never tried before
- The 500th repetition of a known-good strategy with minor token variation

These look identical in the trace tree. The `TracerTraceToTriplet` adapter treats them
identically when building training triplets. The algorithm optimizes both the same way.

This means novel discoveries get overwhelmed by routine high-reward trajectories in the
training data. At scale (128-GPU distributed training), novel behaviors are statistical
noise in a sea of routine exploitation.

### Where It Lands

**Primary integration point:** New span attributes via `EmergenceSpanAttributes` and a
`NoveltyDetector` that annotates spans before they reach the adapter.

**Secondary integration point:** A `NoveltyAwareAdapter` that wraps `TracerTraceToTriplet`
to weight novel trajectories differently.

### Proposed Architecture

```
agentlightning/
  emergence/
    novelty.py          # Novelty detection and annotation
```

#### `novelty.py`: Novelty Detection

```python
class NoveltyDetector:
    """Detect whether a trajectory represents novel or routine behavior.

    Novelty is defined structurally: a trajectory is novel if its shape
    (sequence of tool calls, branching pattern, response structure) has
    not been seen before. Token-level variation within a known shape is
    NOT novelty — it's exploration noise.

    The detector maintains a running codebook of known trajectory shapes.
    New shapes start with high novelty scores that decay as they're seen
    more frequently.
    """

    def __init__(
        self,
        shape_similarity_threshold: float = 0.85,
        novelty_decay_rate: float = 0.95,  # Per-observation decay
        max_codebook_size: int = 1000,
    ):
        self._codebook: Dict[str, ShapeEntry] = {}

    def compute_shape(self, tree: TraceTree) -> TrajectoryShape:
        """Extract structural shape from a trace tree.

        Shape = sequence of (span_name, depth, child_count) tuples.
        Ignores token content, timestamps, and specific attribute values.
        Captures the structural skeleton of what the agent did.
        """

    def score_novelty(self, tree: TraceTree) -> NoveltyScore:
        """Score a trajectory's novelty against the codebook.

        Returns NoveltyScore with:
        - score: 0.0 (completely routine) to 1.0 (never seen)
        - nearest_shape: most similar known shape (if any)
        - similarity_to_nearest: cosine similarity of shapes
        - first_seen: whether this exact shape is being recorded for first time
        - classification: "novel" | "familiar" | "routine"
        """

    def annotate_spans(
        self,
        spans: Sequence[Span],
        tree: TraceTree,
        score: NoveltyScore,
    ) -> Sequence[Span]:
        """Add novelty attributes to spans.

        Adds to root span:
        - agentlightning.emergence.novelty.score: float
        - agentlightning.emergence.novelty.classification: str
        - agentlightning.emergence.novelty.nearest_shape: str
        """

    def get_discovery_rate(self, window_size: int = 100) -> float:
        """Fraction of recent trajectories classified as 'novel'.

        Declining discovery rate is a leading indicator of exploration
        collapse — the system is no longer finding new behavioral patterns.
        """

    def get_codebook_summary(self) -> str:
        """Summarize known trajectory shapes.

        Example:
        'Codebook: 47 known shapes. Top 5 by frequency:
         1. [search → analyze → respond] (seen 234×, 45% of trajectories)
         2. [search → search → analyze → respond] (seen 89×, 17%)
         3. [respond directly] (seen 67×, 13%)
         ...
         Discovery rate (last 100): 0.03 (declining from 0.12 over 5 windows).
         Could indicate behavioral convergence — 3 shapes account for 75%
         of all trajectories.'
        """
```

#### `NoveltyAwareAdapter`: Weighted Training Data

```python
class NoveltyAwareAdapter(TraceAdapter[List[Triplet]]):
    """Wraps TracerTraceToTriplet to weight novel trajectories.

    Does NOT replace the base adapter. Produces the same Triplet format
    with additional metadata and optional sampling weights.

    Novel high-reward trajectories get higher sampling weight.
    Routine high-reward trajectories get standard weight.
    Novel low-reward trajectories get standard weight (exploration is
    not unconditionally good — it needs reward context).
    """

    def __init__(
        self,
        base_adapter: TracerTraceToTriplet,
        novelty_detector: NoveltyDetector,
        novelty_weight_multiplier: float = 2.0,
    ):
        self._base = base_adapter
        self._detector = novelty_detector
        self._weight_multiplier = novelty_weight_multiplier

    def adapt(self, source: Sequence[Span]) -> List[Triplet]:
        """Adapt with novelty annotation.

        Each Triplet.metadata gets:
        - 'novelty_score': float
        - 'novelty_classification': str
        - 'sampling_weight': float (1.0 for routine, multiplier for novel+rewarded)
        """
```

### Integration with Algorithms

**APO**: When computing textual gradients, novel high-reward trajectories can be
highlighted in the prompt to the critic LLM: "This trajectory used a novel approach
(first seen) and achieved high reward. Consider what about this approach is transferable."

**VERL**: Sampling weights from `NoveltyAwareAdapter` directly influence which trajectories
appear more frequently in PPO training batches, giving novel discoveries more gradient
signal without overriding the reward.

---

## The Integrated System

### Module Structure

```
agentlightning/
  emergence/
    __init__.py
    entropy.py              # Gap 1: Exploration decay monitoring
    monitoring.py           # Gap 1: Sliding window collapse detection
    reward_audit.py         # Gap 2: Reward staleness detection
    pareto.py               # Gap 3: Multi-objective tension tracking
    dissolution.py          # Gap 4: Resource TTL and re-validation
    novelty.py              # Gap 5: Novel vs routine distinction
    types.py                # Shared types for emergence module
    semconv.py              # Emergence-specific span attributes
```

### Cross-Module Interactions

The five gaps are not independent. They reinforce each other:

```
                    ┌─────────────────────────┐
                    │  Gap 1: Entropy Monitor  │
                    │  (behavioral diversity)  │
                    └──────────┬──────────────┘
                               │ entropy feeds
                               ▼
┌───────────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ Gap 2: Reward     │◄──│ Gap 3: Pareto   │──►│ Gap 5: Novelty   │
│ Staleness Audit   │   │ Tension Tracker │   │ Detection        │
│ (signal validity) │   │ (trade-offs)    │   │ (discovery rate)  │
└───────────┬───────┘   └────────┬────────┘   └──────────┬───────┘
            │                    │                        │
            │  staleness        │ pareto rank            │ novelty score
            │  informs          │ informs                │ informs
            ▼                    ▼                        ▼
            ┌────────────────────────────────────────────┐
            │          Gap 4: Dissolution Engine         │
            │      (resource lifecycle management)       │
            └────────────────────────────────────────────┘
```

- **Entropy decline** (Gap 1) triggers dissolution condition checks (Gap 4)
- **Reward staleness** (Gap 2) is a dissolution trigger for resources trained under stale rewards
- **Pareto front stagnation** (Gap 3) — when the front stops expanding, it could indicate
  exploration collapse or genuine Pareto optimality. Cross-reference with entropy (Gap 1)
  to distinguish
- **Discovery rate decline** (Gap 5) is both an entropy component (Gap 1) and a dissolution
  condition (Gap 4)

### Generative Friction Points (Deliberately Maintained)

**Friction 1: Novelty vs. Reward**
A novel trajectory with low reward and a routine trajectory with high reward are in
structural tension. The system surfaces both without resolving which is "better."
Resolution depends on context: early in training, novelty should be weighted higher;
late in training, reward matters more. But "early" and "late" are themselves judgment
calls that the system cannot make.

**Friction 2: Dissolution vs. Stability**
Dissolving resources creates exploration pressure but also destroys learned behaviors.
The system tracks both the cost of keeping stale resources (potential reward hacking,
environmental mismatch) and the cost of dissolving them (loss of learned capabilities,
training instability). Neither cost dominates.

**Friction 3: Pareto Front vs. Scalar Reward**
Existing algorithms (APO, VERL) consume scalar rewards. The Pareto tracker preserves
dimensional structure. These disagree on what "best" means. The system maintains both
views — the scalar for optimization, the Pareto for awareness — without resolving the
disagreement.

---

## Dissolution Conditions

Every structural decision in this document carries a condition under which it should be
removed.

### Module: emergence/

**The emergence module as a whole.**
Serves as long as agent-lightning's core training loop lacks built-in exploration
pressure. If future versions of the framework add native entropy regularization,
multi-objective optimization, and resource lifecycle management, this module becomes
redundant scaffolding and should be dissolved.

### Gap 1: Exploration Decay Monitoring

**Trajectory entropy computation.**
Dissolution: When agent-lightning's native metrics (`@tracked` decorator, Prometheus
integration) include behavioral diversity metrics, making external entropy computation
redundant.

**Sliding window collapse detection.**
Dissolution: When algorithms natively implement exploration-exploitation balancing
(e.g., entropy bonus in PPO reward, beam diversity in APO) that makes external collapse
detection unnecessary. Evidence: collapse signals are never acted upon because the
algorithm already prevents collapse.

**Shape-based entropy (vs. token-level).**
Dissolution: When language model behavioral diversity can be measured at the token level
efficiently. Currently, shape-based measurement is a compression that loses information
but is computationally tractable. If token-level entropy becomes cheap, shape-based
measurement adds unnecessary abstraction.

### Gap 2: Reward Function Staleness Detection

**Independent check requirement.**
Serves as long as reward functions are black boxes to the training system. If future
reward models include built-in calibration or uncertainty quantification, external
staleness detection adds redundant computation.

**Distribution shift detection (without independent checks).**
Dissolution: When reward models natively report confidence scores or calibration
metrics, making KL divergence on reward distributions a less informative proxy.

**RewardAuditAdapter.**
Dissolution: When audit data flows through the primary adapter pipeline rather than
requiring a separate adapter. Evidence: all algorithms consume audit records alongside
training data.

### Gap 3: Multi-Objective Tension Support

**Pareto front tracking.**
Dissolution: When agent-lightning's Triplet model natively supports vector rewards
rather than Optional[float]. At that point, Pareto tracking should move from the
emergence module into core.

**Dimensional reward in Triplet.metadata.**
Serves as a backward-compatible shim. Dissolves when the Triplet class gains a
`reward_dimensions: Optional[Dict[str, float]]` field, making the metadata workaround
unnecessary.

**DIMENSIONAL reward match policy.**
Dissolution: When all reward matching policies preserve dimensional structure by
default, making a special policy unnecessary.

**Tension map (pairwise dimension correlation).**
Dissolution: When dashboard visualization handles multi-objective trade-off display
natively, making the text-based tension map redundant.

### Gap 4: Policy Dissolution Mechanism

**TTL-based resource expiry.**
Dissolution: When the training environment is provably stationary (fixed tasks, fixed
evaluation criteria, no distribution shift). In stationary environments, resources
don't go stale and TTL adds unnecessary complexity.

**Validity conditions on resources.**
Dissolution: When all resources are re-validated on every use (streaming validation
rather than point-in-time checks). At that point, validity conditions are redundant
because staleness is detected continuously.

**DissolutionPolicy.EXPLORE mode.**
Dissolution: When the runner natively supports resource-free exploration rollouts,
making explicit "explore mode" switching unnecessary.

**The dissolution engine wrapper around LightningStore.**
Serves as long as dissolution logic is external to the store. If resource lifecycle
management moves into `CollectionBasedLightningStore` natively, the wrapper adds
indirection without benefit.

### Gap 5: Novel vs. Routine Behavior Distinction

**Shape-based novelty detection.**
Dissolution: When agent-lightning's span system includes native trajectory
fingerprinting (e.g., a hash of the trace tree structure), making external shape
computation redundant.

**Codebook-based classification.**
Dissolution: When neural novelty detection (e.g., learned embeddings of trajectory
space) becomes efficient enough for online use. Codebook-based classification is a
discrete approximation that loses continuous similarity information.

**NoveltyAwareAdapter wrapper.**
Dissolution: When the base `TracerTraceToTriplet` adapter supports configurable
weighting functions, making the wrapper unnecessary. Evidence: all users configure
weights through the base adapter rather than using the wrapper.

**Discovery rate metric.**
Dissolution: When entropy monitoring (Gap 1) fully subsumes novelty tracking. Currently,
discovery rate captures information that entropy does not (whether *new shapes* are
appearing, not just whether the distribution is spread). If entropy metrics are extended
to track shape inventory growth, discovery rate becomes redundant.

### Cross-Module Interactions

**The interaction diagram above.**
Dissolution: When the five gaps are sufficiently independent that cross-referencing adds
complexity without insight. Evidence: operators use individual gap tools in isolation and
never benefit from cross-gap signals.

**Generative friction points.**
These dissolve when agents reliably surface competing strategies without explicit
scaffolding. Evidence: removing a friction point does not reduce the diversity of
strategies agents discover.

### "Could Be" Language Pattern

**Hedged output in all emergence modules.**
Dissolution: When the emergence modules produce high-confidence signals that operators
treat as actionable without hedging. At that point, "could indicate" language adds
uncertainty where none exists and should be replaced with direct statements. Evidence:
operators consistently act on emergence signals and report that hedging is noise.

### Best-Effort Enrichment Pattern

**`asyncio.gather(return_exceptions=True)` wrapping.**
Dissolution: When emergence computations are reliable enough that failures drop below
~1%. At that point, switch to hard errors so silent data gaps don't produce silently
misleading output.

---

## What This Is Not

1. **Not an exploration algorithm.** This document does not propose epsilon-greedy,
   UCB, or any other exploration strategy. Those are algorithms that *resolve* the
   exploration-exploitation trade-off. This document proposes mechanisms that *surface*
   the trade-off so it can be resolved by judgment.

2. **Not a replacement for optimization.** Every module is additive. Remove them all
   and agent-lightning works exactly as before. The emergence layer exists alongside
   the optimization loop, not instead of it.

3. **Not prescriptive.** The modules surface signals. They do not automatically modify
   training behavior. An entropy collapse signal does not automatically widen the beam.
   A dissolution trigger does not automatically remove resources. A novelty score does
   not automatically increase sampling weight. Each signal requires a decision.

4. **Not permanent.** Every structure in this document is designed to be removed when
   it's no longer needed. The dissolution conditions section is not decoration — it is
   the most important part of the document.

---

## Implementation Order

If this design is approved, implementation should proceed in this order:

1. **Gap 5: Novelty Detection** — smallest surface area, clearest value signal, tests
   the integration pattern (new adapter wrapping existing adapter).

2. **Gap 1: Entropy Monitoring** — builds on novelty detection's shape computation,
   provides the behavioral diversity baseline that other gaps reference.

3. **Gap 3: Pareto Tension** — extends the reward pipeline, tests the metadata-based
   backward compatibility pattern.

4. **Gap 2: Reward Staleness** — requires the most developer input (independent checks),
   tests the side-channel audit pattern.

5. **Gap 4: Dissolution Engine** — depends on signals from all other gaps, should be
   implemented last so it can reference real signal types.

Each gap is independently useful. The implementation order maximizes the value of
partial completion.

---

## Relationship to Spectra MCP Server Patterns

This design draws directly from patterns validated in the [Spectra Finance MCP Server](https://github.com/Finanzgoblin/mcp-spectra-finance):

| Spectra Pattern | Agent-Lightning Equivalent |
|---|---|
| Layer 3 "could be" hints | Hedged output in all emergence modules |
| `formatVolumeHints` (signals at knowledge boundary) | `ExplorationDecayMonitor.summary()` |
| Generative friction (raw APY vs effective APY) | Pareto front vs scalar reward |
| Dissolution conditions per structural decision | Gap 4 + final section of this document |
| Best-effort enrichment (`Promise.allSettled`) | `asyncio.gather(return_exceptions=True)` |
| Negative signals (surfacing absence) | Discovery rate decline, codebook stagnation |
| Navigation paths between tools | Cross-module interaction diagram |

The key difference: Spectra's emergence patterns operate at the **tool output layer**
(shaping what an LLM sees). Agent-lightning's emergence patterns operate at the
**training loop layer** (shaping what an algorithm learns from). The philosophy is the
same — hold tension open, surface ambiguity, dissolve when no longer needed — but the
engineering target is fundamentally different.

The Spectra server proved that "could be" language and maintained tension produce better
autonomous agent behavior than prescriptive conclusions. This design applies the same
insight to the training process itself: an agent that trains with tension preserved will
be more capable than one that trains with tension resolved.
