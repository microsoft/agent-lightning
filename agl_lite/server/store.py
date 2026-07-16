"""In-memory server state — single-threaded, no locks, plain dict/list.

Route handlers mutate these module-level dictionaries directly on the event loop
thread. See docs/dev_guidelines.md § Concurrency Model.
"""

from __future__ import annotations

from agl_lite.schemas import Event
from agl_lite.schemas import Model
from agl_lite.schemas import Rollout

_rollouts: dict[str, Rollout] = {}
_events: dict[str, dict[str, list[Event]]] = {}
_models: dict[str, dict[str, Model]] = {}

# Append-only list of rollout_ids in the order they reached a terminal state.
# Enables cheap cursor pagination over completed rollouts (see GET
# /api/rollouts/terminal): because completions are appended in completion order,
# an index cursor never misses out-of-order completions and never needs a full
# rescan of _rollouts. Mutated only by the patch handler on terminal transitions.
_terminal_order: list[str] = []
