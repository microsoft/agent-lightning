# Copyright (c) Microsoft. All rights reserved.

"""In-memory server state — single-threaded, no locks, plain dict/list.

Route handlers mutate these module-level dictionaries directly on the event loop
thread. See docs/dev_guidelines.md § Concurrency Model.
"""

from __future__ import annotations

from agentlightning.schemas import Event, Model, Rollout

_rollouts: dict[str, Rollout] = {}
_events: dict[str, dict[str, list[Event]]] = {}
_models: dict[str, dict[str, Model]] = {}

# Deletion leaves a tombstone so existing completion cursors remain valid.
_terminal_order: list[str | None] = []
_terminal_positions: dict[str, int] = {}
