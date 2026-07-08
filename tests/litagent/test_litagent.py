# Copyright (c) Microsoft. All rights reserved.

"""Tests for LitAgent interface boundaries."""

from typing import Any, Dict

from agentlightning.litagent import LitAgent
from agentlightning.types import NamedResources, Rollout


class _BoundaryAgent(LitAgent[Dict[str, Any]]):
    """Minimal LitAgent implementation for interface checks."""

    def rollout(self, task: Dict[str, Any], resources: NamedResources, rollout: Rollout) -> float:
        return 1.0


def test_litagent_no_reverse_references_to_runner_trainer_or_tracer() -> None:
    """Validate LitAgent no longer stores runner/trainer/tracer reverse references."""
    agent = _BoundaryAgent()

    assert not hasattr(agent, "set_trainer")
    assert not hasattr(agent, "get_trainer")
    assert not hasattr(agent, "trainer")
    assert not hasattr(agent, "set_runner")
    assert not hasattr(agent, "get_runner")
    assert not hasattr(agent, "runner")
    assert not hasattr(agent, "get_tracer")
    assert not hasattr(agent, "tracer")
