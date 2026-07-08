# Copyright (c) Microsoft. All rights reserved.

"""Tests for LitAgent interface boundaries."""

import pytest
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


def test_litagent_does_not_store_trained_agents_marker() -> None:
    """Validate the deprecated `trained_agents` argument is removed."""
    agent = _BoundaryAgent()

    assert not hasattr(agent, "trained_agents")

    with pytest.raises(TypeError, match="__init__"):
        _BoundaryAgent(trained_agents="legacy")


def test_litagent_no_legacy_rollout_lifecycle_methods() -> None:
    """Validate deprecated rollout lifecycle hooks are removed from LitAgent."""
    agent = _BoundaryAgent()

    assert not hasattr(agent, "on_rollout_start")
    assert not hasattr(agent, "on_rollout_end")
