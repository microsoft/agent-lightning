"""Tests for runner package public exports."""

import agentlightning.runner as runner_module


def test_runner_exports_do_not_include_legacy_runner() -> None:
    """Legacy runner implementations are not part of the public runner surface."""
    assert "LegacyAgentRunner" not in getattr(runner_module, "__all__", [])
