"""Tests for runner package public exports."""

import agentlightning.runner as runner_module
from agentlightning.runner import Runner


def test_runner_exports_do_not_include_legacy_runner() -> None:
    """Legacy runner implementations are not part of the public runner surface."""
    assert "LegacyAgentRunner" not in getattr(runner_module, "__all__", [])


def test_runner_sync_run_interface_is_removed() -> None:
    """The legacy Runner.run sync fallback is not part of the runner API."""
    assert not hasattr(Runner, "run")
