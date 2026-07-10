# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any

import pytest

import agentlightning as agl
from agentlightning.litagent import LitAgent


class _NoopAgent(LitAgent[object]):
    """Minimal agent used to validate trainer API behavior."""


def test_trainer_with_predefined_tracer() -> None:
    """Test trainer initialization with predefined tracer."""
    algorithm = agl.Baseline()
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        tracer=agl.OtelTracer(),
    )
    # Runner is initialized to be the default runner: LitAgentRunner
    assert isinstance(trainer.runner, agl.LitAgentRunner)
    assert isinstance(trainer.runner.tracer, agl.OtelTracer)


def test_trainer_with_shared_memory_strategy_instance() -> None:
    """Test trainer initialization with an explicit shared-memory strategy."""
    algorithm = agl.Baseline()
    strategy = agl.SharedMemoryExecutionStrategy()
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=1,  # n_runners must be 1 here
        strategy=strategy,
    )
    assert trainer.strategy is strategy


def test_trainer_with_shared_memory_strategy_main_thread() -> None:
    """Test trainer initialization with an explicit strategy allowing n_runners > 1."""
    algorithm = agl.Baseline()
    strategy = agl.SharedMemoryExecutionStrategy(main_thread="algorithm", n_runners=8, managed_store=False)
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        strategy=strategy,
    )
    assert trainer.strategy is strategy
    assert strategy.main_thread == "algorithm"
    assert strategy.managed_store is False


def test_trainer_rejects_conflicting_runner_counts() -> None:
    """Runner parallelism must have one unambiguous source of truth."""
    algorithm = agl.Baseline()
    strategy = agl.SharedMemoryExecutionStrategy(main_thread="algorithm", n_runners=4)
    with pytest.raises(ValueError, match="Configure it in one place"):
        agl.Trainer(
            algorithm=algorithm,
            n_runners=8,
            strategy=strategy,
        )


def test_trainer_uses_explicit_strategy_runner_count() -> None:
    strategy = agl.SharedMemoryExecutionStrategy(main_thread="algorithm", n_runners=4)
    trainer = agl.Trainer(algorithm=agl.Baseline(), strategy=strategy)

    assert trainer.n_runners == 4


def test_trainer_with_client_server_strategy_instance() -> None:
    """Test trainer initialization with an explicit client-server strategy."""
    algorithm = agl.Baseline()
    strategy = agl.ClientServerExecutionStrategy(server_port=9999, n_runners=8)
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        strategy=strategy,
    )
    assert trainer.strategy is strategy
    assert strategy.server_port == 9999


def test_trainer_with_env_vars_for_execution_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that execution strategy supports environment variables to override values."""
    algorithm = agl.Baseline()
    # Execution strategy supports using environment variables to override the values
    monkeypatch.setenv("AGL_SERVER_PORT", "10000")
    monkeypatch.setenv("AGL_CURRENT_ROLE", "algorithm")
    monkeypatch.setenv("AGL_MANAGED_STORE", "0")

    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        strategy=agl.ClientServerExecutionStrategy(n_runners=8),
    )
    assert isinstance(trainer.strategy, agl.ClientServerExecutionStrategy)
    assert trainer.strategy.server_port == 10000
    assert trainer.strategy.role == "algorithm"
    assert trainer.strategy.managed_store is False


def test_trainer_with_adapter_instance() -> None:
    """Test trainer initialization with an explicit adapter instance."""
    algorithm = agl.Baseline()
    adapter = agl.TraceToMessages()
    trainer = agl.Trainer(algorithm=algorithm, n_runners=8, adapter=adapter)
    assert isinstance(trainer.adapter, agl.TraceToMessages)
    assert trainer.adapter is adapter


def test_trainer_with_tracer_triplet_adapter_instance() -> None:
    """Test trainer initialization with an explicit triplet adapter instance."""
    algorithm = agl.Baseline()
    adapter = agl.TracerTraceToTriplet(agent_match="plan_agent", repair_hierarchy=False)
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        adapter=adapter,
    )
    assert trainer.adapter is adapter
    assert adapter.agent_match == "plan_agent"
    assert adapter.repair_hierarchy is False


def test_trainer_rejects_dynamic_component_specs() -> None:
    """Trainer accepts component instances only; dynamic config belongs outside the core API."""
    strategy_alias: Any = "shm"
    strategy_config: Any = {"type": "shm"}
    adapter_path: Any = "agentlightning.adapter.TraceToMessages"
    adapter_config: Any = {"agent_match": "plan_agent"}

    with pytest.raises(TypeError, match="strategy must be an instance of ExecutionStrategy"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, strategy=strategy_alias)

    with pytest.raises(TypeError, match="strategy must be an instance of ExecutionStrategy"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, strategy=strategy_config)

    with pytest.raises(TypeError, match="adapter must be an instance of TraceAdapter"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, adapter=adapter_path)

    with pytest.raises(TypeError, match="adapter must be an instance of TraceAdapter"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, adapter=adapter_config)


def test_trainer_no_longer_has_fit_v0() -> None:
    """Trainer no longer exposes legacy `fit_v0`."""
    assert not hasattr(agl.Trainer, "fit_v0")


def test_trainer_fit_rejects_string_dataset() -> None:
    """Legacy string dataset path is intentionally removed from Trainer.fit."""
    trainer = agl.Trainer(algorithm=agl.Baseline(), n_runners=1)

    with pytest.raises(TypeError, match="no longer accepts string"):
        trainer.fit(_NoopAgent(), train_dataset="http://localhost:8080")


def test_trainer_defaults_to_baseline_algorithm() -> None:
    """Trainer creates a default Baseline algorithm when none is provided."""
    trainer = agl.Trainer(n_runners=1)

    assert isinstance(trainer.algorithm, agl.Baseline)


def test_trainer_rejects_legacy_constructor_args() -> None:
    """Legacy constructor aliases are removed from Trainer.__init__."""
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'n_workers'"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, n_workers=1)  # type: ignore[misc]

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'max_tasks'"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, max_tasks=1)  # type: ignore[misc]

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'daemon'"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, daemon=False)  # type: ignore[misc]

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'triplet_exporter'"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, triplet_exporter=None)  # type: ignore[misc]

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'dev'"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, dev=True)  # type: ignore[misc]

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'port'"):
        agl.Trainer(algorithm=agl.Baseline(), n_runners=1, port=4747)  # type: ignore[misc]
