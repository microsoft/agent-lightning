# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

import agentlightning as agl


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


def test_trainer_with_strategy_alias_shm() -> None:
    """Test trainer initialization with strategy alias 'shm'."""
    algorithm = agl.Baseline()
    # Use strategy alias "shm"
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=1,  # n_runners must be 1 here
        strategy="shm",
    )
    assert isinstance(trainer.strategy, agl.SharedMemoryExecutionStrategy)


def test_trainer_with_strategy_dict_main_thread() -> None:
    """Test trainer initialization with strategy dict allowing n_runners > 1."""
    algorithm = agl.Baseline()
    # Use dict. Now n_runners can be >1 because algorithm is on the main thread
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        strategy={"type": "shm", "main_thread": "algorithm", "managed_store": False},
    )
    assert isinstance(trainer.strategy, agl.SharedMemoryExecutionStrategy)
    assert trainer.strategy.main_thread == "algorithm"
    assert trainer.strategy.managed_store is False


def test_trainer_with_initialized_strategy_ignores_n_runners() -> None:
    """Test that n_runners is ignored when strategy is already initialized."""
    algorithm = agl.Baseline()
    # n_runners is ignored in the trainer because strategy has been initialized with n_runners=4
    strategy = agl.SharedMemoryExecutionStrategy(main_thread="algorithm", n_runners=4)
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        strategy=strategy,
    )
    assert trainer.strategy is strategy
    assert trainer.strategy.n_runners == 4  # type: ignore


def test_trainer_with_client_server_strategy_dict() -> None:
    """Test trainer initialization with client-server strategy dict."""
    algorithm = agl.Baseline()
    # By default, strategy is client-server, but you can also use a string alias to specify it again
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        strategy={
            # This line is optional
            "type": "cs",
            "server_port": 9999,
        },
    )
    assert isinstance(trainer.strategy, agl.ClientServerExecutionStrategy)
    assert trainer.strategy.server_port == 9999


def test_trainer_port_forwarded_to_client_server_strategy() -> None:
    """Test that the top-level port argument configures the client-server strategy."""
    trainer = agl.Trainer(
        algorithm=agl.Baseline(),
        n_runners=4,
        port=8081,
    )

    assert isinstance(trainer.strategy, agl.ClientServerExecutionStrategy)
    assert trainer.strategy.server_port == 8081


def test_trainer_port_ignored_for_non_client_server_strategy() -> None:
    """Test that port has no effect when using a non client-server strategy."""
    trainer = agl.Trainer(
        algorithm=agl.Baseline(),
        n_runners=1,
        port=8082,
        strategy="shm",
    )

    assert isinstance(trainer.strategy, agl.SharedMemoryExecutionStrategy)
    assert not hasattr(trainer.strategy, "server_port")


def test_trainer_port_overrides_existing_client_server_strategy() -> None:
    """Test that provided port overrides an initialized client-server strategy."""
    strategy = agl.ClientServerExecutionStrategy(server_port=9000)

    trainer = agl.Trainer(
        algorithm=agl.Baseline(),
        n_runners=1,
        strategy=strategy,
        port=9100,
    )

    assert trainer.strategy is strategy
    assert trainer.strategy.server_port == 9100  # type: ignore


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
        # This line is optional
        strategy="cs",
    )
    assert isinstance(trainer.strategy, agl.ClientServerExecutionStrategy)
    assert trainer.strategy.server_port == 10000
    assert trainer.strategy.role == "algorithm"
    assert trainer.strategy.managed_store is False


def test_trainer_with_string_adapter() -> None:
    """Test trainer initialization with adapter specified as string."""
    algorithm = agl.Baseline()
    trainer = agl.Trainer(algorithm=algorithm, n_runners=8, adapter="agentlightning.adapter.TraceToMessages")
    assert isinstance(trainer.adapter, agl.TraceToMessages)


def test_trainer_with_adapter_dict_no_type() -> None:
    """Test trainer initialization with adapter dict without type field."""
    algorithm = agl.Baseline()
    # If it's a dict and type is not provided, it will use the default class
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=8,
        adapter={"agent_match": "plan_agent", "repair_hierarchy": False},
    )
    assert isinstance(trainer.adapter, agl.TracerTraceToTriplet)
    assert trainer.adapter.agent_match == "plan_agent"
    assert trainer.adapter.repair_hierarchy is False
