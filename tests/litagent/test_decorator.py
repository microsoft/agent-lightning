# Copyright (c) Microsoft. All rights reserved.

"""Test that @llm_rollout and @rollout decorators preserve function executability."""

import inspect
from typing import Any, cast

import pytest

from agentlightning.litagent import LitAgentLLM, llm_rollout, rollout
from agentlightning.types import LLM, Attempt, AttemptedRollout, NamedResources, ProxyLLM


@llm_rollout
def sample_llm_rollout_func(task: Any, llm: LLM) -> float:
    """A test function with llm_rollout decorator."""
    # Fake a float to bypass the type checker
    return cast(float, f"Processed task: {task} with LLM: {llm}")


@rollout
def sample_rollout_func(task: Any, llm: LLM) -> float:
    """A test function with rollout decorator."""
    return cast(float, f"Processed task: {task} with LLM: {llm}")


def test_llm_rollout_preserves_executability():
    """Test that @llm_rollout decorated functions remain executable."""
    test_task = "Hello World"
    test_llm = "gpt-4"

    # Function should be callable
    assert callable(sample_llm_rollout_func)

    # Function should execute and return expected result
    result = sample_llm_rollout_func(test_task, test_llm)
    expected = f"Processed task: {test_task} with LLM: {test_llm}"
    assert result == expected


def test_llm_rollout_preserves_metadata():
    """Test that @llm_rollout preserves function metadata."""
    # Function name should be preserved
    assert sample_llm_rollout_func.__name__ == "sample_llm_rollout_func"  # type: ignore

    # Docstring should be preserved
    assert sample_llm_rollout_func.__doc__ == "A test function with llm_rollout decorator."


def test_llm_rollout_returns_litagent_instance():
    """Test that @llm_rollout returns a LitAgentLLM instance."""
    assert isinstance(sample_llm_rollout_func, LitAgentLLM)

    # Should have agent methods
    assert hasattr(sample_llm_rollout_func, "rollout")
    assert hasattr(sample_llm_rollout_func, "rollout_async")
    assert hasattr(sample_llm_rollout_func, "training_rollout")


def test_llm_rollout_preserves_signature():
    """Test that @llm_rollout preserves function signature."""
    sig = inspect.signature(sample_llm_rollout_func)
    params = list(sig.parameters.keys())

    # Should have the expected parameters
    assert params == ["task", "llm"]


def test_rollout_preserves_executability():
    """Test that @rollout decorated functions remain executable."""
    test_task = "Hello World"
    test_llm = "gpt-4"

    # Function should be callable
    assert callable(sample_rollout_func)

    # Function should execute and return expected result
    result = sample_rollout_func(test_task, test_llm)  # type: ignore
    expected = f"Processed task: {test_task} with LLM: {test_llm}"
    assert result == expected


def test_rollout_preserves_metadata():
    """Test that @rollout preserves function metadata."""
    # Function name should be preserved
    assert sample_rollout_func.__name__ == "sample_rollout_func"  # type: ignore

    # Docstring should be preserved
    assert sample_rollout_func.__doc__ == "A test function with rollout decorator."


def test_rollout_returns_litagent_instance():
    """Test that @rollout returns a LitAgent instance (actually LitAgentLLM for this pattern)."""
    assert isinstance(sample_rollout_func, LitAgentLLM)

    # Should have agent methods
    assert hasattr(sample_rollout_func, "rollout")
    assert hasattr(sample_rollout_func, "rollout_async")
    assert hasattr(sample_rollout_func, "training_rollout")


def test_rollout_preserves_signature():
    """Test that @rollout preserves function signature."""
    sig = inspect.signature(sample_rollout_func)  # type: ignore
    params = list(sig.parameters.keys())

    # Should have the expected parameters
    assert params == ["task", "llm"]


@pytest.mark.asyncio
async def test_async_function_with_llm_rollout():
    """Test that async functions work with @llm_rollout decorator."""

    @llm_rollout
    async def async_agent(task: Any, llm: LLM) -> float:
        """An async test function."""
        return cast(float, f"Async processed: {task} with {llm}")

    # Should be callable
    assert callable(async_agent)

    # Should preserve async nature when called directly
    result = await async_agent("test", "llm")
    assert result == "Async processed: test with llm"

    # Should be marked as async
    assert async_agent.is_async


@pytest.mark.asyncio
async def test_async_function_with_rollout():
    """Test that async functions work with @rollout decorator."""

    @rollout
    async def async_agent(task: Any, llm: LLM) -> float:
        """An async test function."""
        return cast(float, f"Async processed: {task} with {llm}")

    # Should be callable
    assert callable(async_agent)

    # Should preserve async nature when called directly
    result = await async_agent("test", "llm")  # type: ignore
    assert result == "Async processed: test with llm"

    # Should be marked as async
    assert async_agent.is_async


def test_llm_rollout_strip_proxy_true_strips_proxy_llm():
    """Test that @llm_rollout(strip_proxy=True) strips ProxyLLM to LLM."""

    @llm_rollout(strip_proxy=True)
    def agent_strip_true(task: Any, llm: LLM) -> float:
        """An agent with strip_proxy=True."""
        # Return the type of llm to verify it was stripped
        return cast(float, type(llm).__name__)

    # Create a ProxyLLM resource
    proxy_llm = ProxyLLM(
        endpoint="http://localhost:11434",
        model="test-model",
    )

    # Create an AttemptedRollout
    rollout = AttemptedRollout(
        rollout_id="rollout-123",
        input="test task",
        start_time=0.0,
        attempt=Attempt(
            rollout_id="rollout-123",
            attempt_id="attempt-456",
            sequence_id=1,
            start_time=0.0,
        ),
    )

    # Run rollout with ProxyLLM
    resources = cast(NamedResources, {"llm": proxy_llm})
    result = agent_strip_true.rollout("test", resources, rollout)

    # The LLM should be stripped to regular LLM (not ProxyLLM)
    assert result == "LLM"


def test_llm_rollout_strip_proxy_false_preserves_proxy_llm():
    """Test that @llm_rollout(strip_proxy=False) preserves ProxyLLM."""

    @llm_rollout(strip_proxy=False)
    def agent_strip_false(task: Any, llm: LLM) -> float:
        """An agent with strip_proxy=False."""
        # Return the type of llm to verify it was not stripped
        return cast(float, type(llm).__name__)

    # Create a ProxyLLM resource
    proxy_llm = ProxyLLM(
        endpoint="http://localhost:11434",
        model="test-model",
    )

    # Create an AttemptedRollout
    rollout = AttemptedRollout(
        rollout_id="rollout-123",
        input="test task",
        start_time=0.0,
        attempt=Attempt(
            rollout_id="rollout-123",
            attempt_id="attempt-456",
            sequence_id=1,
            start_time=0.0,
        ),
    )

    # Run rollout with ProxyLLM
    resources = cast(NamedResources, {"llm": proxy_llm})
    result = agent_strip_false.rollout("test", resources, rollout)

    # The LLM should remain as ProxyLLM
    assert result == "ProxyLLM"


def test_llm_rollout_strip_proxy_default_strips_proxy_llm():
    """Test that @llm_rollout defaults to strip_proxy=True."""

    @llm_rollout
    def agent_default(task: Any, llm: LLM) -> float:
        """An agent with default strip_proxy."""
        return cast(float, type(llm).__name__)

    # Create a ProxyLLM resource
    proxy_llm = ProxyLLM(
        endpoint="http://localhost:11434",
        model="test-model",
    )

    # Create an AttemptedRollout
    rollout = AttemptedRollout(
        rollout_id="rollout-123",
        input="test task",
        start_time=0.0,
        attempt=Attempt(
            rollout_id="rollout-123",
            attempt_id="attempt-456",
            sequence_id=1,
            start_time=0.0,
        ),
    )

    # Run rollout with ProxyLLM
    resources = cast(NamedResources, {"llm": proxy_llm})
    result = agent_default.rollout("test", resources, rollout)

    # The LLM should be stripped to regular LLM (default behavior)
    assert result == "LLM"
