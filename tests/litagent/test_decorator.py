# Copyright (c) Microsoft. All rights reserved.

# pyright: reportPrivateUsage=false

"""Test that @llm_rollout and @rollout decorators preserve function executability."""

import inspect
from typing import Any, cast

import pytest

from agentlightning.litagent import llm_rollout, rollout
from agentlightning.litagent.decorator import (
    FunctionalLitAgent,
    _validate_llm_rollout_func,
    _validate_prompt_rollout_func,
    prompt_rollout,
)
from agentlightning.types import LLM, Attempt, AttemptedRollout, NamedResources, PromptTemplate, ProxyLLM


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
    """Test that @llm_rollout returns a FunctionalLitAgent instance."""
    assert isinstance(sample_llm_rollout_func, FunctionalLitAgent)

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
    """Test that @rollout returns a LitAgent instance (actually FunctionalLitAgent for this pattern)."""
    assert isinstance(sample_rollout_func, FunctionalLitAgent)

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
    assert async_agent.is_async()


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
    assert async_agent.is_async()


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


# Tests for prompt_rollout decorator


@prompt_rollout
def sample_prompt_rollout_func(task: Any, prompt_template: PromptTemplate) -> float:
    """A test function with prompt_rollout decorator."""
    return cast(float, f"Processed task: {task} with template: {prompt_template}")


def test_prompt_rollout_preserves_executability():
    """Test that @prompt_rollout decorated functions remain executable."""
    test_task = "Hello World"
    test_template = PromptTemplate(template="Test prompt", engine="f-string")

    # Function should be callable
    assert callable(sample_prompt_rollout_func)

    # Function should execute and return expected result
    result = sample_prompt_rollout_func(test_task, test_template)
    expected = f"Processed task: {test_task} with template: {test_template}"
    assert result == expected


def test_prompt_rollout_preserves_metadata():
    """Test that @prompt_rollout preserves function metadata."""
    # Function name should be preserved
    assert sample_prompt_rollout_func.__name__ == "sample_prompt_rollout_func"  # type: ignore

    # Docstring should be preserved
    assert sample_prompt_rollout_func.__doc__ == "A test function with prompt_rollout decorator."


def test_prompt_rollout_returns_litagent_instance():
    """Test that @prompt_rollout returns a FunctionalLitAgent instance."""
    assert isinstance(sample_prompt_rollout_func, FunctionalLitAgent)

    # Should have agent methods
    assert hasattr(sample_prompt_rollout_func, "rollout")
    assert hasattr(sample_prompt_rollout_func, "rollout_async")
    assert hasattr(sample_prompt_rollout_func, "training_rollout")


def test_prompt_rollout_preserves_signature():
    """Test that @prompt_rollout preserves function signature."""
    sig = inspect.signature(sample_prompt_rollout_func)
    params = list(sig.parameters.keys())

    # Should have the expected parameters
    assert params == ["task", "prompt_template"]


def test_prompt_rollout_with_rollout_method():
    """Test that @prompt_rollout works with the rollout method."""

    @prompt_rollout
    def agent_with_prompt(task: Any, prompt_template: PromptTemplate) -> float:
        """An agent that uses prompt templates."""
        return cast(float, f"Template: {prompt_template.template}, Task: {task}")

    # Create resources with a PromptTemplate
    template = PromptTemplate(template="Test prompt: {input}", engine="f-string")
    resources = cast(NamedResources, {"prompt": template})

    # Create a rollout object
    rollout_obj = AttemptedRollout(
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

    # Run rollout
    result = agent_with_prompt.rollout("my task", resources, rollout_obj)
    assert "Test prompt: {input}" in str(result)
    assert "my task" in str(result)


@pytest.mark.asyncio
async def test_async_function_with_prompt_rollout():
    """Test that async functions work with @prompt_rollout decorator."""

    @prompt_rollout
    async def async_agent(task: Any, prompt_template: PromptTemplate) -> float:
        """An async test function."""
        return cast(float, f"Async processed: {task} with {prompt_template.template}")

    # Should be callable
    assert callable(async_agent)

    # Should preserve async nature when called directly
    template = PromptTemplate(template="Test", engine="f-string")
    result = await async_agent("test", template)
    assert result == "Async processed: test with Test"

    # Should be marked as async
    assert async_agent.is_async()


def test_rollout_detects_prompt_template_pattern():
    """Test that @rollout decorator detects prompt_template pattern."""

    @rollout
    def prompt_agent(task: Any, prompt_template: PromptTemplate) -> float:
        """An agent using prompt templates."""
        return cast(float, f"Task: {task}, Template: {prompt_template.template}")

    # Should be a FunctionalLitAgent
    assert isinstance(prompt_agent, FunctionalLitAgent)

    # Should work when called directly
    template = PromptTemplate(template="Test prompt", engine="f-string")
    result = prompt_agent("test task", template)
    assert "test task" in str(result)
    assert "Test prompt" in str(result)


def test_function_with_rollout_parameter():
    """Test that functions can accept an optional rollout parameter."""

    @llm_rollout
    def agent_with_rollout(task: Any, llm: LLM, rollout: Any) -> float:
        """An agent that accepts rollout parameter."""
        return cast(float, f"Task: {task}, LLM: {llm}, Rollout: {rollout.rollout_id}")

    # Should accept rollout parameter
    assert agent_with_rollout._accepts_rollout()

    # Create resources
    llm = LLM(endpoint="http://test", model="test-model")
    resources = cast(NamedResources, {"llm": llm})

    # Create a rollout object
    rollout_obj = AttemptedRollout(
        rollout_id="rollout-789",
        input="test task",
        start_time=0.0,
        attempt=Attempt(
            rollout_id="rollout-789",
            attempt_id="attempt-abc",
            sequence_id=1,
            start_time=0.0,
        ),
    )

    # Run rollout - should pass rollout parameter
    result = agent_with_rollout.rollout("my task", resources, rollout_obj)
    assert "rollout-789" in str(result)


def test_prompt_rollout_with_rollout_parameter():
    """Test that prompt_rollout functions can accept an optional rollout parameter."""

    @prompt_rollout
    def agent_with_rollout(task: Any, prompt_template: PromptTemplate, rollout: Any) -> float:
        """An agent that accepts rollout parameter."""
        return cast(float, f"Task: {task}, Template: {prompt_template.template}, Rollout: {rollout.rollout_id}")

    # Should accept rollout parameter
    assert agent_with_rollout._accepts_rollout()

    # Create resources
    template = PromptTemplate(template="Test prompt", engine="f-string")
    resources = cast(NamedResources, {"prompt": template})

    # Create a rollout object
    rollout_obj = AttemptedRollout(
        rollout_id="rollout-xyz",
        input="test task",
        start_time=0.0,
        attempt=Attempt(
            rollout_id="rollout-xyz",
            attempt_id="attempt-def",
            sequence_id=1,
            start_time=0.0,
        ),
    )

    # Run rollout - should pass rollout parameter
    result = agent_with_rollout.rollout("my task", resources, rollout_obj)
    assert "rollout-xyz" in str(result)


# Tests for validation functions


def test_validate_llm_rollout_func_valid():
    """Test that _validate_llm_rollout_func accepts valid functions."""

    def valid_func(task: Any, llm: LLM) -> Any:
        return None

    # Should not raise
    assert _validate_llm_rollout_func(valid_func)


def test_validate_llm_rollout_func_with_rollout():
    """Test that _validate_llm_rollout_func accepts functions with rollout parameter."""

    def valid_func(task: Any, llm: LLM, rollout: Any) -> Any:
        return None

    # Should not raise
    assert _validate_llm_rollout_func(valid_func)


def test_validate_llm_rollout_func_missing_task():
    """Test that _validate_llm_rollout_func rejects functions without 'task' as first param."""

    def invalid_func(input: Any, llm: LLM) -> Any:
        return None

    # Should raise ValueError
    with pytest.raises(ValueError, match="must be a positional parameter called 'task'"):
        _validate_llm_rollout_func(invalid_func)


def test_validate_llm_rollout_func_missing_llm():
    """Test that _validate_llm_rollout_func rejects functions without 'llm' parameter."""

    def invalid_func(task: Any, model: LLM) -> Any:
        return None

    # Should raise ValueError
    with pytest.raises(ValueError, match="must have a positional parameter called 'llm'"):
        _validate_llm_rollout_func(invalid_func)


def test_validate_llm_rollout_func_too_few_params():
    """Test that _validate_llm_rollout_func rejects functions with too few parameters."""

    def invalid_func(task: Any) -> Any:
        return None

    # Should raise ValueError
    with pytest.raises(ValueError, match="must have at least 2 parameters"):
        _validate_llm_rollout_func(invalid_func)


def test_validate_prompt_rollout_func_valid():
    """Test that _validate_prompt_rollout_func accepts valid functions."""

    def valid_func(task: Any, prompt_template: PromptTemplate) -> Any:
        return None

    # Should not raise
    assert _validate_prompt_rollout_func(valid_func)


def test_validate_prompt_rollout_func_with_rollout():
    """Test that _validate_prompt_rollout_func accepts functions with rollout parameter."""

    def valid_func(task: Any, prompt_template: PromptTemplate, rollout: Any) -> Any:
        return None

    # Should not raise
    assert _validate_prompt_rollout_func(valid_func)


def test_validate_prompt_rollout_func_missing_task():
    """Test that _validate_prompt_rollout_func rejects functions without 'task' as first param."""

    def invalid_func(input: Any, prompt_template: PromptTemplate) -> Any:
        return None

    # Should raise ValueError
    with pytest.raises(ValueError, match="must be a positional parameter called 'task'"):
        _validate_prompt_rollout_func(invalid_func)


def test_validate_prompt_rollout_func_missing_prompt_template():
    """Test that _validate_prompt_rollout_func rejects functions without 'prompt_template' parameter."""

    def invalid_func(task: Any, template: PromptTemplate) -> Any:
        return None

    # Should raise ValueError
    with pytest.raises(ValueError, match="must have a positional parameter called 'prompt_template'"):
        _validate_prompt_rollout_func(invalid_func)


def test_rollout_decorator_invalid_signature():
    """Test that @rollout raises NotImplementedError for invalid signatures."""

    with pytest.raises(NotImplementedError, match="does not match any known agent patterns"):

        @rollout
        def invalid_agent(task: Any, unknown_param: Any) -> Any:  # type: ignore
            return None
