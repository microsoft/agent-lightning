"""Test that @llm_rollout and @lit_agent decorators preserve function executability."""

import inspect
import pytest
from agentlightning.litagent import llm_rollout, lit_agent, LitAgentLLM


@llm_rollout
def sample_llm_rollout_func(task, llm):
    """A test function with llm_rollout decorator."""
    return f"Processed task: {task} with LLM: {llm}"


@lit_agent
def sample_lit_agent_func(task, llm):
    """A test function with lit_agent decorator."""
    return f"Processed task: {task} with LLM: {llm}"


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
    assert sample_llm_rollout_func.__name__ == "sample_llm_rollout_func"

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


def test_lit_agent_preserves_executability():
    """Test that @lit_agent decorated functions remain executable."""
    test_task = "Hello World"
    test_llm = "gpt-4"

    # Function should be callable
    assert callable(sample_lit_agent_func)

    # Function should execute and return expected result
    result = sample_lit_agent_func(test_task, test_llm)
    expected = f"Processed task: {test_task} with LLM: {test_llm}"
    assert result == expected


def test_lit_agent_preserves_metadata():
    """Test that @lit_agent preserves function metadata."""
    # Function name should be preserved
    assert sample_lit_agent_func.__name__ == "sample_lit_agent_func"

    # Docstring should be preserved
    assert sample_lit_agent_func.__doc__ == "A test function with lit_agent decorator."


def test_lit_agent_returns_litagent_instance():
    """Test that @lit_agent returns a LitAgent instance (actually LitAgentLLM for this pattern)."""
    assert isinstance(sample_lit_agent_func, LitAgentLLM)

    # Should have agent methods
    assert hasattr(sample_lit_agent_func, "rollout")
    assert hasattr(sample_lit_agent_func, "rollout_async")
    assert hasattr(sample_lit_agent_func, "training_rollout")


def test_lit_agent_preserves_signature():
    """Test that @lit_agent preserves function signature."""
    sig = inspect.signature(sample_lit_agent_func)
    params = list(sig.parameters.keys())

    # Should have the expected parameters
    assert params == ["task", "llm"]


@pytest.mark.asyncio
async def test_async_function_with_llm_rollout():
    """Test that async functions work with @llm_rollout decorator."""

    @llm_rollout
    async def async_agent(task, llm):
        """An async test function."""
        return f"Async processed: {task} with {llm}"

    # Should be callable
    assert callable(async_agent)

    # Should preserve async nature when called directly
    result = await async_agent("test", "llm")
    assert result == "Async processed: test with llm"

    # Should be marked as async
    assert async_agent.is_async


@pytest.mark.asyncio
async def test_async_function_with_lit_agent():
    """Test that async functions work with @lit_agent decorator."""

    @lit_agent
    async def async_agent(task, llm):
        """An async test function."""
        return f"Async processed: {task} with {llm}"

    # Should be callable
    assert callable(async_agent)

    # Should preserve async nature when called directly
    result = await async_agent("test", "llm")
    assert result == "Async processed: test with llm"

    # Should be marked as async
    assert async_agent.is_async
