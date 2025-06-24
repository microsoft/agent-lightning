import pytest
import requests
import json
import aiohttp
import pickle
import os
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

from agentlightning.tracer.http import HttpTracer
from agentlightning.tracer.agentops import AgentOpsTracer, LightningSpanProcessor
from agentlightning.tracer.triplet import TripletExporter, TraceTree, RewardMatchPolicy
from agentlightning.tracer.base import BaseTracer
from agentlightning.types import Triplet
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry import trace as trace_api


@pytest.fixture
def http_tracer():
    return HttpTracer(include_headers=True, include_body=True, include_agentlightning_requests=True)


@pytest.fixture
def agentops_tracer():
    return AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)


@pytest.fixture
def triplet_exporter():
    return TripletExporter(
        repair_hierarchy=True,
        llm_call_match=r"openai\.chat\.completion",
        agent_match=None,
        exclude_llm_call_in_reward=True,
        reward_match=RewardMatchPolicy.FIRST_OCCURRENCE,
    )


@pytest.fixture
def mock_readable_span():
    """Create a mock ReadableSpan for testing."""

    def _create_span(
        span_id: int = 1,
        trace_id: int = 1,
        name: str = "test_span",
        attributes: Optional[Dict[str, Any]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        parent_span_id: Optional[int] = None,
    ):
        attributes = attributes or {}
        start_time = start_time or int(time.time() * 1_000_000_000)  # nanoseconds
        end_time = end_time or start_time + 1_000_000_000  # 1 second later

        span = Mock(spec=ReadableSpan)
        span.name = name
        span.attributes = attributes
        span.start_time = start_time
        span.end_time = end_time

        # Mock span context
        span_context = Mock()
        span_context.span_id = span_id
        span_context.trace_id = trace_id
        span_context.trace_flags = Mock()
        span_context.trace_flags.sampled = True
        span.context = span_context
        span.get_span_context.return_value = span_context

        # Mock parent
        if parent_span_id:
            parent = Mock()
            parent.span_id = parent_span_id
            span.parent = parent
        else:
            span.parent = None

        # Mock to_json method
        span.to_json.return_value = json.dumps(
            {
                "name": name,
                "context": {
                    "span_id": trace_api.format_span_id(span_id),
                    "trace_id": trace_api.format_trace_id(trace_id),
                },
                "attributes": attributes,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

        return span

    return _create_span


@pytest.fixture
def sample_llm_spans(mock_readable_span):
    """Create sample LLM call spans for testing."""
    return [
        mock_readable_span(
            span_id=1,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
                "gen_ai.response.id": "resp_1",
                "http.method": "POST",
                "http.status_code": 200,
            },
        ),
        mock_readable_span(
            span_id=2,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [7, 8, 9],
                "response_token_ids": [10, 11, 12],
                "gen_ai.response.id": "resp_2",
                "http.method": "POST",
                "http.status_code": 200,
            },
        ),
    ]


@pytest.fixture
def sample_agent_spans(mock_readable_span):
    """Create sample agent spans for testing."""
    return [
        mock_readable_span(
            span_id=3,
            name="agent_task",
            attributes={
                "agentops.span.kind": "agent",
                "operation.name": "test_agent",
                "agent.name": "TestAgent",
            },
        ),
        mock_readable_span(
            span_id=4,
            name="autogen_agent",
            attributes={
                "recipient_agent_type": "AutogenAgent",
            },
        ),
    ]


@pytest.fixture
def sample_reward_spans(mock_readable_span):
    """Create sample reward spans for testing."""
    return [
        mock_readable_span(
            span_id=5,
            name="reward_span",
            attributes={
                "agentops.task.output": json.dumps({"type": "reward", "value": 0.8}),
            },
        ),
        mock_readable_span(
            span_id=6,
            name="reward_span_2",
            attributes={
                "agentops.entity.output": {"type": "reward", "value": -0.2},
            },
        ),
    ]


# Legacy fixture for backward compatibility
@pytest.fixture
def tracer(http_tracer):
    return http_tracer


class TestAgentOpsTracer:
    """Test suite for AgentOpsTracer functionality."""

    def test_agentops_tracer_initialization(self):
        """Test AgentOpsTracer initialization with different configurations."""
        # Test default initialization
        tracer = AgentOpsTracer()
        assert tracer.agentops_managed is True
        assert tracer.instrument_managed is True
        assert tracer.daemon is True
        assert tracer._lightning_span_processor is None

    def test_agentops_tracer_initialization_custom_config(self):
        """Test AgentOpsTracer initialization with custom configuration."""
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=False)
        assert tracer.agentops_managed is False
        assert tracer.instrument_managed is False
        assert tracer.daemon is False

    def test_agentops_tracer_pickling(self):
        """Test that AgentOpsTracer can be pickled and unpickled."""
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)

        # Test pickling
        pickled = pickle.dumps(tracer)
        unpickled = pickle.loads(pickled)

        assert isinstance(unpickled, AgentOpsTracer)
        assert unpickled.agentops_managed is False
        assert unpickled.instrument_managed is False
        assert unpickled.daemon is True

        # Server manager should be None after unpickling
        assert unpickled._agentops_server_manager is None

    def test_agentops_tracer_getstate_setstate(self):
        """Test the __getstate__ and __setstate__ methods for pickling."""
        tracer = AgentOpsTracer(agentops_managed=True, instrument_managed=True, daemon=True)
        tracer._agentops_server_port_val = 8080

        # Test __getstate__
        state = tracer.__getstate__()
        assert "_agentops_server_manager" not in state or state["_agentops_server_manager"] is None
        assert state["_agentops_server_port_val"] == 8080

        # Test __setstate__
        new_tracer = AgentOpsTracer.__new__(AgentOpsTracer)
        new_tracer.__setstate__(state)
        assert new_tracer._agentops_server_port_val == 8080
        assert new_tracer._agentops_server_manager is None

    @patch("agentlightning.tracer.agentops.agentops")
    def test_agentops_tracer_init_worker_managed(self, mock_agentops):
        """Test init_worker when agentops_managed is True."""
        mock_client = Mock()
        mock_client.initialized = False
        mock_agentops.get_client.return_value = mock_client

        tracer = AgentOpsTracer(agentops_managed=True, instrument_managed=False, daemon=True)
        tracer._agentops_server_port_val = 8080

        # Store original environment
        original_env = dict(os.environ)
        try:
            # Clear environment and run init_worker
            os.environ.clear()
            tracer.init_worker(worker_id=1)

            # Check environment variables were set
            assert os.environ["AGENTOPS_API_KEY"] == "dummy"
            assert os.environ["AGENTOPS_API_ENDPOINT"] == "http://localhost:8080"
            assert os.environ["AGENTOPS_APP_URL"] == "http://localhost:8080/notavailable"
            assert os.environ["AGENTOPS_EXPORTER_ENDPOINT"] == "http://localhost:8080/traces"

            # Check agentops.init was called
            mock_agentops.init.assert_called_once()
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    @patch("agentlightning.tracer.agentops.agentops")
    def test_agentops_tracer_init_worker_not_managed(self, mock_agentops):
        """Test init_worker when agentops_managed is False."""
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)

        tracer.init_worker(worker_id=1)

        # agentops.init should not be called
        mock_agentops.init.assert_not_called()

    def test_agentops_tracer_trace_context_not_initialized(self):
        """Test trace_context raises error when not initialized."""
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)

        with pytest.raises(RuntimeError, match="LightningSpanProcessor is not initialized"):
            with tracer.trace_context():
                pass

    def test_agentops_tracer_get_last_trace_not_initialized(self):
        """Test get_last_trace raises error when not initialized."""
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)

        with pytest.raises(RuntimeError, match="LightningSpanProcessor is not initialized"):
            tracer.get_last_trace()

    @patch("agentops.integration.callbacks.langchain.LangchainCallbackHandler")
    @patch("agentops.get_client")
    def test_agentops_tracer_get_langchain_callback_handler(self, mock_get_client, mock_handler):
        """Test getting Langchain callback handler."""
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)

        # Mock the client
        mock_client = Mock()
        mock_client.initialized = True
        mock_client.config.api_key = "test_api_key"
        mock_get_client.return_value = mock_client

        result = tracer.get_langchain_callback_handler(tags=["test_tag"])

        mock_handler.assert_called_once_with(api_key="test_api_key", tags=["test_tag"])
        assert result == mock_handler.return_value

    @patch("agentops.integration.callbacks.langchain.LangchainCallbackHandler")
    @patch("agentops.get_client")
    def test_agentops_tracer_get_langchain_callback_handler_not_initialized(self, mock_get_client, mock_handler):
        """Test getting Langchain callback handler when client not initialized."""
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)

        # Mock the client
        mock_client = Mock()
        mock_client.initialized = False
        mock_get_client.return_value = mock_client

        result = tracer.get_langchain_callback_handler()

        mock_handler.assert_called_once_with(api_key=None, tags=[])
        assert result == mock_handler.return_value

    def test_agentops_tracer_picklable(self):
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)
        pickled = pickle.dumps(tracer)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, AgentOpsTracer)
        # Should be able to call trace_context (will raise NotImplementedError if not implemented)
        with pytest.raises(RuntimeError):
            with unpickled.trace_context():
                pass


class TestLightningSpanProcessor:
    """Test suite for LightningSpanProcessor functionality."""

    def test_lightning_span_processor_context_manager(self):
        """Test LightningSpanProcessor as context manager."""
        processor = LightningSpanProcessor()

        with processor as p:
            assert p is processor
            assert processor._spans == []

    def test_lightning_span_processor_on_end(self, mock_readable_span):
        """Test span processing on end."""
        processor = LightningSpanProcessor()
        span = mock_readable_span()

        processor.on_end(span)

        assert len(processor.spans()) == 1
        assert processor.spans()[0] == span

    def test_lightning_span_processor_shutdown_and_flush(self):
        """Test shutdown and force_flush methods."""
        processor = LightningSpanProcessor()

        # These should not raise exceptions
        processor.shutdown()
        result = processor.force_flush()
        assert result is True


class TestTripletExporter:
    """Test suite for TripletExporter functionality."""

    def test_triplet_exporter_initialization(self):
        """Test TripletExporter initialization with default parameters."""
        exporter = TripletExporter()

        assert exporter.repair_hierarchy is True
        assert exporter.llm_call_match == r"openai\.chat\.completion"
        assert exporter.agent_match is None
        assert exporter.exclude_llm_call_in_reward is True
        assert exporter.reward_match == RewardMatchPolicy.FIRST_OCCURRENCE

    def test_triplet_exporter_initialization_custom(self):
        """Test TripletExporter initialization with custom parameters."""
        exporter = TripletExporter(
            repair_hierarchy=False,
            llm_call_match=r"custom\.llm\.call",
            agent_match=r"custom_agent",
            exclude_llm_call_in_reward=False,
            reward_match=RewardMatchPolicy.FIRST_SIBLING,
        )

        assert exporter.repair_hierarchy is False
        assert exporter.llm_call_match == r"custom\.llm\.call"
        assert exporter.agent_match == r"custom_agent"
        assert exporter.exclude_llm_call_in_reward is False
        assert exporter.reward_match == RewardMatchPolicy.FIRST_SIBLING

    def test_triplet_exporter_export_basic(self, triplet_exporter, sample_llm_spans):
        """Test basic export functionality."""
        triplets = triplet_exporter.export(sample_llm_spans)

        assert isinstance(triplets, list)
        assert len(triplets) == 2

        for triplet in triplets:
            assert isinstance(triplet, Triplet)
            assert triplet.prompt is not None
            assert triplet.response is not None

    def test_triplet_exporter_export_with_rewards(self, mock_readable_span):
        """Test export functionality with reward spans."""
        # Create LLM call span
        llm_span = mock_readable_span(
            span_id=1,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
                "gen_ai.response.id": "resp_1",
            },
            start_time=1000,
        )

        # Create reward span that comes after LLM call
        reward_span = mock_readable_span(
            span_id=2,
            name="reward_task",
            attributes={
                "agentops.task.output": json.dumps({"type": "reward", "value": 0.8}),
            },
            start_time=2000,
        )

        exporter = TripletExporter(reward_match=RewardMatchPolicy.FIRST_OCCURRENCE)
        triplets = exporter.export([llm_span, reward_span])

        assert len(triplets) == 1
        assert triplets[0].reward == 0.8

    def test_triplet_exporter_export_empty_spans(self, triplet_exporter):
        """Test export with empty span list."""
        with pytest.raises(ValueError, match="No root spans found"):
            triplet_exporter.export([])

    def test_triplet_exporter_export_no_llm_calls(self, triplet_exporter, sample_agent_spans):
        """Test export with no LLM call spans."""
        triplets = triplet_exporter.export(sample_agent_spans)

        assert isinstance(triplets, list)
        assert len(triplets) == 0


class TestTraceTree:
    """Test suite for TraceTree functionality."""

    def test_trace_tree_initialization(self, mock_readable_span):
        """Test TraceTree initialization."""
        span = mock_readable_span()
        tree = TraceTree(id="test_id", span=span)

        assert tree.id == "test_id"
        assert tree.span == span
        assert tree.children == []

    def test_trace_tree_properties(self, mock_readable_span):
        """Test TraceTree start_time and end_time properties."""
        span = mock_readable_span(start_time=1000, end_time=2000)
        tree = TraceTree(id="test_id", span=span)

        assert tree.start_time == 1000
        assert tree.end_time == 2000

    def test_trace_tree_add_child(self, mock_readable_span):
        """Test adding children to TraceTree."""
        parent_span = mock_readable_span(span_id=1)
        child_span = mock_readable_span(span_id=2)

        parent_tree = TraceTree(id="parent", span=parent_span)
        child_tree = TraceTree(id="child", span=child_span)

        parent_tree.add_child(child_tree)

        assert len(parent_tree.children) == 1
        assert parent_tree.children[0] == child_tree

    def test_trace_tree_find_id(self, mock_readable_span):
        """Test finding nodes by ID in TraceTree."""
        parent_span = mock_readable_span(span_id=1)
        child_span = mock_readable_span(span_id=2)

        parent_tree = TraceTree(id="parent", span=parent_span)
        child_tree = TraceTree(id="child", span=child_span)
        parent_tree.add_child(child_tree)

        # Test finding existing nodes
        assert parent_tree.find_id("parent") == parent_tree
        assert parent_tree.find_id("child") == child_tree

        # Test finding non-existing node
        assert parent_tree.find_id("nonexistent") is None

    def test_trace_tree_traverse(self, mock_readable_span):
        """Test traversing TraceTree."""
        parent_span = mock_readable_span(span_id=1)
        child1_span = mock_readable_span(span_id=2)
        child2_span = mock_readable_span(span_id=3)

        parent_tree = TraceTree(id="parent", span=parent_span)
        child1_tree = TraceTree(id="child1", span=child1_span)
        child2_tree = TraceTree(id="child2", span=child2_span)

        parent_tree.add_child(child1_tree)
        parent_tree.add_child(child2_tree)

        traversed = parent_tree.traverse()

        assert len(traversed) == 3
        assert parent_tree in traversed
        assert child1_tree in traversed
        assert child2_tree in traversed

    def test_trace_tree_from_spans_single_root(self, mock_readable_span):
        """Test creating TraceTree from spans with single root."""
        root_span = mock_readable_span(span_id=1, parent_span_id=None)
        child_span = mock_readable_span(span_id=2, parent_span_id=1)

        spans = [root_span, child_span]
        tree = TraceTree.from_spans(spans)

        assert tree.span == root_span
        assert len(tree.children) == 1
        assert tree.children[0].span == child_span

    def test_trace_tree_from_spans_multiple_roots(self, mock_readable_span):
        """Test creating TraceTree from spans with multiple roots."""
        root1_span = mock_readable_span(span_id=1, parent_span_id=None)
        root2_span = mock_readable_span(span_id=2, parent_span_id=None)

        spans = [root1_span, root2_span]
        tree = TraceTree.from_spans(spans)

        # Should create virtual root
        assert tree.span.name == "virtual-root"
        assert len(tree.children) == 2

    def test_trace_tree_agent_name_detection(self, mock_readable_span):
        """Test agent name detection from different span types."""
        # Test OpenAI Agent SDK
        openai_span = mock_readable_span(attributes={"agent.name": "OpenAIAgent"})
        tree = TraceTree(id="test", span=openai_span)
        assert tree.agent_name() == "OpenAIAgent"

        # Test AgentOps decorator
        agentops_span = mock_readable_span(
            attributes={"agentops.span.kind": "agent", "operation.name": "AgentOpsAgent"}
        )
        tree = TraceTree(id="test", span=agentops_span)
        assert tree.agent_name() == "AgentOpsAgent"

        # Test Autogen
        autogen_span = mock_readable_span(attributes={"recipient_agent_type": "AutogenAgent"})
        tree = TraceTree(id="test", span=autogen_span)
        assert tree.agent_name() == "AutogenAgent"

        # Test LangGraph
        langgraph_span = mock_readable_span(attributes={"langchain.chain.type": "LangGraphAgent"})
        tree = TraceTree(id="test", span=langgraph_span)
        assert tree.agent_name() == "LangGraphAgent"

        # Test no agent
        regular_span = mock_readable_span(attributes={})
        tree = TraceTree(id="test", span=regular_span)
        assert tree.agent_name() is None

    def test_trace_tree_reward_detection(self, mock_readable_span):
        """Test reward detection from span attributes."""
        # Test reward with agentops.task.output
        reward_span1 = mock_readable_span(
            attributes={"agentops.task.output": json.dumps({"type": "reward", "value": 0.8})}
        )
        tree = TraceTree(id="test", span=reward_span1)
        assert tree.is_reward_span() is True  # Returns True when it's a reward
        assert tree.maybe_reward_dict() == {"type": "reward", "value": 0.8}

        # Test reward with agentops.entity.output
        reward_span2 = mock_readable_span(attributes={"agentops.entity.output": {"type": "reward", "value": -0.2}})
        tree = TraceTree(id="test", span=reward_span2)
        assert tree.is_reward_span() is True  # Returns True when it's a reward
        assert tree.maybe_reward_dict() == {"type": "reward", "value": -0.2}

        # Test non-reward span
        regular_span = mock_readable_span(attributes={})
        tree = TraceTree(id="test", span=regular_span)
        assert not tree.is_reward_span()  # Returns falsy value when no reward
        assert tree.maybe_reward_dict() == {}


class TestTracerIntegration:
    """Test suite for integration between tracers and triplet exporters."""

    @patch("agentlightning.tracer.agentops.agentops")
    def test_agentops_tracer_with_triplet_exporter(self, mock_agentops, mock_readable_span):
        """Test integration between AgentOpsTracer and TripletExporter."""
        # Setup mocks
        mock_client = Mock()
        mock_client.initialized = False
        mock_agentops.get_client.return_value = mock_client

        # Create tracer and exporter
        tracer = AgentOpsTracer(agentops_managed=False, instrument_managed=False, daemon=True)
        exporter = TripletExporter()

        # Mock the span processor and clear any existing spans
        tracer._lightning_span_processor = LightningSpanProcessor()
        tracer._lightning_span_processor._spans = []  # Clear any existing spans

        # Create mock spans
        llm_span = mock_readable_span(
            span_id=1,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
                "gen_ai.response.id": "resp_1",
            },
        )

        # Simulate span collection
        tracer._lightning_span_processor.on_end(llm_span)

        # Get spans and export triplets
        spans = tracer.get_last_trace()
        assert len(spans) == 1  # Verify we only have one span

        triplets = exporter.export(spans)

        assert len(triplets) == 1
        assert triplets[0].prompt == [1, 2, 3]
        assert triplets[0].response == [4, 5, 6]
        assert triplets[0].metadata["response_id"] == "resp_1"

    def test_http_tracer_with_triplet_exporter_no_llm_calls(self):
        """Test HTTP tracer with triplet exporter when no LLM calls are made."""
        tracer = HttpTracer(include_headers=True, include_body=True)
        exporter = TripletExporter()

        # Simulate HTTP request without LLM calls
        with patch("agentlightning.tracer.http.httprecord"):
            with tracer.trace_context():
                pass  # No actual HTTP calls

        spans = tracer.get_last_trace()
        triplets = exporter.export(spans) if spans else []

        assert isinstance(triplets, list)
        assert len(triplets) == 0

    def test_triplet_data_structure_validation(self, mock_readable_span):
        """Test that exported triplets conform to expected data structure."""
        exporter = TripletExporter()

        llm_span = mock_readable_span(
            span_id=1,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2, 3, 4],
                "response_token_ids": [5, 6, 7, 8],
                "gen_ai.response.id": "test_response_id",
            },
        )

        triplets = exporter.export([llm_span])

        assert len(triplets) == 1
        triplet = triplets[0]

        # Validate Triplet structure
        assert isinstance(triplet, Triplet)
        assert hasattr(triplet, "prompt")
        assert hasattr(triplet, "response")
        assert hasattr(triplet, "reward")
        assert hasattr(triplet, "metadata")

        # Validate data types and values
        assert isinstance(triplet.prompt, list)
        assert isinstance(triplet.response, list)
        assert triplet.reward is None  # No reward span provided
        assert isinstance(triplet.metadata, dict)

        # Validate content
        assert triplet.prompt == [1, 2, 3, 4]
        assert triplet.response == [5, 6, 7, 8]
        assert triplet.metadata["response_id"] == "test_response_id"

    def test_complex_trace_hierarchy_with_rewards(self, mock_readable_span):
        """Test complex trace hierarchy with multiple agents and rewards."""
        exporter = TripletExporter(reward_match=RewardMatchPolicy.FIRST_OCCURRENCE)

        # Create a complex hierarchy: Agent -> LLM Call -> Reward
        agent_span = mock_readable_span(
            span_id=1,
            name="agent_task",
            attributes={
                "agentops.span.kind": "agent",
                "operation.name": "TestAgent",
            },
            start_time=1000,
            parent_span_id=None,
        )

        llm_span = mock_readable_span(
            span_id=2,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
                "gen_ai.response.id": "resp_1",
            },
            start_time=2000,
            parent_span_id=1,
        )

        reward_span = mock_readable_span(
            span_id=3,
            name="reward_evaluation",
            attributes={
                "agentops.task.output": json.dumps({"type": "reward", "value": 0.9}),
            },
            start_time=3000,
            parent_span_id=1,
        )

        spans = [agent_span, llm_span, reward_span]
        triplets = exporter.export(spans)

        assert len(triplets) == 1
        triplet = triplets[0]
        assert triplet.prompt == [1, 2, 3]
        assert triplet.response == [4, 5, 6]
        assert triplet.reward == 0.9
        assert triplet.metadata["agent_name"] == "TestAgent"

    def test_multiple_llm_calls_with_different_rewards(self, mock_readable_span):
        """Test multiple LLM calls with different reward assignments."""
        exporter = TripletExporter(reward_match=RewardMatchPolicy.FIRST_OCCURRENCE)

        # First LLM call
        llm_span1 = mock_readable_span(
            span_id=1,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2],
                "response_token_ids": [3, 4],
                "gen_ai.response.id": "resp_1",
            },
            start_time=1000,
        )

        # First reward
        reward_span1 = mock_readable_span(
            span_id=2,
            name="reward_1",
            attributes={
                "agentops.task.output": json.dumps({"type": "reward", "value": 0.5}),
            },
            start_time=2000,
        )

        # Second LLM call
        llm_span2 = mock_readable_span(
            span_id=3,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [5, 6],
                "response_token_ids": [7, 8],
                "gen_ai.response.id": "resp_2",
            },
            start_time=3000,
        )

        # Second reward
        reward_span2 = mock_readable_span(
            span_id=4,
            name="reward_2",
            attributes={
                "agentops.task.output": json.dumps({"type": "reward", "value": 0.8}),
            },
            start_time=4000,
        )

        spans = [llm_span1, reward_span1, llm_span2, reward_span2]
        triplets = exporter.export(spans)

        assert len(triplets) == 2

        # First triplet should get first reward
        assert triplets[0].prompt == [1, 2]
        assert triplets[0].response == [3, 4]
        assert triplets[0].reward == 0.5

        # Second triplet should get second reward
        assert triplets[1].prompt == [5, 6]
        assert triplets[1].response == [7, 8]
        assert triplets[1].reward == 0.8


class TestHttpTracer:
    def test_basic_http_trace(self, http_tracer):
        with http_tracer.trace_context():
            response = requests.get("https://httpbin.org/get")
            assert response.status_code == 200

        spans = http_tracer.get_last_trace()
        assert len(spans) >= 1
        span = next((s for s in spans if "httpbin.org/get" in s.name), None)
        assert span is not None
        assert span.attributes["http.method"] == "GET"
        assert span.attributes["http.status_code"] == 200

    def test_include_headers(self, http_tracer):
        with http_tracer.trace_context():
            response = requests.get("https://httpbin.org/headers", headers={"X-Test-Header": "pytest"})
            assert response.status_code == 200

        spans = http_tracer.get_last_trace()
        span = next((s for s in spans if "httpbin.org/headers" in s.name), None)
        assert span is not None
        # Check that the custom header is present in the span attributes
        assert "http.request.header.x-test-header" in span.attributes

    def test_include_body(self, http_tracer):
        with http_tracer.trace_context():
            response = requests.post("https://httpbin.org/post", data="pytest-body")
            assert response.status_code == 200

        spans = http_tracer.get_last_trace()
        span = next((s for s in spans if "httpbin.org/post" in s.name), None)
        assert span is not None
        # Check that the request body is present in the span attributes
        assert "http.request.body" in span.attributes
        assert b"pytest-body" in span.attributes["http.request.body"]

    def test_agentlightning_request_filtering(self):
        tracer = HttpTracer(include_agentlightning_requests=False)
        with tracer.trace_context():
            # Simulate a request with the AgentLightning header
            response = requests.get("https://httpbin.org/get", headers={"x-agentlightning-client": "true"})
            assert response.status_code == 200

        spans = tracer.get_last_trace()
        if not spans:
            spans = []
        # Should be empty because the request should be filtered out
        assert all("x-agentlightning-client" not in (attr or "") for span in spans for attr in (span.attributes or {}))

    @pytest.mark.asyncio
    async def test_aiohttp_basic_trace(self, http_tracer):
        async with aiohttp.ClientSession() as session:
            with http_tracer.trace_context():
                async with session.get("https://httpbin.org/get") as response:
                    assert response.status == 200
                    await response.text()
        spans = http_tracer.get_last_trace()
        span = next((s for s in spans if "httpbin.org/get" in s.name), None)
        assert span is not None
        assert span.attributes["http.method"] == "GET"
        assert span.attributes["http.status_code"] == 200

    @pytest.mark.asyncio
    async def test_aiohttp_json_request_response(self, http_tracer):
        json_data = {"foo": "bar"}
        async with aiohttp.ClientSession() as session:
            with http_tracer.trace_context():
                async with session.post("https://httpbin.org/post", json=json_data) as response:
                    assert response.status == 200
                    resp_json = await response.json()
                    assert resp_json["json"] == json_data
        spans = http_tracer.get_last_trace()
        span = next((s for s in spans if "httpbin.org/post" in s.name), None)
        assert span is not None
        # Check that the request body contains the JSON
        assert "http.request.body" in span.attributes
        body_bytes = span.attributes["http.request.body"]
        assert b'"foo": "bar"' in body_bytes or b'"foo":"bar"' in body_bytes
        # Parse and check JSON
        parsed_body = json.loads(body_bytes.decode())
        assert parsed_body == json_data
        # Check that the response body contains the JSON
        assert "http.response.body" in span.attributes
        resp_body_bytes = span.attributes["http.response.body"]
        assert b'"foo": "bar"' in resp_body_bytes or b'"foo":"bar"' in resp_body_bytes
        # Parse and check JSON
        parsed_resp_body = json.loads(resp_body_bytes.decode())
        # httpbin returns a JSON object with a 'json' field
        assert parsed_resp_body["json"] == json_data

    def test_requests_json_request_response(self, http_tracer):
        json_data = {"hello": "world"}
        with http_tracer.trace_context():
            response = requests.post("https://httpbin.org/post", json=json_data)
            assert response.status_code == 200
            resp_json = response.json()
            assert resp_json["json"] == json_data
        spans = http_tracer.get_last_trace()
        span = next((s for s in spans if "httpbin.org/post" in s.name), None)
        assert span is not None
        # Check that the request body contains the JSON
        assert "http.request.body" in span.attributes
        body_bytes = span.attributes["http.request.body"]
        assert b'"hello": "world"' in body_bytes or b'"hello":"world"' in body_bytes
        # Parse and check JSON
        parsed_body = json.loads(body_bytes.decode())
        assert parsed_body == json_data
        # Check that the response body contains the JSON
        assert "http.response.body" in span.attributes
        resp_body_bytes = span.attributes["http.response.body"]
        assert b'"hello": "world"' in resp_body_bytes or b'"hello":"world"' in resp_body_bytes
        # Parse and check JSON
        parsed_resp_body = json.loads(resp_body_bytes.decode())
        assert parsed_resp_body["json"] == json_data

    def test_http_tracer_picklable(self):
        tracer = HttpTracer()
        pickled = pickle.dumps(tracer)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, HttpTracer)
        # Should be able to call trace_context (will not raise, but will not record anything)
        with unpickled.trace_context():
            pass


class TestTracerEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_triplet_exporter_invalid_reward_json(self, mock_readable_span):
        """Test TripletExporter handles invalid JSON in reward spans."""
        llm_span = mock_readable_span(
            span_id=1,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
            },
        )

        # Create reward span with invalid JSON
        reward_span = mock_readable_span(
            span_id=2,
            name="reward_span",
            attributes={
                "agentops.task.output": "invalid json {",
            },
        )

        exporter = TripletExporter()
        triplets = exporter.export([llm_span, reward_span])

        # Should still work, just no reward assigned
        assert len(triplets) == 1
        assert triplets[0].reward is None

    def test_triplet_exporter_missing_token_ids(self, mock_readable_span):
        """Test TripletExporter handles missing token IDs gracefully."""
        llm_span = mock_readable_span(
            span_id=1,
            name="openai.chat.completion",
            attributes={
                # Missing prompt_token_ids and response_token_ids
                "gen_ai.response.id": "resp_1",
            },
        )

        exporter = TripletExporter()
        triplets = exporter.export([llm_span])

        assert len(triplets) == 1
        assert triplets[0].prompt == []  # Should default to empty list
        assert triplets[0].response == []  # Should default to empty list

    def test_trace_tree_repair_hierarchy(self, mock_readable_span):
        """Test TraceTree hierarchy repair functionality."""
        # Create spans with incorrect hierarchy
        root_span = mock_readable_span(
            span_id=1,
            name="root",
            start_time=1000,
            end_time=5000,
            parent_span_id=None,
        )

        # This span should be a child of root but isn't initially
        misplaced_span = mock_readable_span(
            span_id=2,
            name="misplaced",
            start_time=2000,
            end_time=3000,
            parent_span_id=None,  # Should be child of root
        )

        spans = [root_span, misplaced_span]
        tree = TraceTree.from_spans(spans)

        # Before repair, should have virtual root with two children
        assert tree.span.name == "virtual-root"
        assert len(tree.children) == 2

        # After repair, misplaced span should be moved under root
        tree.repair_hierarchy()

        # The repair logic only moves direct children of the root
        # Since misplaced_span is within the time bounds of root_span, it should be moved
        # Find the actual root span in the tree
        root_tree = next((child for child in tree.children if child.span.name == "root"), None)
        misplaced_tree = next((child for child in tree.children if child.span.name == "misplaced"), None)

        # After repair, misplaced should be moved under root
        assert root_tree and len(root_tree.children) == 1
        assert root_tree.children[0].span.name == "misplaced"

    def test_trace_tree_reward_matching_policies(self, mock_readable_span):
        """Test different reward matching policies."""
        # Create LLM call
        llm_span = mock_readable_span(
            span_id=1,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
            },
            start_time=1000,
        )

        # Create multiple reward spans
        reward_span1 = mock_readable_span(
            span_id=2,
            name="reward_1",
            attributes={
                "agentops.task.output": json.dumps({"type": "reward", "value": 0.5}),
            },
            start_time=2000,
        )

        reward_span2 = mock_readable_span(
            span_id=3,
            name="reward_2",
            attributes={
                "agentops.task.output": json.dumps({"type": "reward", "value": 0.8}),
            },
            start_time=3000,
        )

        spans = [llm_span, reward_span1, reward_span2]

        # Test FIRST_OCCURRENCE policy
        exporter_first = TripletExporter(reward_match=RewardMatchPolicy.FIRST_OCCURRENCE)
        triplets_first = exporter_first.export(spans)
        assert len(triplets_first) == 1
        assert triplets_first[0].reward == 0.5  # Should get first reward

        # Test FIRST_SIBLING policy
        exporter_sibling = TripletExporter(reward_match=RewardMatchPolicy.FIRST_SIBLING)
        triplets_sibling = exporter_sibling.export(spans)
        assert len(triplets_sibling) == 1
        # Behavior depends on hierarchy, but should not crash

    def test_trace_tree_agent_filtering(self, mock_readable_span):
        """Test agent filtering in LLM call detection."""
        # Create agent span
        agent_span = mock_readable_span(
            span_id=1,
            name="agent_task",
            attributes={
                "agentops.span.kind": "agent",
                "operation.name": "FilteredAgent",
            },
            parent_span_id=None,
        )

        # Create LLM call under agent
        llm_span = mock_readable_span(
            span_id=2,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
            },
            parent_span_id=1,
        )

        spans = [agent_span, llm_span]

        # Test with agent filter that matches
        exporter_match = TripletExporter(agent_match=r"FilteredAgent")
        triplets_match = exporter_match.export(spans)
        assert len(triplets_match) == 1

        # Test with agent filter that doesn't match
        exporter_no_match = TripletExporter(agent_match=r"DifferentAgent")
        triplets_no_match = exporter_no_match.export(spans)
        assert len(triplets_no_match) == 0

    def test_base_tracer_interface(self):
        """Test that BaseTracer defines the correct interface."""
        tracer = BaseTracer()

        # Should raise NotImplementedError for abstract methods
        with pytest.raises(NotImplementedError):
            with tracer.trace_context():
                pass

    @patch("agentlightning.tracer.agentops.AgentOpsServerManager")
    def test_agentops_tracer_server_manager_error_handling(self, mock_server_manager_class):
        """Test AgentOpsTracer error handling for server manager issues."""
        # Mock server manager that fails to start
        mock_server_manager = Mock()
        mock_server_manager.start.return_value = None
        mock_server_manager.get_port.return_value = None
        mock_server_manager.server_process = None
        mock_server_manager_class.return_value = mock_server_manager

        tracer = AgentOpsTracer(agentops_managed=True, instrument_managed=False, daemon=True)

        with pytest.raises(RuntimeError, match="AgentOps server manager indicates server is not running"):
            tracer.init()

    def test_triplet_exporter_custom_llm_call_pattern(self, mock_readable_span):
        """Test TripletExporter with custom LLM call pattern."""
        # Create span that matches custom pattern
        custom_llm_span = mock_readable_span(
            span_id=1,
            name="custom.llm.call",
            attributes={
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4, 5, 6],
            },
        )

        # Create span that doesn't match custom pattern
        standard_llm_span = mock_readable_span(
            span_id=2,
            name="openai.chat.completion",
            attributes={
                "prompt_token_ids": [7, 8, 9],
                "response_token_ids": [10, 11, 12],
            },
        )

        spans = [custom_llm_span, standard_llm_span]

        # Test with custom pattern
        exporter = TripletExporter(llm_call_match=r"custom\.llm\.call")
        triplets = exporter.export(spans)

        # Should only match the custom span
        assert len(triplets) == 1
        assert triplets[0].prompt == [1, 2, 3]
        assert triplets[0].response == [4, 5, 6]
