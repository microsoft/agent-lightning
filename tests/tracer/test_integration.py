# Copyright (c) Microsoft. All rights reserved.

"""
Integration tests for various agent frameworks with AgentLightning.

This module tests the integration of AgentLightning with:

- Autogen AgentChat
- LangChain/LangGraph
- OpenAI Agent SDK
- AgentOps
- Reward tracking functionality

Uses real agent frameworks but defaults to a mock OpenAI API server.

Set `USE_OPENAI=true`, plus `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables to run
against the real API with an OpenAI model of your choice (`gpt-4.1-nano` by default).
"""

import difflib
import json
import os
import pprint
import re
import textwrap
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Literal, Mapping, Optional, Tuple, Union

import litellm
import pytest
import requests
from agents import Agent, AgentHooks, GuardrailFunctionOutput, InputGuardrail, RunConfig, Runner, function_tool
from agents.mcp import MCPServerStdio
from agents.models.openai_provider import OpenAIProvider
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from fastapi import FastAPI
from openai import OpenAI
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel

from agentlightning.adapter.triplet import TracerTraceToTriplet, TraceTree
from agentlightning.reward import reward
from agentlightning.tracer import Tracer
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import Span
from agentlightning.utils.server_launcher import PythonServerLauncher, PythonServerLauncherArgs

try:
    import langchain  # type: ignore

    LANGCHAIN_INSTALLED = True
except ImportError:
    LANGCHAIN_INSTALLED = False  # type: ignore

if TYPE_CHECKING or LANGCHAIN_INSTALLED:
    from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
    from langchain.chat_models import init_chat_model
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase
    from langchain_core.messages import AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool  # pyright: ignore[reportUnknownVariableType]
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, MessagesState, StateGraph

USE_OPENAI = os.environ.get("USE_OPENAI", "false").lower() == "true"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if USE_OPENAI:
    assert (
        OPENAI_BASE_URL is not None and OPENAI_API_KEY is not None
    ), "OPENAI_BASE_URL and OPENAI_API_KEY must be set when USE_OPENAI is true"


@dataclass
class OpenAISettings:
    base_url: str
    api_key: str
    model: str


class MockOpenAICompatibleServer:
    """
    A mock server that mimics the OpenAI Chat Completions API for testing purposes.
    It provides deterministic, canned responses based on the content of the prompt.
    Now supports replaying from prompt caches.
    """

    def __init__(self) -> None:
        self.app = FastAPI()
        self.server_thread = None
        self.server = None
        self._prev_openai_base_url: Optional[str] = None
        self.prompt_caches = self._load_prompt_caches()
        self._setup_routes()

        self._server_launcher = PythonServerLauncher(self.app, PythonServerLauncherArgs(launch_mode="thread"))

    def _prompt_cache_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "../assets/prompt_caches.jsonl")

    def _load_prompt_caches(self):
        cache_path = self._prompt_cache_path()
        caches: List[Dict[str, Any]] = []
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    try:
                        caches.append(json.loads(line))
                    except Exception:
                        continue
        return caches

    def _find_best_cache_match(self, request_dict: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the cached request with the highest similarity to the incoming request.
        Returns (response, similarity_score) or (None, 0.0) if not found.
        """

        def normalize_messages(msgs: List[Dict[str, Any]]) -> str:
            # Flatten messages to a string for comparison
            if not msgs:
                return ""
            return "\n".join(f"{m.get('role','')}:{m.get('content','')}" for m in msgs)

        req_msgs = request_dict.get("messages", [])
        req_tools = request_dict.get("tools", "")
        req_str = normalize_messages(req_msgs) + f"\ntools:{req_tools}"

        best_score = 0.0
        best_response = None
        for cache in self.prompt_caches:
            cache_req = cache.get("request", {})
            cache_msgs = cache_req.get("messages", [])
            cache_tools = cache_req.get("tools", "")
            cache_str = normalize_messages(cache_msgs) + f"\ntools:{cache_tools}"

            # Use difflib for quick ratio
            score = difflib.SequenceMatcher(None, req_str, cache_str).ratio()
            if score > best_score:
                best_score = score
                best_response = cache.get("response")
        return best_response, best_score

    def _setup_routes(self):
        @self.app.get("/health")
        def health_check():  # pyright: ignore[reportUnusedFunction]
            return {"status": "ok"}

        @self.app.post("/v1/chat/completions")
        def chat_completions(request: Dict[str, Any]):  # pyright: ignore[reportUnusedFunction]
            if USE_OPENAI:
                assert OPENAI_BASE_URL is not None and OPENAI_API_KEY is not None
                # Call Real OpenAI API to get prompt cache
                response = requests.post(
                    OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
                    json=request,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                )
                if response.status_code != 200:
                    raise ValueError(f"Failed to call OpenAI API: {response.status_code} {response.text}")
                response_dict = response.json()
                with open(self._prompt_cache_path(), "a") as f:
                    f.write(json.dumps({"request": request, "response": response_dict}) + "\n")
                return response_dict

            # Try to find the best match in prompt caches
            cached_response, score = self._find_best_cache_match(request)
            if cached_response and score > 0.8:
                time.sleep(0.1)  # Simulate network delay
                # Return the cached response directly
                cached_response["prompt_token_ids"] = [1, 2, 3]
                cached_response["response_token_ids"] = [[4, 5, 6]]
                return cached_response
            raise ValueError("No suitable cached response found. Please ensure the prompt caches are populated.")

    async def __aenter__(self):
        # Start the server manually
        await self._server_launcher.start()
        return OpenAISettings(
            base_url=f"{self._server_launcher.access_endpoint}/v1",
            api_key="dummy",
            model=OPENAI_MODEL,
        )

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._server_launcher.stop()


async def agent_pure_openai(settings: OpenAISettings, tracer: Tracer) -> None:
    """A simple agent using the `openai` library."""
    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    response = client.chat.completions.create(
        model=settings.model, messages=[{"role": "user", "content": "What is the capital of France?"}]
    )
    assert "Paris" in response.choices[0].message.content  # type: ignore


async def agent_litellm(settings: OpenAISettings, tracer: Tracer) -> None:
    """Agent using `litellm` to call the mock server."""
    response = litellm.completion(  # type: ignore
        model="openai/" + settings.model,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        base_url=settings.base_url,
        api_key=settings.api_key,
    )
    assert "4" in response.choices[0].message.content  # type: ignore


async def agent_langchain(settings: OpenAISettings, tracer: Tracer) -> None:
    """A simple LangChain agent."""
    llm = ChatOpenAI(model=settings.model, openai_api_base=settings.base_url, openai_api_key=settings.api_key)  # type: ignore
    prompt = ChatPromptTemplate.from_messages([("human", "{input}")])  # type: ignore
    chain = prompt | llm | StrOutputParser()  # type: ignore
    result = chain.invoke({"input": "What is the capital of France?"})  # type: ignore
    assert "Paris" in result


async def agent_langchain_tooluse(settings: OpenAISettings, tracer: Tracer) -> None:
    """A LangChain agent that uses a calculator tool."""

    @tool
    def multiply(a_and_b: str) -> int:
        """A simple calculator tool that multiplies two integers."""
        a, b = re.search(r"(\d+).*?(\d+)", a_and_b).groups()  # type: ignore
        return int(a) * int(b)

    llm = ChatOpenAI(
        model=settings.model,
        temperature=0,
        openai_api_base=settings.base_url,  # type: ignore
        openai_api_key=settings.api_key,  # type: ignore
        disable_streaming=True,
    )
    tools = [multiply]
    agent = create_agent(  # type: ignore
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the multiply tool to answer math questions.",
    )
    langchain_callback_handler = tracer.get_langchain_handler()
    result = agent.invoke(  # type: ignore
        {"messages": [{"role": "user", "content": "what is 42 * 12"}]},
        {"callbacks": [langchain_callback_handler]} if langchain_callback_handler else None,
    )
    assert "504" in result["messages"][-1].content


async def agent_langgraph(settings: OpenAISettings, tracer: Tracer) -> None:
    """An agent built with LangGraph for stateful, cyclical workflows."""
    llm = init_chat_model(
        "openai:" + settings.model, openai_api_base=settings.base_url, openai_api_key=settings.api_key
    )
    db = SQLDatabase.from_uri("sqlite:///" + os.path.join(os.path.dirname(__file__), "../assets/chinook.db"))  # type: ignore
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    def get_tool(name: str) -> Any:
        return next(t for t in tools if t.name == name)

    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

    def get_schema(state: MessagesState) -> MessagesState:
        """Execute the get_schema tool based on the last message's tool calls."""
        last_message = state["messages"][-1]
        tool_messages: List[Any] = []
        for tool_call in getattr(last_message, "tool_calls", []):
            result = get_schema_tool.invoke(tool_call)  # type: ignore
            tool_messages.append(result)
        return {"messages": tool_messages}

    def run_query(state: MessagesState) -> MessagesState:
        """Execute the run_query tool based on the last message's tool calls."""
        last_message = state["messages"][-1]
        tool_messages: List[Any] = []
        for tool_call in getattr(last_message, "tool_calls", []):
            result = run_query_tool.invoke(tool_call)  # type: ignore
            tool_messages.append(result)
        return {"messages": tool_messages}

    def list_tables(state: MessagesState) -> MessagesState:
        tool_call: Dict[str, Any] = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "abc123",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])

        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        tool_message = list_tables_tool.invoke(tool_call)  # type: ignore
        response = AIMessage(f"Available tables: {tool_message.content}")

        return {"messages": [tool_call_message, tool_message, response]}

    def call_get_schema(state: MessagesState) -> MessagesState:
        # Note that LangChain enforces that all models accept `tool_choice="any"`
        # as well as `tool_choice=<string name of tool>`.
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")  # type: ignore
        response = llm_with_tools.invoke(state["messages"])

        return {"messages": [response]}

    # Generate SQL Query
    def generate_query(state: MessagesState) -> MessagesState:
        prompt = f"""
    You are an agent for SQL ({db.dialect}).
    Write a query to answer the user. Limit results to 5. Do not modify data.
    """
        msg = {"role": "system", "content": prompt}
        llm_with_tools = llm.bind_tools([get_tool("sql_db_query")])  # type: ignore
        resp = llm_with_tools.invoke([msg] + state["messages"])
        return {"messages": [resp]}

    # Double-check SQL Query
    def check_query(state: MessagesState) -> MessagesState:
        prompt = f"""
    You are a SQL expert. Double check the following {db.dialect} query for mistakes.
    Rewrite if needed. Otherwise, output as is.
    """
        user_query = state["messages"][-1].tool_calls[0]["args"]["query"]  # type: ignore
        llm_with_tools = llm.bind_tools([get_tool("sql_db_query")], tool_choice="any")  # type: ignore
        resp = llm_with_tools.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": user_query}])
        resp.id = state["messages"][-1].id  # keep consistent ID for trace
        return {"messages": [resp]}

    # Conditional edge: if query tool-call exists, check query, else done
    def should_continue(state: MessagesState) -> Literal[END, "check_query"]:  # type: ignore
        last = state["messages"][-1]
        return "check_query" if getattr(last, "tool_calls", None) else END

    # 5. Build the agent graph
    builder = StateGraph(MessagesState)
    builder.add_node(list_tables)  # type: ignore
    builder.add_node(call_get_schema)  # type: ignore
    builder.add_node(get_schema)  # type: ignore
    builder.add_node(generate_query)  # type: ignore
    builder.add_node(check_query)  # type: ignore
    builder.add_node(run_query)  # type: ignore
    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges(
        "generate_query",
        should_continue,  # type: ignore
    )
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")
    agent = builder.compile()  # type: ignore

    # 6. Run a sample question
    question = "Which sales agent made the most in sales in 2009?"
    langchain_callback_handler = tracer.get_langchain_handler()
    result = agent.invoke(  # type: ignore
        {"messages": [{"role": "user", "content": question}]},  # type: ignore
        {"callbacks": [langchain_callback_handler]} if langchain_callback_handler else None,
    )
    assert "Steve Johnson" in result["messages"][-1].content
    assert len(result["messages"]) > 5


async def agent_autogen_multiagent(settings: OpenAISettings, tracer: Tracer) -> None:
    """A multi-agent conversation with AutoGen."""

    model_client = OpenAIChatCompletionClient(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
    )

    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    text_termination = TextMentionTermination("APPROVE")

    # Create a team with the primary and critic agents.
    team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination, max_turns=4)

    result = await team.run(task="Write a short poem about the fall season.")
    sources = [msg.source for msg in result.messages]
    assert "primary" in sources
    assert "critic" in sources


async def agent_autogen_mcp(settings: OpenAISettings, tracer: Tracer) -> None:
    """An AutoGen agent using the Multi-agent Conversation Platform (MCP) and a tool (fixed usage)."""
    calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])

    async with McpWorkbench(calculator_mcp_server) as workbench:
        model_client = OpenAIChatCompletionClient(
            model=settings.model,
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
        agent = AssistantAgent(name="calc_agent", model_client=model_client, workbench=workbench)
        # Simulate a tool-use message
        response = await agent.run(task="What is 42 * 12?")
        assert "504" in response.messages[-1].content  # type: ignore


def openai_agents_sdk_run_config(settings: OpenAISettings) -> RunConfig:
    return RunConfig(
        model=settings.model,
        model_provider=OpenAIProvider(api_key=settings.api_key, base_url=settings.base_url, use_responses=False),
    )


async def openai_agents_sdk_eval_hook_and_guardrail(settings: OpenAISettings, tracer: Tracer) -> None:
    class HomeworkOutput(BaseModel):
        is_homework: bool
        reasoning: str

    class EvalHook(AgentHooks):
        @reward
        def evaluate(self, context: Any, agent: Agent, output: Any):
            # Custom reward logic: reward if the answer contains 'homework'
            return 1.0 if output and "no" in str(output).lower() else 0.0

        async def on_end(self, context: Any, agent: Agent, output: Any):
            nonlocal final_reward
            final_reward = final_reward or self.evaluate(context, agent, output)

    guardrail_agent = Agent(
        name="Guardrail check",
        instructions="Check if the user is asking about homework.",
        output_type=HomeworkOutput,
        hooks=EvalHook(),
    )

    async def homework_guardrail(ctx: Any, agent: Agent, input_data: Any):
        result = await Runner.run(
            guardrail_agent, input_data, context=ctx.context, run_config=openai_agents_sdk_run_config(settings)
        )
        final_output = result.final_output_as(HomeworkOutput)
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=not final_output.is_homework,
        )

    main_agent = Agent(
        name="Main Agent",
        instructions="Answer questions. If it's about homework, say so.",
        input_guardrails=[InputGuardrail(guardrail_function=homework_guardrail)],
        hooks=EvalHook(),
    )
    final_reward = None
    result = await Runner.run(
        main_agent,
        "The teacher asks to answer whether hummingbirds are mammals.",
        run_config=openai_agents_sdk_run_config(settings),
    )
    # Should trigger the guardrail and reward should be 1.0
    assert final_reward == 1.0, f"Expected reward to be 1.0, got {final_reward}"
    assert hasattr(result, "final_output")


async def openai_agents_sdk_mcp_tool_use(settings: OpenAISettings, tracer: Tracer) -> None:
    async with MCPServerStdio(params={"command": "uvx", "args": ["mcp-server-calculator"]}) as mcp_server:
        agent = Agent(
            name="MCP Tool Agent",
            instructions="Use the tools to answer the question.",
            mcp_servers=[mcp_server],
        )
        # The actual tool list and invocation will depend on the MCP server implementation
        # Here we just check that the agent can run with the MCP server attached
        result = await Runner.run(agent, "What is 43*57?", run_config=openai_agents_sdk_run_config(settings))
        assert hasattr(result, "final_output")
        assert "2451" in result.final_output_as(str)


async def openai_agents_sdk_handoff_tool_output_type_and_reward(settings: OpenAISettings, tracer: Tracer) -> None:

    class MathOutput(BaseModel):
        answer: int

    @function_tool
    def add(a: int, b: int) -> int:
        return a + b

    class RewardHook(AgentHooks):
        @reward
        async def evaluate(self, context: Any, agent: Agent, output: Any):
            # Use another agent to check the answer and compute reward
            checker = Agent(
                name="Checker",
                instructions="Return 1.0 if the answer is 8, else 0.0.",
                output_type=float,
            )
            result = await Runner.run(
                checker, str(getattr(output, "answer", "")), run_config=openai_agents_sdk_run_config(settings)
            )
            return float(result.final_output)

        async def on_end(self, context: Any, agent: Agent, output: Any):
            nonlocal final_reward
            final_reward = await self.evaluate(context, agent, output)

    math_agent = Agent(
        name="MathAgent",
        instructions="Add two numbers.",
        tools=[add],
        output_type=MathOutput,
        hooks=RewardHook(),
    )

    history_agent = Agent(
        name="HistoryAgent",
        instructions="Answer history questions.",
        output_type=str,
    )

    triage_agent = Agent(
        name="TriageAgent",
        instructions="If the question is about math, handoff to MathAgent. Otherwise, handoff to HistoryAgent.",
        handoffs=[math_agent, history_agent],
    )

    # Math handoff
    final_reward = None
    result = await Runner.run(triage_agent, "What is 3+5?", run_config=openai_agents_sdk_run_config(settings))
    assert isinstance(result.final_output, MathOutput)
    assert result.final_output.answer == 8
    # The reward should be 1.0 (computed by the checker agent)
    assert final_reward == 1.0
    # History handoff
    result2 = await Runner.run(
        triage_agent, "Who was the first president of the US?", run_config=openai_agents_sdk_run_config(settings)
    )
    assert isinstance(result2.final_output, str)
    assert "president" in result2.final_output.lower()


AgentName = Literal[
    "agent_pure_openai",
    "agent_litellm",
    "agent_langchain",
    "agent_langchain_tooluse",
    "agent_langgraph",
    "agent_autogen_multiagent",
    "agent_autogen_mcp",
    "openai_agents_sdk_eval_hook_and_guardrail",
    "openai_agents_sdk_mcp_tool_use",
    "openai_agents_sdk_handoff_tool_output_type_and_reward",
]


AGENTOPS_EXPECTED_TREES: Mapping[AgentName, List[Tuple[str, str]]] = {
    "agent_pure_openai": [("openai.chat.completion", "openai.chat.completion")],
    "agent_litellm": [("openai.chat.completion", "openai.chat.completion")],
    "agent_langchain": [("openai.chat.completion", "openai.chat.completion")],
    "agent_langchain_tooluse": [
        ("chat_model.llm", "openai.chat.completion"),
        ("chat_model.llm", "openai.chat.completion"),
    ],
    "agent_langgraph": [
        ("call_get_schema", "openai.chat.completion"),
        ("generate_query", "openai.chat.completion"),
        ("check_query", "openai.chat.completion"),
        ("run_query", "tool.tool"),
    ],
    "agent_autogen_multiagent": [
        ("primary", "openai.chat.completion"),
        ("critic", "openai.chat.completion"),
    ],
    "agent_autogen_mcp": [
        ("calc_agent", "openai.chat.completion"),
    ],
    "openai_agents_sdk_eval_hook_and_guardrail": [
        ("homework_guardrail", "openai.chat.completion"),
        ("Main Agent", "openai.chat.completion"),
        ("Main Agent", "agentops_reward_operation.task"),
    ],
    "openai_agents_sdk_mcp_tool_use": [
        ("MCP Tool Agent", "openai.chat.completion"),
        ("MCP Tool Agent", "calculate"),
        ("MCP Tool Agent", "openai.chat.completion"),
    ],
    "openai_agents_sdk_handoff_tool_output_type_and_reward": [
        ("TriageAgent", "openai.chat.completion"),
        ("MathAgent", "openai.chat.completion"),
        ("MathAgent", "openai.chat.completion"),
        ("MathAgent", "agentops_reward_operation.task"),
        ("HistoryAgent", "openai.chat.completion"),
    ],
}

AGENTOPS_EXPECTED_TRIPLETS_NUMBER: Mapping[AgentName, int] = {
    "agent_pure_openai": 1,
    "agent_litellm": 1,
    "agent_langchain": 1,
    "agent_langchain_tooluse": 2,
    "agent_langgraph": 4,
    "agent_autogen_multiagent": 4,
    "agent_autogen_mcp": 1,
    "openai_agents_sdk_eval_hook_and_guardrail": 2,
    "openai_agents_sdk_mcp_tool_use": 2,
    "openai_agents_sdk_handoff_tool_output_type_and_reward": 5,
}

AGENTOPS_EXPECTED_REWARDS: Mapping[AgentName, Union[List[float | None], Tuple[List[float | None], ...]]] = {
    "openai_agents_sdk_eval_hook_and_guardrail": ([1.0, None], [None, 1.0]),
    "openai_agents_sdk_handoff_tool_output_type_and_reward": [None, None, 1.0, None, None],
}


AGENT_FUNCTIONS: Mapping[AgentName, Callable[[OpenAISettings, Tracer], Awaitable[Any]]] = {
    "agent_pure_openai": agent_pure_openai,
    "agent_litellm": agent_litellm,
    "agent_langchain": agent_langchain,
    "agent_langchain_tooluse": agent_langchain_tooluse,
    "agent_langgraph": agent_langgraph,
    "agent_autogen_multiagent": agent_autogen_multiagent,
    "agent_autogen_mcp": agent_autogen_mcp,
    "openai_agents_sdk_eval_hook_and_guardrail": openai_agents_sdk_eval_hook_and_guardrail,
    "openai_agents_sdk_mcp_tool_use": openai_agents_sdk_mcp_tool_use,
    "openai_agents_sdk_handoff_tool_output_type_and_reward": openai_agents_sdk_handoff_tool_output_type_and_reward,
}


def assert_expected_pairs_in_tree(root_tuple: Tuple[str, List[Any]], expected_pairs: List[Tuple[str, str]]) -> None:
    """
    Assert that every (ancestor_name, child_name) pair in `expected_pairs`
    occurs somewhere in the tree produced by TraceTree.names_tuple().
    """

    # Collect every node's full path from root → node
    paths: list[tuple[str, ...]] = []  # e.g. [["root", "A", "B"], …]

    def _collect(node_tuple: tuple[str, Any], prefix: list[str]):
        name, children = node_tuple
        cur_path = prefix + [name]
        paths.append(tuple(cur_path))
        for child in children:
            _collect(child, cur_path)

    _collect(root_tuple, [])

    # Greedy—but safe—matching of each expected pair
    used_child_paths: set[tuple[str, ...]] = set()

    for anc_name, child_name in expected_pairs:
        matched = False
        for p in paths:
            if child_name not in p[-1] or tuple(p) in used_child_paths:
                continue
            if any(anc_name in pv for pv in p[:-1]):  # ancestor appears anywhere above
                used_child_paths.add(tuple(p))
                matched = True
                break
        if not matched:
            raise AssertionError(
                f"Expected ancestor/child pair ({anc_name!r}, {child_name!r}) "
                "not found or child already matched.\n"
                f"Root tuple: {pprint.pformat(root_tuple)}\n",
                f"Expected pairs: {expected_pairs}",
            )


@pytest.mark.agentops
@pytest.mark.parametrize("agent_name", list(AGENT_FUNCTIONS.keys()), ids=str)
@pytest.mark.asyncio
async def test_tracer_integration_agentops(agent_name: AgentName):
    if ("langchain" in agent_name or "langgraph" in agent_name) and not LANGCHAIN_INSTALLED:
        pytest.skip("LangChain is not installed. Skip langchain related tests.")

    async with MockOpenAICompatibleServer() as settings:
        tracer = AgentOpsTracer()
        await _run_tracer_with_agent(settings, tracer, agent_name)


@pytest.mark.weave
@pytest.mark.parametrize("agent_name", list(AGENT_FUNCTIONS.keys()), ids=str)
@pytest.mark.asyncio
async def test_tracer_integration_weave(agent_name: AgentName):
    if ("langchain" in agent_name or "langgraph" in agent_name) and not LANGCHAIN_INSTALLED:
        pytest.skip("LangChain is not installed. Skip langchain related tests.")

    from agentlightning.tracer.weave import WeaveTracer

    async with MockOpenAICompatibleServer() as settings:
        tracer = WeaveTracer()
        await _run_tracer_with_agent(settings, tracer, agent_name)


async def _run_tracer_with_agent(settings: OpenAISettings, tracer: Tracer, agent_name: AgentName):
    agent_func = AGENT_FUNCTIONS[agent_name]

    with tracer.lifespan():
        async with tracer.trace_context(name=f"test_integration_{agent_name}"):
            await agent_func(settings, tracer)

        last_trace_normalized = [
            Span.from_opentelemetry(span, "dummy", "dummy", 0) if isinstance(span, ReadableSpan) else span
            for span in tracer.get_last_trace()
        ]
        for span in last_trace_normalized:
            print(">>> rollout_id =", span.rollout_id)
            print("... attempt_id =", span.attempt_id)
            print("... sequence_id =", span.sequence_id)
            print("... trace_id =", span.trace_id)
            print("... span_id =", span.span_id)
            print("... parent_id =", span.parent_id)
            print("... name =", span.name)
            print("... status =", span.status)
            print(
                "... attributes =",
                textwrap.indent(pprint.pformat(span.attributes, width=200, indent=4), "    ").lstrip(),
            )
        tree = TraceTree.from_spans(last_trace_normalized)

        # Visualize the trace tree for debug
        debug_dir = os.path.join(os.path.dirname(__file__), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        tree.visualize(filename=os.path.join(debug_dir, f"{tracer.__class__.__name__}_{agent_name}"))

        tree.repair_hierarchy()

        assert_expected_pairs_in_tree(tree.names_tuple(), AGENTOPS_EXPECTED_TREES[agent_name])

        triplets = TracerTraceToTriplet().adapt(last_trace_normalized)
        assert (
            len(triplets) == AGENTOPS_EXPECTED_TRIPLETS_NUMBER[agent_name]
        ), f"Expected {AGENTOPS_EXPECTED_TRIPLETS_NUMBER[agent_name]} triplets, but got: {triplets}"
        if agent_name in AGENTOPS_EXPECTED_REWARDS:
            expected_reward = AGENTOPS_EXPECTED_REWARDS[agent_name]
            if isinstance(expected_reward, tuple):
                # If the expected rewards are a tuple, make sure at least one of them matches
                assert any([r.reward in expected for r in triplets for expected in expected_reward]), (
                    f"Expected rewards {expected_reward}, " f"but got: {pprint.pformat(triplets)}"
                )
            else:
                assert [r.reward for r in triplets] == expected_reward, (
                    f"Expected rewards {expected_reward}, " f"but got: {pprint.pformat(triplets)}"
                )
