"""
Integration tests for various agent frameworks with AgentLightning.

This module tests the integration of AgentLightning with:
- Autogen AgentChat
- LangChain/LangGraph
- OpenAI Agent SDK
- AgentOps
- Reward tracking functionality

Uses real agent frameworks but mocks only the OpenAI API server.
"""

import pytest
import inspect
import json
import asyncio
import time
import os
import httpx
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional
import threading
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from agentlightning.tracer.agentops import AgentOpsTracer, LightningSpanProcessor
from agentlightning.tracer.triplet import TripletExporter
from agentlightning.reward import reward
from agentlightning.types import Triplet

import openai
from openai import OpenAI, AsyncOpenAI

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain.agents import tool, AgentExecutor, create_react_agent

from langgraph.graph import StateGraph, END, MessagesState
from typing_extensions import TypedDict

import autogen_agentchat
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

from agents import Agent, Runner, AgentHooks, InputGuardrail, GuardrailFunctionOutput, function_tool
from agents.mcp import MCPServerStdio

import litellm

import agentops


OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
OPENAI_MODEL = "gpt-4-mock"
OPENAI_API_KEY = "token-abc123"


class AgentState(TypedDict):
    """State for LangGraph agents."""
    messages: List[BaseMessage]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None


class MockOpenAICompatibleServer:
    """
    A mock server that mimics the OpenAI Chat Completions API for testing purposes.
    It provides deterministic, canned responses based on the content of the prompt.
    """

    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.server_thread = None
        self.server = None
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            last_message_content = request.messages[-1]["content"].lower() if request.messages else ""
            response_text = "This is a default mock response."
            tool_calls = None
            finish_reason = "stop"

            # Check if tools are provided in the request to decide on a tool-calling path.
            if request.tools:
                if "what is 42 * 12" in last_message_content:
                    tool_calls = [
                        {
                            "id": "call_calculator_123",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"a": 42, "b": 12}'},
                        }
                    ]
                elif "search for the weather in sf" in last_message_content:
                    tool_calls = [
                        {
                            "id": "call_search_456",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "weather in SF"}'},
                        }
                    ]
                else:
                    # For ReAct agents, provide a proper action format
                    response_text = """I need to use a tool to solve this problem.

Action: calculator
Action Input: {"a": 42, "b": 12}"""
            else:
                # Handle standard, non-tool-use queries
                if "capital of france" in last_message_content:
                    response_text = "The capital of France is Paris."
                elif "what is 2 + 2" in last_message_content:
                    response_text = "Claro! 2 + 2 is 4."
                elif "thank you note" in last_message_content:
                    response_text = "You're welcome! Here is a thank you note to your friend for the wonderful gift."
                elif "what is 42 * 12" in last_message_content:
                    # For ReAct agents without tool calls, provide the action format
                    response_text = """I need to calculate 42 * 12.

Action: calculator
Action Input: {"a": 42, "b": 12}"""

            # According to OpenAI API, when a tool is called, content is null.
            if tool_calls:
                response_text = None
                finish_reason = "tool_calls"

            mock_choice = {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": finish_reason,
            }
            if tool_calls:
                mock_choice["message"]["tool_calls"] = tool_calls

            response_data = {
                "id": f"chatcmpl-mock-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [mock_choice],
                "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
                # The instrumentation specifically looks for these keys.
                "prompt_token_ids": list(range(15)),
                "response_token_ids": [list(range(25))],
            }

            if request.stream:
                from fastapi.responses import StreamingResponse

                async def stream_generator():
                    # Simplified stream for testing purposes
                    chunk = {
                        "id": response_data["id"],
                        "choices": [{"delta": {"role": "assistant", "content": response_text or ""}}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_generator(), media_type="text/plain")

            return response_data

    async def __aenter__(self):
        # Start the server manually
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()

        # Wait for server to start
        max_wait = 10  # seconds
        wait_time = 0
        while not getattr(self.server, 'started', False) and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1

        if not getattr(self.server, 'started', False):
            raise RuntimeError("Server failed to start within timeout")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.server:
            self.server.should_exit = True
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)


async def run_agent(agent_func):
    """
    Run an agent function with mock server, handling both sync and async functions.

    This function starts a mock OpenAI server and then detects whether the agent
    function is async or sync, executing it appropriately within the server context.

    Args:
        agent_func: The agent function to execute (sync or async)

    Returns:
        The result of the agent function execution
    """
    async with MockOpenAICompatibleServer():
        # Check if the function is async
        if inspect.iscoroutinefunction(agent_func):
            # Handle async function - run directly since we're already in async context
            return await agent_func()
        else:
            # Handle sync function - run without threading
            return agent_func()


def agent_pure_openai():
    """A simple agent using the `openai` library."""
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL, messages=[{"role": "user", "content": "What is the capital of France?"}]
    )
    assert "Paris" in response.choices[0].message.content


def agent_litellm():
    """Agent using `litellm` to call the mock server."""
    response = litellm.completion(
        model="openai/" + OPENAI_MODEL,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
    assert "4" in response.choices[0].message.content


def agent_langchain():
    """A simple LangChain agent."""
    llm = ChatOpenAI(model=OPENAI_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"input": "What is the capital of France?"})
    assert "Paris" in result


def agent_langchain_tooluse():
    """A LangChain agent that uses a calculator tool."""

    @tool
    def calculator(a: int, b: int) -> int:
        """A simple calculator tool that multiplies two integers."""
        return a * b

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    tools = [calculator]
    agent = create_react_agent(llm, tools, hub.pull("hwchase17/react"))
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": "what is 42 * 12"})
    assert "504" in result["output"]


def agent_langgraph():
    """An agent built with LangGraph for stateful, cyclical workflows."""
    # Simplified LangGraph agent that just uses a basic chain
    llm = ChatOpenAI(model=OPENAI_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    def call_model(state: MessagesState):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": messages + [response]}

    # Define the graph structure
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.set_entry_point("agent")
    workflow.set_finish_point("agent")
    app = workflow.compile()

    inputs = {"messages": [HumanMessage(content="search for the weather in sf")]}
    result = app.invoke(inputs)
    assert "weather" in result["messages"][-1].content.lower()


async def agent_autogen_multiagent():
    """A multi-agent conversation with AutoGen (fixed usage)."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

    model_client = OpenAIChatCompletionClient(
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
    assistant = AssistantAgent(name="assistant", model_client=model_client)
    user_proxy = UserProxyAgent(name="user_proxy", model_client=model_client)
    # Use a simple chat
    result = await user_proxy.a_chat(assistant, message="Write a thank you note to my friend.")
    assert "thank you" in result[-1]["content"].lower()


async def agent_autogen_mcp():
    """An AutoGen agent using the Multi-agent Conversation Platform (MCP) and a tool (fixed usage)."""
    calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])

    async with McpWorkbench(calculator_mcp_server) as workbench:
        model_client = OpenAIChatCompletionClient(
            model=OPENAI_MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        )
        agent = AssistantAgent(name="calc_agent", model_client=model_client, workbench=workbench)
        # Simulate a tool-use message
        response = await agent.run(task="What is 42 * 12?")
        assert "calculator" in response.content.lower()


async def openai_agents_sdk_eval_hook_and_guardrail():
    class HomeworkOutput(BaseModel):
        is_homework: bool
        reasoning: str

    class EvalHook(AgentHooks):
        @reward
        def evaluate(self, context, agent, output):
            # Custom reward logic: reward if the answer contains 'homework'
            return 1.0 if output and "homework" in str(output).lower() else 0.0

        async def on_end(self, context, agent, output):
            return self.evaluate(context, agent, output)

    guardrail_agent = Agent(
        name="Guardrail check",
        instructions="Check if the user is asking about homework.",
        output_type=HomeworkOutput,
        hooks=EvalHook(),
    )

    async def homework_guardrail(ctx, agent, input_data):
        result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
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
    result = await Runner.run(main_agent, "Is this homework?")
    # Should trigger the guardrail and reward should be 1.0
    assert result.reward == 1.0
    assert hasattr(result, "final_output")


async def openai_agents_sdk_mcp_tool_use():
    # Simulate a local MCP server (replace with a real one if available)
    async with MCPServerStdio(params={"command": "echo", "args": ["MCP server running"]}) as mcp_server:
        agent = Agent(
            name="MCP Tool Agent",
            instructions="Use the tools to answer the question.",
            mcp_servers=[mcp_server],
        )
        # The actual tool list and invocation will depend on the MCP server implementation
        # Here we just check that the agent can run with the MCP server attached
        result = await Runner.run(agent, "What is 1+1?")
        assert result is not None


async def openai_agents_sdk_handoff_tool_output_type_and_reward():

    class MathOutput(BaseModel):
        answer: int

    @function_tool
    def add(a: int, b: int) -> int:
        return a + b

    class RewardHook(AgentHooks):
        @reward
        async def evaluate(self, context, agent, output):
            # Use another agent to check the answer and compute reward
            checker = Agent(
                name="Checker",
                instructions="Return 1.0 if the answer is 8, else 0.0.",
                output_type=float,
            )
            result = await Runner.run(checker, str(getattr(output, "answer", "")))
            return float(result.final_output)

        async def on_end(self, context, agent, output):
            return await self.evaluate(context, agent, output)

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
    result = await Runner.run(triage_agent, "What is 3+5?")
    assert isinstance(result.final_output, MathOutput)
    assert result.final_output.answer == 8
    # The reward should be 1.0 (computed by the checker agent)
    assert result.reward == 1.0
    # History handoff
    result2 = await Runner.run(triage_agent, "Who was the first president of the US?")
    assert isinstance(result2.final_output, str)
    assert "president" in result2.final_output.lower()


if __name__ == '__main__':
    # Test sync agent
    asyncio.run(run_agent(agent_pure_openai))
    asyncio.run(run_agent(agent_litellm))
    asyncio.run(run_agent(agent_langchain))
    asyncio.run(run_agent(agent_langchain_tooluse))
    asyncio.run(run_agent(agent_langgraph))
    # Test async agents
    asyncio.run(run_agent(agent_autogen_multiagent))
    asyncio.run(run_agent(agent_autogen_mcp))
    asyncio.run(run_agent(openai_agents_sdk_eval_hook_and_guardrail))
    asyncio.run(run_agent(openai_agents_sdk_mcp_tool_use))