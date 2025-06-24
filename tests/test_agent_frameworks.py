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

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain.agents import tool, AgentExecutor, create_react_agent

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
        self.app = FastAPI(lifespan=self.lifespan)
        self.server_thread = None
        self.server = None
        self._setup_routes()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Manages the server's startup and shutdown lifecycle."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self.server = uvicorn.Server(config)
        self.server_thread = threading.Thread(target=self.server.run)
        self.server_thread.start()
        while not self.server.started:
            await asyncio.sleep(0.01)
        yield
        self.server.should_exit = True
        self.server_thread.join()

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
                # Handle standard, non-tool-use queries
                if "capital of france" in last_message_content:
                    response_text = "The capital of France is Paris."
                elif "what is 2 + 2" in last_message_content:
                    response_text = "Claro! 2 + 2 is 4."
                elif "thank you note" in last_message_content:
                    response_text = "You're welcome! Here is a thank you note to your friend for the wonderful gift."

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

                async def stream_generator():
                    # Simplified stream for testing purposes
                    chunk = {
                        "id": response_data["id"],
                        "choices": [{"delta": {"role": "assistant", "content": response_text or ""}}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return asyncio.as_completed(stream_generator())

            return response_data

    async def __aenter__(self):
        # This allows the server to be used in an `async with` block
        await self.lifespan.__wrapped__.__aenter__(self.app)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.lifespan.__wrapped__.__aexit__(self.app, exc_type, exc_val, exc_tb)


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
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        api_base=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
    assert "4" in response.choices[0].message.content


def agent_langchain():
    """A simple LangChain agent."""
    llm = ChatOpenAI(model=OPENAI_MODEL, openai_api_base=OPENAI_BASE_URL, openai_api_key=OPENAI_API_KEY)
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

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_base=OPENAI_BASE_URL, openai_api_key=OPENAI_API_KEY)
    tools = [calculator]
    # This prompt includes the necessary `agent_scratchpad` for ReAct agents.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": "what is 42 * 12"})
    assert "504" in result["output"]


def agent_langgraph():
    """An agent built with LangGraph for stateful, cyclical workflows."""

    @tool
    def web_search(query: str):
        """A mock web search tool that returns a fixed string."""
        print(f"--- Searching for: {query} ---")
        return "The weather in SF is sunny."

    tools = [web_search]
    tool_executor = ToolExecutor(tools)
    llm = ChatOpenAI(model=OPENAI_MODEL, openai_api_base=OPENAI_BASE_URL, openai_api_key=OPENAI_API_KEY)

    def call_model(state):
        messages = state["messages"]
        response = llm.invoke(messages, tools=tools)
        return {"messages": state["messages"] + [response]}

    def call_tool(state):
        last_message = state["messages"][-1]
        action = ToolInvocation(
            tool=last_message.tool_calls[0]["name"],
            tool_input=last_message.tool_calls[0]["args"],
        )
        response = tool_executor.invoke(action)
        tool_message = ToolMessage(content=str(response), tool_call_id=last_message.tool_calls[0]["id"])
        return {"messages": state["messages"] + [tool_message]}

    # Define the graph structure
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
    )
    workflow.add_edge("tools", "agent")
    app = workflow.compile()

    inputs = {"messages": [HumanMessage(content="search for the weather in sf")]}
    result = app.invoke(inputs)
    assert "sunny" in result["messages"][-1].content


async def agent_autogen_multiagent():
    """A multi-agent conversation with AutoGen (fixed usage)."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

    config = {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
    }
    model_client = OpenAIChatCompletionClient(**config)
    assistant = AssistantAgent(name="assistant", model_client=model_client)
    user_proxy = UserProxyAgent(name="user_proxy", model_client=model_client)
    # Use a simple chat
    result = await user_proxy.a_chat(assistant, message="Write a thank you note to my friend.")
    assert "thank you" in result[-1]["content"].lower()


async def agent_autogen_mcp():
    """An AutoGen agent using the Multi-agent Conversation Platform (MCP) and a tool (fixed usage)."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.agents import AssistantAgent

    config = {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
    }
    model_client = OpenAIChatCompletionClient(**config)
    agent = AssistantAgent(name="calc_agent", model_client=model_client)
    # Simulate a tool-use message
    response = await agent.a_generate_reply(messages=[HumanMessage(content="What is 42 * 12?")])
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
