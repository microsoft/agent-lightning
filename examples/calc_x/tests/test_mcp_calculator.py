# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os

import openai
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def assemble_tool_calls_from_stream(stream):
    """从流式响应中组装完整的工具调用结果
    
    vLLM 在处理非流式工具调用时会崩溃，但流式响应可以正常工作。
    这个函数将流式响应组装成与非流式响应相同的格式。
    """
    tool_calls_dict = {}  # 按 index 存储工具调用
    
    for chunk in stream:
        if not chunk.choices:
            continue
            
        choice = chunk.choices[0]
        delta = choice.delta
        
        # 处理工具调用
        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                idx = tool_call_delta.index
                if idx not in tool_calls_dict:
                    tool_calls_dict[idx] = {
                        "id": tool_call_delta.id or "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": ""
                        }
                    }
                
                # 更新工具调用信息
                tool_call = tool_calls_dict[idx]
                if tool_call_delta.id:
                    tool_call["id"] = tool_call_delta.id
                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        tool_call["function"]["name"] = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        tool_call["function"]["arguments"] += tool_call_delta.function.arguments
    
    # 组装成 OpenAI 格式的响应
    tool_calls_list = []
    for idx in sorted(tool_calls_dict.keys()):
        tool_call_data = tool_calls_dict[idx]
        
        # 创建工具调用对象，使其可以通过属性访问
        class MockToolCall:
            def __init__(self, data):
                self.id = data["id"]
                self.type = data["type"]
                self.function = MockFunction(data["function"])
        
        class MockFunction:
            def __init__(self, func_data):
                self.name = func_data["name"]
                self.arguments = func_data["arguments"]
        
        tool_calls_list.append(MockToolCall(tool_call_data))
    
    # 创建类似非流式响应的结构
    class MockMessage:
        def __init__(self):
            self.role = "assistant"
            self.content = None
            self.tool_calls = tool_calls_list
    
    class MockChoice:
        def __init__(self):
            self.message = MockMessage()
            self.finish_reason = "tool_calls"
    
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
    
    return MockResponse()


async def main():
    # 1. Initialize OpenAI client
    client = openai.OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # 2. Prepare MCP stdio connection to the calculator server
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-calculator"],
    )

    # 3. Ask the LLM to calculate an expression via a function call
    # 注意：vLLM 在处理非流式工具调用时会崩溃，必须使用流式响应
    stream = client.chat.completions.create(
        model="/data/aj/llm_model/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": "What is 31415926 * 11415789?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string", "description": "The expression to calculate"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
        stream=True,  # 必须使用流式响应，因为 vLLM 的非流式工具调用会崩溃
    )
    
    # 从流式响应中组装完整的工具调用结果
    chat_resp = assemble_tool_calls_from_stream(stream)
    print(f"组装的响应: tool_calls={chat_resp.choices[0].message.tool_calls}")

    # 4. Extract the expression argument
    func_call = chat_resp.choices[0].message.tool_calls[0]  # type: ignore
    expr = json.loads(func_call.function.arguments)["expression"]  # type: ignore

    # 5. Connect to the MCP server and invoke the 'calculate' tool
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized.")
            result = await session.call_tool("calculate", arguments={"expression": expr})
            # The structured result is under `.structuredContent`
            value = result.structuredContent["result"]  # type: ignore

    # 6. Print out the result
    print(f"{expr} = {value}")


if __name__ == "__main__":
    asyncio.run(main())
