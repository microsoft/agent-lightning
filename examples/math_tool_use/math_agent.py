# Copyright (c) Microsoft. All rights reserved.

"""
Math Reasoning Agent with Calculator Tool

This example demonstrates training an agent to solve grade school math problems
using reinforcement learning. The agent learns to:
1. Break down word problems into steps
2. Use a calculator tool for arithmetic
3. Provide accurate final answers

This is a beginner-friendly example showing Agent-Lightning's core features
with minimal setup requirements.
"""

from __future__ import annotations

import json
# import re
from typing import Any, cast

from openai import AsyncOpenAI

from agentlightning import (
    LLM,
    LitAgent,
    NamedResources,
    Trainer,
    setup_logging,
)
from calculator_tool import calculator_tool
from utils import compute_reward, extract_answer, normalize_number

setup_logging()

# System prompt that teaches the agent how to solve math problems

MATH_AGENT_PROMPT = """You are a helpful assistant that solves grade school math problems step by step.

When solving a problem:
1. Read the problem carefully and identify what is being asked
2. Break down the problem into smaller steps
3. Use the calculator tool for any arithmetic operations (addition, subtraction, multiplication, division)
4. Show your reasoning for each step
5. Provide your final answer wrapped in <answer></answer> tags

Example format:
Problem: "Sarah has 5 apples. She buys 3 more. How many apples does she have?"

Solution:
Let me solve this step by step:
1. Sarah starts with 5 apples
2. She buys 3 more apples
3. I need to add 5 + 3

<tool_call>
{"name": "calculator", "arguments": {"expression": "5 + 3"}}
</tool_call>

Based on the calculation, Sarah has 8 apples.

<answer>8</answer>

Available tools:
- calculator: Evaluates mathematical expressions. Use it for any arithmetic.
  Example: {"name": "calculator", "arguments": {"expression": "24 * 7 + 15"}}

Remember:
- Always use the calculator for arithmetic operations
- Always wrap your final numerical answer in <answer></answer> tags
- Show your step-by-step reasoning

"""


class MathAgent(LitAgent[Any]):

    """
    A math reasoning agent that uses reinforcement learning to improve its
    problem-solving abilities.
    
    The agent learns to:
    - Use the calculator tool effectively
    - Generate well-structured reasoning
    - Provide accurate final answers
    """

    def __init__(self, trained_agents: str | None = None) -> None:
        """
        Initialize the MathAgent.
        
        Args:
            trained_agents: Optional path to previously trained agent checkpoints
        """
        super().__init__(trained_agents=trained_agents)
        
        self.tools = [calculator_tool]
        self.max_iterations = 5  # Maximum tool calls per problem

    async def _call_llm_with_tools(
        self, 
        client: AsyncOpenAI,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Call the LLM with tool support.
        
        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing the response and tool calls
        """
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluates a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate (e.g., '5 + 3 * 2')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }],
            temperature=temperature,
            max_tokens=1024,
        )
        
        return {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls,
            "finish_reason": response.choices[0].finish_reason,
        }

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool execution
        """
        if tool_name == "calculator":
            try:
                expression = arguments.get("expression", "")
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        return "Unknown tool"

    async def training_rollout_async(
        self, 
        task: Any, 
        rollout_id: str, 
        resources: NamedResources
    ) -> float:
        """
        Execute a single training rollout.
        
        This method:
        1. Receives a math problem from the task
        2. Attempts to solve it using the agent
        3. Computes a reward based on the solution quality
        4. Returns the reward for RL optimization
        
        Args:
            task: Dictionary containing 'question' and 'answer'
            rollout_id: Unique identifier for this rollout
            resources: Named resources including the LLM endpoint
            
        Returns:
            Reward value (float between -1 and 1)
        """
        # Get the LLM configuration from resources
        llm: LLM = cast(LLM, resources.get("main_llm"))
        
        # Create OpenAI client pointing to the training endpoint
        client = AsyncOpenAI(
            base_url=llm.endpoint,
            api_key="dummy-key"  # Not used for local vLLM
        )
        
        # Initialize conversation with the math problem
        messages = [
            {"role": "system", "content": MATH_AGENT_PROMPT},
            {"role": "user", "content": f"Problem: {task['question']}"}
        ]
        
        used_calculator = False
        conversation_log = []
        
        # Agent interaction loop
        for iteration in range(self.max_iterations):
            # Get response from LLM
            response = await self._call_llm_with_tools(
                client=client,
                messages=messages,
                model=llm.model,
                temperature=0.7,
            )
            
            # Log the response
            if response["content"]:
                conversation_log.append(f"Assistant: {response['content']}")
                messages.append({
                    "role": "assistant",
                    "content": response["content"]
                })
            
            # Check if agent made tool calls
            if response["tool_calls"]:
                used_calculator = True
                
                for tool_call in response["tool_calls"]:
                    # Parse tool call
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    # Execute tool
                    tool_result = self._execute_tool_call(tool_name, arguments)
                    conversation_log.append(
                        f"Tool Call: {tool_name}({arguments}) = {tool_result}"
                    )
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                        "name": tool_name
                    })
                
                # Continue conversation after tool use
                continue
            
            # No more tool calls - check for final answer
            if response["finish_reason"] == "stop":
                break
        
        # Extract final response
        final_response = messages[-1]["content"] if messages else ""
        
        # Extract the answer from <answer> tags
        predicted_answer = extract_answer(final_response)
        ground_truth = str(task["answer"])
        
        # Compute reward
        reward = compute_reward(
            predicted=predicted_answer,
            ground_truth=ground_truth,
            used_calculator=used_calculator,
            full_response=final_response
        )
        
        # Log results for debugging
        if rollout_id.endswith("0"):  # Log every 10th example
            print(f"\n{'='*60}")
            print(f"Question: {task['question']}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Predicted: {predicted_answer}")
            print(f"Used Calculator: {used_calculator}")
            print(f"Reward: {reward:.3f}")
            print(f"{'='*60}\n")
        
        return reward

    async def validation_rollout_async(
        self, 
        task: Any, 
        rollout_id: str, 
        resources: NamedResources
    ) -> float:
        """
        Execute a validation rollout (same as training but with greedy decoding).
        
        Args:
            task: Dictionary containing 'question' and 'answer'
            rollout_id: Unique identifier for this rollout
            resources: Named resources including the LLM endpoint
            
        Returns:
            Reward value (float between -1 and 1)
        """
        # Use greedy decoding for validation (temperature=0)
        llm: LLM = cast(LLM, resources.get("main_llm"))
        validation_resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0.0},  # Greedy
            )
        }
        return await self.training_rollout_async(
            task, rollout_id, validation_resources
        )


if __name__ == "__main__":
    """
    Entry point for the math agent training.
    
    This starts multiple agent workers that:
    1. Connect to the Lightning Server at localhost:9999
    2. Receive math problems to solve
    3. Execute solutions and report rewards
    4. Get updated with improved model weights
    """
    print("Starting Math Agent Training")
    print("=" * 60)
    print("Configuration:")
    print("  - Workers: 8")
    print("  - Server: http://localhost:9999/")
    print("  - Dataset: GSM8K grade school math")
    print("=" * 60)
    
    # Create and train the agent
    # The Trainer handles:
    # - Distributing tasks to workers
    # - Collecting trajectories
    # - Coordinating with the training server
    trainer = Trainer(n_workers=8)
    agent = MathAgent()
    
    trainer.fit_v0(
        agent=agent,
        server_url="http://localhost:9999/"
    )