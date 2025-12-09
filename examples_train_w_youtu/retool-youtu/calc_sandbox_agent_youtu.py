"""ReTool Calc Sandbox Agent with Youtu-Agent and AgentLightning.

This module implements a rollout function for training LLMs to solve mathematical problems
using a Python code interpreter (sandbox) within the ReTool framework. The agent integrates
Youtu-Agent for multi-turn tool calling and AgentLightning for reinforcement learning training.

Overview:
    - ReTool challenges LLMs with tool-integrated reasoning where a Python code interpreter
      is provided to solve mathematical problems
    - This agent uses Youtu-Agent's TrainingAgent to handle multi-turn interactions
    - AgentLightning's rollout decorator enables RL training with reward computation
    - Sandbox execution is managed via MCP (Model Context Protocol) with HTTP/SSE transport

Prerequisites:
    1. vLLM Service: Deploy a vLLM backend server for model inference
       ```bash
       export BASE_MODEL="YOUR_MODEL_PATH"
       bash vllm_deploy.sh
       ```
    
    2. Sandbox Service: Launch SandboxFusion server for code execution
       ```bash
       export CODESNIP_SERVER_URL="YOUR_SANDBOX_URL"
       ```
       See: https://github.com/bytedance/SandboxFusion
    
    3. Environment Variables:
       - OPENAI_BASE_URL: vLLM service endpoint (default: http://127.0.0.1:8033/v1)
       - OPENAI_API_KEY: API key for the model service
       - MODEL_PATH: Model name/identifier
       - CODESNIP_SERVER_URL: Sandbox service URL

Usage:
    Debug/Testing Mode:
        ```bash
        # Set environment variables
        export OPENAI_BASE_URL="http://127.0.0.1:8033/v1"
        export MODEL_PATH="qwen"
        export CODESNIP_SERVER_URL="YOUR_SANDBOX_URL"
        
        # Run debug function
        python calc_sandbox_agent_youtu.py
        ```
    
    Training Mode:
        This rollout function is used by train_calc_sandbox_agent.py for RL training.
        See the training script for full training setup with Store and Runner services.
        
        For single node training:
        ```bash
        bash run_ray.sh examples_train_w_youtu/retool-youtu/run_qwen2.5_7b_single_node.sh
        ```
        
        For multi-node training (e.g., 4 nodes with 32 GPUs):
        ```bash
        bash run_ray.sh examples_train_w_youtu/retool-youtu/run_qwen2.5_7b.sh
        ```

Key Components:
    - RetoolTask: TypedDict defining the structure of training samples
    - calc_sandbox_agent_youtu: Main rollout function decorated with @agl.rollout
    - compute_reward_with_metrics: Calculates reward and accuracy using retool's scoring
    - emit_reward_with_metrics: Emits reward and metrics to AgentLightning tracer
    - debug: Standalone debugging function for testing without full training setup

Data Format:
    Each task (RetoolTask) contains:
        - data_source: Source identifier (e.g., "aime_2024", "aime_2025")
        - question: The raw problem text
        - prompt: List of message dicts with role and content
        - ability: Problem type (e.g., "MATH")
        - reward_model: Dict containing ground_truth answer
        - agent_name: Name of the agent (e.g., "tool_agent")

Reward Computation:
    The agent computes rewards using retool's compute_score function, which evaluates:
        - Exact match accuracy (acc)
        - Overall score considering correctness 

References:
    - ReTool: https://github.com/ReTool-RL/ReTool
    - Training Dataset: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
    - Testing Dataset: https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024
    - Sandbox Service: https://github.com/bytedance/SandboxFusion
"""
import asyncio
import os
import re
from typing import TypedDict, cast
from pathlib import Path

## youtu-agent
from utu.agents import TrainingAgent
from utu.config import AgentConfig, ConfigLoader

## agentlightning
import agentlightning as agl
from agentlightning.types import SpanNames
from agentlightning.emitter.utils import get_tracer

# Import reward calculation from retool
import sys
sys.path.insert(0, str(Path(__file__).parent))
from retool import compute_score

# set proxy
import os
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")



answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""

def emit_reward_with_metrics(reward: float, metrics: dict) -> None:
    """Emit reward along with additional metrics.
    
    This creates a reward span with the primary reward value and additional
    metrics as attributes.
    
    Args:
        reward: Primary reward value
        metrics: Dictionary of additional metrics (e.g., {"acc": 0.8})
    """
    tracer = get_tracer()
    # Create attributes dict with reward and additional metrics
    attributes = {"reward": reward}
    attributes.update(metrics)
    
    # Create a reward span with all attributes
    span = tracer.start_span(SpanNames.REWARD.value, attributes=attributes)
    with span:
        pass
        
class RetoolTask(TypedDict):
    """This TypedDict defines the structure of each training sample.

    This matches the data format returned by CustomRLHFDataset.map_fn in retool.py:
    - data_source: Source of the data (e.g., "aime_2024", "aime_2025")
    - question: The raw problem text
    - prompt: List of message dicts with role and content
    - ability: Problem type (e.g., "MATH")
    - reward_model: Dict containing ground_truth
    - agent_name: Name of the agent (e.g., "tool_agent")
    """

    data_source: str
    question: str
    prompt: list  # List of {"role": str, "content": str}
    ability: str
    reward_model: dict  # {"ground_truth": str}
    agent_name: str


def compute_reward_with_metrics(solution_str: str, ground_truth: str, num_turns: int) -> dict:
    """Calculate reward and metrics using retool's scoring function.
    
    Returns:
        dict with keys: score, acc, pred
    """
    result = compute_score(
        data_source="custom",
        solution_str=solution_str,
        ground_truth=str(ground_truth),
        extra_info={"num_turns": num_turns},
    )
    return result



def get_youtu_agent(openai_base_url, config, model_settings, model_name):
    agent = TrainingAgent(openai_base_url=openai_base_url, config=config, model_settings=model_settings, model=model_name)
    return agent

AGENT_RUN_TIMEOUT = 180  # 5 minutes


@agl.rollout
async def calc_sandbox_agent_youtu(task: RetoolTask, llm: agl.LLM) -> None:
    """Calc-X Sandbox agent rollout function (HTTP/SSE version).

    It accepts a math problem (RetoolTask format) and a LLM endpoint resource.
    It's expected to return None, and emit reward via `agl.emit_reward`.
    
    Features:
    - Automatic multi-turn tool calling (managed by autogen)
    - Sandbox code execution via MCP protocol (HTTP/SSE transport)
    - Timeout protection to prevent hanging
    - Connects to a running MCP server at SANDBOX_MCP_URL
    """
    
    question = task["question"]
    
    
    # ✅ Use McpWorkbench with HTTP/SSE connection
    # Note: The URL string is used directly instead of StdioServerParams
    try:
        config: AgentConfig = ConfigLoader.load_agent_config("retool/qa_python")
        config.model.model_provider.model = llm.model
        config.model.model_provider.base_url = llm.endpoint
        config.model.model_provider.api_key = os.environ.get("OPENAI_API_KEY", "token-abc123")
        config.max_turns = llm.sampling_parameters.get("max_turns", 8)
        config.model.model_settings.temperature = llm.sampling_parameters.get("temperature", 1.0)
        config.model.model_settings.top_p = llm.sampling_parameters.get("top_p", 1.0)
        print("[CalcSandboxHTTP] llm.model =", llm.model, "llm.endpoint =", llm.endpoint)
        agent = get_youtu_agent(llm.endpoint, config, llm.sampling_parameters, llm.model)

        prompt = f"{question}"
        print(f"\n[CalcSandboxHTTP] Prompt: {prompt}")

        try:
            result = await asyncio.wait_for(
                agent.run(prompt),
                timeout=AGENT_RUN_TIMEOUT,
            )
            final_output = result.final_output
            messages = result.to_input_list()
        except asyncio.TimeoutError:
            print("Failure: Agent run timed out after", AGENT_RUN_TIMEOUT, "seconds")
            final_output = "None"
            messages = []

        tool_call_count = 0
        for message in messages:
            if (type(message) is dict) and ("type" in message) and (message["type"] == "function_call"):
                tool_call_count += 1

        print(f"[CalcSandboxHTTP] (tool_calls: {tool_call_count})\n")
        num_turns = tool_call_count


    except Exception as e:
        print(f"[CalcSandboxHTTP] Error: {e}")
        import traceback
        traceback.print_exc()
        final_output= ""
        num_turns = 0

    ground_truth = task["reward_model"].get("ground_truth")

    result_metrics = compute_reward_with_metrics(final_output, str(ground_truth), num_turns)
    reward_value = float(result_metrics["score"])
    acc_value = float(result_metrics["acc"])

    emit_reward_with_metrics(reward_value, {"acc": acc_value})


async def debug():
    """Here we show a more manual way for debugging, without Trainer.

    We get the data samples on our own, and run the agent with LitAgentRunner.
    You will need an `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variable set
    to run this function.
    
    IMPORTANT: Make sure to start the MCP server first:
        python mcp-server-sandbox-http.py --port 8765
    """
    # Manually create a tracer as Runner will need it.
    # Use a dummy OtelTracer if you don't need to trace anything other than reward.
    tracer = agl.OtelTracer()
    # The runner processes RetoolTask, which matches the agent's task type.
    runner = agl.LitAgentRunner[RetoolTask](tracer)

    # A store is required here to store the data collected.
    store = agl.InMemoryLightningStore()

    # This is what needs to be tuned (i.e., LLM)
    resource = agl.LLM(
        endpoint=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8033/v1"),
        model=os.environ.get("MODEL_PATH", "qwen"),
        sampling_parameters={"temperature": 0}
    )

    made_up_task: RetoolTask = {
        "data_source": "debug",
        "question": f"Calculate the sum of all prime numbers less than 20.{answer_format}",
        "prompt": [{"role": "user", "content": f"Calculate the sum of all prime numbers less than 20.{answer_format}"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": "77"},
        "agent_name": "sandbox_agent_http",
    }

    another_made_up_task: RetoolTask = {
        "data_source": "debug",
        "question": f"What is the value of 15 factorial divided by 12 factorial?{answer_format}",
        "prompt": [{"role": "user", "content": f"What is the value of 15 factorial divided by 12 factorial?{answer_format}"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": "2730"},
        "agent_name": "sandbox_agent_http",
    }


    # The agent here must be the same agent that will be used in the real run.
    with runner.run_context(agent=calc_sandbox_agent_youtu, store=store):
        await runner.step(
            made_up_task,
            resources={
                # The key "main_llm" here can be arbitrary
                "main_llm": resource
            },
        )

        # Run another task
        await runner.step(
            another_made_up_task,
            resources={"main_llm": resource},
        )


if __name__ == "__main__":
    
    asyncio.run(debug())

