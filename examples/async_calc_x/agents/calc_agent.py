"""Calc-X agent — solves math problems using AutoGen + MCP calculator tool.

Standalone container/local agent for agl-lite. No agl-lite imports.

Environment variables (set automatically by agl-lite controller):
  AGL_TASK_INPUT   — JSON object: {"question": "...", "id": "..."}
  OPENAI_BASE_URL  — points to agl-lite gateway
  OPENAI_API_KEY   — auth key for gateway
  AGL_EVENT_URL    — URL to post events (agent_output)

Optional:
  AGL_MODEL_NAME   — model name for LLM calls (default: from --model flag)
  AGL_TEMPERATURE  — sampling temperature (default: 0.7)

Usage (in container):
  python calc_agent.py
  python calc_agent.py --model Qwen/Qwen2.5-1.5B-Instruct

Usage (local runner):
    agl-lite-controller runner_type=local \
            local_agent_class=examples.async_calc_x.agents.calc_agent:CalcXAgent
"""

# AutoGen is an optional runtime dependency for this example container.
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import cast

import httpx

log = logging.getLogger(__name__)

OUTPUT_FORMAT = (
    "Output the answer when you are ready. The answer should be surrounded by "
    "three sharps (`###`), in the form of ### ANSWER: <answer> ###."
)
ANSWER_PATTERN = re.compile(r"###\s*ANSWER:\s*(.+?)(\s*###|$)")
AGENT_TIMEOUT = float(os.environ.get("AGL_AGENT_TIMEOUT", "600.0"))
# Async-rollout gateway pause can hold a single LLM call in 429 backoff for the
# duration of a training phase. Default budget: 5s * 32 = 160s of retries.
MAX_RETRIES = int(os.environ.get("AGL_AGENT_MAX_RETRIES", "32"))


def _setup_logging() -> None:
    log_dir = os.environ.get("AGL_LOG_DIR")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        log_path = Path(log_dir) / "agent.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def post_event(event_type: str, data: dict) -> None:
    """Post an event to agl-lite via AGL_EVENT_URL."""
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        log.warning("AGL_EVENT_URL not set — skipping %s event", event_type)
        return
    api_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        resp = httpx.post(
            event_url,
            json={"event_type": event_type, "data": data},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.error("Failed to post %s event: %s", event_type, e)


def _parse_task(task_input: object) -> tuple[str, str]:
    if not isinstance(task_input, dict) or "question" not in task_input:
        raise ValueError(f"AGL_TASK_INPUT must be a JSON object with 'question' field, got: {type(task_input)}")

    question = task_input["question"]
    if not isinstance(question, str):
        raise ValueError("AGL_TASK_INPUT['question'] must be a string")

    task_id = task_input.get("id", "unknown")
    return question, str(task_id)


class CalcXAgent:
    """Local-runner adapter for the Calc-X agent.

    The local worker instantiates this class with no arguments and calls
    run(task_input). Keep the behavior aligned with the script entrypoint so
    K8s and local modes exercise the same agent logic.
    """

    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        self._model = model
        self._temperature = temperature

    async def run(self, task_input: object) -> None:
        _setup_logging()

        question, task_id = _parse_task(task_input)
        model = self._model or os.environ.get("AGL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
        temperature = self._temperature
        if temperature is None:
            temperature = float(os.environ.get("AGL_TEMPERATURE", "0.7"))

        log.info("Task %s: %s", task_id, question[:100])
        answer, raw_response = await solve(question, model, temperature)
        log.info("Answer: %s", answer)

        post_event("agent_output", {
            "answer": answer,
            "raw_response": raw_response,
            "task_id": task_id,
        })


async def solve(question: str, model: str, temperature: float) -> tuple[str, str]:
    """Run AutoGen agent with MCP calculator to solve a math question.

    Returns (answer, raw_response) where answer is the extracted answer
    string and raw_response is the full last message from the agent.
    """
    # Lazy imports — these are heavy and only needed at solve time.
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.models import ModelFamily
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

    calculator_mcp_server = StdioServerParams(
        command="python",
        args=["-m", "mcp_server_calculator"],
        read_timeout_seconds=120,
    )

    async with McpWorkbench(calculator_mcp_server) as workbench:
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url is None:
            raise RuntimeError("OPENAI_BASE_URL is required")

        model_client = OpenAIChatCompletionClient(
            model=model,
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": ModelFamily.UNKNOWN,
                "structured_output": False,
            },
            temperature=temperature,
            max_retries=MAX_RETRIES,
        )

        agent = AssistantAgent(
            name="calc",
            model_client=model_client,
            workbench=workbench,
            reflect_on_tool_use=True,
        )

        prompt = question + " " + OUTPUT_FORMAT

        try:
            result = await asyncio.wait_for(agent.run(task=prompt), timeout=AGENT_TIMEOUT)
            last_message = cast(str, result.messages[-1].content)  # type: ignore
            match = ANSWER_PATTERN.search(last_message)
            answer = match.group(1).strip() if match else last_message.strip()
        except TimeoutError:
            log.error("Agent timed out after %.0fs", AGENT_TIMEOUT)
            answer = "None"
            last_message = "[TIMEOUT]"
        except Exception as e:
            log.error("Agent failed: %s", e)
            answer = "None"
            last_message = f"[ERROR] {e}"

    return answer, last_message


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Calc-X agent for math problems")
    parser.add_argument(
        "--model",
        default=os.environ.get("AGL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        help="Model name (default: $AGL_MODEL_NAME)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("AGL_TEMPERATURE", "0.7")),
        help="Sampling temperature (default: $AGL_TEMPERATURE or 0.7)",
    )
    args = parser.parse_args()

    # --- Read task input ---
    raw = os.environ.get("AGL_TASK_INPUT")
    if not raw:
        log.error("AGL_TASK_INPUT not set")
        sys.exit(1)

    try:
        task = json.loads(raw)
        asyncio.run(CalcXAgent(model=args.model, temperature=args.temperature).run(task))
    except Exception as e:
        log.error("Agent failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
