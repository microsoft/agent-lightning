# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from typing import Any, Dict, List, cast

import pandas as pd
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings
from metric_utils import compute_scores

import agentlightning as agl

logger = logging.getLogger("rag_agent")

agent_prompt = """You are an assistant who answers questions using Wikipedia retriever. Answer the question using only the retrieved passages. Verify your answer directly against the text.

After each search:
- Summarize findings.
- Decide if info is sufficient.
  - If sufficient: reply in <answer>...</answer> with your answer. The answer must be extremely concise: a single word or a few words only.
  - If not: suggest the next search needed to fill info gaps. The system will return top 3 relevant Wikipedia chunks.
- Explain your reasoning for the chosen action.

Repeat as needed. When done, wrap your final, concise answer in <answer> tags."""


class RAGAgent(agl.LitAgent[Dict[str, Any]]):
    """RAGAgent is an agent that relies on a MCP-based retriever to answer questions."""

    def __init__(self) -> None:
        super().__init__()
        self.mcp_server_url = "http://127.0.0.1:8099/sse"

    async def training_rollout_async(
        self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout
    ) -> float | None:
        # llm resources
        llm = cast(agl.LLM, resources["main_llm"])

        # The rollout should carry an attempt inside
        rollout = cast(agl.AttemptedRollout, rollout)
        base_url = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)

        logger.info(f"Training with model: {llm.model} on endpoint: {base_url}")

        async with MCPServerSse(
            name="wiki_retriever_mcp",
            params={"url": self.mcp_server_url},
        ) as server:
            agent = Agent(
                model=LitellmModel(
                    model="hosted_vllm/" + llm.model,
                    base_url=base_url,
                ),
                model_settings=ModelSettings(
                    max_tokens=2048,
                    temperature=0.7,
                ),
                name="Assistant",
                instructions=agent_prompt,
                mcp_servers=[server],
            )
            result = await Runner.run(agent, task["question"])
            answer = result.final_output

            # reward
            reward = compute_scores(answer, str(task["answer"]))

            logger.info(
                "Question: %s\nAnswer: %s\nGround truth: %s\nReward: %s",
                task["question"],
                answer,
                task["answer"],
                reward,
            )
            return float(reward)  # Convert to float for compatibility with the Runner

    async def validation_rollout_async(
        self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout
    ) -> float | None:
        """Validation rollout will share the same logic as the training rollout."""
        # Same as training rollout, but with different temperature
        llm = cast(agl.LLM, resources["main_llm"])
        rollout = cast(agl.AttemptedRollout, rollout)

        # set temperature
        val_resources: agl.NamedResources = {
            "main_llm": agl.LLM(
                endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
                model=llm.model,
                sampling_parameters={"temperature": 0.7},
            )
        }

        # reuse training rollout for validation
        return await self.training_rollout_async(task, val_resources, rollout)


def debug():
    """Debug the RAGAgent."""

    agl.setup_logging("DEBUG", apply_to=[logger.name])

    # 1. loading dataset
    dataset_path = "data/dataset_tiny.parquet"
    df: pd.DataFrame = pd.read_parquet(dataset_path)  # type: ignore
    data: List[Dict[str, Any]] = df.head(5).to_dict(orient="records")  # type: ignore
    # NOTE: The following dummy data can also be used if you don't have the dataset.
    # data: List[Dict[str, Any]] = [{"question": "What is the capital of France?", "answer": "Paris"}]

    # 2. configuring resources (LLM)
    # Note: You need to start a local service compatible with the OpenAI API (such as vLLM)
    # For example: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
    resources: dict[str, agl.ResourceUnion] = {
        "main_llm": agl.LLM(
            endpoint="http://localhost:8000/v1",  # Replace with your actual vLLM address
            model="Qwen/Qwen2.5-1.5B-Instruct",  # Replace with your actual loaded model name
            sampling_parameters={"temperature": 0.0},
        )
    }

    # 3. run agent
    trainer = agl.Trainer(initial_resources=resources)
    trainer.dev(RAGAgent(), train_dataset=data)  # type: ignore


if __name__ == "__main__":
    debug()
