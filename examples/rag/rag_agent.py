# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from typing import Any, Dict, cast

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings
from utils import compute_scores

import agentlightning as agl

agl.setup_logging()
logger = logging.getLogger(__name__)

agent_prompt = """You are an assistant who answers questions using Wikipedia retriever. Answer the question using only the retrieved passages. Verify your answer directly against the text.

After each search:
- Summarize findings.
- Decide if info is sufficient.
  - If sufficient: reply in <answer>...</answer> with your answer. The answer must be extremely concise: a single word or a few words only.
  - If not: suggest the next search needed to fill info gaps. The system will return top 3 relevant Wikipedia chunks.
- Explain your reasoning for the chosen action.

Repeat as needed. When done, wrap your final, concise answer in <answer> tags."""


class RAGAgent(agl.LitAgent[Dict[str, Any]]):
    def __init__(self, trained_agents: str | None = None) -> None:
        super().__init__(trained_agents=trained_agents)
        self.mcp_server_url = "http://127.0.0.1:8099/sse"

    # (task, resources, rollout)
    async def training_rollout_async(
        self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout
    ) -> float | None:
        # llm resources
        llm = cast(agl.LLM, resources["main_llm"])

        logger.info(f"Training with model: {llm.model} on endpoint: {llm.endpoint}")

        async with MCPServerSse(
            name="wiki_retriever_mcp",
            params={"url": self.mcp_server_url},
        ) as server:
            agent = Agent(
                model=LitellmModel(model="hosted_vllm/" + llm.model, base_url=llm.endpoint),
                model_settings=ModelSettings(
                    max_tokens=4096,
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
                "question:{} answer: {} ground_truth: {} reward: {}".format(
                    task["question"], answer, task["answer"], reward
                )
            )
            return reward

    # (task, resources, rollout)
    async def validation_rollout_async(
        self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout
    ) -> float | None:
        llm = cast(agl.LLM, resources["main_llm"])

        # set temperature
        val_resources: agl.NamedResources = {
            "main_llm": agl.LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0.7},
            )
        }

        # reuse training rollout for validation
        return await self.training_rollout_async(task, val_resources, rollout)


if __name__ == "__main__":

    trainer = agl.Trainer(n_workers=2)

    print("Agent initialized. Please use 'agl.Trainer(...).fit(...)' with a dataset to start training.")
