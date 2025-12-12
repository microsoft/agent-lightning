from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from captioners.add_instruction import add_chat_instruction, add_single_instruction
from captioners.debugging import (
    print_llm_chat_input,
    print_llm_single_input,
)
from agl_envs.simulation import make_env_manager
from contrib.recipes.simulation.captioners import create_prompt_builder

from agentlightning import LLM, LitAgent, NamedResources, Rollout, configure_logger, emit_object, emit_reward, operation
from agentlightning.utils.otel import make_link_attributes

logger = configure_logger(name=__name__, level=logging.ERROR)


class SimulationAgent(LitAgent):
    def __init__(self, config, trained_agents: str | None = None) -> None:
        super().__init__(trained_agents=trained_agents)
        self.config = config
        self.env = None

    def _build_agent(self, llm: LLM, temperature: float):
        model_client = OpenAIChatCompletionClient(
            model=llm.model,
            base_url=llm.endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": ModelFamily.UNKNOWN,
                "structured_output": False,
            },
            temperature=temperature,
        )

        return AssistantAgent(
            name="simulation",
            model_client=model_client,
        )

    def _get_instructed_obs(self, obs, sep="\n\n"):
        """Return instructed observation based on obs_type and captioner type."""
        obs_type = self.config.captioner.obs_type
        cap_type = self.config.captioner.type

        if obs_type == "chat":
            if cap_type == "cot":
                return add_chat_instruction(obs, "cot", sep, self.config.env_name)
            elif cap_type == "naive":
                return add_chat_instruction(obs, "naive", sep)

        elif obs_type == "single":
            if cap_type == "cot":
                return add_single_instruction(obs, "cot", sep, self.config.env_name)
            elif cap_type == "naive":
                return add_single_instruction(obs, "naive", sep, self.config.env_name)

        raise ValueError(f"Unsupported obs_type={obs_type}, type={cap_type}")

    def _print_llm_input(self, obs):
        obs_type = self.config.captioner.obs_type

        if obs_type == "chat":
            return print_llm_chat_input(obs)
        elif obs_type == "single":
            return print_llm_single_input(obs)

    async def rollout_async(
        self,
        task: Dict[str, Any],
        resources: NamedResources,
        rollout: Rollout,
    ) -> float | None:
        rollout_id = rollout.rollout_id
        logger.info(f"[Rollout {rollout_id}] Task: {task}")

        format_penalty = float(self.config["format_penalty"])
        reward_scale = float(self.config["reawrd_scale"])

        # Setup agent
        llm: LLM = resources.get("main_llm")
        print("Training with model:", llm.model, "on endpoint:", llm.endpoint)
        self.agent = self._build_agent(llm, 1.0 if rollout.mode == "train" else 0.4)
        if "max_tokens" in self.config and self.config["max_tokens"] > -1:
            self.agent._model_client.max_tokens = self.config["max_tokens"]

        try:
            # Setup environment
            prompt_builder = create_prompt_builder(self.config.captioner)
            self.env = make_env_manager(self.config.env_name, task, self.config, prompt_builder=prompt_builder)
            obs, pure_env_obs, infos = self.env.reset()
            episode_reward, done = 0.0, False

            step_count = 0
            while not done:
                try:
                    instructed_obs = self._get_instructed_obs(obs)

                    # Main agent step
                    with operation(step_count=step_count):
                        result = await self.agent._model_client.create(instructed_obs)
                    output = result.content
                    logger.info(f"[LLM output]: {output}")

                except Exception as e:
                    logger.error(f"[Rollout {rollout_id}] Error during training rollout: {e}", exc_info=True)
                    break

                if self.config.log_pure_env_obs:
                    emit_object(pure_env_obs, attributes=make_link_attributes({"step_count": str(step_count)}))

                obs, pure_env_obs, executed_action, is_valid, step_reward, terminated, truncated, info = self.env.step(
                    output, use_reasoning=self.config.captioner.type == "cot"
                )

                if rollout.mode == "train":
                    step_reward *= reward_scale

                if format_penalty != 0.0:
                    emit_reward(
                        {
                            "extrinsic_reward": step_reward,
                            "intrinsic_reward": 0.0 if is_valid else -1.0 * format_penalty,
                        },
                        primary_key="extrinsic_reward",
                        attributes=make_link_attributes({"step_count": str(step_count)}),
                    )
                else:
                    emit_reward(step_reward, attributes=make_link_attributes({"step_count": str(step_count)}))

                episode_reward += float(step_reward)
                done = np.logical_or(terminated, truncated)

                step_count += 1

            return episode_reward

        finally:
            if self.env is not None:
                self.env.close()
