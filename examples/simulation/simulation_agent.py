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
from envs import make_env_manager

from agentlightning import LLM, LitAgent, NamedResources, Rollout, configure_logger, emit_object, emit_reward

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
            self.env = make_env_manager(self.config.env_name, task, self.config, render_mode=None)
            obs, pure_env_obs, infos = self.env.reset()
            episode_reward, done = 0.0, False

            while not done:
                try:
                    instructed_obs = self._get_instructed_obs(obs)
                    # print obs
                    if self.config.debugging:
                        self._print_llm_input(instructed_obs)

                    # Main agent step
                    result = await self.agent._model_client.create(instructed_obs)
                    output = result.content
                    logger.info(f"[LLM output]: {output}")

                except Exception as e:
                    logger.error(f"[Rollout {rollout_id}] Error during training rollout: {e}", exc_info=True)
                    break

                if self.config.log_pure_env_obs:
                    emit_object(pure_env_obs)

                obs, pure_env_obs, executed_action, is_valid, step_reward, terminated, truncated, info = self.env.step(
                    output, use_reasoning=self.config.captioner.type == "cot"
                )

                if rollout.mode == "train":
                    step_reward *= reward_scale
                emit_reward(step_reward)

                if format_penalty != 0.0:
                    emit_reward(0.0 if is_valid else -1.0 * format_penalty)

                episode_reward += float(step_reward)
                done = np.logical_or(terminated, truncated)

            return episode_reward

        finally:
            if self.env is not None:
                self.env.close()
