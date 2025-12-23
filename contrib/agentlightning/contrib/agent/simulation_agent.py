from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
from add_instruction import add_chat_instruction, add_single_instruction
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agentlightning import LLM, LitAgent, NamedResources, Rollout, configure_logger, emit_object, emit_reward, operation
from agentlightning.utils.otel import make_link_attributes
from agl_envs.simulation import make_env_manager
from contrib.recipes.simulation.prompt_builder import HistoryPromptBuilder

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

    def _get_instructed_prompt(self, prompt, sep="\n\n"):
        """Return instructed observation based on prompt_type and captioner type."""
        prompt_type = self.config.captioner.prompt_type
        cap_type = self.config.captioner.type

        if prompt_type == "chat":
            if cap_type == "cot":
                return add_chat_instruction(prompt, "cot", sep, self.config.env_name)
            elif cap_type == "naive":
                return add_chat_instruction(prompt, "naive", sep)

        elif prompt_type == "single":
            if cap_type == "cot":
                return add_single_instruction(prompt, "cot", sep, self.config.env_name)
            elif cap_type == "naive":
                return add_single_instruction(prompt, "naive", sep, self.config.env_name)

        raise ValueError(f"Unsupported prompt_type={prompt_type}, type={cap_type}")

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
            prompt_builder = HistoryPromptBuilder(
                max_history=self.config.captioner.max_history, prompt_type=self.config.captioner.prompt_type
            )

            self.env = make_env_manager(self.config.env_name, task, self.config)
            env_obs, infos, available_actions_hint = self.env.reset()

            prompt_builder.init(self.env)
            prompt_builder.update_observation(env_obs)
            prompt_builder.update_admissible_actions(available_actions_hint)

            prompt = prompt_builder.get_prompt()

            episode_reward, done = 0.0, False

            step_count = 0
            while not done:
                try:
                    instructed_prompt = self._get_instructed_prompt(prompt)

                    # Main agent step
                    with operation(step_count=step_count):
                        result = await self.agent._model_client.create(instructed_prompt)
                    output = result.content
                    logger.info(f"[LLM output]: {output}")

                except Exception as e:
                    logger.error(f"[Rollout {rollout_id}] Error during training rollout: {e}", exc_info=True)
                    break

                if self.config.log_env_obs:
                    emit_object(env_obs, attributes=make_link_attributes({"step_count": str(step_count)}))

                env_obs, executed_action, is_valid, step_reward, terminated, truncated, info, available_actions_hint = (
                    self.env.step(output, use_reasoning=self.config.captioner.type == "cot")
                )

                prompt_builder.update_step_count()
                prompt_builder.update_action(executed_action)
                prompt_builder.update_observation(env_obs)
                prompt_builder.update_admissible_actions(available_actions_hint)

                prompt = prompt_builder.get_prompt()

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
