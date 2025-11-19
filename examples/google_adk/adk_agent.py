# Copyright (c) Microsoft. All rights reserved.

"""
Simple example showing how to integrate an ADK agent with observability.
This is based on LitAgent but uses ADK's trace visualization features.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Dict, TypedDict, cast

from agentlightning import LLM, LitAgent, NamedResources
from agentlightning.types import Rollout, RolloutRawResult

logger = logging.getLogger(__name__)

class AdkTask(TypedDict):
    question: str
    app_id: str
    ground_truth: str
    meta: Dict[str, Any] | None


class LitAdkAgent(LitAgent[AdkTask]):

    def rollout(self, task: AdkTask, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:  # type: ignore[override]
        """Synchronous entry point – forwards to ``rollout_async``."""

        if not _HAS_GOOGLE_ADK:  # pragma: no cover
            raise RuntimeError(
                "google-adk>=0.3.0 is required to run this example. "
                "Install it with `pip install google-adk` or enable the "
                "`adk` optional dependency group."
            )

        try:
            return asyncio.run(self.rollout_async(task, resources, rollout))
        except RuntimeError as exc:
            if "asyncio.run()" in str(exc):
                raise RuntimeError(
                    "LitAdkAgent.rollout cannot be executed while an event loop "
                    "is already running. Call `rollout_async` instead."
                ) from exc
            raise

    async def rollout_async(  # type: ignore[override]
        self,
        task: AdkTask,
        resources: NamedResources,
        rollout: Rollout,
    ) -> RolloutRawResult:
        """Runs a single rollout by delegating to ADK's orchestration runtime."""

        if not _HAS_GOOGLE_ADK:
            raise RuntimeError(
                "google-adk>=0.3.0 is required to run this example. "
                "Install it with `pip install google-adk` or enable the "
                "`adk` optional dependency group."
            )

        llm = cast(LLM, resources.get("main_llm"))
        question = task["question"]
        truth = task["ground_truth"]
        app_id = task["app_id"] or "adk_agent_app"

        adk_model = self._build_adk_model(llm)
        adk_agent = self._build_adk_agent(adk_model, task)
        app = App(name=app_id, root_agent=adk_agent)

        runner = InMemoryRunner(app=app)
        try:
            session_state = self._build_session_state(task)
            session = await runner.session_service.create_session(
                app_name=app.name,
                user_id="agent_lightning_user",
                state=session_state,
            )

            message = genai_types.Content(  # type: ignore[union-attr]
                role="user",
                parts=[genai_types.Part.from_text(text=question)],  # type: ignore[union-attr]
            )

            last_response = ""
            async with Aclosing(  # type: ignore[union-attr]
                runner.run_async(
                    user_id=session.user_id,
                    session_id=session.id,
                    new_message=message,
                )
            ) as agen:
                async for event in agen:
                    if event.content and event.content.parts:
                        text = "".join(part.text or "" for part in event.content.parts).strip()
                        if text:
                            last_response = text

            reward = self._compute_reward(last_response, truth)
            return reward
        except Exception as exc:
            logger.exception("ADK rollout failed: %s", exc)
            return 0.0
        finally:
            await runner.close()

    @staticmethod
    def _build_adk_model(llm: LLM) -> LiteLlm:  # type: ignore[valid-type]
        """Create a LiteLLM-backed ADK model from the Agent-Lightning LLM resource."""

        sampling_params = llm.sampling_parameters or {}
        temperature: float | None = None
        top_p: float | None = None
        if isinstance(sampling_params, Mapping):
            temperature = sampling_params.get("temperature")
            top_p = sampling_params.get("top_p")

        llm_kwargs: Dict[str, Any] = {}
        if llm.endpoint:
            llm_kwargs["api_base"] = llm.endpoint
        if llm.api_key:
            llm_kwargs["api_key"] = llm.api_key
        if temperature is not None:
            llm_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            llm_kwargs["top_p"] = float(top_p)

        return LiteLlm(model=llm.model, **llm_kwargs)  # type: ignore[arg-type]

    def _build_adk_agent(self, adk_model: LiteLlm, task: AdkTask) -> LlmAgent:  # type: ignore[valid-type]
        """Construct the ADK agent definition for this rollout."""

        instruction = self._compose_instruction(task)
        agent_name = task["app_id"] or "adk_agent"

        return LlmAgent(
            name=agent_name,
            model=adk_model,
            instruction=instruction,
        )

    @staticmethod
    def _build_session_state(task: AdkTask) -> Dict[str, Any]:
        """Extract session state from the task metadata."""

        session_state: Dict[str, Any] = {}
        meta = task.get("meta")
        if isinstance(meta, Mapping):
            session_state.update(meta)
        return session_state

    def _compose_instruction(self, task: AdkTask) -> str:
        """Compose the instruction prompt for the ADK agent."""

        base_instruction = (
            "You are an enterprise assistant integrated with Google ADK. "
            "Use available tools to reason carefully and produce concise, factual answers. "
            "Explain limitations when information is missing."
        )

        meta = task.get("meta")
        if isinstance(meta, Mapping):
            for key in ("instruction", "goal", "context", "description"):
                if meta.get(key):
                    return f"{base_instruction}\n\nAdditional context:\n{meta[key]}"

        return base_instruction

    @staticmethod
    def _compute_reward(answer: str, truth: str) -> float:
        """Simple lexical reward comparing the ADK answer with ground truth."""

        if not answer or not truth:
            return 0.0

        answer_lower = answer.lower()
        truth_lower = truth.lower()
        return 1.0 if truth_lower in answer_lower else 0.0


# Minimal smoke-test entry point (manual run)
if __name__ == "__main__":
    import sys

    if not _HAS_GOOGLE_ADK:
        sys.exit(
            "google-adk is not installed. Run `pip install google-adk` before executing this script."
        )

    sample_task: AdkTask = {
        "question": "Create a calendar event for Monday 10am titled 'Standup'",
        "app_id": "sample_calendar_app",
        "ground_truth": "create_event",
        "meta": {"priority": "normal"},
    }

    resources: NamedResources = {
        "main_llm": LLM(
            endpoint="http://localhost:8000/v1",
            model="gpt-4.1-mini",
            sampling_parameters={"temperature": 0.0},
            api_key="dummy-key",
        ),
    }

    class DummyRollout:
        pass

    agent = LitAdkAgent()
    result = asyncio.run(agent.rollout_async(sample_task, resources, cast(Rollout, DummyRollout())))
    print("Reward:", result)


