# Copyright (c) Microsoft. All rights reserved.

"""
Simple example showing how to integrate an ADK agent with observability.
This is based on LitAgent but uses ADK's trace visualization features.
"""

from __future__ import annotations
from typing import Any, Dict, TypedDict, cast
from agentlightning import LLM, LitAgent, NamedResources
from agentlightning.types import Rollout, RolloutRawResult


class AdkTask(TypedDict):
    """
    One task item as produced by the dataset.

    Required fields:
    - question: The user instruction for the agent.
    - app_id: The application/environment identifier.
    - ground_truth: The expected action/output (string form) for reward computation.
    Optional fields:
    - meta: Arbitrary metadata.
    """
    question: str
    app_id: str
    ground_truth: str
    meta: Dict[str, Any] | None


class LitAdkAgent(LitAgent[AdkTask]):
    """Basic ADK + LitAgent example."""

    def rollout(self, task: AdkTask, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        # get the llm resource
        llm: LLM = cast(LLM, resources.get("main_llm"))

        question = task["question"]
        app_id = task["app_id"]
        truth = task["ground_truth"]

        # TODO: hook up real ADK orchestration later
        # for now just simulate an action string
        action = f"adk://{app_id}?plan={question}"

        # quick check for correctness
        reward = 1.0 if truth and truth.lower() in action.lower() else 0.0

        return reward


# Minimal smoke-test entry point (manual run)
if __name__ == "__main__":
    # very minimal test run
    sample_task: AdkTask = {
        "question": "Create a calendar event for Monday 10am titled 'Standup'",
        "app_id": "sample_calendar_app",
        "ground_truth": "create_event",
        "meta": {"priority": "normal"},
    }

    resources: NamedResources = {
        "main_llm": LLM(
            endpoint="http://localhost:8000/v1",
            model="meta-llama/Meta-Llama-3-8B-Instruct"
        ),
    }

    class DummyRollout:
        pass

    agent = LitAdkAgent()
    result = agent.rollout(sample_task, resources, cast(Rollout, DummyRollout()))
    print("Reward:", result)


