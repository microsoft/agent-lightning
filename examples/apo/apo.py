# Copyright (c) Microsoft. All rights reserved.

"""This is the APO example written in the legacy client-server style (agent-lightning v0.1).

New users should refer to the `examples/apo/apo.py` for the modern APO example.
"""

import asyncio

from agentlightning.algorithm.base import algo
from agentlightning.reward import get_last_reward
from agentlightning.store.base import LightningStore
from agentlightning.types import NamedResources, PromptTemplate


@algo
async def apo_algorithm(*, store: LightningStore):
    """
    An example of how a prompt optimization works.
    """
    prompt_candidates = [
        "You are a helpful assistant.",
        "You are a knowledgeable AI.",
        "You are a friendly chatbot.",
    ]

    prompt_and_rewards: list[tuple[str, float]] = []

    for prompt in prompt_candidates:
        # 1. The optimization algorithm updates the prompt template
        print(f"\n[Algo] Updating prompt template to: '{prompt}'")
        resources: NamedResources = {"system_prompt": PromptTemplate(template=prompt, engine="f-string")}
        # How the resource is used fully depends on the client implementation.
        await store.update_resources("resource-defult", resources)

        # 2. The algorithm queues up a task from a dataset
        print("[Algo] Queuing task for clients...")
        rollout = await store.enqueue_rollout(input={"prompt": "What is the capital of France?"}, mode="train")
        print(f"[Algo] Task '{rollout.rollout_id}' is now available for clients.")

        # 3. The algorithm waits for clients to process the task
        rollouts = await store.wait_for_rollouts(rollout_ids=[rollout.rollout_id], timeout=30)
        assert rollouts, "Expected a completed rollout from the client."
        print(f"[Algo] Received Result: {rollouts[0]}")
        spans = await store.query_spans(rollout.rollout_id)
        final_reward = get_last_reward(spans)
        assert final_reward is not None, "Expected a final reward from the client."
        print(f"[Algo] Final reward: {final_reward}")
        prompt_and_rewards.append((prompt, final_reward))

    print(f"\n[Algo] All prompts and their rewards: {prompt_and_rewards}")
    best_prompt = max(prompt_and_rewards, key=lambda x: x[1])
    print(f"[Algo] Best prompt found: '{best_prompt[0]}' with reward {best_prompt[1]}")


if __name__ == "__main__":
    # TODO: add @prompt_rollout decorator to implement the rollout function

    asyncio.run(apo_algorithm())
