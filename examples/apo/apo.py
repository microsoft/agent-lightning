import asyncio

from agentlightning.prompt.server import AgentLightningServer
from agentlightning.prompt.resource import NamedResources, PromptTemplate


async def example_apo():
    """
    An example of how a prompt optimization works.
    """
    server = AgentLightningServer(host="127.0.0.1", port=8080)
    await server.start()

    prompt_candidates = [
        "You are a helpful assistant.",
        "You are a knowledgeable AI.",
        "You are a friendly chatbot.",
    ]

    prompt_and_rewards = []

    for prompt in prompt_candidates:
        # 1. The optimization algorithm updates the prompt template
        print(f"\n[Algo] Updating prompt template to: '{prompt}'")
        resources = NamedResources({"system_prompt": PromptTemplate(template=prompt, engine="f-string")})
        # How the resource is used fully depends on the client implementation.
        await server.update_resources(resources)

        # 2. The algorithm queues up a task from a dataset
        print("[Algo] Queuing task for clients...")
        task_id = await server.queue_task(sample={"prompt": "What is the capital of France?"}, is_train=True)
        print(f"[Algo] Task '{task_id}' is now available for clients.")

        # 3. The algorithm waits for clients to process the task
        trajectory = await server.poll_completed_rollout(task_id, timeout=30)
        assert trajectory, "Expected a completed trajectory from the client."
        print(f"[Algo] Received Result: {trajectory}")
        reward = trajectory["transitions"][-1].reward if trajectory["transitions"] else 0
        prompt_and_rewards.append((prompt, reward))

    print(f"\n[Algo] All prompts and their rewards: {prompt_and_rewards}")
    best_prompt = max(prompt_and_rewards, key=lambda x: x[1])
    print(f"[Algo] Best prompt found: '{best_prompt[0]}' with reward {best_prompt[1]}")

    await server.stop()


if __name__ == "__main__":
    asyncio.run(example_apo())
