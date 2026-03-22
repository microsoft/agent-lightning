#!/usr/bin/env python3
"""Mock RL loop — demonstrates the full agl-lite lifecycle with mockai.

Two iterations with a weight update between them:
  Iter 1: register resources + model (v1) → enqueue batch → poll → retrieve → rewards
  Weight update: deregister model → re-register (v2, same endpoint)
  Iter 2: enqueue batch → poll → retrieve → rewards → verify version=2 in events

Event flow per rollout:
  model_request (auto, gateway)  →  agent_output (agent)  →  reward (algorithm)

Usage:
    export AGL_LITE_URL=http://localhost:8080
    export AGL_KEY=<your-key>
    export AGL_K8S_NAMESPACE=<namespace>
    python examples/math-poc/mock_rl_loop.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from agl_lite.client import AglLiteClient, AglLiteError
from agl_lite.schemas.api import EnqueueRolloutRequest, PostEventRequest, RegisterModelRequest
from agl_lite.schemas.rollout import RolloutStatus

# --- Config ---

MOCKAI_MODEL = "mock-llm"
BATCH_SIZE = 5
NUM_ITERATIONS = 2
POLL_INTERVAL = 3
MAX_POLL_TIME = 120

DATA_DIR = Path(__file__).parent / "data"
TEMPLATE_PATH = Path(__file__).parent / "job-template.yaml"


def load_dataset() -> list[dict]:
    """Load GSM8K sample dataset."""
    items = []
    with open(DATA_DIR / "gsm8k_sample.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_tasks(dataset: list[dict], batch_size: int) -> list[dict]:
    """Build task inputs — plain text questions with ground truth for reward.

    The agent receives AGL_TASK_INPUT as a JSON-encoded string (the question).
    Ground truth stays in the algorithm (not sent to the agent).
    """
    tasks = []
    sample = random.sample(dataset, min(batch_size, len(dataset)))
    for item in sample:
        tasks.append({"input": item["question"], "ground_truth": item["answer"]})
    return tasks


def compute_reward(agent_answer: str, ground_truth: str) -> float:
    """Compare agent's extracted answer to ground truth."""
    try:
        return 1.0 if agent_answer.strip() == ground_truth.strip() else 0.0
    except Exception:
        return 0.0


async def wait_for_rollouts(
    client: AglLiteClient,
    rollout_ids: list[str],
    max_time: float = MAX_POLL_TIME,
) -> list:
    """Poll until all rollouts reach a terminal state."""
    terminal = {RolloutStatus.SUCCEEDED, RolloutStatus.TERMINAL_FAILED, RolloutStatus.CANCELLED}
    start = time.time()
    while time.time() - start < max_time:
        rollouts = await client.query_rollouts(ids=rollout_ids, limit=len(rollout_ids))
        done = all(r.status in terminal for r in rollouts)
        if done:
            return rollouts

        running = sum(1 for r in rollouts if r.status == RolloutStatus.RUNNING)
        queuing = sum(1 for r in rollouts if r.status == RolloutStatus.QUEUING)
        print(f"  Waiting... {running} running, {queuing} queuing, {time.time() - start:.0f}s elapsed")
        await asyncio.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Rollouts did not complete within {max_time}s")


async def run_iteration(
    client: AglLiteClient,
    resources_id: str,
    dataset: list[dict],
    iteration: int,
    expected_version: int,
) -> dict:
    """Run one batch of rollouts, collect agent outputs, compute and post rewards."""
    print(f"\n{'='*60}")
    print(f"  ITERATION {iteration} (model version={expected_version})")
    print(f"{'='*60}")

    # Build tasks
    tasks = build_tasks(dataset, BATCH_SIZE)

    # Enqueue rollouts
    requests = [
        EnqueueRolloutRequest(
            resources_id=resources_id,
            input=t["input"],
            config={},
        )
        for t in tasks
    ]
    rollouts = await client.enqueue_rollouts(requests)
    rollout_ids = [r.rollout_id for r in rollouts]
    print(f"  Enqueued {len(rollout_ids)} rollouts")

    # Wait for completion
    completed = await wait_for_rollouts(client, rollout_ids)
    succeeded = [r for r in completed if r.status == RolloutStatus.SUCCEEDED]
    failed = [r for r in completed if r.status == RolloutStatus.TERMINAL_FAILED]
    print(f"  Results: {len(succeeded)} succeeded, {len(failed)} failed")

    # Retrieve events, compute rewards, post reward events
    total_reward = 0.0
    model_request_count = 0
    agent_output_count = 0
    reward_count = 0
    versions_seen = set()

    for i, rollout in enumerate(succeeded):
        events = await client.get_events(rollout.rollout_id)

        # Find model_request events → check version
        for event in events:
            if event.event_type == "model_request":
                model_request_count += 1
                server = event.data.get("server", {})
                v = server.get("version")
                if v is not None:
                    versions_seen.add(v)

        # Find agent_output event → get the answer
        agent_answer = None
        attempt_id = None
        for event in events:
            if event.event_type == "agent_output":
                agent_output_count += 1
                agent_answer = event.data.get("answer", "")
                attempt_id = event.attempt_id
                break

        # Find ground truth — match by question text
        matching_task = next(
            (t for t in tasks if t["input"] == rollout.input),
            None,
        )
        gt = matching_task["ground_truth"] if matching_task else ""

        # Compute reward
        reward = compute_reward(agent_answer or "", gt) if agent_answer else 0.0
        total_reward += reward

        # Post reward event
        if attempt_id:
            try:
                await client.post_event(
                    rollout.rollout_id,
                    attempt_id,
                    PostEventRequest(event_type="reward", data={"value": reward, "ground_truth": gt, "agent_answer": agent_answer}),
                )
                reward_count += 1
            except AglLiteError as e:
                print(f"  Warning: failed to post reward for {rollout.rollout_id}: {e}", file=sys.stderr)

    avg_reward = total_reward / len(succeeded) if succeeded else 0.0
    print(f"  Events: {model_request_count} model_request, {agent_output_count} agent_output, {reward_count} reward")
    print(f"  Versions seen: {versions_seen}")
    print(f"  Average reward: {avg_reward:.2f}")

    return {
        "iteration": iteration,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "model_requests": model_request_count,
        "agent_outputs": agent_output_count,
        "rewards": reward_count,
        "versions": versions_seen,
        "avg_reward": avg_reward,
    }


async def main() -> None:
    base_url = os.environ.get("AGL_LITE_URL", "http://localhost:8080")
    agl_key = os.environ.get("AGL_KEY")
    namespace = os.environ.get("AGL_K8S_NAMESPACE", "agl")

    mockai_url = f"http://mockai.{namespace}.svc.cluster.local:5002/v1"

    print(f"agl-lite URL: {base_url}")
    print(f"mockai URL (in-cluster): {mockai_url}")

    client = AglLiteClient(base_url=base_url, agl_key=agl_key)

    try:
        # --- Setup: register resources ---
        print("\n--- Registering resources ---")
        with open(TEMPLATE_PATH) as f:
            job_template = yaml.safe_load(f)

        res = await client.add_resources({"job_template": job_template})
        resources_id = res.resources_id
        print(f"  Resources: {resources_id}")

        # --- Load dataset ---
        dataset = load_dataset()
        print(f"  Dataset: {len(dataset)} problems")

        # --- Register model server (v1) ---
        print("\n--- Registering model server (v1) ---")
        await client.register_models([
            RegisterModelRequest(model=MOCKAI_MODEL, endpoint=mockai_url, version=1),
        ])
        print(f"  Registered: {MOCKAI_MODEL} → {mockai_url} (v1)")

        # --- Iteration 1 ---
        results = []
        r1 = await run_iteration(client, resources_id, dataset, iteration=1, expected_version=1)
        results.append(r1)

        # --- Weight update: v1 → v2 ---
        print(f"\n{'='*60}")
        print("  WEIGHT UPDATE: v1 → v2")
        print(f"{'='*60}")

        print("  Deregistering model (simulating weight update window)...")
        await client.delete_model(MOCKAI_MODEL)
        print("  Model pool empty — gateway returns 503 for new requests")

        await asyncio.sleep(2)

        print("  Re-registering model (v2)...")
        await client.register_models([
            RegisterModelRequest(model=MOCKAI_MODEL, endpoint=mockai_url, version=2),
        ])
        print(f"  Registered: {MOCKAI_MODEL} → {mockai_url} (v2)")

        # --- Iteration 2 ---
        r2 = await run_iteration(client, resources_id, dataset, iteration=2, expected_version=2)
        results.append(r2)

        # --- Summary ---
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        total_succeeded = sum(r["succeeded"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        total_model_requests = sum(r["model_requests"] for r in results)
        total_agent_outputs = sum(r["agent_outputs"] for r in results)
        total_rewards = sum(r["rewards"] for r in results)
        all_versions = set()
        for r in results:
            all_versions.update(r["versions"])

        print(f"  Iterations: {len(results)}")
        print(f"  Rollouts: {total_succeeded} succeeded, {total_failed} failed")
        print(f"  Events: {total_model_requests} model_request, {total_agent_outputs} agent_output, {total_rewards} reward")
        print(f"  Versions seen: {all_versions}")

        # --- Verify ---
        ok = True
        if total_succeeded < BATCH_SIZE:
            print(f"\n  ❌ Too many failures")
            ok = False
        if total_model_requests == 0:
            print(f"\n  ❌ No model_request events captured")
            ok = False
        if total_agent_outputs == 0:
            print(f"\n  ❌ No agent_output events — agent not reporting results")
            ok = False
        if total_rewards == 0:
            print(f"\n  ❌ No reward events posted")
            ok = False

        if ok:
            print(f"\n  ✅ Math PoC completed successfully!")
        else:
            print(f"\n  ❌ Math PoC had issues — check above")
            sys.exit(1)

    finally:
        await client.close()


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
