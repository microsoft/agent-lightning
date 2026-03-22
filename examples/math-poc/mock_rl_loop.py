#!/usr/bin/env python3
"""Mock RL loop — demonstrates the full agl-lite lifecycle with mockai.

Two iterations with a weight update between them:
  Iter 1: register resources + model (v1) → enqueue batch → poll → retrieve → rewards
  Weight update: deregister model → re-register (v2, same endpoint)
  Iter 2: enqueue batch → poll → retrieve → rewards → verify version=2 in events

Usage:
    export AGL_LITE_URL=http://localhost:8080
    export AGL_KEY=<your-key>
    export AGL_K8S_NAMESPACE=<namespace>
    python examples/math-poc/mock_rl_loop.py

Requires: agl-lite serve + controller + mockai running in K8s.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Add repo root to path so we can import agl_lite.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from agl_lite.client import AglLiteClient, AglLiteError
from agl_lite.schemas.api import EnqueueRolloutRequest, RegisterModelRequest
from agl_lite.schemas.rollout import RolloutStatus

# --- Config ---

MOCKAI_MODEL = "mock-llm"
AGENT_IMAGE = "math-agent:dev"
AGENT_COMMAND = ["python", "/app/qa_agent.py"]
BATCH_SIZE = 5  # rollouts per iteration
NUM_ITERATIONS = 2
POLL_INTERVAL = 3  # seconds between polls
MAX_POLL_TIME = 120  # seconds before giving up

DATA_DIR = Path(__file__).parent / "data"
TEMPLATE_PATH = Path(__file__).parent / "job-template.yaml"


def load_dataset() -> list[dict]:
    """Load GSM8K sample dataset."""
    items = []
    with open(DATA_DIR / "gsm8k_sample.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_prompts(dataset: list[dict], batch_size: int) -> list[dict]:
    """Build prompts with embedded answers (some correct, some wrong).

    Mockai echo mode returns the prompt verbatim. The reward function
    extracts the embedded answer and compares to ground truth.
    """
    prompts = []
    sample = random.sample(dataset, min(batch_size, len(dataset)))
    for item in sample:
        if random.random() < 0.5:
            # Embed correct answer
            answer = item["answer"]
        else:
            # Embed wrong answer
            try:
                answer = str(int(item["answer"]) + random.randint(1, 10))
            except ValueError:
                answer = "999"

        prompt = f"Solve: {item['question']} The answer is {answer}."
        prompts.append({
            "prompt": prompt,
            "model": MOCKAI_MODEL,
            "ground_truth": item["answer"],
            "embedded_answer": answer,
        })
    return prompts


def compute_reward(response_text: str, ground_truth: str) -> float:
    """Extract embedded answer from echoed response and compare to ground truth."""
    match = re.search(r"The answer is (\S+)\.", response_text)
    if match and match.group(1).strip() == ground_truth.strip():
        return 1.0
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
        statuses = {r.status for r in rollouts}
        done = all(s in terminal for s in statuses)
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
    """Run one batch of rollouts and collect results."""
    print(f"\n{'='*60}")
    print(f"  ITERATION {iteration} (model version={expected_version})")
    print(f"{'='*60}")

    # Build prompts
    prompts = build_prompts(dataset, BATCH_SIZE)

    # Enqueue rollouts
    requests = [
        EnqueueRolloutRequest(
            resources_id=resources_id,
            input=p,
            config={"image": AGENT_IMAGE, "command": AGENT_COMMAND},
        )
        for p in prompts
    ]
    rollouts = await client.enqueue_rollouts(requests)
    rollout_ids = [r.rollout_id for r in rollouts]
    print(f"  Enqueued {len(rollout_ids)} rollouts")

    # Wait for completion
    completed = await wait_for_rollouts(client, rollout_ids)
    succeeded = [r for r in completed if r.status == RolloutStatus.SUCCEEDED]
    failed = [r for r in completed if r.status == RolloutStatus.TERMINAL_FAILED]
    print(f"  Results: {len(succeeded)} succeeded, {len(failed)} failed")

    # Retrieve events and compute rewards
    total_reward = 0.0
    event_count = 0
    versions_seen = set()

    for rollout in succeeded:
        events = await client.get_events(rollout.rollout_id)
        event_count += len(events)

        for event in events:
            if event.event_type == "model_request":
                # Check version
                server = event.data.get("server", {})
                v = server.get("version")
                if v is not None:
                    versions_seen.add(v)

                # Compute reward from echoed response
                response = event.data.get("response", {})
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    # Find matching prompt by rollout_id
                    matching = [p for p in prompts if True]  # all prompts for now
                    gt = rollout.input.get("ground_truth", "")
                    reward = compute_reward(content, gt)
                    total_reward += reward

    avg_reward = total_reward / len(succeeded) if succeeded else 0.0
    print(f"  Events: {event_count}, Versions seen: {versions_seen}")
    print(f"  Average reward: {avg_reward:.2f}")

    return {
        "iteration": iteration,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "events": event_count,
        "versions": versions_seen,
        "avg_reward": avg_reward,
    }


async def main() -> None:
    # --- Config from env ---
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

        await asyncio.sleep(2)  # Simulate weight loading

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
        total_events = sum(r["events"] for r in results)
        all_versions = set()
        for r in results:
            all_versions.update(r["versions"])

        print(f"  Iterations: {len(results)}")
        print(f"  Total rollouts: {total_succeeded} succeeded, {total_failed} failed")
        print(f"  Total events: {total_events}")
        print(f"  Versions seen: {all_versions}")

        # --- Verify ---
        ok = True
        if total_succeeded < BATCH_SIZE * NUM_ITERATIONS * 0.5:
            print(f"\n  ❌ Too many failures: {total_failed}/{BATCH_SIZE * NUM_ITERATIONS}")
            ok = False
        if not all_versions:
            print("\n  ❌ No version info in events")
            ok = False
        if total_events == 0:
            print("\n  ❌ No events captured")
            ok = False

        if ok:
            print(f"\n  ✅ Math PoC completed successfully!")
        else:
            print(f"\n  ❌ Math PoC had issues — check above")
            sys.exit(1)

    finally:
        await client.close()


if __name__ == "__main__":
    random.seed(42)  # Reproducible prompt generation
    asyncio.run(main())
