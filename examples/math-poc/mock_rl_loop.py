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
MAX_POLL_TIME = 180

DATA_DIR = Path(__file__).parent / "data"
TEMPLATE_PATH = Path(__file__).parent / "job-template.yaml"


def log(msg: str) -> None:
    """Print a log message without timestamp (reproducible output)."""
    print(msg, flush=True)


def load_dataset() -> list[dict]:
    """Load GSM8K sample dataset."""
    items = []
    with open(DATA_DIR / "gsm8k_sample.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_tasks(dataset: list[dict], batch_size: int, offset: int = 0) -> list[dict]:
    """Build task inputs with embedded \\boxed{} answers for mockai echo mode.

    Mockai echoes the last user message, so we embed \\boxed{answer} in the
    question text. The agent's parser extracts it as the "model response".

    Alternating pattern: even index → correct answer, odd index → wrong answer.
    This gives deterministic, reproducible mixed rewards (no randomness).
    """
    WRONG_ANSWER = "WRONG"
    tasks = []
    for i in range(batch_size):
        idx = (offset + i) % len(dataset)
        item = dataset[idx]
        correct = i % 2 == 0  # even=correct, odd=wrong
        boxed_value = item["answer"] if correct else WRONG_ANSWER
        # Append \boxed{} to question so mockai echo returns it
        augmented_input = item["question"] + f"\n\\boxed{{{boxed_value}}}"
        tasks.append({
            "input": augmented_input,
            "ground_truth": item["answer"],
            "expect_correct": correct,
            "boxed_value": boxed_value,
        })
    return tasks


async def wait_for_rollouts(
    client: AglLiteClient,
    rollout_ids: list[str],
    max_time: float = MAX_POLL_TIME,
) -> list:
    """Poll until all rollouts reach a terminal state."""
    terminal = {RolloutStatus.SUCCEEDED, RolloutStatus.TERMINAL_FAILED, RolloutStatus.CANCELLED}
    start = time.time()
    poll_count = 0
    while time.time() - start < max_time:
        rollouts = await client.query_rollouts(ids=rollout_ids, limit=len(rollout_ids))
        done = all(r.status in terminal for r in rollouts)

        status_counts = {}
        for r in rollouts:
            status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1

        poll_count += 1
        if done:
            log(f"  [poll #{poll_count}] All done: {status_counts}")
            return rollouts

        log(f"  [poll #{poll_count}] {status_counts}")
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
    log(f"")
    log(f"{'='*60}")
    log(f"  ITERATION {iteration} (expected model version={expected_version})")
    log(f"{'='*60}")

    # Build tasks (deterministic: iteration 1 uses items 0-4, iteration 2 uses 5-9)
    task_offset = (iteration - 1) * BATCH_SIZE
    tasks = build_tasks(dataset, BATCH_SIZE, offset=task_offset)

    log(f"  Tasks ({len(tasks)}):")
    for i, t in enumerate(tasks):
        q_preview = t["input"][:60] + "..." if len(t["input"]) > 60 else t["input"]
        tag = "✓" if t["expect_correct"] else "✗"
        log(f"    [{i}] {tag} Q: {q_preview}")
        log(f"         A: {t['ground_truth']}, boxed: {t['boxed_value']}")

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

    log(f"  Enqueued {len(rollout_ids)} rollouts:")
    for i, r in enumerate(rollouts):
        log(f"    [{i}] {r.rollout_id} → {r.status.value}")

    # Wait for completion
    log(f"  Polling for completion...")
    completed = await wait_for_rollouts(client, rollout_ids)
    succeeded = [r for r in completed if r.status == RolloutStatus.SUCCEEDED]
    failed = [r for r in completed if r.status == RolloutStatus.TERMINAL_FAILED]

    log(f"  Completed: {len(succeeded)} succeeded, {len(failed)} failed")
    for r in failed:
        log(f"    FAILED: {r.rollout_id} — {r.error_message}")

    # Retrieve events, compute rewards, post reward events
    total_reward = 0.0
    model_request_count = 0
    agent_output_count = 0
    reward_count = 0
    versions_seen = set()
    assertion_failures = []

    log(f"  Collecting events and computing rewards:")
    for rollout in succeeded:
        events = await client.get_events(rollout.rollout_id)

        # Classify events
        mr_events = [e for e in events if e.event_type == "model_request"]
        ao_events = [e for e in events if e.event_type == "agent_output"]
        model_request_count += len(mr_events)
        agent_output_count += len(ao_events)

        # Check version from model_request
        for event in mr_events:
            server = event.data.get("server", {})
            v = server.get("version")
            if v is not None:
                versions_seen.add(v)

        # Get agent answer
        agent_answer = None
        attempt_id = None
        for event in ao_events:
            agent_answer = event.data.get("answer", "")
            attempt_id = event.attempt_id
            break

        # Find ground truth and expected pattern
        matching_task = next(
            (t for t in tasks if t["input"] == rollout.input),
            None,
        )
        gt = matching_task["ground_truth"] if matching_task else "?"
        expect_correct = matching_task["expect_correct"] if matching_task else None
        boxed_value = matching_task["boxed_value"] if matching_task else "?"

        # Compute reward
        reward = 1.0 if agent_answer and agent_answer.strip() == gt.strip() else 0.0
        total_reward += reward

        # Assert agent extracted the embedded boxed value
        expected_reward = 1.0 if expect_correct else 0.0
        answer_ok = agent_answer is not None and agent_answer.strip() == boxed_value.strip()
        reward_ok = reward == expected_reward

        q_preview = str(rollout.input)[:40] + "..." if len(str(rollout.input)) > 40 else str(rollout.input)
        tag = "✓" if expect_correct else "✗"
        log(f"    {rollout.rollout_id}: [{tag}]")
        log(f"      events: {len(mr_events)} model_request, {len(ao_events)} agent_output")
        log(f"      boxed={boxed_value!r}, agent_answer={agent_answer!r}, ground_truth={gt!r}")
        log(f"      reward={reward} (expected={expected_reward})")
        if not answer_ok:
            log(f"      ⚠ ANSWER MISMATCH: expected boxed_value={boxed_value!r}, got {agent_answer!r}")
            assertion_failures.append(f"{rollout.rollout_id}: answer {agent_answer!r} != boxed {boxed_value!r}")
        if not reward_ok:
            log(f"      ⚠ REWARD MISMATCH: expected {expected_reward}, got {reward}")
            assertion_failures.append(f"{rollout.rollout_id}: reward {reward} != expected {expected_reward}")

        # Post reward event
        if attempt_id:
            try:
                await client.post_event(
                    rollout.rollout_id,
                    attempt_id,
                    PostEventRequest(
                        event_type="reward",
                        data={"value": reward, "ground_truth": gt, "agent_answer": agent_answer},
                    ),
                )
                reward_count += 1
            except AglLiteError as e:
                log(f"      WARNING: failed to post reward: {e}")

    avg_reward = total_reward / len(succeeded) if succeeded else 0.0
    expect_correct_count = sum(1 for t in tasks if t["expect_correct"])
    expect_wrong_count = len(tasks) - expect_correct_count
    expected_avg = expect_correct_count / len(tasks) if tasks else 0.0

    log(f"  --- Iteration {iteration} Summary ---")
    log(f"  Events: {model_request_count} model_request, {agent_output_count} agent_output, {reward_count} reward")
    log(f"  Versions seen: {sorted(versions_seen)}")
    log(f"  Pattern: {expect_correct_count} correct + {expect_wrong_count} wrong → expected avg={expected_avg:.2f}")
    log(f"  Average reward: {avg_reward:.2f}")
    if assertion_failures:
        log(f"  ⚠ Assertion failures ({len(assertion_failures)}):")
        for f in assertion_failures:
            log(f"    - {f}")
    else:
        log(f"  ✅ All assertions passed")

    return {
        "iteration": iteration,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "model_requests": model_request_count,
        "agent_outputs": agent_output_count,
        "rewards": reward_count,
        "versions": versions_seen,
        "avg_reward": avg_reward,
        "assertion_failures": assertion_failures,
    }


async def main() -> None:
    base_url = os.environ.get("AGL_LITE_URL", "http://localhost:8080")
    agl_key = os.environ.get("AGL_KEY")
    namespace = os.environ.get("AGL_K8S_NAMESPACE", "agl")

    mockai_url = f"http://mockai.{namespace}.svc.cluster.local:5002/v1"

    log(f"=== Math PoC — Mock RL Loop ===")
    log(f"  agl-lite:  {base_url}")
    log(f"  mockai:    {mockai_url}")
    log(f"  namespace: {namespace}")

    client = AglLiteClient(base_url=base_url, agl_key=agl_key)

    try:
        # --- Setup: register resources ---
        log(f"")
        log(f"--- Setup ---")
        with open(TEMPLATE_PATH) as f:
            job_template = yaml.safe_load(f)
        log(f"  Job template: {TEMPLATE_PATH.name}")
        log(f"    containers: {[c['name'] for c in job_template.get('containers', [])]}")

        res = await client.add_resources({"job_template": job_template})
        resources_id = res.resources_id
        log(f"  Resources registered: {resources_id}")

        dataset = load_dataset()
        log(f"  Dataset loaded: {len(dataset)} problems from {DATA_DIR.name}/")

        # --- Register model server (v1) ---
        log(f"")
        log(f"--- Register model server (v1) ---")
        await client.register_models([
            RegisterModelRequest(model=MOCKAI_MODEL, endpoint=mockai_url, version=1),
        ])
        log(f"  {MOCKAI_MODEL} → {mockai_url} (version=1)")

        # --- Iteration 1 ---
        results = []
        r1 = await run_iteration(client, resources_id, dataset, iteration=1, expected_version=1)
        results.append(r1)

        # --- Weight update: v1 → v2 ---
        log(f"")
        log(f"{'='*60}")
        log(f"  WEIGHT UPDATE: v1 → v2")
        log(f"{'='*60}")

        log(f"  Deregistering model '{MOCKAI_MODEL}'...")
        await client.delete_model(MOCKAI_MODEL)
        log(f"  Model pool empty — gateway returns 503 for new requests")

        log(f"  Simulating weight loading (2s)...")
        await asyncio.sleep(2)

        log(f"  Re-registering model (v2)...")
        await client.register_models([
            RegisterModelRequest(model=MOCKAI_MODEL, endpoint=mockai_url, version=2),
        ])
        log(f"  {MOCKAI_MODEL} → {mockai_url} (version=2)")

        # --- Iteration 2 ---
        r2 = await run_iteration(client, resources_id, dataset, iteration=2, expected_version=2)
        results.append(r2)

        # --- Final Summary ---
        log(f"")
        log(f"{'='*60}")
        log(f"  FINAL SUMMARY")
        log(f"{'='*60}")
        total_succeeded = sum(r["succeeded"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        total_mr = sum(r["model_requests"] for r in results)
        total_ao = sum(r["agent_outputs"] for r in results)
        total_rw = sum(r["rewards"] for r in results)
        all_versions = set()
        for r in results:
            all_versions.update(r["versions"])

        log(f"  Iterations: {len(results)}")
        log(f"  Rollouts: {total_succeeded} succeeded, {total_failed} failed")
        log(f"  Events: {total_mr} model_request, {total_ao} agent_output, {total_rw} reward")
        log(f"  Versions seen: {sorted(all_versions)}")
        for r in results:
            log(f"  Iter {r['iteration']}: reward={r['avg_reward']:.2f}, versions={sorted(r['versions'])}")

        # Collect assertion failures
        all_assertion_failures = []
        for r in results:
            all_assertion_failures.extend(r.get("assertion_failures", []))

        # --- Verify ---
        checks = []
        checks.append(("Rollouts succeeded", total_succeeded >= BATCH_SIZE, f"{total_succeeded}/{BATCH_SIZE * NUM_ITERATIONS}"))
        checks.append(("model_request events", total_mr > 0, str(total_mr)))
        checks.append(("agent_output events", total_ao > 0, str(total_ao)))
        checks.append(("reward events", total_rw > 0, str(total_rw)))
        checks.append(("Version 1 seen", 1 in all_versions, str(all_versions)))
        checks.append(("Version 2 seen", 2 in all_versions, str(all_versions)))
        checks.append(("Answer assertions", len(all_assertion_failures) == 0, f"{len(all_assertion_failures)} failures"))

        log(f"")
        log(f"  Checks:")
        all_ok = True
        for name, passed, detail in checks:
            status = "✅" if passed else "❌"
            log(f"    {status} {name}: {detail}")
            if not passed:
                all_ok = False

        log(f"")
        if all_ok:
            log(f"  ✅ Math PoC completed successfully!")
        else:
            log(f"  ❌ Math PoC had failures — check above")
            sys.exit(1)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
