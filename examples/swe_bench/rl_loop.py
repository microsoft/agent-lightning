#!/usr/bin/env python3
"""SWE-bench RL loop — task-agnostic orchestration.

Algorithm contract — for each rollout, the algorithm only sets:
  - input:        raw SWE-bench JSONL row (instance_id, problem_statement, etc.)
  - resources_id: link to the registered resource snapshot (job_template)
  - metadata:     algorithm control indexes (batch_idx, sample_idx_in_batch)

Everything else is handled by hooks loaded into the agl-lite server:
  - on_enqueue:   reads input, sets per-instance Docker image, generates eval_script,
                  injects env vars (AGL_TASK_INPUT, AGL_EVAL_SCRIPT, AGL_EVAL_META)
  - on_succeeded: reads test_output artifact from disk, grades using official
                  swebench get_eval_report(), posts reward event

Usage:
    export AGL_LITE_URL=http://localhost:8080
    export AGL_KEY=<your-key>
    export AGL_MODEL_NAME=<model>
    export AGL_MODEL_ENDPOINT=<endpoint>
    python examples/swe_bench/rl_loop.py
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

from agl_lite.client import AglLiteClient
from agl_lite.schemas.api import EnqueueRolloutRequest, RegisterModelRequest
from agl_lite.schemas.rollout import RolloutStatus

# --- Config (from env) ---

MODEL_NAME = os.environ.get("AGL_MODEL_NAME", "")
MODEL_ENDPOINT = os.environ.get("AGL_MODEL_ENDPOINT", "")
BATCH_SIZE = int(os.environ.get("AGL_BATCH_SIZE", "5"))
NUM_ITERATIONS = int(os.environ.get("AGL_NUM_ITERATIONS", "1"))
POLL_INTERVAL = int(os.environ.get("AGL_POLL_INTERVAL_SEC", "30"))
MAX_POLL_TIME = int(os.environ.get("AGL_MAX_POLL_TIME", "7200"))

EXAMPLE_DIR = Path(__file__).parent


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dataset() -> list[dict]:
    """Load SWE-bench instances from JSONL."""
    items = []
    dataset_path = EXAMPLE_DIR / "swebench_samples.jsonl"
    with open(dataset_path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


async def wait_for_rollouts(
    client: AglLiteClient,
    rollout_ids: list[str],
    max_time: float = MAX_POLL_TIME,
) -> list:
    terminal = {RolloutStatus.SUCCEEDED, RolloutStatus.TERMINAL_FAILED, RolloutStatus.CANCELLED}
    start = time.time()
    poll_count = 0
    while time.time() - start < max_time:
        rollouts = await client.query_rollouts(ids=rollout_ids, limit=len(rollout_ids))
        done = all(r.status in terminal for r in rollouts)

        status_counts: dict[str, int] = {}
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
) -> dict:
    """Run one batch of SWE-bench instances."""
    log(f"")
    log(f"{'='*60}")
    log(f"  ITERATION {iteration}")
    log(f"{'='*60}")

    # Build batch from dataset
    task_offset = (iteration - 1) * BATCH_SIZE
    batch = []
    for i in range(BATCH_SIZE):
        idx = (task_offset + i) % len(dataset)
        batch.append(dataset[idx])

    log(f"  Instances ({len(batch)}):")
    for i, item in enumerate(batch):
        iid = item["instance_id"]
        ps = item.get("problem_statement", "")[:60]
        log(f"    [{i}] {iid}: {ps}...")

    # Enqueue rollouts — algorithm only sets input, resources_id, metadata
    requests = [
        EnqueueRolloutRequest(
            resources_id=resources_id,
            input=item,  # full JSONL row — hooks handle image, eval_script, etc.
            metadata={"batch_idx": iteration, "sample_idx_in_batch": i},
        )
        for i, item in enumerate(batch)
    ]
    rollouts = await client.enqueue_rollouts(requests)
    rollout_ids = [r.rollout_id for r in rollouts]
    log(f"  Enqueued {len(rollout_ids)} rollouts")

    # Wait for completion
    log(f"  Polling for completion...")
    completed = await wait_for_rollouts(client, rollout_ids)
    succeeded = [r for r in completed if r.status == RolloutStatus.SUCCEEDED]
    failed = [r for r in completed if r.status == RolloutStatus.TERMINAL_FAILED]

    log(f"  Completed: {len(succeeded)} succeeded, {len(failed)} failed")
    for r in failed:
        log(f"    FAILED: {r.rollout_id} -- {r.error_message}")

    # Collect results from events (rewards posted by on_succeeded hook)
    resolved_count = 0
    total_rollouts = len(succeeded) + len(failed)

    log(f"  Results:")
    for rollout in completed:
        events = await client.get_events(rollout.rollout_id)
        rw_events = [e for e in events if e.event_type == "reward"]

        if rw_events:
            rw = rw_events[-1]
            resolved = rw.data.get("resolved", False)
            instance_id = rw.data.get("instance_id", "?")
            reason = rw.data.get("reason", "")
            patch_size = rw.data.get("patch_size", 0)

            tag = "✓" if resolved else "✗"
            if resolved:
                resolved_count += 1
            log(f"    {tag} {instance_id}: {reason} (patch: {patch_size}B)")
        else:
            instance_id = rollout.input.get("instance_id", "?") if isinstance(rollout.input, dict) else "?"
            log(f"    ? {instance_id}: no reward event")

    log(f"")
    log(f"  --- Iteration {iteration} Summary ---")
    log(f"  Resolved: {resolved_count}/{total_rollouts}")
    log(f"  Succeeded: {len(succeeded)}, Failed: {len(failed)}")

    return {
        "iteration": iteration,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "resolved": resolved_count,
        "total": total_rollouts,
    }


async def main() -> None:
    base_url = os.environ.get("AGL_LITE_URL", "http://localhost:8080")
    agl_key = os.environ.get("AGL_KEY")

    log(f"=== SWE-bench RL Loop ===")
    log(f"  agl-lite:  {base_url}")
    log(f"  model:     {MODEL_NAME}")
    if MODEL_ENDPOINT:
        log(f"  endpoint:  {MODEL_ENDPOINT}")
    log(f"  batch:     {BATCH_SIZE}, iterations: {NUM_ITERATIONS}")

    client = AglLiteClient(base_url=base_url, agl_key=agl_key)

    try:
        # --- Setup: register resources ---
        log(f"")
        log(f"--- Setup ---")
        template_path = EXAMPLE_DIR / "job-template.yaml"
        with open(template_path) as f:
            job_template = yaml.safe_load(f)
        log(f"  Job template: {template_path}")

        res = await client.add_resources({"job_template": job_template})
        resources_id = res.resources_id
        log(f"  Resources registered: {resources_id}")

        dataset = load_dataset()
        log(f"  Dataset loaded: {len(dataset)} instances")

        # --- Register model server ---
        if MODEL_ENDPOINT and MODEL_NAME:
            log(f"")
            log(f"--- Register model server ---")
            await client.register_models([
                RegisterModelRequest(model=MODEL_NAME, endpoint=MODEL_ENDPOINT, version=1),
            ])
            log(f"  {MODEL_NAME} -> {MODEL_ENDPOINT} (version=1)")

        # --- Run iterations ---
        results = []
        for i in range(1, NUM_ITERATIONS + 1):
            r = await run_iteration(client, resources_id, dataset, iteration=i)
            results.append(r)

        # --- Final Summary ---
        log(f"")
        log(f"{'='*60}")
        log(f"  FINAL SUMMARY")
        log(f"{'='*60}")
        total_resolved = sum(r["resolved"] for r in results)
        total_instances = sum(r["total"] for r in results)
        total_succeeded = sum(r["succeeded"] for r in results)
        total_failed = sum(r["failed"] for r in results)

        log(f"  Iterations: {len(results)}")
        log(f"  Total instances: {total_instances}")
        log(f"  Resolved: {total_resolved}/{total_instances} ({total_resolved/max(total_instances,1):.1%})")
        log(f"  Succeeded: {total_succeeded}, Failed: {total_failed}")
        for r in results:
            log(f"  Iter {r['iteration']}: {r['resolved']}/{r['total']} resolved")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
