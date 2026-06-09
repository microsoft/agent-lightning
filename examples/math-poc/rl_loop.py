#!/usr/bin/env python3
"""Unified RL loop — task-agnostic orchestration for math-poc.

Algorithm contract — for each rollout, the algorithm only sets:
  - input:     raw dataset row (full JSONL content, e.g., {"question": ..., "answer": ...})
    - metadata:  algorithm control indexes (batch_idx, etc.)

Everything else is handled by hooks loaded into the agl-lite server:
  - on_startup:   loads pod spec template from AGL_POD_SPEC_TEMPLATE
  - on_enqueue:   reads input, builds pod spec, injects AGL_TASK_INPUT into container env
  - on_succeeded: reads rollout.input for ground_truth, extracts answer from events,
                  computes reward, posts reward event to store."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agl_lite.client import AglLiteClient
from agl_lite.schemas import Model, RolloutCreate, RolloutState

# --- Config (from env, set by run.sh) ---
MODEL_NAME = os.environ.get("AGL_MODEL_NAME", "mock-llm")
MODEL_ENDPOINT = os.environ.get("AGL_MODEL_ENDPOINT", "")
BATCH_SIZE = int(os.environ.get("AGL_BATCH_SIZE", "5"))
NUM_ITERATIONS = int(os.environ.get("AGL_NUM_ITERATIONS", "1"))
POLL_INTERVAL = int(os.environ.get("AGL_POLL_INTERVAL_SEC", "5"))
MAX_POLL_TIME = int(os.environ.get("AGL_MAX_POLL_TIME", "300"))
LOG_DIR = os.environ.get("AGL_LOG_DIR")
DATA_DIR = Path(__file__).resolve().parent / "data"

_log_file = None


def _setup_log_file() -> None:
    global _log_file
    if LOG_DIR:
        log_path = Path(LOG_DIR) / "rl_loop.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115


def log(msg: str) -> None:
    print(msg, flush=True)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()


def load_dataset() -> list[dict]:
    items = []
    with open(DATA_DIR / "gsm8k_sample.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


async def wait_for_rollouts(
    client: AglLiteClient,
    rollout_ids: list[str],
    max_time: float = MAX_POLL_TIME,
) -> list:
    terminal = {RolloutState.SUCCEEDED, RolloutState.FAILED}
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
    dataset: list[dict],
    iteration: int,
) -> dict:
    """Run one batch of rollouts, collect results from events (rewards posted by hooks)."""
    log("")
    log(f"{'=' * 60}")
    log(f"  ITERATION {iteration}")
    log(f"{'=' * 60}")

    # Build batch from dataset
    task_offset = (iteration - 1) * BATCH_SIZE
    batch = []
    for i in range(BATCH_SIZE):
        idx = (task_offset + i) % len(dataset)
        batch.append(dataset[idx])

    log(f"  Tasks ({len(batch)}):")
    for i, item in enumerate(batch):
        q = item["question"][:70] + "..." if len(item["question"]) > 70 else item["question"]
        log(f"    [{i}] Q: {q}")
        log(f"         GT: {item['answer']}")

    # Enqueue rollouts — algorithm sets input and metadata only.
    # on_enqueue hook assembles the pod spec and injects per-sample env vars.
    requests = [
        RolloutCreate(
            input=item,
            metadata={"batch_idx": iteration},
        )
        for item in batch
    ]
    rollouts = await client.enqueue_rollouts(requests)
    rollout_ids = [r.rollout_id for r in rollouts]
    log(f"  Enqueued {len(rollout_ids)} rollouts")

    # Wait for completion
    log("  Polling for completion...")
    completed = await wait_for_rollouts(client, rollout_ids)
    succeeded = [r for r in completed if r.status == RolloutState.SUCCEEDED]
    failed = [r for r in completed if r.status == RolloutState.FAILED]

    log(f"  Completed: {len(succeeded)} succeeded, {len(failed)} failed")
    for r in failed:
        log(f"    FAILED: {r.rollout_id} -- {r.error_message}")

    # Collect results from events (rewards already posted by hooks)
    total_reward = 0.0
    reward_count = 0
    model_request_count = 0

    log("  Results:")
    for rollout in succeeded:
        events = await client.get_events(rollout.rollout_id)

        mr_events = [e for e in events if e.event_type == "model_request"]
        rw_events = [e for e in events if e.event_type == "reward"]
        model_request_count += len(mr_events)

        if rw_events:
            rw = rw_events[-1]  # last reward event
            reward = rw.data.get("value", 0.0)
            total_reward += reward
            reward_count += 1
            gt = rw.data.get("ground_truth", "?")
            answer = rw.data.get("agent_answer", "?")
            reason = rw.data.get("reason", "")
            tag = "+" if reward > 0 else "-"
            detail = reason if reason else f"answer={answer!r}, gt={gt!r}"
            log(f"    {rollout.rollout_id}: [{tag}] {detail}")
        else:
            log(f"    {rollout.rollout_id}: [?] no reward event (hook may have failed)")

    avg_reward = total_reward / len(succeeded) if succeeded else 0.0

    log("")
    log(f"  --- Iteration {iteration} Summary ---")
    log(f"  Rollouts: {len(succeeded)} succeeded, {len(failed)} failed")
    log(f"  Events: {model_request_count} model_request, {reward_count} reward")
    log(f"  Average reward: {avg_reward:.2f} ({int(total_reward)}/{len(succeeded)} correct)")

    return {
        "iteration": iteration,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "model_requests": model_request_count,
        "rewards": reward_count,
        "avg_reward": avg_reward,
        "total_reward": total_reward,
    }


async def main() -> None:
    _setup_log_file()
    base_url = os.environ.get("AGL_BASE_URL", "http://localhost:8080")
    agl_key = os.environ.get("AGL_KEY")
    mode = os.environ.get("AGL_MODEL_MODE", "vllm")

    log(f"=== Math PoC -- RL Loop ({mode}) ===")
    log(f"  agl-lite:  {base_url}")
    log(f"  model:     {MODEL_NAME}")
    if LOG_DIR:
        log(f"  log dir:   {LOG_DIR}")
    if MODEL_ENDPOINT:
        log(f"  endpoint:  {MODEL_ENDPOINT}")
    log(f"  batch:     {BATCH_SIZE}, iterations: {NUM_ITERATIONS}")

    client = AglLiteClient(base_url=base_url, agl_key=agl_key)

    try:
        log("")
        log("--- Setup ---")
        dataset = load_dataset()
        log(f"  Dataset loaded: {len(dataset)} problems")

        # --- Register model server ---
        if MODEL_ENDPOINT:
            log("")
            log("--- Register model server ---")
            await client.register_models(
                [
                    Model(model=MODEL_NAME, endpoint=MODEL_ENDPOINT, version=1),
                ]
            )
            log(f"  {MODEL_NAME} -> {MODEL_ENDPOINT} (version=1)")

        # --- Run iterations ---
        results = []
        for i in range(1, NUM_ITERATIONS + 1):
            r = await run_iteration(client, dataset, iteration=i)
            results.append(r)

        # --- Final Summary ---
        log("")
        log(f"{'=' * 60}")
        log("  FINAL SUMMARY")
        log(f"{'=' * 60}")
        total_succeeded = sum(r["succeeded"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        total_correct = sum(r["total_reward"] for r in results)

        log(f"  Iterations: {len(results)}")
        log(f"  Rollouts: {total_succeeded} succeeded, {total_failed} failed")
        if total_succeeded:
            log(f"  Accuracy: {int(total_correct)}/{total_succeeded} = {total_correct / total_succeeded:.1%}")
        for r in results:
            log(f"  Iter {r['iteration']}: reward={r['avg_reward']:.2f} ({int(r['total_reward'])}/{r['succeeded']})")

        # --- Verify ---
        checks = []
        checks.append(
            ("Rollouts succeeded", total_succeeded > 0, f"{total_succeeded}/{total_succeeded + total_failed}")
        )
        checks.append(
            ("Reward events", sum(r["rewards"] for r in results) > 0, str(sum(r["rewards"] for r in results)))
        )

        log("")
        log("  Checks:")
        all_ok = True
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            log(f"    [{status}] {name}: {detail}")
            if not passed:
                all_ok = False

        log("")
        if all_ok:
            log(f"  Math PoC ({mode}) completed successfully!")
        else:
            log(f"  Math PoC ({mode}) had failures -- check above")
            sys.exit(1)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
