#!/usr/bin/env python3
"""RL loop — real inference with vLLM on GSM8K math problems.

Single iteration (model frozen in Phase 4b):
  Register resources + model (v1) → enqueue batch → poll → retrieve → rewards

The agent calls a real LLM (e.g., Qwen2.5-1.5B-Instruct) via the gateway.
Reward = 1.0 if the model's \\boxed{answer} matches ground truth numerically.

Event flow per rollout:
  model_request (auto, gateway)  →  agent_output (agent)  →  reward (algorithm)

Usage:
    export AGL_LITE_URL=http://localhost:8080
    export AGL_KEY=<your-key>
    export AGL_K8S_NAMESPACE=<namespace>
    export AGL_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
    export AGL_MODEL_ENDPOINT=http://host.minikube.internal:8010/v1
    python examples/math-poc/rl_loop.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from agl_lite.client import AglLiteClient, AglLiteError
from agl_lite.schemas.api import EnqueueRolloutRequest, PostEventRequest, RegisterModelRequest
from agl_lite.schemas.rollout import RolloutStatus

# --- Config (from env, set by run.sh from deploy/.env) ---

MODEL_NAME = os.environ.get("AGL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
BATCH_SIZE = int(os.environ.get("AGL_BATCH_SIZE", "5"))
NUM_ITERATIONS = int(os.environ.get("AGL_NUM_ITERATIONS", "1"))
POLL_INTERVAL = 5  # real inference is slower
MAX_POLL_TIME = 300  # 5 min — real models take longer

DATA_DIR = Path(__file__).parent / "data"
TEMPLATE_PATH = Path(__file__).parent / "job-template.yaml"


def log(msg: str) -> None:
    """Print a log message without timestamp (reproducible structure)."""
    print(msg, flush=True)


def load_dataset() -> list[dict]:
    """Load GSM8K sample dataset."""
    items = []
    with open(DATA_DIR / "gsm8k_sample.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def normalize_number(s: str) -> float | None:
    """Try to parse a string as a number, handling common formats.

    Handles: "18", "18.0", "18.00", "$18", "18,000", "1,234.56", "-5"
    Returns None if unparseable.
    """
    if not s:
        return None
    # Strip whitespace, $, commas
    cleaned = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def compute_reward(agent_answer: str | None, ground_truth: str) -> tuple[float, str]:
    """Compute reward by numeric comparison.

    Returns (reward, reason).
    """
    if agent_answer is None:
        return 0.0, "no answer extracted"

    agent_num = normalize_number(agent_answer)
    gt_num = normalize_number(ground_truth)

    if agent_num is None:
        return 0.0, f"agent answer not numeric: {agent_answer!r}"
    if gt_num is None:
        return 0.0, f"ground truth not numeric: {ground_truth!r}"

    if abs(agent_num - gt_num) < 1e-6:
        return 1.0, "correct"
    else:
        return 0.0, f"wrong: {agent_num} != {gt_num}"


def build_tasks(dataset: list[dict], batch_size: int, offset: int = 0) -> list[dict]:
    """Build task inputs — plain questions for real model inference.

    No \\boxed{} embedding — the model generates its own answer.
    """
    tasks = []
    for i in range(batch_size):
        idx = (offset + i) % len(dataset)
        item = dataset[idx]
        tasks.append({"input": item["question"], "ground_truth": item["answer"]})
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
    model_version: int,
) -> dict:
    """Run one batch of rollouts, collect agent outputs, compute and post rewards."""
    log(f"")
    log(f"{'='*60}")
    log(f"  ITERATION {iteration} (model version={model_version})")
    log(f"{'='*60}")

    # Build tasks
    task_offset = (iteration - 1) * BATCH_SIZE
    tasks = build_tasks(dataset, BATCH_SIZE, offset=task_offset)

    log(f"  Tasks ({len(tasks)}):")
    for i, t in enumerate(tasks):
        q_preview = t["input"][:70] + "..." if len(t["input"]) > 70 else t["input"]
        log(f"    [{i}] Q: {q_preview}")
        log(f"         GT: {t['ground_truth']}")

    # Enqueue rollouts
    requests = [
        EnqueueRolloutRequest(
            resources_id=resources_id,
            input=t["input"],
            config={"environment_variables": {"AGL_MODEL_NAME": MODEL_NAME}},
        )
        for t in tasks
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
        log(f"    FAILED: {r.rollout_id} — {r.error_message}")

    # Retrieve events, compute rewards, post reward events
    total_reward = 0.0
    model_request_count = 0
    agent_output_count = 0
    reward_count = 0
    versions_seen = set()
    structural_failures = []

    first_mr_logged = False

    log(f"  Results:")
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

        # --- Log first model_request event in detail ---
        for event in mr_events:
            d = event.data
            req = d.get("request", {})
            resp = d.get("response")
            srv = d.get("server", {})

            if not first_mr_logged:
                log(f"")
                log(f"  ── First model_request event (sample) ──")
                log(f"    server:   model={srv.get('model')}, version={srv.get('version')}")
                log(f"    request.model:    {req.get('model')}")
                log(f"    request.stream:   {req.get('stream')}")
                log(f"    request.messages: {len(req.get('messages', []))} messages")
                if isinstance(resp, list) and resp:
                    # Reassemble streamed content
                    assembled = ""
                    for chunk in resp:
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            assembled += delta.get("content", "")
                    log(f"    response: {len(resp)} SSE chunks (streaming)")
                    log(f"      content ({len(assembled)} chars):")
                    # Print first 300 chars of LLM reasoning
                    for line in assembled[:300].split("\n"):
                        log(f"        {line}")
                    if len(assembled) > 300:
                        log(f"        ...")
                elif isinstance(resp, dict):
                    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                    log(f"    response: non-streaming, {len(content)} chars")
                log(f"  ── end sample ──")
                log(f"")
                first_mr_logged = True

            # Structural checks on every model_request
            has_request = "request" in d
            has_response = "response" in d
            has_server = "server" in d
            has_model = "model" in req
            has_messages = "messages" in req and len(req["messages"]) >= 2
            has_stream = "stream" in req
            is_streaming = req.get("stream") is True
            has_version = "version" in srv and srv["version"] is not None
            resp_nonempty = bool(resp)

            checks = [
                ("has request", has_request),
                ("has response", has_response),
                ("has server", has_server),
                ("has model in request", has_model),
                ("has ≥2 messages", has_messages),
                ("has stream flag", has_stream),
                ("stream=True", is_streaming),
                ("has version in server", has_version),
                ("response non-empty", resp_nonempty),
            ]
            for check_name, ok in checks:
                if not ok:
                    structural_failures.append(
                        f"{rollout.rollout_id}: model_request: {check_name}"
                    )

        # Get agent answer
        agent_answer = None
        attempt_id = None
        for event in ao_events:
            agent_answer = event.data.get("answer", "")
            attempt_id = event.attempt_id
            break

        # Find ground truth
        matching_task = next(
            (t for t in tasks if t["input"] == rollout.input),
            None,
        )
        gt = matching_task["ground_truth"] if matching_task else "?"

        # Compute reward (numeric comparison)
        reward, reason = compute_reward(agent_answer, gt)
        total_reward += reward

        tag = "✓" if reward > 0 else "✗"
        log(f"    {rollout.rollout_id}: [{tag}] answer={agent_answer!r}, gt={gt!r} → {reason}")

        # Post reward event
        if attempt_id:
            try:
                await client.post_event(
                    rollout.rollout_id,
                    attempt_id,
                    PostEventRequest(
                        event_type="reward",
                        data={"value": reward, "ground_truth": gt, "agent_answer": agent_answer, "reason": reason},
                    ),
                )
                reward_count += 1
            except AglLiteError as e:
                log(f"      WARNING: failed to post reward: {e}")

    avg_reward = total_reward / len(succeeded) if succeeded else 0.0

    log(f"")
    log(f"  --- Iteration {iteration} Summary ---")
    log(f"  Rollouts: {len(succeeded)} succeeded, {len(failed)} failed")
    log(f"  Events: {model_request_count} model_request, {agent_output_count} agent_output, {reward_count} reward")
    log(f"  Versions seen: {sorted(versions_seen)}")
    log(f"  Average reward: {avg_reward:.2f} ({int(total_reward)}/{len(succeeded)} correct)")
    if structural_failures:
        log(f"  ⚠ Structural failures ({len(structural_failures)}):")
        for f in structural_failures:
            log(f"    - {f}")
    else:
        log(f"  ✅ All structural checks passed")

    return {
        "iteration": iteration,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "model_requests": model_request_count,
        "agent_outputs": agent_output_count,
        "rewards": reward_count,
        "versions": versions_seen,
        "avg_reward": avg_reward,
        "total_reward": total_reward,
        "structural_failures": structural_failures,
    }


async def main() -> None:
    base_url = os.environ.get("AGL_LITE_URL", "http://localhost:8080")
    agl_key = os.environ.get("AGL_KEY")
    namespace = os.environ.get("AGL_K8S_NAMESPACE", "agl")
    model_endpoint = os.environ["AGL_MODEL_ENDPOINT"]  # required for vllm mode

    log(f"=== Math PoC — RL Loop (vLLM) ===")
    log(f"  agl-lite:  {base_url}")
    log(f"  model:     {MODEL_NAME} → {model_endpoint}")
    log(f"  namespace: {namespace}")
    log(f"  batch:     {BATCH_SIZE}, iterations: {NUM_ITERATIONS}")

    client = AglLiteClient(base_url=base_url, agl_key=agl_key)

    try:
        # --- Setup: register resources ---
        log(f"")
        log(f"--- Setup ---")
        with open(TEMPLATE_PATH) as f:
            job_template = yaml.safe_load(f)
        log(f"  Job template: {TEMPLATE_PATH.name}")

        res = await client.add_resources({"job_template": job_template})
        resources_id = res.resources_id
        log(f"  Resources registered: {resources_id}")

        dataset = load_dataset()
        log(f"  Dataset loaded: {len(dataset)} problems")

        # --- Register model server ---
        log(f"")
        log(f"--- Register model server (v1) ---")
        await client.register_models([
            RegisterModelRequest(model=MODEL_NAME, endpoint=model_endpoint, version=1),
        ])
        log(f"  {MODEL_NAME} → {model_endpoint} (version=1)")

        # --- Run iterations ---
        results = []
        for i in range(1, NUM_ITERATIONS + 1):
            r = await run_iteration(client, resources_id, dataset, iteration=i, model_version=1)
            results.append(r)

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
        total_correct = sum(r["total_reward"] for r in results)
        all_versions = set()
        all_structural = []
        for r in results:
            all_versions.update(r["versions"])
            all_structural.extend(r.get("structural_failures", []))

        log(f"  Iterations: {len(results)}")
        log(f"  Rollouts: {total_succeeded} succeeded, {total_failed} failed")
        log(f"  Events: {total_mr} model_request, {total_ao} agent_output, {total_rw} reward")
        log(f"  Accuracy: {int(total_correct)}/{total_succeeded} = {total_correct/total_succeeded:.1%}" if total_succeeded else "  Accuracy: N/A")
        for r in results:
            log(f"  Iter {r['iteration']}: reward={r['avg_reward']:.2f} ({int(r['total_reward'])}/{r['succeeded']})")

        # --- Verify ---
        checks = []
        checks.append(("Rollouts succeeded", total_succeeded > 0, f"{total_succeeded}/{total_succeeded + total_failed}"))
        checks.append(("model_request events", total_mr > 0, str(total_mr)))
        checks.append(("agent_output events", total_ao > 0, str(total_ao)))
        checks.append(("reward events", total_rw > 0, str(total_rw)))
        checks.append(("Version 1 seen", 1 in all_versions, str(all_versions)))
        checks.append(("Structural checks", len(all_structural) == 0, f"{len(all_structural)} failures"))
        # Sanity: at least some rewards should be non-zero (model should get easy ones right)
        checks.append(("Some correct answers", total_correct > 0, f"{int(total_correct)} correct"))

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
            log(f"  ✅ Math PoC (vLLM) completed successfully!")
        else:
            log(f"  ❌ Math PoC (vLLM) had failures — check above")
            sys.exit(1)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
