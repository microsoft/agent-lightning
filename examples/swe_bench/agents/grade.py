#!/usr/bin/env python3
"""SWE-bench grading and event reporting.

Reads outputs from a local directory, posts agent_output event
(patch summary), then grades test output and posts reward event.

Usage:
    python3 grade.py <output_dir> <artifact_path>

Arguments:
    output_dir     — local directory containing patch.diff and test_output.txt
    artifact_path  — relative path (rollout_id/attempt_id) for artifact location

Expected env vars:
    AGL_EVAL_META   — JSON with instance_id, repo, version, FAIL_TO_PASS, PASS_TO_PASS
    AGL_EVENT_URL   — URL to post events
    AGL_KEY         — API key (fallback: OPENAI_API_KEY)
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace


def post_event(event_type: str, data: dict) -> None:
    """Post an event to the agl-lite server."""
    event_url = os.environ.get("AGL_EVENT_URL", "")
    if not event_url:
        print(f"WARNING: AGL_EVENT_URL not set, skipping {event_type} event")
        return

    api_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY", "")
    payload = json.dumps({"event_type": event_type, "data": data}).encode()

    req = urllib.request.Request(
        event_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception as e:
        print(f"WARNING: Failed to post {event_type} event: {e}")


def report_agent_output(output_dir: Path, artifact_path: str, instance_id: str) -> None:
    """Post agent_output event with patch content and artifact path."""
    patch_file = output_dir / "patch.diff"
    patch = patch_file.read_text() if patch_file.exists() else ""

    post_event("agent_output", {
        "patch": patch,
        "instance_id": instance_id,
        "patch_size": len(patch),
        "artifact_path": artifact_path,
    })


def grade(test_output_path: str, eval_meta: dict) -> dict:
    """Grade test output using official swebench tools.

    Returns dict with reward, resolved, reason.
    """
    instance_id = eval_meta["instance_id"]

    test_spec = SimpleNamespace(
        instance_id=instance_id,
        repo=eval_meta.get("repo", ""),
        version=eval_meta.get("version", ""),
        FAIL_TO_PASS=eval_meta["FAIL_TO_PASS"],
        PASS_TO_PASS=eval_meta["PASS_TO_PASS"],
    )

    try:
        from swebench.harness.grading import get_eval_report
    except ImportError:
        raise ImportError("swebench not installed — run: python3 -m pip install swebench")

    prediction = {
        "instance_id": instance_id,
        "model_patch": "",
        "model_name_or_path": "agl-lite",
    }

    try:
        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        resolved = report.get(instance_id, {}).get("resolved", False)
        return {
            "reward": 1.0 if resolved else 0.0,
            "resolved": resolved,
            "reason": "resolved" if resolved else "not resolved",
        }
    except Exception as e:
        return {"reward": 0.0, "resolved": False, "reason": f"grading error: {e}"}


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: grade.py <output_dir> <artifact_path>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    artifact_path = sys.argv[2]

    eval_meta_raw = os.environ.get("AGL_EVAL_META", "")
    eval_meta = json.loads(eval_meta_raw) if eval_meta_raw else {}
    instance_id = eval_meta.get("instance_id", "")

    # 1. Report agent output (patch summary + artifact path).
    report_agent_output(output_dir, artifact_path, instance_id)

    # 2. Grade test output and post reward.
    test_output_path = output_dir / "test_output.txt"
    if not eval_meta or not test_output_path.exists():
        print("Skipping grading: no test output or eval meta")
        result = {"reward": 0.0, "resolved": False, "reason": "no test output or eval meta"}
    else:
        result = grade(str(test_output_path), eval_meta)

    print(f"Grade: reward={result['reward']} resolved={result['resolved']} reason={result['reason']}")
    post_event("reward", {
        "value": result["reward"],
        "resolved": result["resolved"],
        "instance_id": instance_id,
        "reason": result["reason"],
    })


if __name__ == "__main__":
    main()
