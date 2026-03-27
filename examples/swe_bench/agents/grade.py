#!/usr/bin/env python3
"""SWE-bench grading and reward posting.

Grades test output using official swebench get_eval_report(),
posts a reward event to the agl-lite server, and archives
the test log to the shared volume.

Expected env vars:
  AGL_EVAL_META     — JSON with instance_id, repo, version, FAIL_TO_PASS, PASS_TO_PASS
  AGL_EVENT_URL     — URL to post events
  AGL_KEY           — API key (fallback: OPENAI_API_KEY)
  AGL_ROLLOUT_ID    — rollout ID for artifact path
"""

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

TEST_OUTPUT_FILE = "/tmp/test_output.txt"
ARTIFACT_DIR = Path(f"/data/artifacts/{os.environ.get('AGL_ROLLOUT_ID', 'unknown')}")


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


def post_reward(result: dict, patch_size: int) -> None:
    """Post reward event to agl-lite server."""
    import urllib.request

    event_url = os.environ.get("AGL_EVENT_URL", "")
    if not event_url:
        print("WARNING: AGL_EVENT_URL not set, skipping reward post")
        return

    api_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY", "")
    eval_meta = json.loads(os.environ.get("AGL_EVAL_META", "{}"))
    instance_id = eval_meta.get("instance_id", "")

    payload = json.dumps({
        "event_type": "reward",
        "data": {
            "value": result["reward"],
            "resolved": result["resolved"],
            "instance_id": instance_id,
            "patch_size": patch_size,
            "reason": result["reason"],
        },
    }).encode()

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
        print(f"WARNING: Failed to post reward event: {e}")


def archive_test_log() -> None:
    """Copy test log to shared volume for debugging."""
    if not Path(TEST_OUTPUT_FILE).exists():
        return
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TEST_OUTPUT_FILE, ARTIFACT_DIR / "test_output.txt")
        print(f"Archived test output to {ARTIFACT_DIR}/test_output.txt")
    except Exception as e:
        print(f"WARNING: Failed to archive test output: {e}")


def main() -> None:
    patch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    eval_meta_raw = os.environ.get("AGL_EVAL_META", "")
    if not eval_meta_raw or not Path(TEST_OUTPUT_FILE).exists():
        print("Skipping grading: no test output or eval meta")
        result = {"reward": 0.0, "resolved": False, "reason": "no test output or eval meta"}
    else:
        eval_meta = json.loads(eval_meta_raw)
        result = grade(TEST_OUTPUT_FILE, eval_meta)

    print(f"Grade: reward={result['reward']} resolved={result['resolved']} reason={result['reason']}")

    post_reward(result, patch_size)
    archive_test_log()


if __name__ == "__main__":
    main()
