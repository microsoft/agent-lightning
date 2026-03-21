"""Minimal QA agent — reads a task, makes one LLM call, prints the result.

Environment variables (set automatically by agl-lite controller):
  AGL_TASK_INPUT   — JSON string with task payload (must contain "prompt")
  OPENAI_BASE_URL  — points to agl-lite gateway
  OPENAI_API_KEY   — auth key for gateway

Optional:
  CRASH_ON_FIRST   — if "1", exit non-zero on first attempt (for retry testing).
                     Creates a marker file so only the first attempt crashes.

Does NOT import agl-lite — proves language-agnostic contract.
Any program that reads env vars and calls an OpenAI-compatible endpoint works.
"""

import json
import os
import sys

from openai import OpenAI


def main() -> None:
    # --- Crash-on-first support (for K8s retry testing) ---
    marker = "/tmp/.agl_crash_done"
    if os.environ.get("CRASH_ON_FIRST") == "1" and not os.path.exists(marker):
        # First attempt: create marker and crash.
        with open(marker, "w") as f:
            f.write("crashed")
        print("CRASH_ON_FIRST: simulating failure on first attempt", file=sys.stderr)
        sys.exit(1)

    # --- Read task input ---
    raw = os.environ.get("AGL_TASK_INPUT")
    if not raw:
        print("ERROR: AGL_TASK_INPUT not set", file=sys.stderr)
        sys.exit(1)

    task = json.loads(raw)
    prompt = task.get("prompt")
    if not prompt:
        print("ERROR: task has no 'prompt' field", file=sys.stderr)
        sys.exit(1)

    # --- Call LLM via gateway ---
    # OPENAI_BASE_URL and OPENAI_API_KEY are set by the controller.
    # The openai SDK reads them automatically.
    client = OpenAI(max_retries=5, timeout=120.0)

    model = task.get("model", "mock-llm")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.choices[0].message.content
    print(result)


if __name__ == "__main__":
    main()
