"""Minimal QA agent — reads a task, makes one LLM call, parses the answer, reports it.

Environment variables (set automatically by agl-lite controller):
  AGL_TASK_INPUT   — JSON dict with "question" field (plain text math problem)
  OPENAI_BASE_URL  — points to agl-lite gateway
  OPENAI_API_KEY   — auth key for gateway
  AGL_EVENT_URL    — URL to post events (agent_output)

Optional:
  CRASH_ON_FIRST   — if "1", exit non-zero on first attempt (for retry testing).
                     Creates a marker file so only the first attempt crashes.

Does NOT import agl-lite — proves language-agnostic contract.
"""

import json
import os
import re
import sys

import httpx
from openai import OpenAI

PROMPT_TEMPLATE = (
    "You're a helpful math assistant. "
    "For every given problem, put the answer in the format \\boxed{answer}.\n\n"
    "Question: {question}"
)


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def post_event(event_type: str, data: dict) -> None:
    """Post an event to agl-lite via AGL_EVENT_URL."""
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        print(f"AGL_EVENT_URL not set — skipping {event_type} event", file=sys.stderr)
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        resp = httpx.post(
            event_url,
            json={"event_type": event_type, "data": data},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to post {event_type} event: {e}", file=sys.stderr)


def main() -> None:
    # --- Crash-on-first support (for K8s retry testing) ---
    marker = "/tmp/.agl_crash_done"
    if os.environ.get("CRASH_ON_FIRST") == "1" and not os.path.exists(marker):
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
    question = task.get("question") if isinstance(task, dict) else str(task)
    if not question:
        print("ERROR: task has no 'question' field", file=sys.stderr)
        sys.exit(1)

    # --- Build prompt ---
    prompt = PROMPT_TEMPLATE.format(question=question)

    # --- Call LLM via gateway ---
    client = OpenAI(max_retries=5, timeout=120.0)
    model = task.get("model", "mock-llm") if isinstance(task, dict) else "mock-llm"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or ""
    print(f"LLM response: {content}")

    # --- Parse answer ---
    answer = extract_boxed_answer(content)
    if answer is None:
        print("WARNING: could not extract \\boxed{answer} from response", file=sys.stderr)
        answer = content.strip()

    print(f"Extracted answer: {answer}")

    # --- Report agent_output event ---
    post_event("agent_output", {"answer": answer, "raw_response": content})


if __name__ == "__main__":
    main()
