"""Minimal QA agent — reads a task, makes one LLM call, parses the answer, reports it.

Environment variables (set automatically by agl-lite controller):
  AGL_TASK_INPUT   — plain text question (JSON-encoded string)
  OPENAI_BASE_URL  — points to agl-lite gateway
  OPENAI_API_KEY   — auth key for gateway
  AGL_EVENT_URL    — URL to post events (agent_output)

Optional:
  CRASH_ON_FIRST   — if "1", exit non-zero on first attempt (for retry testing).

Does NOT import agl-lite — proves language-agnostic contract.

Usage:
  python qa_agent.py --model mock-llm
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
from openai import OpenAI


def _setup_logging() -> None:
    """Write logs to stdout and, if AGL_LOG_DIR is set, to $AGL_LOG_DIR/agent.log."""
    log_dir = os.environ.get("AGL_LOG_DIR")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        log_path = Path(log_dir) / "agent.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


_setup_logging()
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You're a helpful math assistant. "
    "For every given problem, put your final answer in \\boxed{answer} format."
)


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def post_event(event_type: str, data: dict) -> None:
    """Post an event to agl-lite via AGL_EVENT_URL."""
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        log.warning("AGL_EVENT_URL not set — skipping %s event", event_type)
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
        log.error("Failed to post %s event: %s", event_type, e)


def main() -> None:
    parser = argparse.ArgumentParser(description="QA agent for math problems")
    parser.add_argument(
        "--model",
        default=os.environ.get("AGL_MODEL_NAME", "mock-llm"),
        help="Model name to use (default: $AGL_MODEL_NAME or mock-llm)",
    )
    args = parser.parse_args()

    # --- Crash-on-first support (for K8s retry testing) ---
    marker = "/tmp/.agl_crash_done"
    if os.environ.get("CRASH_ON_FIRST") == "1" and not os.path.exists(marker):
        with open(marker, "w") as f:
            f.write("crashed")
        log.error("CRASH_ON_FIRST: simulating failure on first attempt")
        sys.exit(1)

    # --- Read task input ---
    raw = os.environ.get("AGL_TASK_INPUT")
    if not raw:
        log.error("AGL_TASK_INPUT not set")
        sys.exit(1)

    question = json.loads(raw)
    if not isinstance(question, str):
        log.error("AGL_TASK_INPUT should be a plain text string, got %s", type(question))
        sys.exit(1)

    # --- Call LLM via gateway ---
    client = OpenAI(max_retries=5, timeout=120.0)

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stream=True,
    )

    # Collect streamed chunks
    content_parts: list[str] = []
    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            content_parts.append(delta.content)
    content = "".join(content_parts)
    log.info("LLM response: %s", content)

    # --- Parse answer ---
    answer = extract_boxed_answer(content)
    if answer is None:
        log.warning("could not extract \\boxed{answer} from response")
        answer = content.strip()

    log.info("Extracted answer: %s", answer)

    # --- Report agent_output event ---
    post_event("agent_output", {"answer": answer, "raw_response": content})


if __name__ == "__main__":
    main()
