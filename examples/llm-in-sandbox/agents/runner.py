#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Agent Lightning adapter for the llm-in-sandbox container entrypoint."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from importlib import import_module
from pathlib import Path
from typing import Any

ANSWER_RE = re.compile(r"##########\s*(.*?)\s*##########", re.DOTALL)
REWARD_MODULES = {
    "biomed_mini": "llm_in_sandbox.benchmark.biomed.reward",
    "chem_mini": "llm_in_sandbox.benchmark.chem.reward",
    "instruct_pretrain": "llm_in_sandbox.benchmark.instruct_pretrain.reward",
    "long_context_mini": "llm_in_sandbox.benchmark.instruct_pretrain.reward",
    "math_mini": "llm_in_sandbox.benchmark.math.reward",
}
DEFAULT_MAX_TOKENS_PER_CALL = "20000"


def log(message: str) -> None:
    print(message, flush=True)


def load_task_input() -> dict[str, Any]:
    raw = os.environ.get("AGL_TASK_INPUT", "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def env_or_task(name: str, task: dict[str, Any], key: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    extra = task.get("extra_info") if isinstance(task.get("extra_info"), dict) else {}
    value = extra.get(key) if isinstance(extra, dict) else None
    return str(value) if value is not None else ""


def data_file_path(folder_name: str, filename: str) -> Path:
    return Path("/data") / folder_name / filename


def load_sample(folder_name: str, filename: str, index: int) -> dict[str, Any]:
    path = data_file_path(folder_name, filename)
    with path.open(encoding="utf-8") as file:
        samples = json.load(file)
    sample = samples[index]
    if not isinstance(sample, dict):
        raise TypeError(f"sample at index {index} is not an object: {path}")
    return sample


def openai_model_name(model_name: str) -> str:
    explicit = os.environ.get("LLM_NAME")
    if explicit:
        return explicit
    prefix = os.environ.get("AGL_OPENAI_MODEL_PREFIX", "openai/")
    if not prefix:
        return model_name
    if model_name.startswith(("openai/", "anthropic/", "azure/", "hosted_vllm/")):
        return model_name
    return f"{prefix}{model_name}"


def configure_llm_env() -> None:
    model_name = os.environ.get("AGL_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
    os.environ["LLM_NAME"] = openai_model_name(model_name)
    os.environ["LLM_BASE_URL"] = (
        os.environ.get("LLM_BASE_URL") or os.environ.get("AGL_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    )
    os.environ["LLM_API_KEY"] = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
    os.environ["LLM_TEMPERATURE"] = os.environ.get("LLM_TEMPERATURE") or os.environ.get("AGL_LLM_TEMPERATURE", "1.0")
    os.environ["OPENAI_TIMEOUT"] = os.environ.get("OPENAI_TIMEOUT", "900")
    os.environ["MAX_TOKENS_PER_CALL"] = os.environ.get("MAX_TOKENS_PER_CALL") or DEFAULT_MAX_TOKENS_PER_CALL


def run_llm_in_sandbox() -> tuple[int, str]:
    proc = subprocess.Popen(
        ["llm-in-sandbox", "run_in_container"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return proc.wait(), "".join(lines)


def extract_answer(output: str) -> str:
    match = ANSWER_RE.search(output)
    if not match:
        return ""
    return match.group(1).strip()


def compute_reward(sample: dict[str, Any], answer: str) -> tuple[float, str]:
    ground_truth = str(sample.get("reward_model", {}).get("ground_truth", ""))
    data_source = str(sample.get("data_source", ""))
    reward_module_name = REWARD_MODULES.get(data_source)
    if reward_module_name is None:
        return 0.0, f"unsupported data_source: {data_source}"

    extra_info = dict(sample.get("extra_info", {}))
    extra_info.pop("ground_truth", None)
    try:
        reward_module = import_module(reward_module_name)
        compute_score = reward_module.compute_score
        reward = compute_score(answer, ground_truth, **extra_info)
    except Exception as exc:
        return 0.0, f"reward error: {exc}"
    return float(reward), "computed by llm-in-sandbox reward"


def post_event(event_type: str, data: dict[str, Any]) -> bool:
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        log(f"AGL_EVENT_URL is not set; skip event {event_type}")
        return False
    body = json.dumps({"event_type": event_type, "data": data}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    agl_key = os.environ.get("AGL_KEY")
    if agl_key:
        headers["Authorization"] = f"Bearer {agl_key}"
    request = urllib.request.Request(
        event_url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
        return True
    except (urllib.error.URLError, TimeoutError) as exc:
        log(f"failed to post {event_type} event: {exc}")
        return False


def main() -> int:
    task = load_task_input()
    folder_name = env_or_task("DATA_FOLDER_NAME", task, "data_folder_name")
    filename = env_or_task("DATA_FILENAME", task, "data_filename")
    data_index_raw = env_or_task("DATA_INDEX", task, "data_index")
    if not folder_name or not filename or data_index_raw == "":
        raise RuntimeError("DATA_FOLDER_NAME, DATA_FILENAME, and DATA_INDEX are required")
    data_index = int(data_index_raw)

    sample = load_sample(folder_name, filename, data_index)
    configure_llm_env()

    log(f"llm-in-sandbox sample: {folder_name}/{filename}[{data_index}]")
    log(f"llm-in-sandbox model: {os.environ['LLM_NAME']}")
    returncode, output = run_llm_in_sandbox()
    answer = extract_answer(output)
    reward, reason = compute_reward(sample, answer)

    extra = sample.get("extra_info", {}) if isinstance(sample.get("extra_info"), dict) else {}
    event_base = {
        "data_source": sample.get("data_source"),
        "data_id": extra.get("id"),
        "data_folder_name": folder_name,
        "data_filename": filename,
        "data_index": data_index,
        "returncode": returncode,
    }
    post_event("agent_output", {**event_base, "answer": answer})
    post_event(
        "reward",
        {
            **event_base,
            "value": reward,
            "reason": reason,
            "source": "agent",
            "ground_truth": sample.get("reward_model", {}).get("ground_truth"),
            "agent_answer": answer,
        },
    )
    log(f"reward={reward:.4f} reason={reason}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
