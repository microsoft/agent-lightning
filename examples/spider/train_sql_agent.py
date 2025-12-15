# Copyright (c) Microsoft. All rights reserved.

"""Train an SQL agent on the Spider dataset using Agent-lightning.

This module provides a training script for SQL agents using different model configurations.
The script supports three different training configurations:

1. 'fast' - A lightweight configuration optimized for CI testing with reduced epochs
2. 'qwen' - Standard configuration using Qwen-2.5-Coder-1.5B-Instruct model
3. 'llama' - Configuration using LLaMA-3.2-1B-Instruct model with JSON formatting

Usage:
    python train_sql_agent.py fast    # Fast training for CI/testing
    python train_sql_agent.py qwen    # Standard Qwen model training
    python train_sql_agent.py llama   # LLaMA model training

The script uses reinforcement learning with VERL framework
to train agents on the Spider dataset for text-to-SQL generation tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sql_agent import LitSQLAgent

import agentlightning as agl
RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/train_spider.parquet",
        "val_files": "data/test_dev.parquet",
        "train_batch_size": 2,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 1,
            "log_prob_micro_batch_size_per_gpu": 1,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.6,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 2,
            "ppo_micro_batch_size_per_gpu": 1,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 1,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": False,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "spider",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
        "log_interval": 1,
    },
}

DEFAULT_LOCAL_QWEN05_CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "local_qwen05.json"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def collect_hardware_snapshot(run_dir: Path) -> None:
    """Collect a one-off hardware snapshot for the current node."""

    lines = [f"captured_at: {datetime.now().isoformat()}"]
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"python: {platform.python_version()}")

    lines.append(f"cpu_count: {os.cpu_count()}")
    try:
        import psutil

        cpu_freq = psutil.cpu_freq()
        lines.append(f"cpu_freq_mhz: {cpu_freq.current if cpu_freq else 'unknown'}")
        vm = psutil.virtual_memory()
        lines.append(f"ram_total_gb: {round(vm.total / 1e9, 2)}")
    except Exception:
        try:
            load1, load5, load15 = os.getloadavg()
            lines.append(f"cpu_load: {load1},{load5},{load15}")
        except Exception:
            lines.append("cpu_info: unavailable")

    try:
        import torch

        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_version: {torch.version.cuda}")
            lines.append(f"n_gpus: {torch.cuda.device_count()}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                lines.append(
                    f"gpu_{idx}: name={props.name}, total_mem_gb={round(props.total_memory/1e9,2)}, "
                    f"capability={props.major}.{props.minor}"
                )
    except Exception:
        lines.append("torch_cuda_info: unavailable")

    try:
        smi = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        if smi.stdout:
            lines.append("nvidia_smi_devices:")
            lines.append(smi.stdout.strip())
    except Exception:
        lines.append("nvidia_smi: unavailable")

    _write_text(run_dir / "hardware.txt", "\n".join(lines) + "\n")


def write_objectives(run_dir: Path) -> None:
    """Persist high-level goals/constraints (user-editable placeholder)."""

    goal = os.getenv("RUN_OBJECTIVE", "not specified")
    oom_policy = os.getenv("OOM_POLICY", "not specified")
    notes = [
        "Objectives/constraints for this run (edit RUN_OBJECTIVE/OOM_POLICY envs to override):",
        f"- objective: {goal}",
        f"- oom_risk_tolerance: {oom_policy}",
        "- priority: maximize throughput unless specified",
    ]
    _write_text(run_dir / "objectives.txt", "\n".join(notes) + "\n")


def _sample_gpu_usage(path: Path, stop_event: threading.Event, duration_s: int, interval_s: float) -> None:
    """Poll nvidia-smi for a short window and store CSV samples."""

    header = "timestamp,name,util.gpu,util.mem,mem.used,mem.total,pcie.gen,pcie.width"
    _write_text(path, header + "\n")
    start = time.time()
    while not stop_event.is_set() and (time.time() - start) <= duration_s:
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if res.stdout:
                with path.open("a", encoding="utf-8") as f:
                    f.write(res.stdout.strip() + "\n")
        except FileNotFoundError:
            break
        except Exception:
            break
        time.sleep(interval_s)


def _sample_cpu_usage(path: Path, stop_event: threading.Event, duration_s: int, interval_s: float) -> None:
    """Poll CPU/memory usage for a short window and store CSV samples."""

    header = "timestamp,cpu_percent,load_avg,mem_percent"
    _write_text(path, header + "\n")
    start = time.time()
    while not stop_event.is_set() and (time.time() - start) <= duration_s:
        ts = datetime.now().isoformat()
        cpu_percent = "na"
        mem_percent = "na"
        load_avg = "na"
        try:
            import psutil

            cpu_percent = str(psutil.cpu_percent(interval=None))
            mem_percent = str(psutil.virtual_memory().percent)
        except Exception:
            pass
        try:
            load_avg_vals = os.getloadavg()
            load_avg = ",".join(str(round(v, 2)) for v in load_avg_vals)
        except Exception:
            pass
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{ts},{cpu_percent},{load_avg},{mem_percent}\n")
        time.sleep(interval_s)


def start_system_monitors(run_dir: Path, duration_s: int = 180, interval_s: float = 1.0) -> threading.Event:
    """Start short-lived background monitors for GPU/CPU; returns a stop flag."""

    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_sample_gpu_usage,
            args=(run_dir / "gpu_monitor.csv", stop_event, duration_s, interval_s),
            daemon=True,
        ),
        threading.Thread(
            target=_sample_cpu_usage,
            args=(run_dir / "cpu_monitor.csv", stop_event, duration_s, interval_s),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
    return stop_event


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config from disk (relative paths are resolved from this file)."""

    raw_path = Path(config_path)
    candidates = [raw_path] if raw_path.is_absolute() else [
        Path.cwd() / raw_path,
        Path(__file__).resolve().parent / raw_path,
    ]

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Config file not found. Tried: {', '.join(str(p) for p in candidates)}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pyyaml is required for YAML configs") from exc
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def config_train_fast() -> Dict[str, Any]:
    """A fast training run for CI testing purposes."""

    # `EXPERIMENT_NAME="spider_$(date +%Y%m%d%H%M%S)"`
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"spider_{timestamp}"

    # `PROJECT_NAME=AgentLightningCI`
    PROJECT_NAME = "AgentLightningCI"

    # Simulate writing to $GITHUB_OUTPUT if it’s set
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print("Set environment variables:")
    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["actor_rollout_ref"]["model"]["path"] = "/home/lthpc/student/LiTengfei/LLaMA-Factory/models/Qwen2.5-Coder-0.5B-Instruct"
    config["data"]["val_files"] = "data/test_dev.parquet"
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_local_qwen05() -> Dict[str, Any]:
    """Train with local Qwen2.5-Coder-0.5B-Instruct weights (configurable via file/env)."""

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"spider_local05_{timestamp}"
    PROJECT_NAME = "AgentLightningLocal"

    external_path = os.getenv("LOCAL_QWEN05_CONFIG_FILE")
    base_config: Dict[str, Any]
    try:
        base_config = load_config_file(external_path) if external_path else load_config_file(DEFAULT_LOCAL_QWEN05_CONFIG_FILE)
    except Exception:
        base_config = deepcopy(RL_TRAINING_CONFIG)

    config = deepcopy(base_config)
    config["actor_rollout_ref"]["model"]["path"] = os.getenv(
        "LOCAL_QWEN_MODEL_PATH", config["actor_rollout_ref"]["model"].get("path")
    )
    config["trainer"]["experiment_name"] = config["trainer"].get("experiment_name", EXPERIMENT_NAME)
    config["trainer"]["project_name"] = config["trainer"].get("project_name", PROJECT_NAME)
    return config


def config_train_qwen() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5B."""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config


def config_train_npu() -> Dict[str, Any]:
    """A configuration for training with NPU."""

    config = deepcopy(RL_TRAINING_CONFIG)
    del config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["enable_auto_tool_choice"]
    del config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"]
    del config["trainer"]["logger"][1]
    config["actor_rollout_ref"]["actor"]["use_torch_compile"] = False
    config["trainer"]["val_before_train"] = False
    config["trainer"]["save_freq"] = 256
    config["trainer"]["device"] = "npu"
    return config


def config_train_llama() -> Dict[str, Any]:
    """A configuration for training with LLaMA-3.2-1B-Instruct.

    You will need a `HF_TOKEN` set to run with this config.
    """

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] = "llama3_json"
    config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"] = "llama3_json"
    config["actor_rollout_ref"]["model"]["path"] = "meta-llama/Llama-3.2-1B-Instruct"
    return config


def prepare_run_outputs(config: Dict[str, Any], run_label: str) -> Path:
    """Create per-run folder and inject log/config paths into the config."""

    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_label = run_label.replace("/", "_")
    safe_run_name = f"{timestamp}_config_{base_label}"

    run_dir = log_dir / safe_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    progress_log = run_dir / "progress.txt"
    config_dump = run_dir / "config.json"

    config["trainer"]["progress_log_file"] = str(progress_log)
    config["trainer"]["sequence_log_file"] = str(run_dir / "sequence_lengths.csv")
    # Make sure summary lines are emitted regularly and captured by progress_log_file.
    config["trainer"].setdefault("log_interval", 10)
    config["trainer"]["run_dir"] = str(run_dir)

    with open(config_dump, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[log] summary metrics will be written to {progress_log}")
    print(f"[log] saved run config to {config_dump}")
    print(f"[log] run artifacts directory: {run_dir}")

    collect_hardware_snapshot(run_dir)
    write_objectives(run_dir)
    start_system_monitors(run_dir)
    return run_dir


def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """Train the SQL agent with the given configuration."""

    agent = LitSQLAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=10, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train an SQL agent on the Spider dataset using different model configurations"
    )

    parser.add_argument(
        "config",
        choices=["fast", "local_qwen05", "qwen", "llama", "npu"],
        help="Training configuration: 'fast' (CI testing), 'local_qwen05' (local 0.5B), 'qwen' (Qwen-2.5-Coder-1.5B), 'llama' (LLaMA-3.2-3B),'npu' (Train with NPU)",
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Optional JSON/YAML config file to load (relative paths resolved from this script).",
    )

    args = parser.parse_args()

    # Get the appropriate configuration
    config_functions = {
        "fast": config_train_fast,
        "local_qwen05": config_train_local_qwen05,
        "qwen": config_train_qwen,
        "llama": config_train_llama,
        "npu": config_train_npu,
    }
    run_label = args.config
    if args.config_file:
        config = load_config_file(args.config_file)
        config.setdefault("trainer", {})
        if not config["trainer"].get("experiment_name"):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            config["trainer"]["experiment_name"] = f"{args.config}_{timestamp}"
        if not config["trainer"].get("project_name"):
            config["trainer"]["project_name"] = "AgentLightningCustom"
        run_label = Path(args.config_file).stem
    else:
        config = config_functions[args.config]()
    run_dir = prepare_run_outputs(config, run_label)

    # Set active agent - use provided value or default based on config choice
    active_agent = args.active_agent

    print(f"Starting training with '{args.config}' configuration...")
    print(f"Active agent: {active_agent}")

    try:
        train(config, active_agent)
    except Exception as exc:  # noqa: BLE001
        error_path = Path(run_dir) / "error.txt"
        _write_text(error_path, f"exception: {exc}\n")
        raise


if __name__ == "__main__":
    main()
