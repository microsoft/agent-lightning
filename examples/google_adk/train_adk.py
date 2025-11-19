# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import pandas as pd
from rich.console import Console

from agentlightning import LLM, Trainer, configure_logger

from adk_agent import LitAdkAgent

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ADK Agent with Agent-Lightning.")
    parser.add_argument("--train-file", type=str, default="data/train.parquet", help="Path to training parquet file.")
    parser.add_argument("--val-file", type=str, default="data/test.parquet", help="Path to validation parquet file.")
    parser.add_argument("--model", type=str, default=None, help="Model name for the rollout LLM.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Reduce workload for CI (smaller dataset slice, fewer runners).",
    )
    parser.add_argument(
        "--ci-fast",
        action="store_true",
        help="Ultra-fast CI mode (tiny dataset slice, 1 runner).",
    )
    parser.add_argument(
        "--external-store-address",
        type=str,
        default=None,
        help="Use an external LightningStore address for debugging (e.g., http://localhost:4747).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "agent-lightning-adk"),
        help="Weights & Biases project name (printed for CI validation).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (printed for CI validation).",
    )
    return parser.parse_args()


def load_dataset(path: str) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict("records")


def main() -> None:
    configure_logger()
    args = parse_args()

    # CI modes adjust workload
    n_runners = 2
    train_limit = None
    val_limit = None
    if args.ci_fast:
        n_runners = 1
        train_limit = 16
        val_limit = 16
    elif args.ci:
        n_runners = 2
        train_limit = 128
        val_limit = 64

    # Print WandB info for CI validation
    console.print(f"[bold]WandB Project[/bold]: {args.wandb_project}")
    console.print(f"[bold]WandB Run Name[/bold]: {args.wandb_run_name or 'auto'}")

    # Load datasets
    train_data = load_dataset(args.train_file)
    val_data = load_dataset(args.val_file)
    if train_limit is not None:
        train_data = train_data[: train_limit]
    if val_limit is not None:
        val_data = val_data[: val_limit]

    # Rollout LLM resource (VERL convention uses key 'main_llm')
    # Model default: pull from env OPENAI_MODEL or a sensible default
    model_name = args.model or os.environ.get("OPENAI_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    rollout_llm = LLM(endpoint=args.endpoint, model=model_name, sampling_parameters={"temperature": 0.0})
    resources: Dict[str, Any] = {"main_llm": rollout_llm}

    # Trainer and Agent
    trainer_kwargs: Dict[str, Any] = {"n_runners": n_runners}
    if args.external_store_address:
        # In v0.2 API, the Trainer may accept a store argument or an address in algorithm-specific ways.
        # Here we pass it through as a generic kwarg in case your environment supports it.
        trainer_kwargs["external_store_address"] = args.external_store_address

    trainer = Trainer(**trainer_kwargs)
    agent = LitAdkAgent()

    console.print(
        f"Starting training with n_runners={n_runners}, train={len(train_data)} samples, val={len(val_data)} samples"
    )
    # Fit accepts either agent + datasets or legacy fit_v0 with store URL.
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data, resources=resources)  # type: ignore


if __name__ == "__main__":
    main()


