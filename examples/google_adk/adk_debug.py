# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, cast

import pandas as pd
from rich.console import Console

from agentlightning import LLM, configure_logger

from adk_agent import AdkTask, LitAdkAgent

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADK agent sanity check (single-run, no training).")
    parser.add_argument("--file", type=str, default="data/test.parquet", help="Path to parquet file.")
    parser.add_argument("--index", type=int, default=0, help="Row index to run as a single task.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        help="Model name to use for rollout.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logger()
    args = parse_args()

    if not os.path.exists(args.file):
        console.print(f"[red]Dataset file not found:[/red] {args.file}")
        raise SystemExit(1)

    df = pd.read_parquet(args.file)
    if df.empty:
        console.print("[red]Dataset file is empty.[/red]")
        raise SystemExit(1)

    row_idx = max(0, min(args.index, len(df) - 1))
    task: AdkTask = cast(AdkTask, df.iloc[row_idx].to_dict())

    resources: Dict[str, Any] = {
        "main_llm": LLM(endpoint=args.endpoint, model=args.model, sampling_parameters={"temperature": 0.0})
    }

    agent = LitAdkAgent()
    # Minimal stub for Rollout (the runner would provide this normally)
    class _R:
        pass

    reward = agent.rollout(task, resources, cast(Any, _R()))
    console.print(f"[bold]Sanity check reward:[/bold] {reward}")


if __name__ == "__main__":
    main()


