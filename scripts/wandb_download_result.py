# Copyright (c) Microsoft. All rights reserved.

"""Usage example:

python scripts/wandb_download_result.py AgentLightning \
    --runs spider_agl_v0_2 \
    --metrics training/reward val/reward \
    --out docs/assets/sql-agent-training-result.json \
    --step 16
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch metrics from Weights & Biases runs and output Chart.js-ready JSON. "
            "Aggregates by step bins to tame long x-axes."
        )
    )
    p.add_argument(
        "project",
        help="W&B project name (e.g., 'my-project'). Uses your default entity unless --entity is set.",
    )
    p.add_argument(
        "--entity",
        default=None,
        help="W&B entity (team/user). If omitted, uses wandb.Api().default_entity.",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run names (display names) to include. Example: --runs a b c",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Metric keys to fetch. Example: --metrics train/loss val/acc",
    )
    p.add_argument(
        "--step",
        type=int,
        default=1,
        help="Aggregate step size in _step units (e.g., 16 groups steps into bins of 16). Default: 1 (no binning).",
    )
    p.add_argument(
        "--out",
        default="wandb_result.json",
        help="Output file name. Default: 'wandb_result.json'",
    )
    p.add_argument(
        "--label-format",
        default="{run}:{metric}",
        help="Dataset label format. You can use {run} and {metric}. Default: '{run}:{metric}'",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="If set, exit with nonzero code when a run or metric is missing.",
    )
    return p.parse_args()


def fetch_runs(api: wandb.Api, entity: str, project: str, run_names: List[str]) -> Dict[str, wandb.Run]:
    """
    Fetch runs by displayName matching any in run_names.
    """
    name_set = set(run_names)
    found: Dict[str, wandb.Run] = {}

    # W&B filtering supports 'displayName'
    # We fetch all runs in the project once, then pick matching ones to be robust across filters/backends.
    # If the project is huge, you can optimize to paginate/stop early—here we walk until we’ve found all.
    for run in api.runs(f"{entity}/{project}"):
        dn = getattr(run, "name", None) or getattr(run, "displayName", None)
        # run.name is usually the short name; W&B Python public API exposes it as .name
        if dn in name_set and dn not in found:
            found[dn] = run
            if len(found) == len(name_set):
                break

    return found


def aggregate_history(df: pd.DataFrame, metrics: List[str], step: int) -> pd.DataFrame:
    """
    Given a history dataframe with '_step' and metric columns,
    aggregate by floor(_step/step)*step and average metric values per bin.
    """
    if "_step" not in df.columns:
        raise ValueError("History dataframe missing required '_step' column.")

    if step < 1:
        step = 1

    # Drop rows where all requested metrics are NaN to avoid empty bins
    keep_mask = df[metrics].notna().any(axis=1)
    df = df.loc[keep_mask].copy()

    # Compute bin: bin is rounded to the nearest multiples of step
    df["_bin"] = np.round(df["_step"] / step) * step

    # Group by bin and average each metric
    grouped = df.groupby("_bin", as_index=False)[metrics].mean()

    # Ensure bins are sorted
    grouped = grouped.sort_values("_bin").reset_index(drop=True)
    return grouped


def build_chartjs(
    per_run_metric_df: Dict[Tuple[str, str], pd.DataFrame],
    label_format: str,
) -> Dict[str, Any]:
    """
    Build a Chart.js line chart dataset:
      labels: union of all bins across runs (sorted)
      datasets: one per (run, metric) pair, aligned to labels, with None for missing points
    """
    # Union of all bins
    all_bins = set()
    for df in per_run_metric_df.values():
        all_bins.update(df["_bin"].tolist())
    labels = sorted(all_bins)

    # Chart.js wants arrays of primitive x labels (we'll use the bin starts)
    # If you want to render actual x=_step values, labels are these bin starts.
    datasets = []
    for (run_name, metric), df in per_run_metric_df.items():
        series_map = dict(zip(df["_bin"].tolist(), df[metric].tolist()))
        data = [series_map.get(b, None) for b in labels]
        datasets.append(
            {
                "label": label_format.format(run=run_name, metric=metric),
                "data": data,
                # Chart.js can infer styles; consumers can style further on the frontend
                "spanGaps": True,  # nicer lines across missing bins
            }
        )

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "interaction": {"mode": "nearest", "intersect": False},
            "plugins": {
                "legend": {"display": True, "position": "top"},
                "title": {"display": True, "text": "W&B Metrics (binned by step)"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "Step (bin start)"}},
                "y": {"title": {"display": True, "text": "Value"}},
            },
        },
    }


def main():
    args = parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if not entity:
        print("::error::Unable to determine W&B entity. Pass --entity.", file=sys.stderr)
        sys.exit(1)

    runs = fetch_runs(api, entity, args.project, args.runs)
    missing = [r for r in args.runs if r not in runs]
    if missing:
        msg = f"Runs not found: {', '.join(missing)}"
        if args.strict:
            print(f"::error::{msg}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"::warning::{msg}", file=sys.stderr)

    if not runs:
        print("::error::No matching runs found.", file=sys.stderr)
        sys.exit(1)

    per_run_metric_df: Dict[Tuple[str, str], pd.DataFrame] = {}

    for run_name, run in runs.items():
        # Fetch each metric separately to avoid losing sparse metrics due to row intersection.
        for metric in args.metrics:
            hist = run.history(keys=["_step", metric], pandas=True)
            if hist is None or hist.empty:
                msg = f"No history for run '{run_name}' (metric '{metric}')."
                if args.strict:
                    print(f"::error::{msg}", file=sys.stderr)
                    sys.exit(1)
                else:
                    print(f"::warning::{msg}", file=sys.stderr)
                    continue
            # Ensure numeric _step
            if "_step" not in hist.columns:
                print(
                    f"::warning::Run '{run_name}' has no '_step' column; skipping metric '{metric}'.",
                    file=sys.stderr,
                )
                continue

            # Clean to numeric where possible
            hist["_step"] = pd.to_numeric(hist["_step"], errors="coerce")
            hist = hist.dropna(subset=["_step"])
            hist["_step"] = hist["_step"].astype(int)
            # Aggregate per metric; dense metrics can be tamed with --step (e.g., 16)
            grouped = aggregate_history(hist, [metric], args.step)
            if metric not in grouped.columns:
                msg = f"Metric '{metric}' not found in run '{run_name}'."
                if args.strict:
                    print(f"::error::{msg}", file=sys.stderr)
                    sys.exit(1)
                else:
                    print(f"::warning::{msg}", file=sys.stderr)
                    continue
            # Keep only _bin and the single metric for simpler merging later
            per_run_metric_df[(run_name, metric)] = grouped[["_bin", metric]].copy()

    if not per_run_metric_df:
        print("::error::No data collected for any run/metric.", file=sys.stderr)
        sys.exit(1)

    chart = build_chartjs(per_run_metric_df, args.label_format)

    payload = json.dumps(chart, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"Wrote Chart.js JSON to: {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
