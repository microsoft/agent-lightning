import os
from datetime import datetime, timedelta

import pandas as pd
import wandb

api = wandb.Api()

entity = "ultmaster"
project = "AgentZero"
run_names = [
    "c38c8374-aware-panther-train_calc_instrument",
    "720dfb66-aware-panther-train_calc",
    "9d3076ad-bold-alien-train_calc_g1",
]

# Build filter query using wandb public API filters
filters = {
    "name": {"$in": run_names},
}

keys = [
    "_step",
    "val/reward",
    "val/turn_count",
    "val/mean_response_length",
    "prompt_length/mean",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "critic/rewards/mean",
]

runs = api.runs(path=f"{entity}/{project}", filters=filters)

for run in runs:
    print(f"Processing run {run.id} (started {run.created_at})")
    history = run.history(keys=None, x_axis="_step", samples=1000, pandas=True)
    df = history[keys].set_index("_step")
    df["_runtime"] = run.summary["_wandb"]["runtime"]

    # Save to CSV
    fname = f"{project}_{run.id}_metrics.csv"
    df.to_csv(fname)
    print(f"Saved {fname}, {df.shape[0]} rows.")
