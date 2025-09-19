# Copyright (c) Microsoft. All rights reserved.

import wandb
import sys

if len(sys.argv) != 3:
    print("Usage: python validate_example_wandb.py <project> <run_name>")

project = sys.argv[1]
run_name = sys.argv[2]
api = wandb.Api()
entity_name = api.default_entity
print("Default entity:", entity_name)
print("Project:", project)
print("Run name:", run_name)

runs = api.runs(f"{entity_name}/{project}", filters={"displayName": run_name})
for run in runs:
    print(f"Found run: {run.name} (ID: {run.id})")
    if run.name == run_name:
        break
else:
    print(f"::error::Run with name '{run_name}' not found in project '{project}'.")
    sys.exit(1)

hist = run.history(keys=["val/reward"], pandas=True)
print("History:", hist)
if hist.empty:
    print("::error::No history found for the run.")
    sys.exit(1)
else:
    first, last = hist["val/reward"].iloc[0], hist["val/reward"].iloc[-1]
    if last <= first:
        print(
            f"::warning title=Training no improvement::No improvement (run_name={run_name} start={first:.4f}, end={last:.4f})"
        )
    else:
        print(
            f"::notice title=Training completed::Run has improved (run_name={run_name} start={first:.4f}, end={last:.4f})"
        )
