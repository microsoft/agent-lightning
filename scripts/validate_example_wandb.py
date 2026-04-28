# Copyright (c) Microsoft. All rights reserved.

import argparse
import sys

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a Weights & Biases run for reward/trace rollouts.")
    parser.add_argument("project", help="W&B project name")
    parser.add_argument("run_name", help="W&B run display name")
    parser.add_argument(
        "--reward-tolerance",
        type=int,
        default=0,
        help="Allowed difference between first and last val/n_rollouts_w_reward",
    )
    parser.add_argument(
        "--trace-tolerance",
        type=int,
        default=0,
        help="Allowed difference between first and last val/n_rollouts_w_trace",
    )
    return parser.parse_args()


args = parse_args()

project = args.project
run_name = args.run_name
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

hist = run.history(keys=["val/reward", "val/n_rollouts_w_reward", "val/n_rollouts_w_trace"], pandas=True)
print("History:", hist)
if hist.empty:
    print("::error::No history found for the run.")
    sys.exit(1)
else:
    # Check whether all rollouts have (approximately) succeeded
    first_row = hist.iloc[0]
    last_row = hist.iloc[-1]

    first_reward_rollouts = first_row["val/n_rollouts_w_reward"]
    last_reward_rollouts = last_row["val/n_rollouts_w_reward"]
    reward_diff = abs(first_reward_rollouts - last_reward_rollouts)

    if reward_diff > args.reward_tolerance or (first_reward_rollouts == 0 and last_reward_rollouts == 0):
        print(
            "::error::Some rollouts have failed to produce rewards: "
            f"{first_reward_rollouts} -> {last_reward_rollouts} "
            f"(tolerance={args.reward_tolerance})"
        )
        sys.exit(1)
    elif first_reward_rollouts != last_reward_rollouts:
        print(
            "::warning::First and last val/n_rollouts_w_reward are different: "
            f"{first_reward_rollouts} -> {last_reward_rollouts}"
        )

    first_trace_rollouts = first_row["val/n_rollouts_w_trace"]
    last_trace_rollouts = last_row["val/n_rollouts_w_trace"]
    trace_diff = abs(first_trace_rollouts - last_trace_rollouts)

    if trace_diff > args.trace_tolerance or (first_trace_rollouts == 0 and last_trace_rollouts == 0):
        print(
            "::error::Some rollouts have failed to produce traces: "
            f"{first_trace_rollouts} -> {last_trace_rollouts} "
            f"(tolerance={args.trace_tolerance})"
        )
        sys.exit(1)
    elif first_trace_rollouts != last_trace_rollouts:
        print(
            "::warning::First and last val/n_rollouts_w_trace are different: "
            f"{first_trace_rollouts} -> {last_trace_rollouts}"
        )

    first_reward, last_reward = first_row["val/reward"], last_row["val/reward"]
    if last_reward <= first_reward:
        print(
            f"::warning title=Training no improvement::No improvement (run_name={run_name} start={first_reward:.4f}, end={last_reward:.4f})"
        )
    else:
        print(
            f"::notice title=Training completed::Run has improved (run_name={run_name} start={first_reward:.4f}, end={last_reward:.4f})"
        )
