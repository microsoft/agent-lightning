# Copyright (c) Microsoft. All rights reserved.

"""Logging utility module for SWE-bench evaluation runs.

This module provides a simple logging utility function that writes evaluation
results and logs to timestamped files organized by run ID and instance ID.
It supports structured logging for tracking the progress and outcomes of
SWE-bench evaluation experiments.

Key features:
- Timestamped logging with datetime formatting
- Run-specific directory organization
- Simple text-based logging for evaluation outputs
"""

import datetime
import os


def logger(run_id: str, instance_id: str, text: str) -> None:
    os.makedirs(f"./logs/{run_id}", exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"./logs/{run_id}/{instance_id}", mode="a") as f:
        print(f"\n\n{current_time}\n{text}\n", file=f)
