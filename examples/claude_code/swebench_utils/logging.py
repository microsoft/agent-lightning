# Copyright (c) Microsoft. All rights reserved.

"""Logging utility module for SWE-bench evaluation runs.

This module provides a simple logging utility function that writes evaluation
results and logs to timestamped files organized by run ID and instance ID.
"""

import datetime
import os


def log_for_evaluation(run_id: str, instance_id: str, text: str) -> None:
    """Log a message for evaluation purposes of SWE-Bench.

    The format follows the SWE-Bench evaluation framework.

    Args:
        run_id: The run ID of the evaluation.
        instance_id: The instance ID of the evaluation.
        text: The text to log.
    """
    os.makedirs(f"./logs/{run_id}", exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"./logs/{run_id}/{instance_id}", mode="a") as f:
        print(f"\n\n{current_time}\n{text}\n", file=f)
