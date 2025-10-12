# Search_R1 Example

## Overview

This example originally runs on a single node with eight GPUs, each requiring at least 40GB of memory.

### Prepare Data and Environment

Run `bash data_process.sh` to download database, training and testing data, and build a retriever environment.

### Prepare retrieval Server

Run `bash retrieval_launch.sh` to launch the retriever server.

### Run RL training (GRPO) with Llama-3.2-3b-base.

1. Start ray: `bash ../../scripts/restart_ray.sh`. To use Wandb, you need to set the WANDB_API_KEY environment variable before starting ray.

2. Run the agent: `python search_r1_agent.py`. It automatically launches 128 agent workers by default.

3. In another terminal, launch the training server: `bash train.sh`.
