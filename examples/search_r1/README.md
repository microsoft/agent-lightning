# Search-R1 Example

## Overview

This example implements **Search R1** within Agent Lightning. It also serves as a demonstration of a **framework-free agent training pipeline**, showing how to run end-to-end RL training without relying on specialized frameworks. **It's tested and compatible with Agent-lightning v0.1.x**.

The example is designed to run on a single node with 8 GPUs, each having at least 40 GB of memory.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `data_process.sh` | Prepares the Wikipedia corpus, datasets, and `retriever` conda environment |
| `retrieval_launch.sh` | Launches the retrieval service backed by the processed corpus |
| `retrieval_server.py` | FastAPI server that powers document retrieval during training |
| `search_r1_agent.py` | Agent-Lightning rollout script implementing the Search-R1 workflow |
| `train.sh` | Starts the RL training server that coordinates GRPO optimization |
| `qa_em.py` | Exact-match evaluation utilities for validating model predictions |

---

## Prepare Data and Environment

Run the following script once to prepare data and the retriever environment:

```bash
bash data_process.sh
```

This script performs the following steps:

* Creates a new conda environment named **`retriever`**.
* Downloads the **Wikipedia data** used to build the retrieval database.
* Downloads the **training and testing datasets**.
* Stores all data under the newly created **`data/`** directory.

The environment setup and data-processing logic are adapted from [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1).

---

## Prepare Retrieval Server

To start the retrieval server, run:

```bash
bash retrieval_launch.sh
```

This script activates the previously created **`retriever`** environment and starts a **retrieval server** at `http://127.0.0.1:8000` using the downloaded Wikipedia data. The server receives user queries and returns a ranked list of retrieved text passages.

The retrieval server implementation is based on `search_r1/search/retrieval_server.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py).

> ⚠️ **Note:** Keep the retrieval server running during training (for example, in a separate `tmux` session or terminal window).

---

## Run RL Training (GRPO) with Llama-3.2-3b-base

1. **Start Ray**

   ```bash
   bash ../../scripts/restart_ray.sh
   ```

   > If you plan to use WandB for experiment tracking, set the environment variable
   > `WANDB_API_KEY` before starting Ray.

2. **Launch the Agent**

   ```bash
   python search_r1_agent.py
   ```

   This script automatically launches **128 agent workers** by default. Each agent follows the Search-R1 workflow, retrieving information from the database and generating answers accordingly.


3. **Start the Training Server**
   In another terminal, run:

   ```bash
   bash train.sh
   ```

   This script starts the RL training server.

---

## Evaluation

Evaluation scripts and benchmark results will be released soon.
