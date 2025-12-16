# Search-R1 Example

## Overview

This example implements **Search R1** within Agent Lightning. It also serves as a demonstration of a **framework-free agent training pipeline**, showing how to run end-to-end RL training without relying on specialized frameworks. **It's tested and compatible with Agent-lightning v0.1.2**.

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

## Benchmark Results

We evaluated Search-R1 across seven diverse question-answering benchmarks, covering both General QA (NQ, TriviaQA, PopQA) and complex multi-hop reasoning tasks (HotpotQA, 2WikiMultiHopQA, Musique, and Bamboogle).

The following tables compare the performance of the original Search-R1 implementation and the Agent-Lightning version across various base models.

| Model | Source | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B-Instruct** | **Search-R1 (Original)** | 34.1 | 54.5 | 37.8 | 32.4 | 31.9 | 10.3 | 26.4 |
| | **Agent-Lightning** | **45.3** | **61.7** | **43.8** | **42.6** | **36.4** | **17.1** | **37.6** |
| **Qwen2.5-7B-Instruct** | **Search-R1 (Original)** | 39.3 | 61.0 | 39.7 | 37.0 | 41.4 | 14.6 | 36.8 |
| | **Agent-Lightning** | **46.5** | **65.9** | **46.8** | **43.7** | **46.2** | **20.3** | **47.2** |
| **Llama-3.2-3B** | **Search-R1 (Reproduced)** | 26.3 | 49.0 | 23.0 | 21.6 | 27.3 | 4.5 | 9.7 |
| | **Agent-Lightning** | **29.6** | **51.9** | **25.7** | **23.2** | **28.3** | **5.8** | 9.6 |
