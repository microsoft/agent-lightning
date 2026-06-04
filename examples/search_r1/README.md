# Search-R1 Example - VERL Training on agl-lite

Train a Search-R1 style retrieval-augmented QA agent using VERL on agl-lite.
This example follows the workflow from Agent Lightning's upstream
`contrib/recipes/search_r1` recipe, adapted to the `examples/calc_x` local-runner
shape in this repository.

## Architecture

```text
data_process.sh
  -> creates/uses conda env: retriever
  -> downloads wiki-18 corpus, E5 FAISS index parts, train/test parquet
  -> writes examples/search_r1/data/wiki-18.jsonl
  -> writes examples/search_r1/data/e5_Flat.index
  -> writes examples/search_r1/data/train.parquet and test.parquet

retrieval_launch.sh
  -> activates retriever env
  -> starts retrieval_server.py at http://127.0.0.1:8000/retrieve

run.sh
  -> agl-lite-server (:8080)                         background
  -> agl-lite-controller runner_type=local           background
  -> train_search_r1_agent.py -> run_ppo()           foreground
                                |
                                v
                     AglLiteRolloutBridge
                                |
                                v
                     enqueue Search-R1 rollouts
                                |
                                v
           local controller starts SearchR1Agent subprocesses
                                |
                                +-> LLM calls -> agl-lite proxy -> VERL vLLM
                                |
                                +-> search calls -> retrieval endpoint (:8000)
                                |
                                +-> POST reward event -> agl-lite server
```

Search-R1 is multi-turn. Each rollout alternates between model responses,
`<search>...</search>` actions, `<information>...</information>` retrieval
feedback, and a final `<answer>...</answer>`. The training config uses agl-lite
trajectory aggregation so the multi-turn trace is trained as one trajectory.

## Step 1: Prepare Data and Retriever Environment

Run this once from the repository root:

```bash
examples/search_r1/data_process.sh
```

This mirrors the upstream Search-R1 preparation flow. It:

- creates a conda environment named `retriever` if it does not exist;
- installs PyTorch, FAISS GPU, `transformers`, `datasets`, `pyserini`,
  `fastapi`, `uvicorn`, and `tqdm` into that environment;
- downloads the Wikipedia corpus from `PeterJinGo/wiki-18-corpus`;
- downloads the E5 FAISS index parts from `PeterJinGo/wiki-18-e5-index`;
- downloads train/test QA parquet files from `PeterJinGo/nq_hotpotqa_train`;
- combines `part_*` into `data/e5_Flat.index`;
- decompresses `data/wiki-18.jsonl.gz` into `data/wiki-18.jsonl`.

Expected files after preparation:

```text
examples/search_r1/data/wiki-18.jsonl
examples/search_r1/data/e5_Flat.index
examples/search_r1/data/train.parquet
examples/search_r1/data/test.parquet
```

Useful overrides:

```bash
export SEARCH_R1_DATA_DIR=/path/to/search_r1_data
export SEARCH_R1_RETRIEVER_ENV=retriever
export SEARCH_R1_RETRIEVER_PYTHON=3.10
export SEARCH_R1_PYTORCH_CUDA=12.1
export SEARCH_R1_SKIP_RETRIEVER_INSTALL=1  # skip installs when env is ready
```

## Step 2: Start Retrieval Endpoint

Start the retrieval endpoint in a separate terminal and keep it running during
training:

```bash
examples/search_r1/retrieval_launch.sh
```

By default this starts:

```text
http://127.0.0.1:8000/retrieve
```

The endpoint accepts the Search-R1 request shape:

```json
{
  "queries": ["query text"],
  "topk": 3,
  "return_scores": true
}
```

and returns:

```json
{
  "result": [
    [
      {
        "document": {
          "title": "...",
          "text": "...",
          "contents": "..."
        },
        "score": 0.0
      }
    ]
  ]
}
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Useful overrides:

```bash
export SEARCH_R1_DATA_DIR=/path/to/search_r1_data
export SEARCH_R1_RETRIEVER_ENV=retriever
export SEARCH_R1_RETRIEVAL_HOST=0.0.0.0
export SEARCH_R1_RETRIEVAL_PORT=8000
export SEARCH_R1_TOPK=3
export SEARCH_R1_RETRIEVER_NAME=e5
export SEARCH_R1_RETRIEVER_MODEL=intfloat/e5-base-v2
export SEARCH_R1_RETRIEVER_DEVICE=auto  # auto, cuda, cuda:0, or cpu
```

The training agent reads the retrieval URL from:

```bash
export SEARCH_R1_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve
```

## Step 3: Run Local Training

In another terminal, from the repository root:

```bash
examples/search_r1/run.sh
```

`run.sh` starts `agl-lite-server`, waits for `/healthz`, starts
`agl-lite-controller runner_type=local`, then launches
`train_search_r1_agent.py`.

For a short smoke configuration:

```bash
examples/search_r1/run.sh --ci
```

To use a different model while keeping the server proxy and trainer model in
sync:

```bash
SEARCH_R1_MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct examples/search_r1/run.sh --ci
```

Default full-training model:

```text
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

`--ci` defaults to:

```text
Qwen/Qwen2.5-Coder-0.5B-Instruct
```

unless `--model` or `SEARCH_R1_MODEL` is provided.

## Standalone Training

If `agl-lite-server`, `agl-lite-controller runner_type=local`, VERL, and the
retrieval endpoint are already running:

```bash
python examples/search_r1/train_search_r1_agent.py \
    --train-file examples/search_r1/data/train.parquet \
    --val-file examples/search_r1/data/test.parquet \
    --agl-base-url http://localhost:8080 \
    --agl-key dummy
```

The script accepts dotlist config overrides after known arguments, matching the
`calc_x` style:

```bash
python examples/search_r1/train_search_r1_agent.py --ci \
    trainer.total_training_steps=2 \
    actor_rollout_ref.rollout.n=2
```

## Runtime Knobs

Agent-side knobs read by each local rollout subprocess:

```bash
export SEARCH_R1_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve
export SEARCH_R1_TOPK=3
export SEARCH_R1_MAX_TURNS=4
export SEARCH_R1_MAX_TOKENS=500
export SEARCH_R1_TEMPERATURE=1.0
```

Dataset paths can be overridden without moving files:

```bash
examples/search_r1/run.sh \
    --train-file /path/to/train.parquet \
    --val-file /path/to/test.parquet
```

Rows are expected to contain at least:

| Column | Description |
|--------|-------------|
| `question` | QA prompt shown to the agent |
| `golden_answers` | Accepted exact-match answers; string or list |

## Files

| File | Description |
|------|-------------|
| `data_process.sh` | Creates retriever env and downloads/prepares corpus, index, train/test data |
| `retrieval_launch.sh` | Activates retriever env and starts the retrieval endpoint |
| `retrieval_server.py` | Search-R1 compatible dense retrieval FastAPI service |
| `run.sh` | E2E local entrypoint: agl-lite server, local controller, training |
| `train_search_r1_agent.py` | Loads parquet data, builds VERL config, calls `run_ppo()` |
| `agents/search_r1_agent.py` | Local runner agent: LLM loop, retrieval calls, reward event |
| `agents/qa_em.py` | Exact-match reward helpers adapted from upstream Search-R1 |
| `data/` | Dataset, corpus, and FAISS index directory |
