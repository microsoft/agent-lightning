# Search-R1

| GPU | Model | Controller Mode | Trainer Mode | Code |
|---|---|---|---|---|
| 8× A100 40GB | `meta-llama/Llama-3.2-3B-Instruct` | Local | Sync only | [Source](https://github.com/microsoft/agent-lightning/tree/main/examples/search_r1) |

Search-R1 trains a retrieval-augmented question-answering agent with `verl` and Agent Lightning >=v1.0. During each multi-turn rollout, the agent alternates between model responses and Wikipedia searches before producing a final answer.

This example is based on [*Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*](https://arxiv.org/abs/2503.09516) by Jin et al. (2025).

This example uses the local controller in synchronous trainer mode. Each rollout runs as a local process, calls the policy model through the AGL Gateway, and queries a separate FAISS retrieval service.

The example supports two API styles:

1. **Chat Completions API:** the standard text-in/text-out API used by default.
2. **Token-in/token-out Completions API:** the agent sends prompt token IDs and receives generated token IDs while preserving the multi-turn token sequence.

## Data Preparation

Prepare the Wikipedia corpus, E5 FAISS index, training data, and retriever environment from the repository root:

```bash
examples/search_r1/data_process.sh
```

The script creates a Conda environment named `retriever` and prepares these files:

- `examples/search_r1/data/wiki-18.jsonl`
- `examples/search_r1/data/e5_Flat.index`
- `examples/search_r1/data/train.parquet`
- `examples/search_r1/data/test.parquet`

Set `SEARCH_R1_DATA_DIR` before running the script to use a different data directory.

## Retrieval Service

Start the retrieval service in a separate terminal and keep it running during training:

```bash
examples/search_r1/retrieval_launch.sh
```

The service listens at `http://127.0.0.1:8000/retrieve` by default. Check that it is ready with:

```bash
curl http://127.0.0.1:8000/healthz
```

Common retrieval settings include:

| Setting | Default | Description |
|---|---|---|
| `SEARCH_R1_DATA_DIR` | `examples/search_r1/data` | Corpus and FAISS index directory |
| `SEARCH_R1_RETRIEVAL_PORT` | `8000` | Retrieval service port |
| `SEARCH_R1_TOPK` | `3` | Documents returned for each search |
| `SEARCH_R1_RETRIEVER_DEVICE` | `auto` | Retriever device: `auto`, `cuda`, `cuda:0`, or `cpu` |

## Training

With the retrieval service running, start local training from the repository root:

```bash
examples/search_r1/run.sh
```

`run.sh` starts `agl-server`, the local `agl-controller`, and the `verl` trainer. The script cleans up the server, controller, and Ray processes when it exits.

The default agent uses the Chat Completions API. To use the token-in/token-out Completions API instead:

```bash
examples/search_r1/run.sh --api-type completion
```

To use different dataset files:

```bash
examples/search_r1/run.sh \
    --train-file /path/to/train.parquet \
    --val-file /path/to/test.parquet
```

Agent runtime settings include:

| Setting | Default | Description |
|---|---|---|
| `SEARCH_R1_RETRIEVAL_URL` | `http://127.0.0.1:8000/retrieve` | Retrieval endpoint used by rollout agents |
| `SEARCH_R1_MAX_TURNS` | `4` | Maximum model/search turns per rollout |
| `SEARCH_R1_MAX_TOKENS` | `500` | Maximum generated tokens per model response |
| `SEARCH_R1_TEMPERATURE` | `1.0` | Sampling temperature |
