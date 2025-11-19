# ADK Example

This folder hosts the runnable sample that pairs the ADK agent with Agent-lightning’s VERL integration. For architectural details, please read the [Train ADK Agent how-to](../../docs/how-to/train-adk-agent.md). This README focuses only on installing dependencies and running the scripts.

## Install

```bash
cd examples/google_adk
pip install "agentlightning[verl,adk]" "google-adk>=0.3.0"
# or: uv sync
```

You’ll need a machine with a 40 GB GPU (A100 or similar) for full training; CPU + smaller GPUs are fine for CI modes.

## Prepare data

Create `data/train.parquet` and `data/test.parquet` that match the `AdkTask` schema (`question`, `app_id`, `ground_truth`, optional `meta`). To download and convert the Spider dataset (same as `examples/spider`):

```bash
uv run python prepare_dataset.py --download --outdir data
```

Alternatively, convert your own JSON/CSV files using `--train` and `--test` flags. See the how-to guide for details.

## Run training

```bash
python train_adk.py \
  --train-file data/train.parquet \
  --val-file data/test.parquet \
  --model ${OPENAI_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct} \
  --endpoint ${OPENAI_API_BASE:-http://localhost:8000/v1}
```

Useful flags:

- `--ci` / `--ci-fast` shrink runner count and dataset slices for smoke tests.
- `--external-store-address` connects to an existing LightningStore service.
- `--wandb-project` / `--wandb-run-name` enable Weights & Biases logging.

Environment variables the scripts read:

- `OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_MODEL`
- `HF_TOKEN` (required for VERL checkpoints hosted on Hugging Face)

## Quick debug loop

Before spending GPU hours, run:

```bash
python adk_debug.py --file data/test.parquet --index 0
```

This executes a single rollout with the same ADK wiring used in training, letting you confirm credentials, dataset rows, and trace emission without launching VERL. Use `--model` and `--endpoint` overrides to point at different LLM backends.
