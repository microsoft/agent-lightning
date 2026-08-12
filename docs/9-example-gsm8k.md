# GSM8K

| GPU | Model | Controller Mode | Trainer Mode |
|---|---|---|---|
| 1× A100 80GB | `Qwen/Qwen2.5-1.5B-Instruct` | Local | Sync only |

GSM8K trains a grade-school math reasoning agent on the `openai/gsm8k` dataset with VERL and Agent Lightning >=v1.0.
This example runs in local mode and demonstrates support for two API styles:

1. **Chat Completions API:** the commonly used text-in/text-out API, where the agent sends structured chat messages and receives generated text.
2. **Token-in/token-out Completions API:** the agent sends prompt token IDs directly and receives generated token IDs.

## Data Preparation

Download the dataset into `~/dataset/gsm8k`:

```bash
hf download openai/gsm8k --repo-type dataset --local-dir ~/dataset/gsm8k
```

The example reads these files by default:

- `~/dataset/gsm8k/main/train-00000-of-00001.parquet`
- `~/dataset/gsm8k/main/test-00000-of-00001.parquet`

Training uses all samples from `main/train`. Validation uses 100 random samples from `main/test` with seed `42` by default.

## Training

Make sure you have activated the project environment and installed the example dependencies:

```bash
source .venv/bin/activate
uv pip install \
    datasets \
    openai \
    httpx
```

Then start training:

```bash
source .venv/bin/activate
cd examples/gsm8k
bash run_local.sh
```

You can change the validation sample count or seed with:

```bash
bash run_local.sh --val-size 100 --seed 42
```

The local example uses `ChatAgent` with the standard Chat Completions API by default. To demonstrate the token-in/token-out Completions API, use `CompletionAgent` instead:

```bash
bash run_local.sh --api completion
```

In token-in/token-out mode, the agent tokenizes the prompt with the configured model tokenizer, sends prompt token IDs to the OpenAI-compatible Completions endpoint, receives response token IDs, and decodes them locally for answer evaluation.

`run_local.sh` starts `agl-server`, `agl-controller`, and Ray locally, and writes server/controller logs under `/tmp/`.
When the script exits, it cleans up the local server, controller, and Ray process it started.
