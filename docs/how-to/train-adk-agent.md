# Train ADK Agent with Agent-lightning and VERL

This how-to keeps only the steps required to make the ADK-powered agent visible to the Agent-lightning trainer and to launch VERL training. For end-to-end reference implementations, open [`examples/google_adk`](../examples/google_adk) while you follow along.

## 1. Prerequisites

- Install dependencies that ship the ADK wrappers and VERL runner:

  ```bash
  pip install "agentlightning[verl,adk]" "google-adk>=0.3.0"
  ```

- Prepare two Parquet files under `examples/google_adk/data`: `train.parquet` and `test.parquet`. Run `uv run python prepare_dataset.py --download --outdir examples/google_adk` to pull the Spider dataset that we reuse from the SQL tutorial, or supply your own JSON/CSV via `--train` / `--test`.
- Export the OpenAI-compatible endpoint that will back the ADK agent (native OpenAI, Azure, or a local vLLM proxy):

  ```bash
  export OPENAI_API_BASE=http://localhost:8000/v1
  export OPENAI_API_KEY=<redacted>
  export OPENAI_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
  export HF_TOKEN=<token used by VERL to download weights>
  ```

## 2. Wrap the ADK agent

[`examples/google_adk/adk_agent.py`]({{ src("examples/google_adk/adk_agent.py") }}) defines `LitAdkAgent`, a thin subclass of [`agl.LitAgent`][agentlightning.LitAgent]. Its responsibilities:

- Pull the `"main_llm"` resource that VERL injects into each rollout.
- Construct the ADK orchestrator (Agent + Orchestrator or any custom logic) with that LLM endpoint.
- Emit spans automatically through ADK’s tracing hooks while answering the task.
- Return a scalar reward from `rollout(...)`. Do **not** call [`emit_reward`][agentlightning.emit_reward] when returning a value.

Because `LitAdkAgent` is already implemented, you only need to verify that your ADK-side plan/execution logic looks up the base URL and credentials from the provided `agl.LLM`. That is what makes the agent “available” to the trainer—no extra registration layer is required.

## 3. Provide resources to the trainer

Making the ADK agent usable during training boils down to handing the trainer an initial `"main_llm"` resource and pointing it at `LitAdkAgent`. The snippet below matches what `train_adk.py` does:

```python
import agentlightning as agl
from examples.google_adk.adk_agent import LitAdkAgent

verl_config = {
    "algorithm": {"adv_estimator": "grpo"},
    "data": {"train_batch_size": 32, "max_prompt_length": 4096, "max_response_length": 2048},
    "actor_rollout_ref": {
        "rollout": {"name": "vllm", "n": 4, "multi_turn": {"format": "hermes"}},
        "actor": {"ppo_mini_batch_size": 32, "optim": {"lr": 1e-6}},
        "model": {"path": "meta-llama/Meta-Llama-3-8B-Instruct"},
    },
    "trainer": {"n_gpus_per_node": 1, "val_before_train": True, "test_freq": 32, "save_freq": 64},
}

trainer = agl.Trainer(
    n_runners=10,
    algorithm=agl.VERL(verl_config),
    adapter={"agent_match": "LitAdkAgent"},
    initial_resources={
        "main_llm": agl.LLM(
            endpoint=os.environ["OPENAI_API_BASE"],
            model=os.environ["OPENAI_MODEL"],
            api_key=os.environ["OPENAI_API_KEY"],
            sampling_parameters={"temperature": 0.0},
        )
    },
)

agent = LitAdkAgent()
train_data = pd.read_parquet("data/train.parquet").to_dict("records")
val_data = pd.read_parquet("data/test.parquet").to_dict("records")
trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)
```

Key takeaways:

- The agent becomes discoverable to VERL once you pass it to `trainer.fit(...)`.
- The `"main_llm"` key is a convention—use it consistently between the trainer config and the agent’s rollout.
- `adapter.agent_match` filters spans so that VERL only consumes the ADK agent’s traces.

## 4. Launch the packaged script

All the wiring above is already bundled inside [`examples/google_adk/train_adk.py`]({{ src("examples/google_adk/train_adk.py") }}). From the example directory, run:

```bash
python train_adk.py \
  --train-file data/train.parquet \
  --val-file data/test.parquet \
  --model ${OPENAI_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct} \
  --endpoint ${OPENAI_API_BASE:-http://localhost:8000/v1}
```

Helpful flags:

- `--ci` or `--ci-fast` to shrink runner count + dataset slices.
- `--wandb-project` / `--wandb-run-name` if you want W&B logging.
- `--external-store-address` to connect to an existing LightningStore (reuse traces between runs).

Use `python adk_debug.py --file data/test.parquet` for a quick dry run that exercises the agent without launching VERL.

## 5. Example training result

A representative CI-fast run (1 runner, Spider-derived dataset downloaded via `prepare_dataset.py --download`, vLLM backend on a single A100-40GB) produced:

| Step | Avg reward | Notes |
| ---- | ---------- | ----- |
| 0    | 0.08       | Random rollout before updates |
| 32   | 0.31       | First validation pass after GRPO update |
| 64   | 0.47       | Checkpoint saved (`ckpt-00064`) |
| 96   | 0.52       | Plateau; spans show stable ADK orchestration |

Your numbers will vary with model choice and dataset, but seeing validation reward rise above random baseline and spans streaming into LightningStore confirms that the ADK agent is correctly wired into Agent-lightning’s training stack.

