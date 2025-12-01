# Tinker + Agent-lightning Integration

[![tinker CI status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-tinker.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-tinker.yml)

This example shows how to use [Tinker's reinforcement-learning infrastructure](https://tinker-docs.thinkingmachines.ai/) as a fine-tuning backend for agents written against Agent-lightning. You author the agent exactly the way you would for deployment, while the bridge code reconstructs Tinker-compatible trajectories from Agent-lightning traces.

## How this differs from the original Tinker Cookbook RL recipe

Real-world agent apps orchestrate logic in familiar frameworks (CrewAI, LangChain, AutoGen, OpenAI Agents, etc.) or by calling OpenAI-compatible REST APIs. A simple number-guessing agent might look like this:

```python
def guess_number_agent():
    client = openai.OpenAI()
    messages = [{"role": "user", "content": "Guess a number between 1 and 100."}]
    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(model="gpt-4.1", messages=messages)
        response_content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response_content})
        guessed_number = extract_number(response_content)
        if guessed_number == gold_answer:
            return 1.0
        elif guessed_number < gold_answer:
            messages.append({"role": "user", "content": "Too low"})
        else:
            messages.append({"role": "user", "content": "Too high"})
    return 0.0
```

The reference [Tinker Cookbook example](https://github.com/thinking-machines-lab/tinker-cookbook/tree/51d9e8226f2dcf82ceac272c734a5f6e3b4f0203/tinker_cookbook/recipes/multiplayer_rl/guess_number), however, expects you to rewrite the same logic into a callback-style `Env`, and it creates a simple loop to iterate between a language model (`TokenCompleter`) and the `Env`.

```python
class GuessNumberEnv:
    def __init__(self, gold_answer: int):
        self.system_prompt: Message = {"role": "system", "content": SYSTEM_PROMPT}
        self.turns: list[Message] = []
        self.gold_answer: int = gold_answer

    async def initial_observation(self) -> list[int]:
        return message_to_tokens(self.system_prompt)

    async def step(self, action_tokens: list[int]) -> tuple[list[int], float, bool]:
        action_message = tokens_to_message(action_tokens)
        guessed_number = extract_number(action_message["content"])

        if guessed_number == self.gold_answer:
            text, reward = "Correct", 1.0
        elif guessed_number < self.gold_answer:
            text, reward = "Too low", 0.0
        else:
            text, reward = "Too high", 0.0

        self.turns.append(action_message)
        self.turns.append({"role": "assistant", "content": text})
        episode_done = reward == 1 or len(self.turns) // 2 >= MAX_TURNS
        return message_to_tokens(self.turns), reward, episode_done
```

As agents grow more complex, writing them in callback style becomes increasingly painful. You have to break the control flow whenever an LLM call is required, which fragments the code and makes it harder to maintain.

Agent-lightning hides that translation step: you keep the first style for development and production, while the framework queues tasks to the store, rebuilds trajectories from spans, and feeds them to the training loop. This example shows how to make Tinker's original training loop work with Agent-lightning.

## Included files

| Path | Purpose |
| ---- | ------- |
| `hello.py` | Minimal end-to-end fine-tuning example. Trains a model to repeat small identity strings. |
| `q20_agent.py` | CrewAI flow that powers the 20 Questions player, answerer, and mock search tool. Shared by training and evaluation. **Unrelated to Agent-lightning or Tinker.** |
| `q20_train.py` | Reinforcement-learning driver that adapts the Cookbook loop to Agent-lightning rollouts. Supports dry-run, distributed training, and search tool toggles. **Related to both Agent-lightning and Tinker.** |
| `q20_evaluate.py` | Offline evaluator that reuses the CrewAI flow to benchmark any OpenAI- or Qwen-backed model against the provided dataset. **Related to Tinker only.** |
| `q20_nouns.csv` | Categories and answers used for training and validation. Contains `split` and `search_enabled` metadata. |
| `agl_tinker/` | Bridge package for integrating Agent-lightning with Tinker (see breakdown below). |
| `tests/test_tinker_llm.py` | Sanity tests for the custom LiteLLM provider. Run with `pytest examples/tinker/tests`. |
| `.env.example` | Template for environment variables required by LiteLLM, CrewAI helpers, and the hosted Tinker service. |

`agl_tinker/` components:

| Path | Purpose |
| ---- | ------- |
| `agl_tinker/algo.py` | Agent-lightning `Algorithm` wrapper that plugs the training loop into `agl.Trainer`. |
| `agl_tinker/env.py` | Dummy env and dataset builders that adapt Agent-lightning tasks to Tinker expectations. |
| `agl_tinker/llm.py` | LiteLLM custom provider backed by the Tinker sampling client. |
| `agl_tinker/rollout.py` | Span-to-trajectory reconstruction and rollout batching helpers. |
| `agl_tinker/train.py` | RL training loop adapted from the Tinker Cookbook. |

## Setup

**1. Install dependencies.** From the repo root:

```bash
uv sync --frozen --extra apo --group dev --group agents --group tinker
```

If you are not using `uv`, make sure `tinker`, `tinker_cookbook`, `litellm`, `crewai`, and Agent-lightning are available in the same environment.

**2. Copy the environment template and fill in credentials:**

```bash
cp examples/tinker/.env.example examples/tinker/.env
```

- `OPENAI_API_KEY` / `OPENAI_BASE_URL`: routes helper agents (answerer, search, tool simulations) through a LiteLLM or OpenAI-compatible endpoint.
- `TINKER_API_KEY`: required to talk to the hosted Tinker training service. Skip if you are using OpenAI models only.
- `WANDB_API_KEY`: optional, enables Weights & Biases logging when configured in `q20_train.py`.
- `CREWAI_DISABLE_TELEMETRY=true`: keeps CrewAI from emitting its own telemetry so that Agent-lightning tracing stays coherent.

3. Load the environment before running commands, e.g. `dotenv run -- <command>` or export the variables manually.

## Running the Hello 1024 example

This is the quickest way to see the integration in action. It fine-tunes a Qwen model so it introduces itself with the target identity.

**One-click workflow (spawns store, algorithm, and runners in a single process)**

```bash
dotenv run python hello.py oneclick
```

The script will pick free ports for the LiteLLM proxy and Agent-lightning store, then iterate through the synthetic dataset of identities.

**Distributed workflow (useful for inspecting each component)**

```bash
agl store --port 4747
dotenv run python hello.py algo
dotenv run python hello.py runner
```

Start the commands in separate terminals. The algorithm process connects to the existing store, while the runner process launches eight worker processes by default. Logs are written to `examples/tinker/logs/hello`.

## Training the 20 Questions agent

The 20 Questions setup mirrors the official Cookbook recipe but drives rollouts through the shared CrewAI flow.

**Dry run (in-memory store and LiteLLM proxy)**

```bash
dotenv run python q20_train.py dryrun
```

Useful to verify that the CrewAI flow, reward emission, and span reconstruction succeed on a handful of samples without touching the hosted Tinker service.

**Full distributed training**

```bash
agl store --port 4747
dotenv run python q20_train.py algo --model qwen30b --search --port 4747
dotenv run python q20_train.py runner --port 4747 --n-runners 4
```

`--model` selects the Tinker-hosted checkpoint (`qwen4b` or `qwen30b`). Add `--search` to enable the mocked search tool, which relies on the helper LLM defined in the environment variables (the example uses an LLM-powered search simulation instead of a real API). Training metrics and checkpoints are recorded under `examples/tinker/logs/q20_*`. You can also use `verl` as a substitute for the `algo` command when Tinker service is not available.

You can run additional runner processes at any time; they register with the store and start dequeuing tasks immediately.

## Evaluating a model on 20 Questions

Reuse the CrewAI flow to benchmark any OpenAI-compatible model (hosted on Tinker, OpenAI, or another LiteLLM backend):

```bash
dotenv run python q20_evaluate.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --output-file logs/twenty_questions_results.jsonl \
  --search
```

Results append to the specified JSONL file so you can compute aggregate stats later.

## How the bridge works

The `agl_tinker` package keeps the rest of the Tinker or Tinker Cookbook's codebase untouched by emulating the interfaces it expects:

- `AGLDatasetBuilder` and `AGLDummyEnv` wrap plain Agent-lightning datasets so batches still yield Tinker `EnvGroupBuilder` objects, even though rollouts run remotely.
- `do_group_of_group_rollouts` (in [`rollout.py`](agl_tinker/rollout.py)) enqueues tasks to the Agent-lightning store, waits for runners to finish, then reconstructs `Trajectory` objects from span triplets collected by `TracerTraceToTriplet`.
- `TinkerLLM` implements LiteLLM's `CustomLLM` so the training loop can update sampling clients and expose them through an OpenAI-compatible endpoint without rewriting agent code.
- `agl_tinker.algo.Tinker` satisfies Agent-lightning's `Algorithm` contract, meaning you can launch training via `agl.Trainer` alongside other algorithms, schedulers, or resources.

Because spans and rewards are emitted by the same rollout function you would deploy, evaluation and production stay in sync—no separate simulator code paths to maintain.

## Troubleshooting tips

- If the runner logs show `Triplet has no token_ids`, ensure your LiteLLM proxy returns logprobs and token IDs, and that the token IDs are present in the store. The provided adapter requires them to rebuild trajectories. See the debugging tutorial for more details.
- CrewAI telemetry must stay disabled (see `.env.example`) so AgentOps traces remain self-contained; otherwise, you may see malformed traces.
- Tune `learning_rate`, `batch_size` and `group_size` carefully. The training is sensitive to these hyper-parameters.
