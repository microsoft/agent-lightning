# Train ADK Agent with Agent-lightning and VERL

This walkthrough builds upon the **Agent-lightning v0.2 ADK Agent** example and explains how the system components integrate: an **ADK agent** wrapped as a [`LitAgent`][agentlightning.LitAgent], the **[`VERL`][agentlightning.algorithm.verl.VERL] reinforcement learning (RL) algorithm**, and the **[`Trainer`][agentlightning.Trainer]**, which coordinates both training and debugging.

The command-line interface in [`examples/google_adk/train_adk.py`]({{ src("examples/google_adk/train_adk.py") }}) provides a complete runnable example. However, this document focuses on understanding the underlying architecture so you can effectively adapt the workflow to your own agents.

## ADK Agent Architecture

Agent-lightning integrates seamlessly with various orchestration frameworks, including [Agent Framework](https://github.com/microsoft/agent-framework), [AutoGen](https://github.com/microsoft/autogen), [CrewAI](https://www.crewai.com/), [LangGraph](https://github.com/langchain-ai/langgraph), and the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). It can also interoperate with custom Python logic.

**ADK (Application Development Kit)** is an observability framework for agent traces that provides span visualization and Cloud Trace integration. This example demonstrates how to wrap ADK's agent orchestration capabilities in a [`LitAgent`][agentlightning.LitAgent], similar to how the spider example wraps LangGraph workflows.

In this example, the **ADK agent** uses ADK's agent orchestration capabilities to build workflows that automatically capture and visualize spans (individual operations like LLM calls, tool executions) within a rollout. These spans can be organized as sequential, parallel, or nested structures, providing rich observability into agent behavior.

The visualization shows how spans are organized within a rollout, making it easier to understand agent execution patterns and debug issues. ADK's Cloud Trace integration enables distributed tracing across multiple services and components.

## Bridging ADK and Agent-lightning

!!! tip

    Keep [`adk_agent.py`]({{ src("examples/google_adk/adk_agent.py") }}) open on the side while reading this section. This will help you understand how the code snippets shown here work in practice.

The **`LitAdkAgent`** class defined in [`adk_agent.py`]({{ src("examples/google_adk/adk_agent.py") }}) acts as the bridge. It subclasses [`agl.LitAgent`][agentlightning.LitAgent], allowing the runner to provision shared resources (e.g., [LLMs][agentlightning.LLM]) for each rollout. Similar to how `LitSQLAgent` wraps LangGraph workflows in the spider example, `LitAdkAgent` wraps ADK's agent orchestration capabilities.

Below is a simplified illustration of the key logic:

```python
class LitAdkAgent(agl.LitAgent[AdkTask]):

    def rollout(
        self,
        task: AdkTask,
        resources: agl.NamedResources,
        rollout: agl.Rollout
    ) -> float | None:
        # 1) Obtain the LLM resource (injected by the algorithm/trainer)
        llm: agl.LLM = cast(agl.LLM, resources.get("main_llm"))

        # 2) Unpack task fields
        question = task["question"]
        app_id = task["app_id"]
        ground_truth = task["ground_truth"]

        # 3) Build your ADK agent and perform the action
        # In a real integration, you would use ADK's agent orchestration:
        # from adk import Agent, Orchestrator
        # adk_agent = Agent(
        #     base_url=llm.endpoint,
        #     model=llm.model,
        #     api_key=llm.api_key,
        #     enable_tracing=True  # Enables Cloud Trace integration
        # )
        # orchestrator = Orchestrator(agent=adk_agent)
        # action = orchestrator.plan_and_execute(
        #     question=question,
        #     app_id=app_id,
        #     temperature=llm.sampling_parameters.get("temperature", 0.0)
        # )
        # ADK automatically creates spans for each operation, showing sequential,
        # parallel, or nested structures in the trace visualization.
        
        # For demonstration, we synthesize a naive "action" string:
        action = f"adk://{app_id}?plan={question}"

        # 4) Compute reward based on action matching ground truth
        reward = 1.0 if ground_truth and ground_truth.lower() in action.lower() else 0.0

        # 5) Return the reward
        return reward
```

The `LitAdkAgent` serves as a wrapper that extracts the LLM resource, uses ADK's orchestration capabilities to process the task, and returns an evaluation result as a reward signal. ADK's observability features automatically capture spans for visualization, making it easy to understand agent execution patterns.

The `"main_llm"` resource key is a convention between the agent and [VERL][agentlightning.algorithm.verl.VERL]. It is used to inject an OpenAI-compatible endpoint from the [VERL][agentlightning.algorithm.verl.VERL] algorithm during rollout. Two approaches are supported to use this [agentlightning.LLM][] resource:

1. **Direct access** – Use [`llm.endpoint`][agentlightning.LLM.endpoint] for a simple integration.
2. **Context-aware access** – Use [`get_base_url`][agentlightning.ProxyLLM.get_base_url] with [`rollout.rollout_id`][agentlightning.Rollout.rollout_id] and [`rollout.attempt.attempt_id`][agentlightning.Attempt.attempt_id].
   This approach enables per-caller trace attribution, improving trace collection per rollout or attempt when runner-side tracers are unavailable. For details, see [Working with Traces](../tutorials/traces.md).

## Task Structure

The ADK agent expects tasks with the following structure:

```python
class AdkTask(TypedDict):
    question: str        # The user instruction for the agent
    app_id: str         # The application/environment identifier
    ground_truth: str   # The expected action/output for reward computation
    meta: Dict[str, Any] | None  # Optional arbitrary metadata
```

Each task represents a single user instruction that the agent should process to generate an appropriate action. ADK's observability features will automatically capture spans for each operation, organizing them for visualization.

## Reward Signal and Evaluation

The reward computation in the agent provides the reward mechanism for RL training. In this example, a simple reward is computed by checking if the generated action contains the ground truth pattern. However, in a production setup, you would implement more sophisticated evaluation logic.

For example, you might:

- Execute the generated action and compare the results to expected outcomes
- Use semantic similarity metrics to compare the generated action to the ground truth
- Apply domain-specific validation rules based on the app_id and question type
- Leverage ADK's trace visualization to understand execution patterns and identify optimization opportunities

!!! attention

    The ground-truth actions must **never** be exposed to the agent during training to prevent data leakage.

In this setup, the reward is returned directly from the [`rollout`][agentlightning.LitAgent.rollout] method, enabling the runner to forward it back to the RL algorithm.

!!! warning

    Avoid using [`emit_reward`][agentlightning.emit_reward] in conjunction with returning a reward value. Doing both will cause the algorithm to receive duplicate reward signals, leading to inconsistent training behavior.

## Configuring VERL for Reinforcement Learning

View [`examples/google_adk/train_adk.py`]({{ src("examples/google_adk/train_adk.py") }}) for a full reinforcement learning configuration, which is a plain Python dictionary. It mirrors (and actually *is*) the [shell arguments](https://verl.readthedocs.io/en/latest/index.html) used to launch training in the VERL framework but is easier to tweak programmatically:

```python
verl_config: Dict[str, Any] = {
    "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
    "data": {
        # train_files and val_files are no longer needed here
        # because data are read in agl.Trainer
        ...,
        # Controls how many tasks are pooled per step
        # (multiplied by actor_rollout_ref.rollout.n)
        "train_batch_size": 32,
        # Prompt and responses larger than these lengths are truncated
        "max_prompt_length": 4096,
        "max_response_length": 2048,
    },
    "actor_rollout_ref": {
        "rollout": {
            # Only vLLM is supported currently
            "name": "vllm",
            # Equals to group size of GRPO
            "n": 4,
            # Used to enable tool call parser in vLLM
            "multi_turn": {"format": "hermes"},
            ...
        },
        "actor": {"ppo_mini_batch_size": 32, "optim": {"lr": 1e-6}, ...},
        "model": {
            # Config your preferred LLM here
            "path": "meta-llama/Meta-Llama-3-8B-Instruct",
            ...
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        # Validation once before training starts
        "val_before_train": True,
        # Validation every N training steps
        "test_freq": 32,
        # Save checkpoints every N training steps
        "save_freq": 64,
        # Go through the train dataset this many times
        "total_epochs": 2
    },
}
```

This is equivalent to the following CLI invocation:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.path=meta-llama/Meta-Llama-3-8B-Instruct \
    trainer.n_gpus_per_node=1 \
    trainer.val_before_train=True \
    trainer.test_freq=32 \
    trainer.save_freq=64 \
    trainer.total_epochs=2
```

!!! warning
    We used to provide a CLI called `python -m agentlightning.verl` to launch training in v0.1. This is no longer the recommended approach. Instead, use [`agl.Trainer`][agentlightning.Trainer] to run VERL and agent runners together, or follow the [debugging tutorial](../tutorials/debug.md) if you want an isolated experience similar to v0.1.

## Orchestrating Training with [`Trainer`][agentlightning.Trainer]

[`Trainer`][agentlightning.Trainer] is the high-level orchestrator that integrates the agent, algorithm, dataset, and distributed runners. The key benefits of using the [`Trainer`][agentlightning.Trainer] are:

1. It allows you to launch everything with a single line of code: `trainer.fit(...)`.
2. It exposes configuration options such as `n_runners` to control parallelism and `adapter` to define how algorithms interpret the trace data produced by the agent.

An example usage is shown below:

```python
import agentlightning as agl
import pandas as pd

agent = LitAdkAgent()
algorithm = agl.VERL(verl_config)
trainer = agl.Trainer(
    n_runners=10,
    algorithm=algorithm,
    adapter={"agent_match": "LitAdkAgent"},
)
train_data = pd.read_parquet("data/train.parquet").to_dict("records")
val_data = pd.read_parquet("data/test.parquet").to_dict("records")
trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)
```

First, `agl.VERL(verl_config)` launches the [`VERL`][agentlightning.algorithm.verl.VERL] algorithm and its OpenAI-compatible proxy. The `train_data` and `val_data` are passed into [`VERL`][agentlightning.algorithm.verl.VERL], which enqueues tasks to a centralized task queue managed by the [`LightningStore`][agentlightning.LightningStore], accessible to all runners.

When [`Trainer.fit`][agentlightning.Trainer.fit] is called, it launches 10 concurrent runners (as specified by `n_runners=10`). Each runner pulls tasks from the centralized task queue, executes the agent's [`rollout`][agentlightning.LitAgent.rollout] method, collects traces, and returns rewards to VERL for training.

The [`Adapter`][agentlightning.Adapter], as discussed earlier, is used at the algorithm side, and receives the traces emitted by the agent and runners. The `agent_match` parameter ensures [`VERL`][agentlightning.algorithm.verl.VERL] only ingests spans from the specific agent you want to optimize.

## Dry-Run the Pipeline with [`Trainer.dev`][agentlightning.Trainer.dev]

Before committing hours of GPU time, you can **dry-run** the agent with [`Trainer.dev()`][agentlightning.Trainer.dev]. This method swaps in the lightweight [`Baseline`][agentlightning.Baseline] algorithm, enqueues up to ten tasks, and prints every span emitted by the agent. Because it uses the same runner stack as full training, it's ideal for verifying ADK agent orchestration, Cloud Trace integration, and agent logic.

To begin, the agent needs a valid OpenAI-compatible endpoint since VERL is not active in this mode. You can use OpenAI's official API or your own local LLM endpoint. Wrap it as follows:

```python
trainer = agl.Trainer(
    n_runners=1,
    initial_resources={
        "main_llm": agl.LLM(
            endpoint=os.environ["OPENAI_API_BASE"],
            model="gpt-4.1-nano",
            sampling_parameters={"temperature": 0.7},
        )
    },
)
```

Then, call [`trainer.dev(...)`][agentlightning.Trainer.dev] with a small number of tasks:

```python
dev_data = pd.read_parquet("data/test.parquet").to_dict("records")[:10]
trainer.dev(agent, dev_dataset=dev_data)
```

Run this in a Python session or adapt your script to include a `--dev` flag. Once the spans appear healthy and the rewards are non-zero, switch back to [`trainer.fit(...)`][agentlightning.Trainer.fit] for full RL training. See the [debugging tutorial](../tutorials/debug.md) for more tips on how to debug the agent.

## Running the Sample Code

The following tutorial explains how to run the complete example in [`examples/google_adk`]({{ src("examples/google_adk") }}).

### Dataset

The trainer expects Parquet files inside `examples/google_adk/data`:
`train.parquet` and `test.parquet`.

#### Dataset Schema

Each record in the dataset should contain:

- `question` (str): The user instruction/task.
- `app_id` (str): Application/environment identifier.
- `ground_truth` (str): The expected action/output for this task.
- `meta` (object, optional): Arbitrary metadata dictionary.

#### Preparing the Dataset

Use the dataset preparation script to convert your data to Parquet format:

```bash
cd examples/google_adk

# Option A: Convert from your source files (JSONL/JSON/CSV/Parquet)
uv run python prepare_dataset.py \
  --train path/to/train.jsonl \
  --test path/to/test.jsonl \
  --outdir .

# Option B: Generate a toy dataset for local smoke tests
uv run python prepare_dataset.py --generate-toy --outdir .
```

This creates:
- `data/train.parquet` - Training dataset
- `data/test.parquet` - Validation dataset

#### Optional: Host on Google Drive for CI

For CI workflows that need to download the dataset automatically:

1. Compress your `data` directory into a zip file and upload it to Google Drive.
2. Make the file publicly accessible and get the shareable link.
3. Use `gdown` in your CI workflow to download and extract the dataset:

```bash
cd examples/google_adk
uv run gdown --fuzzy https://drive.google.com/file/d/YOUR_FILE_ID/view
unzip adk-data.zip -d data
```

### Dependencies

Create a clean virtual environment, activate it, and install Agent-lightning with the VERL extras required by [this tutorial](../tutorials/installation.md). Install Google ADK dependencies:

```bash
pip install "agentlightning[verl,adk]"
```

Or install separately:

```bash
pip install "google-adk>=0.3.0" "fastapi" "uvicorn" "pytest"
```

ADK provides observability features including Cloud Trace integration for span visualization, making it easy to understand how agent operations (LLM calls, tool executions) are organized within rollouts.

For full training profiles, plan to use a GPU with at least **40 GB** of memory.

### Launch Training

From [`examples/google_adk`]({{ src("examples/google_adk") }}), run the training script:

```bash
python train_adk.py --train-file data/train.parquet --val-file data/test.parquet
```

The script supports several CLI options:

- `--train-file`: Path to training Parquet file (default: `data/train.parquet`)
- `--val-file`: Path to validation Parquet file (default: `data/test.parquet`)
- `--model`: Model name for the rollout LLM (default: from `OPENAI_MODEL` env var or `meta-llama/Meta-Llama-3-8B-Instruct`)
- `--endpoint`: OpenAI-compatible base URL (default: from `OPENAI_API_BASE` env var or `http://localhost:8000/v1`)
- `--ci`: Reduce workload for CI (smaller dataset slice, fewer runners)
- `--ci-fast`: Ultra-fast CI mode (tiny dataset slice, 1 runner)
- `--external-store-address`: Use an external LightningStore address for debugging (e.g., `http://localhost:4747`)
- `--wandb-project`: Weights & Biases project name (default: `agent-lightning-adk`)
- `--wandb-run-name`: Weights & Biases run name (optional)

**Example with CI mode:**

```bash
python train_adk.py --ci-fast --wandb-project my-project --wandb-run-name test-run
```

**Example with external store for debugging:**

```bash
python train_adk.py --external-store-address http://localhost:4747
```

The script instantiates `LitAdkAgent` and launches [`trainer.fit`][agentlightning.Trainer.fit].

For training with VERL, export an `HF_TOKEN` before running so VERL can download the model weights.

!!! tip "Troubleshooting"

    If you have got some Ray worker errors on either `WANDB_API_KEY` not set, or `HF_TOKEN` not set, or data not found, please try to restart the Ray cluster with the helper script: [scripts/restart_ray.sh]({{ src("scripts/restart_ray.sh") }}), which essentially stops the ray cluster if any, and starts a new one:

    ```bash
    env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0
    ```

### Debugging the Agent without VERL

[`adk_debug.py`]({{ src("examples/google_adk/adk_debug.py") }}) provides a sanity check script to test the agent without training. This allows you to verify that the agent configuration, ADK orchestration, and dataset are correct before starting a full training run.

To test the agent without training:

```bash
python adk_debug.py
```

You can customize the test with CLI options:

- `--file`: Path to Parquet file (default: `data/test.parquet`)
- `--index`: Row index to test (default: `0`)
- `--model`: Model name (default: from `OPENAI_MODEL` env var)
- `--endpoint`: API endpoint (default: from `OPENAI_API_BASE` env var or `http://localhost:8000/v1`)

**Example:**

```bash
python adk_debug.py --file data/train.parquet --index 0 --endpoint http://localhost:8000/v1 --model meta-llama/Meta-Llama-3-8B-Instruct
```

Set the following environment variables if needed:

```bash
export OPENAI_API_BASE=<your_api_base>
export OPENAI_API_KEY=<your_api_key>
export OPENAI_MODEL=<your_model_name>
```

This allows you to verify that the agent logic and reward computation behave as expected before reinforcement learning is introduced.

