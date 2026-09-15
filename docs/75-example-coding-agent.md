# Coding Agent

| GPU | Model | Controller Mode | Trainer Mode | Code |
|---|---|---|---|---|
| 4× B200 | `Qwen/Qwen3.5-35B-A3B` | K8s | Sync, Megatron + R3 | [Source](https://github.com/microsoft/agent-lightning/tree/main/examples/swe_smith) |

The Coding Agent example trains a software-engineering agent on SWE-smith tasks with `verl` and Agent Lightning >=v1.0. Each rollout runs as a Kubernetes Job inside a repository-specific image, edits an isolated checkout, executes tests, and reports the resulting reward to the AGL Gateway.

This example uses two machines:

- **Machine A — Kubernetes Controller machine:** connects to the Kubernetes cluster, prepares repository images in the node-accessible Docker runtime, and runs `agl-controller` to create rollout Jobs.
- **Machine B — GPU training machine:** provides the GPUs and runs both `agl-server` (the AGL Gateway) and the `verl` trainer with its model backend.

Machine B's AGL Gateway address must be reachable from Machine A and from the rollout pods in the Kubernetes cluster.

## Environment Preparation

On **Machine A (Kubernetes Controller machine)**, activate the project environment and install the dependency used to prepare repository images:

```bash
source .venv/bin/activate
uv pip install -r examples/swe_smith/requirements.txt
```

Machine A also requires Docker, `kubectl`, and access to the Kubernetes cluster.

On **Machine B (GPU training machine)**, install the project and GPU training environment described in the project installation guide. The SWE-smith image-preparation requirements above are not needed on Machine B.

The default trainer uses VERL's Megatron backend and MBridge. Install the
`mcore` extra that matches the supported VERL version, and verify the imports
before starting a run:

```bash
uv pip install --python .venv/bin/python "verl[mcore]==0.8.0"
.venv/bin/python -c "import mbridge, megatron.core"
```

R3 must receive the expert choices made by the rollout engine. Qwen3.5 uses
hybrid attention, for which VERL requires vLLM 0.22.0 or newer. Replace the
generic vLLM 0.20.2 setup pin with a CUDA-compatible vLLM version in that range
before launching this example. See [Router replay and runtime
requirements](#router-replay-and-runtime-requirements) for the required
capabilities and preflight checks.

## Data Preparation

The provided splits are derived from the original SWE-smith dataset, which contains 59,136 executable software-engineering tasks from 128 Python repositories. We build the training data with the following filtering pipeline:

1. Remove tasks with an empty problem statement. The original release contains 18,033 such records.
2. Remove tasks whose corresponding problem branch is missing from the provided repository image. This affects 1,265 records.
3. Remove tasks requiring more than 200 tests, which avoids examples with prohibitively expensive test suites.
4. Run Qwen3.5-9B four times on every remaining candidate as a difficulty probe.
5. Remove tasks solved in all four probe rollouts because they provide little learning signal.
6. Retain tasks with a mixture of successful and failed probe rollouts, yielding approximately 5,000 examples.
7. Add a sample of 1,000 tasks that fail all four probes so the training set is not biased toward easier tasks.

The resulting data contains approximately 6,000 training examples and 400 validation examples. `train_dataset_mixed.jsonl` contains the mixed-difficulty training set, while `val_dataset_filtered.jsonl` contains the filtered validation set.

Download the pre-split dataset archive from [Google Drive](https://drive.google.com/file/d/1q19DP53l4rldvBR2dkUhbaPI_mHVBVL1/view?usp=drive_link) on **both machines**, then extract it into `examples/swe_smith/`:

- **Machine A** reads the datasets to determine which repository images must be prepared.
- **Machine B** reads the datasets to construct the training and validation inputs.

The example reads these files by default:

- `examples/swe_smith/train_dataset_mixed.jsonl`
- `examples/swe_smith/val_dataset_filtered.jsonl`

When using `run.sh`, custom paths can be selected with the `AGL_TRAIN_DATASET_PATH` and `AGL_VAL_DATASET_PATH` environment variables read by the launcher.

## Repository Image Preparation

On Machine A, prepare the repository images in the Docker daemon used by the Kubernetes nodes before starting the Controller:

```bash
python examples/swe_smith/pull_images.py \
    --dataset examples/swe_smith/train_dataset_mixed.jsonl \
    --dataset examples/swe_smith/val_dataset_filtered.jsonl
```

This command installs the OpenAI client into each required SWE-smith base image and creates the `:openai` tags expected by `job-template-openai.yaml`. Run it again if the datasets introduce new repository images.

## Training

The distributed launcher has three roles and must be started in this order:

```text
server → controller → trainer
```

On **Machine B (GPU training machine)**, start the Gateway:

```bash
export AGL_SERVER_PUBLIC_HOST=<address-reachable-from-controller-and-pods>
export AGL_KEY=<shared-secret>
export AGL_MODEL_NAME=Qwen/Qwen3.5-35B-A3B
examples/swe_smith/run.sh server
```

On **Machine A (Kubernetes Controller machine)**, start the Controller:

```bash
export AGL_SERVER_PUBLIC_HOST=<gateway-address>
export AGL_KEY=<same-shared-secret>
export AGL_NAMESPACE=agents
examples/swe_smith/run.sh controller
```

After the Gateway and Controller are ready, start the trainer on **Machine B (GPU training machine)**:

```bash
export AGL_KEY=<same-shared-secret>
export AGL_MODEL_NAME=Qwen/Qwen3.5-35B-A3B
examples/swe_smith/run.sh trainer
```

The launcher passes additional arguments to `train_smith_agent_megatron.py`,
including `verl` dotlist overrides:

```bash
examples/swe_smith/run.sh trainer \
    trainer.total_training_steps=100 \
    actor_rollout_ref.rollout.n=4
```

Use `--ci` for a one-step smoke run. It keeps the 35B model and four-GPU
topology, but caps both datasets at two rows and reduces the batch and context
sizes:

```bash
examples/swe_smith/run.sh trainer --ci
```

## Router Replay and Runtime Requirements

Qwen3.5-35B-A3B is a mixture-of-experts model. The experts selected for a token
during vLLM generation can differ from those selected when Megatron recomputes
that token during the policy update. VERL's R3 router replay records the rollout
expert indices and replays them for both log-probability calculation and the
actor update.

R3 requires both sides of the configuration:

```text
actor_rollout_ref.actor.megatron.router_replay.mode=R3
actor_rollout_ref.rollout.enable_rollout_routing_replay=true
```

The trainer also uses rollout-level advantages and Agent Lightning's
`per_rollout_mean` policy loss. Rollout importance sampling and rejection
sampling are explicitly disabled. Two similarly named asynchronous settings
serve different purposes:

- `actor_rollout_ref.rollout.mode=async` keeps vLLM in the server mode required
  by the tool-calling agent.
- `agentlightning.async_rollout.enabled=false` makes Agent Lightning wait for a
  complete rollout batch before each update.

The checked-in configuration is a conservative starting point for one 4× B200
node: PP=1, TP=1, EP=4, ETP=1 for the Megatron actor and TP=1 for vLLM, with
parameter, optimizer, and gradient offload enabled. Override these values only
as a consistent topology; the product of the relevant parallel dimensions must
fit the available GPU world size.

R3 needs a vLLM build that exposes the routed experts in completion outputs and
accepts `--enable-return-routed-experts`. The support landed in
[vllm-project/vllm#28284](https://github.com/vllm-project/vllm/pull/28284), with
the EPLB fix in
[vllm-project/vllm#33013](https://github.com/vllm-project/vllm/pull/33013). For
Qwen3.5's hybrid-attention layout, use vLLM >=0.22.0 as required by VERL's
[router replay guide](https://github.com/verl-project/verl/tree/main/examples/router_replay).
Verify the runtime before starting a multi-hour run:

```bash
.venv/bin/python -c "from importlib.metadata import version; print(version('vllm'))"
.venv/bin/vllm serve --help 2>&1 | grep -i return-routed-experts
```

## Preventing Reward Hacking

A coding agent may obtain the reference fix without solving the task, for example by inspecting Git history, downloading upstream source code with `curl` or `wget`, installing the original package with `pip`, or using Python networking libraries such as `urllib`.

The SWE agent limits these reward-hacking paths in two ways:

- **Repository isolation:** before the agent starts, the harness checks out the task branch and moves `.git` outside the visible testbed. Agent commands that invoke Git, access the hidden Git metadata, install packages, download files, or modify the test harness are blocked.
- **Network isolation:** we strongly recommend adding a Kubernetes network policy that denies all outbound traffic from agent pods except connections to the AGL Gateway. Without this restriction, an agent may retrieve upstream source code or other external information and obtain reward without solving the task as intended.

The final reward is computed by running the task-specific `FAIL_TO_PASS` and `PASS_TO_PASS` tests inside the isolated repository environment. These controls are part of the training setup: weakening them can allow the agent to recover reference code and corrupt the reward signal.
