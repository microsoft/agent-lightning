# Coding Agent

| GPU | Model | Controller Mode | Trainer Mode |
|---|---|---|---|
| 4× B200 | `Qwen/Qwen3.5-9B` | K8s | Sync and async |

The Coding Agent example trains a software-engineering agent on SWE-smith tasks with VERL and Agent Lightning >=v1.0. Each rollout runs as a Kubernetes Job inside a repository-specific image, edits an isolated checkout, executes tests, and reports the resulting reward to the AGL Gateway.

This example uses two machines:

- **Machine A — Kubernetes Controller machine:** connects to the Kubernetes cluster, prepares repository images in the node-accessible Docker runtime, and runs `agl-controller` to create rollout Jobs.
- **Machine B — GPU training machine:** provides the GPUs and runs both `agl-server` (the AGL Gateway) and the VERL trainer with its model backend.

Machine B's AGL Gateway address must be reachable from Machine A and from the rollout pods in the Kubernetes cluster.

## Environment Preparation

On **Machine A (Kubernetes Controller machine)**, activate the project environment and install the dependency used to prepare repository images:

```bash
source .venv/bin/activate
uv pip install -r examples/swe_smith/requirements.txt
```

Machine A also requires Docker, `kubectl`, and access to the Kubernetes cluster.

On **Machine B (GPU training machine)**, install the project and GPU training environment described in the project installation guide. The SWE-smith image-preparation requirements above are not needed on Machine B.

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
export AGL_MODEL_NAME=Qwen/Qwen3.5-9B
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
export AGL_MODEL_NAME=Qwen/Qwen3.5-9B
examples/swe_smith/run.sh trainer
```

The launcher passes additional arguments to `train_smith_agent.py`, including VERL dotlist overrides:

```bash
examples/swe_smith/run.sh trainer \
    trainer.total_training_steps=100 \
    actor_rollout_ref.rollout.n=4
```

## Preventing Reward Hacking

A coding agent may obtain the reference fix without solving the task, for example by inspecting Git history, downloading upstream source code with `curl` or `wget`, installing the original package with `pip`, or using Python networking libraries such as `urllib`.

The SWE agent limits these reward-hacking paths in two ways:

- **Repository isolation:** before the agent starts, the harness checks out the task branch and moves `.git` outside the visible testbed. Agent commands that invoke Git, access the hidden Git metadata, install packages, download files, or modify the test harness are blocked.
- **Network isolation:** we strongly recommend adding a Kubernetes network policy that denies all outbound traffic from agent pods except connections to the AGL Gateway. Without this restriction, an agent may retrieve upstream source code or other external information and obtain reward without solving the task as intended.

The final reward is computed by running the task-specific `FAIL_TO_PASS` and `PASS_TO_PASS` tests inside the isolated repository environment. These controls are part of the training setup: weakening them can allow the agent to recover reference code and corrupt the reward signal.
