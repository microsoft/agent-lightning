# ScienceWorld

| GPU | Model | Controller Mode | Trainer Mode |
|---|---|---|---|
| 8× A100 40GB | `Qwen/Qwen2.5-7B-Instruct` | Local | Async only |

ScienceWorld trains an agent with VERL and Agent Lightning >=v1.0 to solve text-based science tasks from AllenAI's [ScienceWorld](https://github.com/allenai/ScienceWorld).

This example uses the local controller in asynchronous trainer mode. Each rollout runs as a local process that interacts with a ScienceWorld environment, calls the model through the AGL Gateway, and reports the final reward. It does not require K8s, Docker, or Minikube.

## Environment Preparation

Install Java and the example dependencies:

```bash
sudo apt-get install -y default-jre
uv pip install scienceworld openai
```

ScienceWorld starts a JVM for each rollout, so Java 1.8 or later is required.

## Training

Start local training from the repository root:

```bash
examples/science_world/run_local.sh
```

`run_local.sh` starts `agl-lite-server`, the local `agl-lite-controller`, and the VERL trainer. The controller launches each rollout as a local process, and the script cleans up the server, controller, and Ray processes when it exits.

The training dataset is generated automatically from ScienceWorld task names and variation indices. To train on selected tasks or change the number of variations per task:

```bash
examples/science_world/run_local.sh \
    --task-names find-non-living-thing,find-living-thing \
    --variations-per-task 50
```

Available runtime settings include:

| Setting | Default | Description |
|---|---|---|
| `--task-names` | `all` | Comma-separated task names, or all ScienceWorld tasks |
| `--variations-per-task` | `50` | Maximum variations per task |
| `--simplification` | `easy` | ScienceWorld simplification preset |
| `SW_MAX_STEPS` | `30` | Maximum model turns per rollout |
| `SW_ENV_STEP_LIMIT` | `100` | ScienceWorld environment step limit |
| `AGL_MAX_TOKENS` | `256` | Maximum tokens per model completion |
