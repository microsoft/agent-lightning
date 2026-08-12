# LLM-in-Sandbox

| GPU | Model | Controller Mode | Trainer Mode |
|---|---|---|---|
| 4× A100 80GB | `Qwen/Qwen3-4B-Instruct-2507` | K8s | Sync only |

LLM-in-Sandbox trains a general instruction-following agent with VERL and Agent Lightning >=v1.0. The agent can manage files, execute code, and use external resources inside an isolated container sandbox.

This example is based on [*Computer Environments Elicit General Agentic Intelligence in LLMs*](https://arxiv.org/abs/2601.16206) by Cheng et al. (2026).

This example uses the K8s controller in synchronous trainer mode. Each rollout runs as a Kubernetes Job, while model calls pass through the AGL Gateway to the VERL-managed vLLM server. The agent dependencies remain isolated from the trainer environment.

## Environment Preparation

Use Python 3.12 and install the project environment before running the example. You also need:

- Docker
- Minikube
- `kubectl`
- Image build support inside Minikube

The bundled Minikube setup is intended for testing only. For production deployments, replace it with a production-grade Kubernetes cluster.

## Data Preparation

The public training and validation data is hosted in the [`daixuancheng/llm-in-sandbox-rl`](https://huggingface.co/datasets/daixuancheng/llm-in-sandbox-rl) dataset on Hugging Face. The upstream [`llm-in-sandbox-rl`](https://github.com/llm-in-sandbox/llm-in-sandbox-rl) repository provides the conversion script used to generate the JSON files expected by this example.

From the repository root, clone the upstream repository and convert all dataset configurations:

```bash
git clone --depth 1 https://github.com/llm-in-sandbox/llm-in-sandbox-rl.git /tmp/llm-in-sandbox-rl
python /tmp/llm-in-sandbox-rl/examples/llm_in_sandbox/convert_llm_sandbox_dataset.py \
    --all \
    --output-dir examples/llm-in-sandbox/data
```

The converter downloads the following Hugging Face configurations:

- Training: `instruct_pretrain` (`train` split, 3,600 samples)
- Validation: `math_mini`, `biomed_mini`, and `long_context_mini` (`test` splits)

The default files used by this example are:

| Split | Path |
|---|---|
| Training | `examples/llm-in-sandbox/data/llm_sandbox_instruct_pretrain/train_verl.json` |
| Validation | `examples/llm-in-sandbox/data/llm_sandbox_math_mini/test_verl.json` |
| Validation | `examples/llm-in-sandbox/data/llm_sandbox_biomed_mini/test_verl.json` |
| Validation | `examples/llm-in-sandbox/data/llm_sandbox_long_context_mini/test_verl.json` |

The command above creates these directories directly; no manual file move is needed. If you generate or download the files separately, place `train_verl.json` and `test_verl.json` in their corresponding directories, or pass those directories to the launcher.

For validation, select any one or more of `math_mini`, `biomed_mini`, and `long_context_mini`. Separate multiple directories with commas:

```bash
examples/llm-in-sandbox/run.sh \
    --train-data-dir /path/to/train-data \
    --val-data-dir /path/to/math-data,/path/to/biomed-data,/path/to/long-context-data
```

## Training

Start training from the repository root:

```bash
examples/llm-in-sandbox/run.sh
```

The launcher:

1. creates a local Minikube cluster;
2. builds the `llm-in-sandbox-agent:dev` image;
3. starts `agl-server` and the K8s `agl-controller`;
4. starts the VERL trainer;
5. cleans up the server, controller, and Ray processes when it exits.

The controller creates one Kubernetes Job for each rollout. Inside the Job, the adapter runs the sandbox agent, routes model calls through the AGL Gateway, evaluates the final answer, and reports the reward.

Additional VERL settings can be passed as dotlist overrides:

```bash
examples/llm-in-sandbox/run.sh \
    trainer.total_epochs=2 \
    actor_rollout_ref.rollout.n=2
```

Use `Ctrl+C` to stop training and clean up the processes started by the launcher.
