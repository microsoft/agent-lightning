# SHAPER: Skill-Harness Evolution

SHAPER evolves a planner skill and an executable context harness around a
frozen embodied agent. The implementation includes runnable VLABench and
ESI-Bench adapters.

Paper: [arXiv:2608.11350](https://arxiv.org/abs/2608.11350)

## Install

Install Agent Lightning and the SHAPER runtime extension from the repository
root:

```bash
python -m pip install -e .
python -m pip install -e contrib/agentlightning/contrib/shaper
```

Check out the benchmark revisions used by the adapters:

```bash
export AGL_ROOT="$PWD"
export BENCH_ROOT="$AGL_ROOT/shaper-benchmarks"
bash contrib/recipes/shaper/scripts/checkout_shaper_benchmarks.sh "$BENCH_ROOT"
```

### VLABench

VLABench uses separate simulator and OpenPI actor environments.

```bash
conda create -n shaper-vlabench python=3.10 pip -y
conda activate shaper-vlabench
PYTHON="$CONDA_PREFIX/bin/python" \
  bash "$AGL_ROOT/contrib/recipes/shaper/scripts/bootstrap_shaper_environment.sh" \
  vlabench-simulator "$AGL_ROOT" "$BENCH_ROOT/VLABench" \
  "$BENCH_ROOT/OpenPI" --download-assets
```

Install the actor and checkpoint, then start its websocket service:

```bash
bash "$AGL_ROOT/contrib/recipes/shaper/scripts/bootstrap_shaper_environment.sh" \
  vlabench-actor "$AGL_ROOT" "$BENCH_ROOT/OpenPI" \
  /absolute/path/models/pi0-primitive-10task --download-checkpoint

bash "$AGL_ROOT/contrib/recipes/shaper/scripts/start_shaper_vlabench_actor.sh" \
  "$AGL_ROOT" "$BENCH_ROOT/OpenPI" \
  /absolute/path/models/pi0-primitive-10task 8000 vlabench-base
```

### ESI-Bench

ESI-Bench uses a controller environment and an isolated Python 3.11
OmniGibson worker environment. Use a supported 20/30/40-series NVIDIA GPU.

```bash
python3 -m venv /absolute/path/envs/shaper-esi-controller
PYTHON=/absolute/path/envs/shaper-esi-controller/bin/python \
  bash "$AGL_ROOT/contrib/recipes/shaper/scripts/bootstrap_shaper_environment.sh" \
  esi-controller "$AGL_ROOT"

conda create -n shaper-esi-worker python=3.11 pip -y
conda activate shaper-esi-worker
export OMNIGIBSON_DATA_PATH=/absolute/path/omnigibson-data
export SHAPER_ACCEPT_NVIDIA_EULA=YES
export SHAPER_ACCEPT_BEHAVIOR_DATASET_TOS=YES
PYTHON="$CONDA_PREFIX/bin/python" \
  bash "$AGL_ROOT/contrib/recipes/shaper/scripts/bootstrap_shaper_environment.sh" \
  esi-worker "$AGL_ROOT" "$BENCH_ROOT/ESI-Bench" \
  "$BENCH_ROOT/BEHAVIOR-1K" --install-behavior
```

### Planner Service

Both recipes use an OpenAI-compatible multimodal planner. To run one locally
with vLLM:

```bash
python -m pip install "vllm>=0.19.0"
bash contrib/recipes/shaper/scripts/start_shaper_planner_vllm.sh
curl http://127.0.0.1:8001/v1/models
```

The Qwen3.6-27B defaults follow its official vLLM and thinking-mode settings:
8-way tensor parallelism, a 262,144-token context, the `qwen3` reasoning
parser, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`,
`presence_penalty=0`, and `repetition_penalty=1`. Set `SHAPER_MODEL`,
`SHAPER_VLLM_TP_SIZE`, `SHAPER_VLLM_MAX_MODEL_LEN`, or the planner sampling
variables in the run scripts when needed. Extra arguments are forwarded to
`vllm serve`. To use a hosted API instead, configure the same run scripts
directly:

```bash
export SHAPER_PLANNER_ENDPOINT="https://provider.example.com/v1"
export SHAPER_MODEL="<multimodal-model>"
export OPENAI_API_KEY="<api-key>"
```

## Run

Runtime configuration is collected at the top of two executable scripts:

- `contrib/recipes/shaper/scripts/run_vlabench.sh`
- `contrib/recipes/shaper/scripts/run_esi_bench.sh`

Edit their `Configuration` blocks when your paths or endpoints differ. The
defaults assume benchmark checkouts under `./shaper-benchmarks`, a planner at
`http://127.0.0.1:8001/v1`, and the OpenPI actor at `127.0.0.1:8000`.
`run_esi_bench.sh` also contains the worker-Python and OmniGibson data paths.
Shell environment variables override every value in the scripts. API keys are
never stored in them; export `OPENAI_API_KEY` when the planner requires one.

Check the configured environment before starting a run:

```bash
bash contrib/recipes/shaper/scripts/run_vlabench.sh check
bash contrib/recipes/shaper/scripts/run_esi_bench.sh check
```

Train:

```bash
bash contrib/recipes/shaper/scripts/run_vlabench.sh train
bash contrib/recipes/shaper/scripts/run_esi_bench.sh train
```

Each training run writes `shaper_run.json`, `best_skill.txt`, and
`best_harness.py` under `outputs/shaper/<benchmark>`.

The main training options are:

| Option | Default | Meaning |
|---|---:|---|
| `--n-runners` | `1` | Process-isolated simulator workers |
| `--validation-size` | full split | Fixed validation subset size |
| `--gradient-batch-size` | `4` | Rollouts summarized per optimizer update |
| `--beam-width` | `3` | Candidates retained after validation |
| `--branch-factor` | `2` | Proposals generated per parent |
| `--skill-rounds` | `2` | Skill evolution rounds |
| `--harness-rounds` | `2` | Harness evolution rounds |
| `--rollout-batch-timeout` | `3600` | Seconds allowed per concurrent rollout wave |
| `--role-max-completion-tokens` | planner setting | Judger, summarizer, and optimizer output limit |

Evaluate the resulting artifact pair on the configured validation split:

```bash
bash contrib/recipes/shaper/scripts/run_vlabench.sh eval
bash contrib/recipes/shaper/scripts/run_esi_bench.sh eval
```

Additional CLI options are forwarded to the underlying command. For example:

```bash
SHAPER_N_RUNNERS=4 bash contrib/recipes/shaper/scripts/run_vlabench.sh train \
  --beam-width 2 --skill-rounds 3

bash contrib/recipes/shaper/scripts/run_esi_bench.sh eval \
  --start-index 0 --limit 20
```

To evaluate the bundled ESI-Bench reporting subset:

```bash
ESI_VALIDATION_SPLIT="$PWD/contrib/recipes/shaper/esi_bench/splits/reported_eval231.txt" \
  bash contrib/recipes/shaper/scripts/run_esi_bench.sh eval
```

Evaluation writes aggregate reward and one record per episode to
`evaluation.json`.
