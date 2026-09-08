#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$recipe_dir/../../.." && pwd)"

# Configuration. Edit these defaults or export the variables before running.
export SHAPER_BENCH_ROOT="${SHAPER_BENCH_ROOT:-$repo_root/shaper-benchmarks}"
export SHAPER_PLANNER_ENDPOINT="${SHAPER_PLANNER_ENDPOINT:-http://127.0.0.1:8001/v1}"
export SHAPER_MODEL="${SHAPER_MODEL:-Qwen/Qwen3.6-27B}"
export SHAPER_API_KEY_ENV="${SHAPER_API_KEY_ENV:-OPENAI_API_KEY}"
export SHAPER_PLANNER_MAX_TOKENS="${SHAPER_PLANNER_MAX_TOKENS:-32768}"
export SHAPER_OPTIMIZER_MAX_TOKENS="${SHAPER_OPTIMIZER_MAX_TOKENS:-65536}"
export SHAPER_PLANNER_TIMEOUT="${SHAPER_PLANNER_TIMEOUT:-300}"
export SHAPER_PLANNER_RETRIES="${SHAPER_PLANNER_RETRIES:-2}"
export SHAPER_PLANNER_TEMPERATURE="${SHAPER_PLANNER_TEMPERATURE:-1.0}"
export SHAPER_PLANNER_TOP_P="${SHAPER_PLANNER_TOP_P:-0.95}"
export SHAPER_PLANNER_PRESENCE_PENALTY="${SHAPER_PLANNER_PRESENCE_PENALTY:-0.0}"
export SHAPER_PLANNER_EXTRA_BODY="${SHAPER_PLANNER_EXTRA_BODY:-{\"top_k\":20,\"min_p\":0.0,\"repetition_penalty\":1.0}}"

export ESI_BENCH_ROOT="${ESI_BENCH_ROOT:-$SHAPER_BENCH_ROOT/ESI-Bench}"
export ESI_BEHAVIOR_ROOT="${ESI_BEHAVIOR_ROOT:-$SHAPER_BENCH_ROOT/BEHAVIOR-1K}"
export ESI_WORKER_PYTHON="${ESI_WORKER_PYTHON:-$HOME/miniconda3/envs/shaper-esi-worker/bin/python}"
export ESI_OMNIGIBSON_DATA_ROOT="${ESI_OMNIGIBSON_DATA_ROOT:-$HOME/omnigibson-data}"
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$ESI_OMNIGIBSON_DATA_ROOT}"
export ESI_QUESTIONS_JSONL="${ESI_QUESTIONS_JSONL:-$ESI_BENCH_ROOT/hf_dataset/data/questions.jsonl}"
export ESI_MAKE_MAPS_PATH="${ESI_MAKE_MAPS_PATH:-$ESI_BEHAVIOR_ROOT/asset_pipeline/b1k_pipeline/usd_conversion/make_maps.py}"
export ESI_TRAIN_SPLIT="${ESI_TRAIN_SPLIT:-$recipe_dir/esi_bench/splits/recipe_train10.txt}"
export ESI_VALIDATION_SPLIT="${ESI_VALIDATION_SPLIT:-$recipe_dir/esi_bench/splits/recipe_validation10.txt}"

export ESI_MAX_STEPS="${ESI_MAX_STEPS:-30}"
export ESI_MIN_STEPS="${ESI_MIN_STEPS:-3}"
export ESI_CONFIDENCE_THRESHOLD="${ESI_CONFIDENCE_THRESHOLD:-0.85}"
export ESI_MAX_NEW_TOKENS="${ESI_MAX_NEW_TOKENS:-32768}"
export ESI_TEMPERATURE="${ESI_TEMPERATURE:-$SHAPER_PLANNER_TEMPERATURE}"
export ESI_TOP_P="${ESI_TOP_P:-$SHAPER_PLANNER_TOP_P}"
export ESI_ROBOT="${ESI_ROBOT:-R1}"
export ESI_EPISODE_TIMEOUT="${ESI_EPISODE_TIMEOUT:-1800}"
export ESI_ENVIRONMENT_RETRIES="${ESI_ENVIRONMENT_RETRIES:-1}"

export SHAPER_HARNESS_TIMEOUT="${SHAPER_HARNESS_TIMEOUT:-3}"
export SHAPER_HARNESS_MEMORY_MB="${SHAPER_HARNESS_MEMORY_MB:-768}"
export SHAPER_HARNESS_MAX_OUTPUT_CHARS="${SHAPER_HARNESS_MAX_OUTPUT_CHARS:-24000000}"

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "Set PYTHON to a Python interpreter." >&2
  exit 2
fi
output_dir="${SHAPER_OUTPUT_DIR:-$repo_root/outputs/shaper/esi_bench}"
export ESI_OUTPUT_ROOT="${ESI_OUTPUT_ROOT:-$output_dir/runner}"
n_runners="${SHAPER_N_RUNNERS:-1}"
cd "$repo_root"
command="${1:-}"
if [[ -n "$command" ]]; then
  shift
fi

usage() {
  cat <<EOF
Usage: $0 {check|train|eval} [additional arguments]

Defaults are in the Configuration block at the top of this file.
Set OPENAI_API_KEY (or the variable named by SHAPER_API_KEY_ENV) for a
remote authenticated planner endpoint.
EOF
}

case "$command" in
  check)
    exec "$python_bin" -m contrib.recipes.shaper.esi_bench.check_env "$@"
    ;;
  train)
    exec "$python_bin" -m contrib.recipes.shaper.esi_bench.train \
      --model "$SHAPER_MODEL" \
      --output-dir "$output_dir" \
      --n-runners "$n_runners" \
      "$@"
    ;;
  eval)
    skill_path="${SHAPER_SKILL_PATH:-$output_dir/best_skill.txt}"
    harness_path="${SHAPER_HARNESS_PATH:-$output_dir/best_harness.py}"
    exec "$python_bin" -m contrib.recipes.shaper.esi_bench.evaluate \
      --split validation \
      --skill-path "$skill_path" \
      --harness-path "$harness_path" \
      --n-runners "$n_runners" \
      --output "$output_dir/evaluation.json" \
      "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
