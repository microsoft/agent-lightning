from __future__ import annotations

import argparse
import importlib.resources
import json
import os
from pathlib import Path
from pprint import pprint
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DATA_SOURCE = "swe_smith"
HF_DATASET = "SWE-bench/SWE-smith"
TRAIN_BACKEND = "fsdp"

INSTANCE_FIELDS = (
    "instance_id",
    "problem_statement",
    "image_name",
    "repo",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)

def log(message: str) -> None:
    print(message, flush=True)

def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)

def _passes_curation(
    instance: dict[str, Any], min_f2p: int, max_f2p: int, require_pr: bool
) -> bool:
    f2p = instance.get("FAIL_TO_PASS") or []
    if not (min_f2p <= len(f2p) <= max_f2p):
        return False
    if require_pr and ".pr_" not in instance.get("instance_id", ""):
        return False
    return True

def _project(instance: dict[str, Any]) -> dict[str, Any]:
    row = {field: instance.get(field) for field in INSTANCE_FIELDS}
    row["data_source"] = DATA_SOURCE
    row["data_id"] = str(instance.get("instance_id", ""))
    return row

def load_instances(
    *,
    dataset_path: str | None,
    max_instances: int,
    min_f2p: int,
    max_f2p: int,
    max_repos: int | None,
) -> list[dict[str, Any]]:
    if dataset_path:
        rows: Any = [json.loads(line) for line in Path(dataset_path).open() if line.strip()]
        require_pr = False
    else:
        from datasets import load_dataset

        rows = load_dataset(HF_DATASET, split="train", streaming=True)
        require_pr = True

    selected: list[dict[str, Any]] = []
    repos: set[str] = set()
    for instance in rows:
        if not _passes_curation(instance, min_f2p, max_f2p, require_pr):
            continue
        repo = instance.get("repo", "")
        if max_repos is not None and repo not in repos and len(repos) >= max_repos:
            continue
        repos.add(repo)
        selected.append(_project(instance))
        if len(selected) >= max_instances:
            break

    if not selected:
        raise ValueError(
            f"No SWE-smith instances matched ({min_f2p} <= len(FAIL_TO_PASS) <= {max_f2p})"
        )
    return selected

def split_dataset(
    instances: list[dict[str, Any]],
    val_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_val = max(1, int(len(instances) * val_fraction))
    n_train = len(instances) - n_val
    if n_train <= 0:
        raise ValueError(
            f"{len(instances)} instances after val_fraction={val_fraction} leaves no train rows"
        )
    return instances[:n_train], instances[n_train:]

def load_split_file(
    path: str,
    *,
    max_instances: int | None = None,
) -> list[dict[str, Any]]:
    """Load a pre-split, pre-curated JSONL dataset and project to the verl schema.

    Unlike load_instances(), this applies no FAIL_TO_PASS curation and performs no
    train/val split -- the file is consumed as-is (optionally capped to max_instances).
    """
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    selected = [_project(row) for row in rows]
    if max_instances:
        selected = selected[:max_instances]
    if not selected:
        raise ValueError(f"No instances loaded from {path}")
    return selected

def verl_default_config() -> dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            # bypass_mode: reuse the rollout-time log-probs as old_log_prob for
            # importance sampling instead of recomputing them (saves the old-logprob
            # forward pass). Requires rollout_log_probs in the batch, which the proxy
            # emits in train mode. Community-standard for async RL.
            "rollout_correction": {
                "bypass_mode": True,
                "loss_type": "ppo_clip",
                "rollout_is": None,
                "rollout_rs": None,
                "rollout_rs_threshold": None,
            },
        },
        "data": {
            "train_batch_size": 16,
            "max_prompt_length": 65536,
            "max_response_length": 65536,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "mode": "async",
                "tensor_model_parallel_size": 1,
                "n": 8,
                "log_prob_micro_batch_size_per_gpu": 1,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.8,
                "max_model_len": 81920,
                "max_num_batched_tokens": 8192,
                "enforce_eager": False,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "chat_template": str(EXAMPLE_DIR / "swe_smith_chat_template.jinja"),
                        # vLLM 0.20 FlashInfer MoE uses a blocked runtime layout
                        # that is not compatible with bucketed IPC weight refit.
                        "moe_backend": "triton",
                    }
                },
                "temperature": 1,
                "val_kwargs": {"temperature": 0.7, "do_sample": True},
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,

                "checkpoint_engine": {"update_weights_bucket_megabytes": 4096},
            },
            "actor": {
                "ppo_mini_batch_size": 16,

                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 16384,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,

                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                    "entropy_from_logits_with_chunking": True,
                },
                "loss_agg_mode": "token-mean",
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": DEFAULT_MODEL,
                "use_remove_padding": True,
                "use_fused_kernels": True,
                "fused_kernel_options": {"impl_backend": "torch"},
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 4,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agl-lite",
            "experiment_name": "swe_smith",
            "nnodes": 1,

            "nccl_timeout": 1800,
            "test_freq": 16,
            "save_freq": 16,
            "total_epochs": 4,
            "total_training_steps": 1000,
        },
        "agentlightning": {
            "agl_base_url": "http://localhost:8080",
            "agl_key": "",
            "is_shuffle": False,

            "rollout_timeout_seconds": 5400,
            "reward_fillna_value": 0.0,
            "max_ppo_update_times": 2,
            "cleanup_namespace": os.environ.get("AGL_NAMESPACE", "default"),
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 65536,
                "trajectory_max_response_length": 65536,
            },
            "async_rollout": {

                "enabled": False,
                "async_train_batch_size": 50,
            },
            "k8s": {
                "job_template_path": str(EXAMPLE_DIR / "job-template-openai.yaml"),
            },
        },
    }

def build_config(
    *,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
    ci: bool = False,
) -> DictConfig:
    verl_pkg = importlib.resources.files("agl_lite.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    overrides = verl_default_config()

    if model:
        overrides["actor_rollout_ref"]["model"]["path"] = model
    if agl_base_url:
        overrides["agentlightning"]["agl_base_url"] = agl_base_url
    if agl_key is not None:
        overrides["agentlightning"]["agl_key"] = agl_key

    rollout_mode = overrides["actor_rollout_ref"]["rollout"]["mode"]
    model_path = overrides["actor_rollout_ref"]["model"]["path"]
    overrides["trainer"]["experiment_name"] = (
        f"swe_smith_{rollout_mode}_{model_path.split('/')[-1]}_{TRAIN_BACKEND}"
    )
    if run_name:
        overrides["trainer"]["experiment_name"] = f'{overrides["trainer"]["experiment_name"]}_{run_name}'

    if ci:
        overrides["trainer"]["project_name"] = "agl-lite-CI"
        overrides["trainer"]["total_epochs"] = 1
        overrides["trainer"]["total_training_steps"] = 5
        overrides["trainer"]["test_freq"] = -1
        overrides["trainer"].pop("save_freq", None)
        overrides["trainer"]["n_gpus_per_node"] = 1

        if not model:
            overrides["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen3-8B"
        ci_model = overrides["actor_rollout_ref"]["model"]["path"]
        overrides["trainer"]["experiment_name"] = (
            f"swe_smith_{rollout_mode}_{ci_model.split('/')[-1]}_{TRAIN_BACKEND}_ci"
        )
        if run_name:
            overrides["trainer"]["experiment_name"] = f'{overrides["trainer"]["experiment_name"]}_{run_name}'
        overrides["trainer"]["logger"] = ["console"]
        overrides["data"]["train_batch_size"] = 1
        overrides["data"]["max_prompt_length"] = 8192
        overrides["data"]["max_response_length"] = 8192
        overrides["actor_rollout_ref"]["rollout"]["n"] = 2
        overrides["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
        overrides["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 1
        overrides["actor_rollout_ref"]["rollout"]["max_model_len"] = 8192
        overrides["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 1
        overrides["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1

        overrides["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] = False
        overrides["actor_rollout_ref"]["actor"]["fsdp_config"] = {
            "param_offload": True,
            "optimizer_offload": True,
            "entropy_from_logits_with_chunking": True,
        }
        overrides["actor_rollout_ref"]["rollout"]["log_prob_use_dynamic_bsz"] = False
        overrides["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 1
        overrides["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 1
        overrides["agentlightning"]["rollout_timeout_seconds"] = 1800
        overrides["agentlightning"]["trace_aggregator"]["trajectory_max_prompt_length"] = 4096
        overrides["agentlightning"]["trace_aggregator"]["trajectory_max_response_length"] = 4096

        overrides["agentlightning"]["async_rollout"]["async_train_batch_size"] = 2

    override_conf = OmegaConf.create(overrides)
    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, override_conf, cli_override_conf)
    OmegaConf.set_struct(config, False)
    return config

def dump_subset(
    *,
    out_path: str,
    dataset_path: str | None,
    max_instances: int,
    min_f2p: int,
    max_f2p: int,
    max_repos: int | None,
) -> None:
    instances = load_instances(
        dataset_path=dataset_path,
        max_instances=max_instances,
        min_f2p=min_f2p,
        max_f2p=max_f2p,
        max_repos=max_repos,
    )

    out = Path(out_path)
    if not out.is_absolute():
        out = EXAMPLE_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in instances:
            fh.write(json.dumps(row) + "\n")
    distinct_images = sorted({row["image_name"] for row in instances})
    log(f"Wrote {len(instances)} instances to {out}")
    log(f"  distinct images (to prepare): {len(distinct_images)}")

def train(
    *,
    dataset_path: str | None,
    train_dataset_path: str | None = None,
    val_dataset_path: str | None = None,
    max_val_instances: int | None = None,
    max_instances: int,
    min_f2p: int,
    max_f2p: int,
    max_repos: int | None,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
    ci: bool = False,
) -> None:
    from agl_lite.verl.entrypoint import run_ppo

    if not agl_key:
        raise RuntimeError("AGL_KEY is required")

    use_explicit = bool(
        train_dataset_path
        and val_dataset_path
        and Path(train_dataset_path).is_file()
        and Path(val_dataset_path).is_file()
    )
    if use_explicit:
        train_cap = max_instances if ci else None
        val_cap = max(2, max_instances // 8) if ci else max_val_instances
        train_dataset = load_split_file(train_dataset_path, max_instances=train_cap)
        val_dataset = load_split_file(val_dataset_path, max_instances=val_cap)
        instances = train_dataset + val_dataset
    else:
        instances = load_instances(
            dataset_path=dataset_path,
            max_instances=max_instances,
            min_f2p=min_f2p,
            max_f2p=max_f2p,
            max_repos=max_repos,
        )
        train_dataset, val_dataset = split_dataset(instances)
    distinct_repos = sorted({row["repo"] for row in instances})

    log("=== Preflight ===")
    log(f"  agl-lite:     {agl_base_url or 'http://localhost:8080'}")
    log(f"  model:        {model or DEFAULT_MODEL}")
    if use_explicit:
        log(f"  data mode:    explicit pre-split files (no curation/split)")
        log(f"  train file:   {train_dataset_path}")
        log(f"  val file:     {val_dataset_path}")
    else:
        log(f"  data mode:    single file + {{train,val}} split")
    log(f"  instances:    {len(instances)}  (train {len(train_dataset)} / val {len(val_dataset)})")
    log(f"  distinct repos (images to prepare): {len(distinct_repos)}")

    config = build_config(
        model=model,
        agl_base_url=agl_base_url,
        agl_key=agl_key,
        run_name=run_name,
        config_overrides=config_overrides,
        ci=ci,
    )
    log("\n=== VERL config ===")
    pprint(OmegaConf.to_container(config, resolve=True))

    log("\n=== Start VERL training ===")
    run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train a SWE-smith agent with VERL/GRPO via agl-lite")
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("AGL_DATASET_PATH", str(EXAMPLE_DIR / "subset0.jsonl")),
        help="Local JSONL subset (default: subset0.jsonl, generated via --dump-subset). "
        "Use '' to stream from HF. Ignored when --train-dataset-path/--val-dataset-path exist.",
    )

    parser.add_argument(
        "--train-dataset-path",
        default=os.environ.get("AGL_TRAIN_DATASET_PATH", str(EXAMPLE_DIR / "train_dataset.jsonl")),
        help="Pre-split training JSONL, used as-is (no FAIL_TO_PASS curation, no train/val split). "
        "Takes precedence over --dataset-path when this and --val-dataset-path both exist. "
        "Set '' to fall back to the single-file split path.",
    )
    parser.add_argument(
        "--val-dataset-path",
        default=os.environ.get("AGL_VAL_DATASET_PATH", str(EXAMPLE_DIR / "val_dataset.jsonl")),
        help="Pre-split validation JSONL, used as-is. Pairs with --train-dataset-path.",
    )
    parser.add_argument(
        "--max-val-instances",
        type=int,
        default=env_int("AGL_MAX_VAL_INSTANCES", 0) or None,
        help="Optional cap on validation instances (default: all). Each validation eval "
        "runs ALL val instances at the test_freq cadence, so capping bounds eval time.",
    )

    parser.add_argument("--max-instances", type=int, default=env_int("AGL_MAX_INSTANCES", 735))
    parser.add_argument("--min-f2p", type=int, default=2)
    parser.add_argument("--max-f2p", type=int, default=5)
    parser.add_argument("--max-repos", type=int, default=env_int("AGL_MAX_REPOS", 0) or None)
    parser.add_argument("--model", default=os.environ.get("AGL_MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--agl-base-url", default=os.environ.get("AGL_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--agl-key", default=os.environ.get("AGL_KEY", ""))
    parser.add_argument("--run-name", default=os.environ.get("AGL_VERL_EXPERIMENT_NAME", None))
    parser.add_argument("--ci", action="store_true", help="Run a small smoke training loop")
    parser.add_argument(
        "--dump-subset",
        nargs="?",
        const="subset0.jsonl",
        default=None,
        help="Curate the subset and write it to this JSONL, then exit (no training). "
        "Relative paths resolve under the example dir; bare --dump-subset writes "
        "subset0.jsonl there. pull_images.py reads it to prepare exactly the "
        "training images.",
    )
    args, config_overrides = parser.parse_known_args()
    return args, config_overrides

def main() -> None:
    args, config_overrides = parse_args()
    if args.dump_subset:
        dump_subset(
            out_path=args.dump_subset,
            dataset_path=args.dataset_path or None,
            max_instances=args.max_instances,
            min_f2p=args.min_f2p,
            max_f2p=args.max_f2p,
            max_repos=args.max_repos,
        )
        return
    train(
        dataset_path=args.dataset_path or None,
        train_dataset_path=args.train_dataset_path or None,
        val_dataset_path=args.val_dataset_path or None,
        max_val_instances=args.max_val_instances,
        max_instances=args.max_instances,
        min_f2p=args.min_f2p,
        max_f2p=args.max_f2p,
        max_repos=args.max_repos,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        config_overrides=config_overrides,
        ci=args.ci,
    )

if __name__ == "__main__":
    main()
