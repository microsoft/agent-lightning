# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Dict, cast

from calc_agent import MathProblem, calc_agent
from datasets import Dataset as HuggingFaceDataset

import agentlightning as agl


def verl_default_config() -> Dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 32,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
            },
            "actor": {
                "ppo_mini_batch_size": 32,
                "ppo_micro_batch_size_per_gpu": 4,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 8,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": "Qwen/Qwen2.5-0.5B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "AgentLightningCI",
            "experiment_name": "train_verl_v0_2",
            "nnodes": 1,
            "save_freq": 3,
            "test_freq": 3,
            "total_epochs": 1,
            "total_training_steps": 3,
        },
    }


def train(*, train_file: str, val_file: str, model: str, llm_proxy: bool, ci: bool, n_runners: int):
    # TODO: use train_file and val_file from arguments
    train_dataset = cast(agl.Dataset[MathProblem], HuggingFaceDataset.from_parquet("data/train.parquet").to_list())
    val_dataset = cast(agl.Dataset[MathProblem], HuggingFaceDataset.from_parquet("data/test_mini.parquet").to_list())

    print("First 5 rows of train dataset:")
    print(train_dataset[:5])  # type: ignore
    print("First 5 rows of val dataset:")
    print(val_dataset[:5])  # type: ignore

    config = verl_default_config()
    # TODO: augment config based on function arguments

    algorithm = agl.VERL(config)

    if llm_proxy:
        # We deliberately used a dummy OtelTracer and handles all tracing via LLM Proxy.
        tracer = agl.OtelTracer()
        adapter = agl.LlmProxyTraceToTriplet()

        trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, tracer=tracer, adapter=adapter)
    else:
        trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners)

    trainer.fit(calc_agent, train_dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    main()
