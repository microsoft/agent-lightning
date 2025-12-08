# Copyright (c) Microsoft. All rights reserved.

"""Train a Youtu-agent on the ASearcher dataset using Agent-lightning.

This module provides a training script for local-wiki-search agents using different model configurations.
The script supports three different training configurations:

1. 'fast' - A lightweight configuration optimized for CI testing with reduced epochs
2. 'qwen' - Standard configuration using Qwen-2.5-Coder-1.5B-Instruct model
3. 'llama' - Configuration using LLaMA-3.2-1B-Instruct model with JSON formatting

Usage:
    python train_sql_agent.py fast    # Fast training for CI/testing
    python train_sql_agent.py qwen    # Standard Qwen model training
    python train_sql_agent.py llama   # LLaMA model training

The script uses reinforcement learning with VERL framework
to train agents on the ASearacher dataset for information searching tasks with local wiki retrieval service.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from search_agent import SearchYoutuAgent, SearchAgent
import agentlightning as agl
from agentlightning.verl.dataset import AgentDataset, LoadedDataset


assert "DATA_ROOT_PATH" in os.environ, "Environment variable DATA_ROOT_PATH must be set"
assert "MODEL_ROOT_PATH" in os.environ, "Environment variable MODEL_ROOT_PATH must be set"
assert "SAVE_ROOT_PATH" in os.environ, "Environment variable SAVE_ROOT_PATH must be set"

DATA_ROOT_PATH = os.getenv("DATA_ROOT_PATH")
MODEL_ROOT_PATH = os.getenv("MODEL_ROOT_PATH")
SAVE_ROOT_PATH = os.getenv("SAVE_ROOT_PATH")

train_asearcher_base=f"{DATA_ROOT_PATH}/ASearcher-train-data/base/ASearcher-Base-35k_train.parquet"
train_asearcher_lrm=f"{DATA_ROOT_PATH}/ASearcher-train-data/lrm/ASearcher-LRM-35k_train.parquet"

## single hop qa
##  Natural Questions [15], TriviaQA [12] and PopQA
test_nq=f"{DATA_ROOT_PATH}/ASearcher-test-data/NQ_rand1000_test.parquet"
test_triviaqa=f"{DATA_ROOT_PATH}/ASearcher-test-data/TriviaQA_rand1000_test.parquet"
test_popqa=f"{DATA_ROOT_PATH}/ASearcher-test-data/PopQA_rand1000_test.parquet"

## multihop qa
# HotpotQA [44], 2WikiMultiHopQA [10], MuSiQue [36], and Bamboogle
test_hotpotqa=f"{DATA_ROOT_PATH}/ASearcher-test-data/HotpotQA_rand1000_test.parquet"
test_2wikimultihopqa=f"{DATA_ROOT_PATH}/ASearcher-test-data/2WikiMultihopQA_rand1000_test.parquet"
test_musique=f"{DATA_ROOT_PATH}/ASearcher-test-data/Musique_rand1000_test.parquet"
test_bamboogle=f"{DATA_ROOT_PATH}/ASearcher-test-data/Bamboogle_test.parquet"

## deep search
test_frames=f"{DATA_ROOT_PATH}/ASearcher-test-data/frames_test.parquet"
test_gaia=f"{DATA_ROOT_PATH}/ASearcher-test-data/GAIA_test.parquet"
test_xbench=f"{DATA_ROOT_PATH}/ASearcher-test-data/xbench-deepsearch_test.parquet"


TRAIN_FILES=[train_asearcher_base]
TEST_FILES=[test_nq, test_triviaqa, test_popqa, test_hotpotqa, test_2wikimultihopqa, test_musique, test_bamboogle]


RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": TRAIN_FILES,
        "val_files": TEST_FILES,
        "train_batch_size": 32,
        "val_batch_size": 32,
        "max_prompt_length": 16384,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 2,
            "n": 16,
            "log_prob_micro_batch_size_per_gpu": 1,
            "multi_turn": {"format": "hermes", "max_user_turns": 5, "max_assistant_turns": 5},
            "temperature": 1.0,
            "top_p": 1.0,
            "name": "vllm",
            "gpu_memory_utilization": 0.7,
            "val_kwargs":{
                "top_p": 1.0,
                "temperature":0.0,
                "n":1,
            }
        },
        "actor": {
            "ppo_mini_batch_size": 128,
            "ppo_micro_batch_size_per_gpu": 1,
            "optim": {"lr": 1e-6},
            "use_kl_loss": True, # False,
            "kl_loss_coef": 0.001, # 0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "ulysses_sequence_parallel_size": 2,
            "use_dynamic_bsz": False,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 1,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": f"{MODEL_ROOT_PATH}/Qwen2.5-1.5B-Instruct_Qwen",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "nnodes": 1,
        "n_gpus_per_node": 2,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "asearcher-n16-v0.2",
        "save_freq": 5,
        "test_freq": 10,
        "total_epochs": 20,
        "balance_batch": False,
    },
}



def config_train_qwen_3b() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5."""
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = f"{MODEL_ROOT_PATH}/Qwen2.5-3B-Instruct_Qwen"
    config["data"]["max_prompt_length"] = 20000
    config["data"]["train_batch_size"] = 128
    config["data"]["val_batch_size"] = 128
    config["trainer"]["val_before_train"] = True
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 128
    config["trainer"]["experiment_name"] = config["trainer"]["experiment_name"] + "_3b_32K_onpolicy-1110"
    config["data"]["val_files"] = [test_nq, test_triviaqa, test_popqa, test_hotpotqa, test_2wikimultihopqa, test_musique, test_bamboogle]
    config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 2
    config["actor_rollout_ref"]["actor"]["ulysses_sequence_parallel_size"] = 2
    config["trainer"]["val_before_train"] = False

    # #### debug only
    # config["data"]["train_batch_size"] = 8
    # config["data"]["val_batch_size"] = 4
    # config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 4

    config["trainer"]["save_freq"] = 5
    config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = 32000
    config["actor_rollout_ref"]["ref"]["log_prob_max_token_len_per_gpu"] = 32000
    return config



def config_train_qwen_3b_debug() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5."""
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = f"{MODEL_ROOT_PATH}/Qwen2.5-3B-Instruct_Qwen"
    config["data"]["max_prompt_length"] = 20000
    # #### debug only
    config["data"]["train_batch_size"] = 4
    config["data"]["val_batch_size"] = 4
    config["actor_rollout_ref"]["rollout"]["n"] = 1
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["max_assistant_turns"] = 2
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["max_user_turns"] = 2
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 4
    config["trainer"]["experiment_name"] = config["trainer"]["experiment_name"] + "_3b_32K_onpolicy-debug"
    config["data"]["val_files"] = [test_nq, test_triviaqa, test_popqa, test_hotpotqa, test_2wikimultihopqa, test_musique, test_bamboogle]
    config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 2
    config["actor_rollout_ref"]["actor"]["ulysses_sequence_parallel_size"] = 1
    config["trainer"]["save_freq"] = 5
    config["trainer"]["val_before_train"] = False
    config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = 32000
    config["actor_rollout_ref"]["ref"]["log_prob_max_token_len_per_gpu"] = 32000
    return config




def config_train_qwen_7b() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5."""
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = f"{MODEL_ROOT_PATH}/Qwen2.5-7B-Instruct_Qwen"
    config["trainer"]["experiment_name"] = config["trainer"]["experiment_name"] + "_7b_1127"
    config["data"]["val_files"] = [test_nq, test_triviaqa, test_popqa, test_hotpotqa, test_2wikimultihopqa, test_musique, test_bamboogle]
    config["actor_rollout_ref"]["actor"]["ulysses_sequence_parallel_size"] = 2
    config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 4
    config["data"]["max_prompt_length"] = 30000
    config["data"]["train_batch_size"] = 128
    config["data"]["val_batch_size"] = 128
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 128
    config["trainer"]["nnodes"] = 16
    config["trainer"]["n_gpus_per_node"] = 8
    config["trainer"]["val_before_train"] = True
    config["trainer"]["val_before_train"] = False
    config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = 32000
    config["actor_rollout_ref"]["ref"]["log_prob_max_token_len_per_gpu"] = 32000
    return config




def config_train_qwen_32b() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5."""
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["actor"]["ulysses_sequence_parallel_size"] = 4
    config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 8
    config["actor_rollout_ref"]["model"]["path"] = f"{MODEL_ROOT_PATH}/Qwen2.5-32B-Instruct_Qwen"
    config["trainer"]["experiment_name"] = config["trainer"]["experiment_name"] + "_32b_onpolicy_32K-1110"
    config["data"]["max_prompt_length"] = 30000
    config["data"]["train_batch_size"] = 128
    config["data"]["val_batch_size"] = 128
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 128
    config["data"]["val_files"] = [test_nq, test_triviaqa, test_popqa, test_hotpotqa, test_2wikimultihopqa, test_musique, test_bamboogle]
    config["trainer"]["nnodes"] = 16
    config["trainer"]["n_gpus_per_node"] = 8
    config["trainer"]["val_before_train"] = True
    config["trainer"]["save_freq"] = 1
    config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = 32000
    config["actor_rollout_ref"]["ref"]["log_prob_max_token_len_per_gpu"] = 32000
    return config



def train(config: Dict[str, Any], llm_proxy: bool, active_agent: Optional[str], n_runners: int,\
    external_store_address: Optional[str], youtu: bool) -> None:
    """Train the search-wiki Youtu-agent with the given configuration."""
    if youtu:
        print("Use Youtu Agent")
        config["trainer"]["experiment_name"] += "_youtu"
        ## 采用LitAgent agent实现方式
        agent = SearchYoutuAgent()
    else:
        print("Use Autogen Agent")
        config["trainer"]["experiment_name"] += "_autogen"
        ## 采用autogen实现方式
        agent = SearchAgent()

    root_dir = f"{SAVE_ROOT_PATH}/checkpoints"
    default_local_dir = os.path.join(root_dir, config["trainer"]["project_name"], config["trainer"]["experiment_name"])
    os.makedirs(default_local_dir, exist_ok=True)
    config["trainer"]["default_local_dir"] = default_local_dir
    rollout_data_dir = os.path.join(default_local_dir, "rollout")
    validation_data_dir = os.path.join(default_local_dir, "validation")
    os.makedirs(rollout_data_dir, exist_ok=True)
    os.makedirs(validation_data_dir, exist_ok=True)
    config["trainer"]["rollout_data_dir"] = rollout_data_dir
    config["trainer"]["validation_data_dir"] = validation_data_dir

    algorithm = agl.VERL(config)
    train_dataset = []
    val_dataset = []
    if external_store_address:
        store: Optional[agl.LightningStore] = agl.LightningStoreClient(external_store_address)
    else:
        store = None
    # 35583
    for train_file in config["data"]["train_files"]:
        train_dataset += pd.read_parquet(train_file).to_dict(orient="records")  # type: ignore
    # 6125
    for val_file in config["data"]["val_files"]:
        val_dataset += pd.read_parquet(val_file).to_dict(orient="records")  # type: ignore
    print("Training set size:", len(train_dataset))
    print("Testing set size:", len(val_dataset))
    if llm_proxy:
        tracer = agl.OtelTracer()  # dummy tracer for LLM Proxy
        adapter = agl.LlmProxyTraceToTriplet()
        trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=store, tracer=tracer, adapter=adapter)
    else:
        trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=store)

    trainer.fit(agent, train_dataset=train_dataset, val_dataset=val_dataset)  # type: ignore
    return



def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train a Search R1 agent on the Asearcher dataset using different model configurations"
    )

    parser.add_argument(
        "config",
        choices=["qwen3b", "qwen3bdebug", "qwen7b", "qwen32b"],
        help="Training configuration:'qwen' (Qwen-2.5)",
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )
    parser.add_argument("--llm-proxy", action="store_true", help="Enable LLM Proxy tracing/adapter")
    parser.add_argument(
        "--external-store-address",
        type=str,
        default="",
        help="Connect to an external store instead of creating a new one in memory",
    )
    parser.add_argument("--n-runners", type=int, default=16, help="Number of runners for Trainer")
    parser.add_argument("--youtu", action="store_true", help="Use youtu agent")
    args = parser.parse_args()
    
    if args.external_store_address:
        print(f"Connecting to external store at: {args.external_store_address}")
        if not os.getenv("AGL_MANAGED_STORE"):
            raise ValueError(
                "When using an external store, please set the environment variable AGL_MANAGED_STORE=0. "
                "Otherwise the trainer will still try to manage the store lifecycle for you!"
            )

    # Get the appropriate configuration
    config_functions = {"qwen3b": config_train_qwen_3b,
        "qwen3bdebug": config_train_qwen_3b_debug, "qwen7b": config_train_qwen_7b,
            "qwen32b": config_train_qwen_32b}
    config = config_functions[args.config]()
    # Set active agent - use provided value or default based on config choice
    active_agent = args.active_agent
    print(f"Starting training with '{args.config}' configuration...")
    print(f"Active agent: {active_agent}")
    train(config, active_agent=args.active_agent,\
        llm_proxy=args.llm_proxy, n_runners=args.n_runners,
            external_store_address=args.external_store_address, youtu=args.youtu)
    return



if __name__ == "__main__":
    main()
