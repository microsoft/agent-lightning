# Copyright (c) Microsoft. All rights reserved.

"""The training helper script for Calc Sandbox Agent with VERL algorithm.

Example usage:

```bash
python train_calc_sandbox_agent.py --train-file data/train.parquet --val-file data/test.parquet --llm-proxy
```

To use an external store, run a store server first:

```bash
agl store --port 9999
```

Then run the training script with the external store address:

```bash
AGL_MANAGED_STORE=0 python train_calc_sandbox_agent.py --external-store-address http://localhost:9999
```

Alternatively, you can also run algorithms and runners separately if needed:

```bash
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=algorithm python train_calc_sandbox_agent.py --external-store-address http://localhost:9999
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=runner python train_calc_sandbox_agent.py --external-store-address http://localhost:9999
```
"""

import argparse
import ast
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from calc_sandbox_agent_youtu import calc_sandbox_agent_youtu

import agentlightning as agl


def parse_override_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type.
    
    Args:
        value_str: String representation of the value
        
    Returns:
        Parsed value (int, float, bool, list, dict, or str)
    """
    # Handle boolean values
    if value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False
    
    # Try to parse as Python literal (list, dict, int, float, etc.)
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If it fails, return as string
        return value_str


def apply_config_overrides(config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """Apply configuration overrides to the base config.
    
    Args:
        config: Base configuration dictionary
        overrides: List of override strings in format "key.subkey=value"
        
    Returns:
        Updated configuration dictionary
    """
    for override in overrides:
        if '=' not in override:
            continue
            
        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to the nested dictionary
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        final_key = keys[-1]
        parsed_value = parse_override_value(value_str)
        current[final_key] = parsed_value
        print(f"Override: {key_path} = {parsed_value}")
    
    return config


def verl_default_config() -> Dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_files": [],  # Will be set by train() function
            "val_files": [],  # Will be set by train() function
            "train_batch_size": 8,
            "max_prompt_length": 2048,
            "max_response_length": 14336,  # 16384 - 2048 = 14336 to stay within max_token_len
            "return_raw_chat": True,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "dataloader_num_workers": 0,  # Required by VERL
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 8,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.9,
                "val_kwargs": {
                    "temperature": 1.0,
                    "top_p": 0.6,
                    "n": 16,
                },
            },
            "actor": {
                "ppo_mini_batch_size": 2,
                "ppo_micro_batch_size_per_gpu": 4,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "clip_ratio_c": 10.0,
                "use_dynamic_bsz": True,
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
                "path": "Qwen/Qwen2.5-1.5B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 2,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "agent_lightning_retool",
            "experiment_name": "calc_sandbox_agent",
            "nnodes": 1,
            "save_freq": 5,
            "test_freq": 6,
            "total_epochs": 1,
            "total_training_steps": 10,
            "log_val_generations": 4,
        },
    }


def train(
    *,
    train_file: str,
    val_file: str,
    model: Optional[str],
    llm_proxy: bool,
    ci: bool,
    n_runners: int,
    external_store_address: str,
    val_temperature: float,
    val_top_p: float,
    val_n: int,
    custom_dataset_path: Optional[str],
    custom_dataset_name: Optional[str],
    config_overrides: Optional[list] = None,
):
    """The training entrypoint function for Calc Sandbox Agent with VERL algorithm.

    Args:
        train_file: The path to the training parquet file.
        val_file: The path to the validation parquet file.
        model: The HF model id or path to override the default model.
        llm_proxy: Whether to enable LLM Proxy tracing/adapter.
        ci: Whether to run a minimal CI-style training loop.
        n_runners: The number of runners for the Trainer.
        external_store_address: Connects to an external store instead of creating a new one in memory.
        val_temperature: Temperature for validation sampling.
        val_top_p: Top-p for validation sampling.
        val_n: Number of responses per prompt for validation.
        custom_dataset_path: Path to custom dataset class module (optional).
        custom_dataset_name: Name of custom dataset class (optional).
        config_overrides: List of config overrides in format "key.subkey=value" (optional).
    """
    # Get default config first so we can configure data paths
    config = verl_default_config()
    
    # Apply config overrides from command line (Hydra-style)
    if config_overrides:
        print("\n=== Applying Configuration Overrides ===")
        config = apply_config_overrides(config, config_overrides)
        print("=========================================\n")
    
    # Configure data file paths - VERL will load them using create_rl_dataset
    config["data"]["train_files"] = [train_file]
    config["data"]["val_files"] = [val_file]
    print(f"Configured train_files: {config['data']['train_files']}")
    print(f"Configured val_files: {config['data']['val_files']}")

    if model:
        config["actor_rollout_ref"]["model"]["path"] = model

    # Update validation kwargs
    config["actor_rollout_ref"]["rollout"]["val_kwargs"]["temperature"] = val_temperature
    config["actor_rollout_ref"]["rollout"]["val_kwargs"]["top_p"] = val_top_p
    config["actor_rollout_ref"]["rollout"]["val_kwargs"]["n"] = val_n
    
    # Configure custom dataset if provided
    if custom_dataset_path and custom_dataset_name:
        config["data"]["custom_cls"] = {
            "path": custom_dataset_path,
            "name": custom_dataset_name,
        }
        print(f"Using custom dataset: {custom_dataset_name} from {custom_dataset_path}")
    else:
        print("Using default VERL dataset class (will be determined by create_rl_dataset)")

    # CI toggle keeps everything else the same but you can tweak the lightweight bits here if desired
    if ci:
        # Config the experiment name and project name so that they are available to CI
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        EXPERIMENT_NAME = f"calc_sandbox_{timestamp}"

        PROJECT_NAME = "AgentLightningCI"

        # Simulate writing to $GITHUB_OUTPUT if it’s set
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"project_name={PROJECT_NAME}\n")
                f.write(f"run_name={EXPERIMENT_NAME}\n")

        print("Set environment variables:")
        print(f"PROJECT_NAME={PROJECT_NAME}")
        print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

        # Keep it tiny/light without adding new knobs
        config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.9
        config["trainer"]["total_epochs"] = 1
        config["trainer"]["total_training_steps"] = 6
        config["trainer"]["test_freq"] = 6
        config["trainer"]["experiment_name"] = EXPERIMENT_NAME
        config["trainer"]["project_name"] = PROJECT_NAME
        config["trainer"].pop("save_freq", None)
    import pprint
    pprint.pprint(config)
    algorithm = agl.VERL(config)
    import logging
    agl.configure_logger(logging.WARNING)
    
    if external_store_address:
        store: Optional[agl.LightningStore] = agl.LightningStoreClient(external_store_address)
    else:
        store = None

    if llm_proxy:
        tracer = agl.OtelTracer()  # dummy tracer for LLM Proxy
        adapter = agl.LlmProxyTraceToTriplet()
        trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=store, tracer=tracer, adapter=adapter)
    else:
        trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=store)

    # Pass None for datasets - let VERL's create_rl_dataset load them based on config
    # This allows custom dataset classes to be used properly
    print("Datasets will be loaded by VERL's create_rl_dataset based on config")
    trainer.fit(calc_sandbox_agent_youtu, train_dataset=None, val_dataset=None)


def main():
    parser = argparse.ArgumentParser(
        description="Train a calc sandbox agent with Agent-lightning + VERL.",
        epilog="""
        You can also override any config value using Hydra-style syntax:
        
        Examples:
            python train_calc_sandbox_agent.py --train-file data/train.parquet \\
                algorithm.adv_estimator=grpo \\
                trainer.total_epochs=10 \\
                actor_rollout_ref.actor.optim.lr=1e-6
        """
    )
    parser.add_argument("--train-file", type=str, default="data/train.parquet", help="Path to train parquet file")
    parser.add_argument("--val-file", type=str, default="data/test.parquet", help="Path to val parquet file")
    parser.add_argument("--model", type=str, default=None, help="HF model id or path (optional)")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for vLLM server (optional)")
    parser.add_argument("--llm-proxy", action="store_true", help="Enable LLM Proxy tracing/adapter")
    parser.add_argument("--ci", action="store_true", help="Run a minimal CI-style training loop")
    parser.add_argument("--n-runners", type=int, default=15, help="Number of runners for Trainer")
    parser.add_argument(
        "--external-store-address",
        type=str,
        default="",
        help="Connect to an external store instead of creating a new one in memory",
    )
    parser.add_argument("--val-temperature", type=float, default=1.0, help="Temperature for validation sampling")
    parser.add_argument("--val-top-p", type=float, default=0.6, help="Top-p for validation sampling")
    parser.add_argument("--val-n", type=int, default=16, help="Number of responses per prompt for validation")
    parser.add_argument("--custom-dataset-path", type=str, default=None, help="Path to custom dataset class module")
    parser.add_argument("--custom-dataset-name", type=str, default=None, help="Name of custom dataset class")

    # Parse known args and capture unknown args as config overrides
    args, unknown_args = parser.parse_known_args()
    
    # Filter out unknown args that look like config overrides (contain '=')
    config_overrides = [arg for arg in unknown_args if '=' in arg]
    
    # Warn about unknown args that don't look like config overrides
    invalid_args = [arg for arg in unknown_args if '=' not in arg]
    if invalid_args:
        print(f"Warning: Unknown arguments (ignored): {invalid_args}")
    
    if config_overrides:
        print(f"Detected {len(config_overrides)} config overrides from command line")

    if args.external_store_address:
        print(f"Connecting to external store at: {args.external_store_address}")
        if not os.getenv("AGL_MANAGED_STORE"):
            raise ValueError(
                "When using an external store, please set the environment variable AGL_MANAGED_STORE=0. "
                "Otherwise the trainer will still try to manage the store lifecycle for you!"
            )
    
    train(
        train_file=args.train_file,
        val_file=args.val_file,
        model=args.model,
        llm_proxy=args.llm_proxy,
        ci=args.ci,
        n_runners=args.n_runners,
        external_store_address=args.external_store_address,
        val_temperature=args.val_temperature,
        val_top_p=args.val_top_p,
        val_n=args.val_n,
        custom_dataset_path=args.custom_dataset_path,
        custom_dataset_name=args.custom_dataset_name,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()
