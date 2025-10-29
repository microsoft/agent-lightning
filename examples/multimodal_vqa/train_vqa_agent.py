# Copyright (c) Microsoft. All rights reserved.

"""
Training script for Multimodal VQA Agent with VERL.

This script trains vision-language models using reinforcement learning (VERL/GRPO) on VQAv2 dataset.

Architecture:
    Data → Runner (vLLM/SGLang) → Agent → Reward → VERL (GRPO) → Update Model

Key Features:
- End-to-end vision-language model training
- VERL with GRPO algorithm
- Single GPU optimization (24GB VRAM)
- Distributed rollout with LLM proxy
- Vision tokens processed locally
- Supports both vLLM and SGLang serving frameworks

Requirements:
- GPU: RTX 4090 / A6000 (24GB VRAM)
- Dataset: VQAv2 (download with download_vqav2.sh)
- Framework: vLLM or SGLang
- Model: Any supported vision-language model (see README.md)

Usage:
    # Single machine training (no proxy)
    python train_vqa_agent.py --n-runners 4

    # Distributed rollout (with proxy for parallel workers)
    python train_vqa_agent.py --llm-proxy --n-runners 10

    # Custom configuration
    python train_vqa_agent.py \\
        --n-runners 8 \\
        --max-steps 500 \\
        --model Qwen/Qwen2-VL-2B-Instruct \\
        --data-path data/vqav2_train.parquet

    # Try different models (works with both vLLM and SGLang)
    python train_vqa_agent.py --model llava-hf/llava-v1.6-vicuna-7b-hf  # LLaVA
    python train_vqa_agent.py --model OpenGVLab/InternVL2_5-2B  # InternVL
    python train_vqa_agent.py --model meta-llama/Llama-3.2-11B-Vision-Instruct  # Llama Vision
"""

import argparse
import asyncio
import os
from pathlib import Path

import agentlightning as agl
from vqa_agent import VQATask, vqa_agent


def load_vqa_dataset(data_path: str, split: str = "train", limit: int | None = None) -> list[VQATask]:
    """
    Load VQAv2 dataset from parquet files.

    Args:
        data_path: Path to parquet file (e.g., data/vqav2_train.parquet)
        split: Dataset split ("train" or "val")
        limit: Maximum number of tasks to load (for debugging)

    Returns:
        List of VQA tasks

    Expected parquet schema:
        - id: str
        - image_path: str (local path to COCO image)
        - question: str
        - answer: str (most common answer from VQAv2)

    Note:
        Use data_utils.py to convert VQAv2 JSON to parquet format
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for loading parquet files. Install with: pip install pandas pyarrow")

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}\n"
            f"Please run: bash download_vqav2.sh\n"
            f"Or use data_utils.py to prepare the dataset"
        )

    print(f"Loading {split} dataset from {data_path}...")
    df = pd.read_parquet(data_path)

    if limit:
        df = df.head(limit)

    tasks = []
    for _, row in df.iterrows():
        task: VQATask = {
            "id": str(row["id"]),
            "image": str(row["image_path"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
        }
        tasks.append(task)

    print(f"Loaded {len(tasks)} tasks from {split} split")
    return tasks


def get_verl_config(
    model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
    train_batch_size: int = 16,
    gpu_memory_utilization: float = 0.6,
) -> dict:
    """
    Get VERL configuration optimized for single GPU (24GB VRAM).

    Args:
        model_path: HuggingFace model path for any vision-language model
        train_batch_size: Training batch size (reduce if OOM)
        gpu_memory_utilization: vLLM/SGLang memory utilization (0.0-1.0)

    Returns:
        VERL configuration dict

    Single GPU Optimizations:
    - FSDP with parameter and optimizer offloading
    - Gradient checkpointing for memory efficiency
    - Smaller batch size (16 for 2B model on 24GB GPU)
    - Lower GPU memory utilization (0.6) to leave room for training

    Supported Frameworks:
        VERL can manage both vLLM and SGLang backends automatically.
        The framework is selected based on availability and model compatibility.

    Note:
        For multi-GPU training, disable offloading and increase batch size
    """
    return {
        "algorithm": {
            "adv_estimator": "grpo",  # GRPO (Group Relative Policy Optimization)
        },
        "data": {
            "train_batch_size": train_batch_size,
            "max_prompt_length": 2048,  # Increased for vision tokens
            "max_response_length": 512,
        },
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "gpu_memory_utilization": gpu_memory_utilization,
                "log_requests": False,
                "enable_prefix_caching": True,  # Cache vision tokens
            },
            "model": {
                "path": model_path,
            },
            "actor": {
                # Single GPU optimization with FSDP offloading
                "fsdp_config": {
                    "param_offload": True,  # Offload parameters to CPU
                    "optimizer_offload": True,  # Offload optimizer states to CPU
                    "grad_offload": False,  # Keep gradients on GPU for speed
                },
                "optim": {
                    "lr": 1e-6,  # Lower LR for vision-language models
                },
                "ppo_mini_batch_size": 4,  # Smaller mini-batch for memory
                "ppo_micro_batch_size": 1,
            },
        },
    }


async def train_vqa_agent(
    model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
    data_path: str = "data/vqav2_train.parquet",
    val_data_path: str = "data/vqav2_val.parquet",
    n_runners: int = 4,
    max_steps: int = 1000,
    use_llm_proxy: bool = False,
    debug_mode: bool = False,
) -> None:
    """
    Train VQA agent with VERL algorithm.

    Args:
        model_path: HuggingFace model path for any vision-language model
                   Examples: "Qwen/Qwen2-VL-2B-Instruct", "llava-hf/llava-v1.6-vicuna-7b-hf"
        data_path: Path to training parquet file
        val_data_path: Path to validation parquet file
        n_runners: Number of parallel runners
        max_steps: Maximum training steps
        use_llm_proxy: Whether to use LLM proxy for distributed rollout
        debug_mode: If True, use small subset for quick testing

    Training Flow:
        1. Load VQAv2 dataset
        2. Initialize VERL algorithm
        3. Create Trainer with vLLM/SGLang rollout (auto-selected by VERL)
        4. Train with reward feedback
        5. Evaluate on validation set
        6. Save trained model

    Supported Models:
        Works with any vision-language model supported by vLLM or SGLang.
        See README.md for complete list.
    """
    print("=" * 80)
    print("Multimodal VQA Agent Training (VERL)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Training data: {data_path}")
    print(f"Validation data: {val_data_path}")
    print(f"Runners: {n_runners}")
    print(f"Max steps: {max_steps}")
    print(f"LLM Proxy: {use_llm_proxy}")
    print(f"Debug mode: {debug_mode}")
    print("=" * 80)

    # Load datasets
    train_limit = 100 if debug_mode else None
    val_limit = 20 if debug_mode else 500

    train_tasks = load_vqa_dataset(data_path, split="train", limit=train_limit)
    val_tasks = load_vqa_dataset(val_data_path, split="val", limit=val_limit)

    # Initialize VERL algorithm
    verl_config = get_verl_config(
        model_path=model_path,
        train_batch_size=16 if not debug_mode else 4,
        gpu_memory_utilization=0.6,
    )

    print("\nInitializing VERL algorithm...")
    print(f"Algorithm: {verl_config['algorithm']['adv_estimator']}")
    print(f"Train batch size: {verl_config['data']['train_batch_size']}")
    print(f"FSDP offloading: param={verl_config['actor_rollout_ref']['actor']['fsdp_config']['param_offload']}, "
          f"optimizer={verl_config['actor_rollout_ref']['actor']['fsdp_config']['optimizer_offload']}")

    # Create VERL algorithm
    # Note: VERL entrypoint already supports multimodal LLMs (agentlightning/verl/entrypoint.py:90)
    verl_algo = agl.VERL(verl_config)

    # Create Trainer
    trainer = agl.Trainer(
        agent=vqa_agent,
        tasks=train_tasks,
        val_tasks=val_tasks,
        n_runners=n_runners,
        max_steps=max_steps,
        algorithm=verl_algo,
    )

    # Configure LLM proxy if requested
    if use_llm_proxy:
        print("\nStarting LLM proxy for distributed rollout...")
        print("Workers can connect to this proxy for parallel task execution")
        # LLM proxy will be started automatically by Trainer
        # Workers should set OPENAI_BASE_URL to the proxy endpoint

    # Train
    print("\nStarting training...\n")
    await trainer.fit()

    # Save trained model
    output_dir = f"outputs/vqa_agent_{model_path.split('/')[-1]}"
    print(f"\nSaving trained model to {output_dir}...")
    await trainer.save_model(output_dir)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Model saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal VQA Agent with VERL")

    # Model and data
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model path for vision-language model (default: Qwen/Qwen2-VL-2B-Instruct). "
             "Examples: llava-hf/llava-v1.6-vicuna-7b-hf, OpenGVLab/InternVL2_5-2B, meta-llama/Llama-3.2-11B-Vision-Instruct",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/vqav2_train.parquet",
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default="data/vqav2_val.parquet",
        help="Path to validation parquet file",
    )

    # Training configuration
    parser.add_argument(
        "--n-runners",
        type=int,
        default=4,
        help="Number of parallel runners (default: 4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps (default: 1000)",
    )
    parser.add_argument(
        "--llm-proxy",
        action="store_true",
        help="Use LLM proxy for distributed rollout",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with small dataset subset",
    )

    args = parser.parse_args()

    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("[WARNING] No GPU detected! VERL requires GPU for training.")
            print("This script will fail without CUDA-capable GPU.")
    except ImportError:
        print("[WARNING] PyTorch not installed. Cannot check GPU availability.")

    # Run training
    asyncio.run(
        train_vqa_agent(
            model_path=args.model,
            data_path=args.data_path,
            val_data_path=args.val_data_path,
            n_runners=args.n_runners,
            max_steps=args.max_steps,
            use_llm_proxy=args.llm_proxy,
            debug_mode=args.debug,
        )
    )


if __name__ == "__main__":
    main()
