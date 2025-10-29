# Copyright (c) Microsoft. All rights reserved.

"""
Multimodal VQA Agent with VERL (RL Training).

This agent demonstrates:
1. Using Qwen2-VL with vLLM for multimodal inference
2. RL training with VERL algorithm (GRPO/PPO)
3. Vision tokens processed by local model (not API)
4. End-to-end fine-tuning of vision-language model

Supported Frameworks:
- vLLM (recommended, deep Agent Lightning integration)
- SGLang (alternative, supports more recent models)

Supported Models (both vLLM and SGLang):
- Qwen-VL series: Qwen2-VL-2B/7B, Qwen2.5-VL, Qwen3-VL
- LLaVA series: LLaVA-v1.5/v1.6, LLaVA-NeXT, LLaVA-OneVision
- Llama 3.2 Vision (11B)
- DeepSeek-VL2
- MiniCPM-V
- Phi-4-multimodal
- GLM-4V series

See README.md for complete list and framework-specific models

Usage:
    # Debug mode (single task)
    python vqa_agent.py

    # Training mode (see train_vqa_agent.py)
    python train_vqa_agent.py --llm-proxy --n-runners 10

Architecture:
    Task → vLLM (Qwen2-VL) → Answer → Reward → VERL → Update Model
"""

import asyncio
import os
import re
from pathlib import Path
from typing import TypedDict

from openai import AsyncOpenAI

import agentlightning as agl


class VQATask(TypedDict):
    """Visual Question Answering task structure for RL training.

    Attributes:
        id: Unique task identifier
        image: **Local file path** to the image (required for vLLM)
        question: Question about the image
        answer: Ground truth answer for evaluation (single answer)

    Note:
        - image must be a local file path, not URL
        - vLLM will load image using file:// protocol
        - For VQAv2 dataset, answers should be the most common answer
    """

    id: str
    image: str  # Local file path, e.g., "data/vqav2/images/train2014/COCO_train2014_000000123456.jpg"
    question: str
    answer: str  # Expected answer for reward computation


def compute_answer_similarity(predicted: str, expected: str) -> float:
    """
    Compute similarity between predicted and expected answers.

    Simple string matching for VQA evaluation. For more accurate evaluation,
    consider using VQA evaluation metrics from the official VQAv2 toolkit.

    Args:
        predicted: The agent's answer
        expected: Ground truth answer

    Returns:
        Similarity score from 0.0 to 1.0
    """
    predicted_lower = predicted.lower().strip()
    expected_lower = expected.lower().strip()

    # Exact match
    if predicted_lower == expected_lower:
        return 1.0

    # Contains match
    if expected_lower in predicted_lower:
        return 0.8

    # Word overlap
    pred_words = set(predicted_lower.split())
    exp_words = set(expected_lower.split())
    if not exp_words:
        return 0.0

    overlap = len(pred_words & exp_words) / len(exp_words)
    return min(overlap, 1.0)


@agl.rollout
async def vqa_agent(task: VQATask, llm: agl.LLM) -> None:
    """
    Multimodal VQA agent for RL training with VERL.

    This agent:
    1. Receives task with local image path
    2. Calls vLLM/SGLang endpoint (running vision-language model) via OpenAI-compatible API
    3. Computes reward based on answer quality
    4. Emits reward for VERL to collect

    Args:
        task: VQA task with image path, question, and expected answer
        llm: LLM resource pointing to vLLM/SGLang endpoint with vision-language model

    Note:
        - llm.endpoint should be vLLM or SGLang OpenAI-compatible endpoint (e.g., http://localhost:8000/v1)
        - llm.model should be the model name (e.g., "Qwen/Qwen2-VL-2B-Instruct", "llava-hf/llava-v1.6-vicuna-7b-hf")
        - Image must be accessible from vLLM/SGLang server (use shared filesystem or absolute paths)

    Example task:
        {
            "id": "vqa_001",
            "image": "/path/to/coco/train2014/COCO_train2014_000000001234.jpg",
            "question": "What color is the car?",
            "answer": "red"
        }
    """
    # Create OpenAI client pointing to vLLM
    # vLLM provides OpenAI-compatible API at /v1
    client = AsyncOpenAI(
        base_url=llm.endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),  # vLLM doesn't require real key
    )

    # Get sampling parameters
    temperature = llm.sampling_parameters.get("temperature", 0.7)
    max_tokens = llm.sampling_parameters.get("max_tokens", 100)

    # Construct multimodal message
    # For vLLM/SGLang with vision-language models:
    # - Use file:// protocol for local images
    # - Server will process image through model's vision encoder
    # - Vision tokens are generated automatically
    image_path = task["image"]
    if not image_path.startswith("file://") and not image_path.startswith("http"):
        # Convert to absolute path and add file:// prefix
        if not Path(image_path).is_absolute():
            image_path = str(Path(image_path).resolve())
        image_path = f"file://{image_path}"

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": task["question"]},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_path  # Local file path with file:// protocol
                },
            },
        ],
    }

    try:
        # Call vLLM/SGLang with vision-language model
        # Server automatically:
        # 1. Loads image from file path
        # 2. Processes it through model's vision encoder
        # 3. Generates vision tokens
        # 4. Concatenates with text tokens
        # 5. Runs through LLM decoder
        response = await client.chat.completions.create(
            model=llm.model,
            messages=[message],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content or ""

        # Compute reward based on answer similarity
        reward = compute_answer_similarity(answer, task["answer"])

        # Emit reward for VERL to collect
        # This is crucial for RL training - VERL uses these rewards to update the model
        agl.emit_reward(reward)

        # Print for debugging
        print(f"Task ID: {task['id']}")
        print(f"Question: {task['question']}")
        print(f"Expected: {task['answer']}")
        print(f"Got: {answer}")
        print(f"Reward: {reward:.3f}\n")

    except Exception as e:
        print(f"Error processing task {task['id']}: {e}")
        # Emit 0 reward on error
        agl.emit_reward(0.0)


async def debug():
    """
    Debug function to test agent with a single task.

    This demonstrates manual rollout without Trainer, useful for:
    - Testing vLLM/SGLang endpoint connectivity
    - Verifying image loading
    - Debugging reward computation

    Requirements:
    - vLLM or SGLang server running with a vision-language model
    - Set OPENAI_BASE_URL to server endpoint
    - Image file must exist locally

    Usage:
        # For vLLM
        export OPENAI_BASE_URL=http://localhost:8000/v1
        python vqa_agent.py

        # For SGLang
        export OPENAI_BASE_URL=http://localhost:30000/v1
        python vqa_agent.py
    """
    # Create tracer and store
    tracer = agl.OtelTracer()
    runner = agl.LitAgentRunner[VQATask](tracer)
    store = agl.InMemoryLightningStore()

    # LLM resource pointing to vLLM/SGLang
    server_endpoint = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    llm_resource = agl.LLM(
        endpoint=server_endpoint,
        model="Qwen/Qwen2-VL-2B-Instruct",  # Model name loaded in vLLM/SGLang
        sampling_parameters={"temperature": 0.7, "max_tokens": 100},
    )

    # Debug task
    # TODO: Update this with a real image path on your server
    debug_task: VQATask = {
        "id": "debug_001",
        "image": "data/sample_images/cat.jpg",  # Replace with actual path
        "question": "What animal is in this image?",
        "answer": "cat",
    }

    print("[DEBUG] Running single task...")
    print(f"Server endpoint: {server_endpoint}")
    print(f"Model: {llm_resource.model}")
    print(f"Image path: {debug_task['image']}")
    print(f"Question: {debug_task['question']}\n")

    # Run agent
    with runner.run_context(agent=vqa_agent, store=store):
        rollout = await runner.step(
            debug_task,
            resources={"main_llm": llm_resource},
        )

        # Query results from store
        spans = await store.query_spans(rollout.rollout_id)
        reward = agl.find_final_reward(spans)

        print(f"[DEBUG] Final reward from trace: {reward}")
        print(f"[DEBUG] Captured {len(spans)} spans")


if __name__ == "__main__":
    print("=" * 60)
    print("Multimodal VQA Agent (VERL Mode)")
    print("=" * 60)

    # Check if server endpoint is configured
    if not os.environ.get("OPENAI_BASE_URL"):
        print("\n[WARNING] OPENAI_BASE_URL not set!")
        print("Please set it to your vLLM or SGLang endpoint:")
        print("  vLLM:    export OPENAI_BASE_URL=http://localhost:8000/v1")
        print("  SGLang:  export OPENAI_BASE_URL=http://localhost:30000/v1")
        print("\nUsing default: http://localhost:8000/v1")
        print()

    # Run debug
    asyncio.run(debug())
