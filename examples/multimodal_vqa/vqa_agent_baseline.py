# Copyright (c) Microsoft. All rights reserved.

"""
Baseline VQA Agent with OpenAI API (for comparison with RL-trained models).

This is a simplified baseline agent that uses OpenAI's GPT-4V API directly.
It's designed for comparison with the RL-trained Qwen2-VL model.

Key Differences from RL Version:
- Uses GPT-4V API (no local GPU needed)
- No RL training - just inference
- Simpler setup - only requires OPENAI_API_KEY
- Used as baseline to evaluate RL training improvements

Usage:
    export OPENAI_API_KEY="your-api-key"
    python vqa_agent_baseline.py
"""

import asyncio
import os
from pathlib import Path
from typing import TypedDict

from openai import AsyncOpenAI


class VQATask(TypedDict):
    """Visual Question Answering task structure (same as RL version).

    Attributes:
        id: Unique task identifier
        image: Local file path or URL to the image
        question: Question about the image
        answer: Ground truth answer for evaluation
    """
    id: str
    image: str
    question: str
    answer: str


def compute_answer_similarity(predicted: str, expected: str) -> float:
    """
    Compute similarity between predicted and expected answers.

    Same evaluation metric as RL version for fair comparison.
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


async def vqa_baseline(
    task: VQATask,
    model: str = "gpt-4o",
    api_key: str | None = None
) -> tuple[str, float]:
    """
    Baseline VQA agent using OpenAI API.

    Args:
        task: VQA task with image, question, and expected answer
        model: OpenAI model name (default: gpt-4o)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)

    Returns:
        Tuple of (answer, reward)

    Example:
        task = {
            "id": "test_001",
            "image": "https://example.com/cat.jpg",
            "question": "What animal is this?",
            "answer": "cat"
        }
        answer, reward = await vqa_baseline(task)
    """
    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # Prepare image URL
    image_url = task["image"]
    if not image_url.startswith("http"):
        # Local file - convert to absolute path
        if not Path(image_url).is_absolute():
            image_url = str(Path(image_url).resolve())
        # For OpenAI API, we need actual URLs or base64
        # For simplicity, this baseline expects URLs or will fail gracefully
        print(f"[WARNING] Local file path detected: {image_url}")
        print("[WARNING] OpenAI API requires URLs or base64. Consider using remote URLs.")

    # Construct message
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": task["question"]},
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
    }

    try:
        # Call OpenAI API
        response = await client.chat.completions.create(
            model=model,
            messages=[message],
            max_tokens=100,
            temperature=0.7,
        )

        answer = response.choices[0].message.content or ""
        reward = compute_answer_similarity(answer, task["answer"])

        # Print results
        print(f"Task ID: {task['id']}")
        print(f"Question: {task['question']}")
        print(f"Expected: {task['answer']}")
        print(f"Got: {answer}")
        print(f"Reward: {reward:.3f}\n")

        return answer, reward

    except Exception as e:
        print(f"Error processing task {task['id']}: {e}")
        return "", 0.0


async def debug():
    """
    Debug function to test baseline agent.

    Requirements:
    - Set OPENAI_API_KEY environment variable
    - Use image URLs (local files not directly supported by OpenAI API)

    Usage:
        export OPENAI_API_KEY="sk-..."
        python vqa_agent_baseline.py
    """
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set!")
        print("Please set it to your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        return

    # Debug task - using a public URL for testing
    # TODO: Update this with actual VQAv2 image URLs
    debug_task: VQATask = {
        "id": "baseline_debug_001",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "question": "What animal is in this image?",
        "answer": "cat",
    }

    print("=" * 60)
    print("Baseline VQA Agent (OpenAI API)")
    print("=" * 60)
    print(f"Model: gpt-4o")
    print(f"Image: {debug_task['image']}")
    print(f"Question: {debug_task['question']}\n")

    # Run agent
    answer, reward = await vqa_baseline(debug_task)

    print(f"[DEBUG] Final reward: {reward:.3f}")
    print(f"[DEBUG] Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(debug())
