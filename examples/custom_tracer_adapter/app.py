# Copyright (c) Microsoft. All rights reserved.

"""Minimal Agent-Lightning example demonstrating custom tracer and adapter patterns.

This example shows how to:
1. Create a custom tracer for observability (e.g., DataDog, custom metrics)
2. Build a custom adapter to transform traces into rewards
3. Train an agent with <200 lines of code

Run with: python minimal_agent.py --train
Debug with: python minimal_agent.py --debug
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

import agentlightning as agl


# 
# 1. CUSTOM TRACER - Plug in your own observability



@dataclass
class CustomSpan:
    """Custom span format for your observability system."""

    name: str
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]


class CustomTracer(agl.Tracer):

    """Custom tracer that captures spans in your preferred format.

    This demonstrates how to integrate Agent-Lightning with:
    - Custom observability platforms (DataDog, New Relic, etc.)
    - Internal monitoring systems
    - Custom analytics pipelines
    
    """

    def __init__(self):
        self.spans: List[CustomSpan] = []
        self.current_span: Optional[CustomSpan] = None

    def start_span(self, name: str, **attributes) -> Any:
        """Start a new span with custom attributes."""
        self.current_span = CustomSpan(name=name, attributes=attributes, events=[])
        return self.current_span

    def end_span(self, span: Any) -> None:
        """Finalize and store the span."""
        if span:
            self.spans.append(span)

    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the current span."""
        if self.current_span:
            self.current_span.events.append({"name": name, **attributes})

    def get_spans(self) -> List[CustomSpan]:
        """Retrieve all captured spans."""
        return self.spans


# 2. CUSTOM ADAPTER - Transform traces into rewards


class CustomAdapter(agl.Adapter):
    """Custom adapter that extracts rewards from your trace format.

    This shows how to:
    - Parse custom trace structures
    - Compute rewards from multiple signals
    - Aggregate metrics for RL
    """

    def extract(self, trace: Any) -> Optional[agl.Triplet]:
        """Extract (prompt, response, reward) from custom trace."""
        if not isinstance(trace, CustomSpan):
            return None

        # Extract prompt from span attributes
        prompt = trace.attributes.get("prompt", "")

        # Extract response from events
        response = ""
        for event in trace.events:
            if event["name"] == "response":
                response = event.get("content", "")

        # Calculate reward from multiple signals
        reward = self._compute_reward(trace)

        return agl.Triplet(prompt=prompt, response=response, reward=reward)

    def _compute_reward(self, span: CustomSpan) -> float:
        """Compute reward from span signals.

        Demonstrates reward shaping from:
        - Correctness (from events)
        - Latency (from attributes)
        - Token efficiency (from attributes)
        """
        reward = 0.0

        # Base reward from correctness
        for event in span.events:
            if event["name"] == "reward":
                reward += event.get("value", 0.0)

        # Penalty for high latency
        latency = span.attributes.get("latency", 0.0)
        if latency > 10.0:  # seconds
            reward -= 0.1

        # Bonus for token efficiency
        tokens = span.attributes.get("total_tokens", 1000)
        if tokens < 500:
            reward += 0.05

        return reward



# 3.  AGENT - Simple math solver


@agl.rollout
async def simple_math_agent(task: Dict[str, str], llm: agl.LLM) -> None:

    """Minimal agent that solves simple math problems.

    Args:
        task: Dict with 'question' and 'answer' keys
        llm: LLM endpoint configuration
    """
    client = AsyncOpenAI(base_url=llm.endpoint, api_key="dummy")

    # Simple prompt
    prompt = f"Solve this math problem: {task['question']}\nAnswer with just the number."

    # Get response
    response = await client.chat.completions.create(
        model=llm.model, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=100
    )

    answer = response.choices[0].message.content or ""

    # Compute reward (1.0 if correct, 0.0 otherwise)
    correct = answer.strip() == task["answer"].strip()
    reward = 1.0 if correct else 0.0

    # Emit reward
    agl.emit_reward(reward)

    print(f"Q: {task['question']} | A: {answer} | Expected: {task['answer']} | R: {reward}")


# 4. TRAINING & DEBUGGING


def create_dataset() -> List[Dict[str, str]]:
    """Create a minimal dataset for demonstration."""

    return [

        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "What is 5 * 3?", "answer": "15"},
        {"question": "What is 10 - 7?", "answer": "3"},
        {"question": "What is 12 / 4?", "answer": "3"},
        {"question": "What is 8 + 6?", "answer": "14"},

    ]


async def debug_mode():

    """Debug mode: Test agent without training."""

    print("=" * 60)
    print("DEBUG MODE: Testing agent without training")
    print("=" * 60)

    # Use custom tracer and adapter
    tracer = CustomTracer()
    adapter = CustomAdapter()

    # Create runner
    runner = agl.LitAgentRunner(tracer)
    store = agl.InMemoryLightningStore()

    #  LLM (replace with your endpoint, i used ollama to inference)

    llm = agl.LLM(endpoint="http://localhost:113/v1", model="llama3.2:3b")

    # Run a few test cases
    test_tasks = create_dataset()[:2]

    with runner.run_context(agent=simple_math_agent, store=store):
        for task in test_tasks:
            await runner.step(task, resources={"main_llm": llm})

    # Show captured spans

    print("\nCaptured Spans:")
    for i, span in enumerate(tracer.get_spans()):

        print(f"\nSpan {i + 1}: {span.name}")
        print(f"  Attributes: {json.dumps(span.attributes, indent=4)}")
        print(f"  Events: {json.dumps(span.events, indent=4)}")


def train_mode():

    """Training mode: Full RL training with custom tracer/adapter."""

    print("=" * 60)
    print("TRAINING MODE: Custom observability example")
    print("=" * 60)

    # Create datasets

    train_data = create_dataset()
    val_data = train_data[:2]  # Small validation set

    # Configure VERL algorithm (minimal config)

    config = {
        "algorithm": {"adv_estimator": "grpo"},
        "data": {"train_batch_size": 4},
        "actor_rollout_ref": {
            "model": {"path": "Qwen/Qwen2.5-1.5B-Instruct"},
            "rollout": {"n": 2, "name": "vllm"},
        },
        "trainer": {
            "total_epochs": 1,
            "total_training_steps": 2,
            "project_name": "MinimalExample",
        },
    }

    algorithm = agl.VERL(config)

    # Create trainer with custom tracer and adapter
    tracer = CustomTracer()
    adapter = CustomAdapter()

    trainer = agl.Trainer(algorithm=algorithm, n_runners=2, tracer=tracer, adapter=adapter)

    # Train
    trainer.fit(simple_math_agent, train_data, val_dataset=val_data)

    print("\n" + "=" * 60)
    print("Training complete! Check captured traces in your observability system.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Minimal Agent-Lightning example with custom tracer/adapter")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--debug", action="store_true", help="Run debug mode (test without training)")
    args = parser.parse_args()

    if args.train:
        train_mode()
    elif args.debug:
        import asyncio

        asyncio.run(debug_mode())
    else:
        print("Usage: python minimal_agent.py --train  OR  python minimal_agent.py --debug")


if __name__ == "__main__":
    main()