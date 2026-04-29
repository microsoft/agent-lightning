# Copyright (c) Microsoft Corporation.

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

from ddtrace import tracer as dd_tracer
from openai import AsyncOpenAI
from opentelemetry.sdk.trace import ReadableSpan

import agentlightning as agl


class DataDogTracer(agl.Tracer):
    """Tracer that sends all spans to DataDog APM.
    
    This tracer is passed to LitAgentRunner, which manages its lifecycle:
    - Runner calls start_span() when beginning a rollout trace
    - Agent calls agl.emit_reward() which triggers add_event()
    - Runner calls end_span() when completing the trace
    
    Usage:
        tracer = DataDogTracer()
        runner = agl.LitAgentRunner(tracer)
        with runner.run_context(agent=my_agent, store=store):
            await runner.step(task, resources=resources)
    """

    def __init__(self, service: str = "agent-lightning"):
        self.service = service
        self.current_span: Optional[Any] = None
        self.span_start_time: float = 0.0
        print(f"[DataDogTracer] Initialized for service: {service}")

    def start_span(self, name: str, **attributes) -> Any:
        """Called by Runner when starting to trace a rollout."""
        print(f"[DataDogTracer] start_span called by Runner: {name}")
        
        # Create DataDog span
        self.current_span = dd_tracer.trace(
            name=name,
            service=self.service,
            resource=attributes.get("task_id", name),
        )
        
        # Add custom tags for filtering in DataDog UI
        for key, value in attributes.items():
            self.current_span.set_tag(key, value)
        
        # Track timing for metrics
        self.span_start_time = time.time()
        
        return self.current_span

    def end_span(self, span: Any) -> None:
        """Called by Runner when finishing the rollout trace."""
        if span:
            print(f"[DataDogTracer] end_span called by Runner")
            
            # Add duration metric
            duration = time.time() - self.span_start_time
            span.set_tag("duration_seconds", duration)
            
            # Finish the span (sends to DataDog)
            span.finish()
            print(f"[DataDogTracer] Span sent to DataDog (duration: {duration:.2f}s)")

    def add_event(self, name: str, **attributes) -> None:
        """Called when agent uses agl.emit_reward(), agl.emit_message(), etc.
        
        This is triggered by the agent code calling agl.emit_*() functions
        during the rollout execution.
        """
        if self.current_span:
            print(f"[DataDogTracer] add_event called (triggered by agl.emit_*): {name}")
            
            # Add event as span tag
            event_data = {k: v for k, v in attributes.items()}
            self.current_span.set_tag(f"event.{name}", str(event_data))
            
            # Special handling for rewards (send as metric)
            if name == "reward":
                reward_value = attributes.get("value", 0.0)
                self.current_span.set_metric("agent.reward", reward_value)
            
            # Special handling for errors
            if name == "error":
                self.current_span.set_tag("error", True)
                self.current_span.set_tag("error.msg", attributes.get("message", ""))


# custom adapter

class MultiSignalAdapter(agl.TraceAdapter[List[agl.Triplet]]):
    """Adapter that computes rewards from correctness + latency + tokens.
    
    This demonstrates reward shaping for production systems where you
    care about accuracy, speed, and cost simultaneously.
    """

    def adapt(self, spans: List[ReadableSpan]) -> List[agl.Triplet]:
        """Transform OpenTelemetry spans into RL training data."""
        print(f"[MultiSignalAdapter] Processing {len(spans)} spans")
        
        triplets: List[agl.Triplet] = []
        
        for span in spans:
            attrs = dict(span.attributes) if span.attributes else {}
            
            # Look for LLM interaction spans (have gen_ai.* attributes)
            if "gen_ai.prompt" in attrs:
                prompt = str(attrs.get("gen_ai.prompt", ""))
                response = str(attrs.get("gen_ai.response", ""))
                
                # Extract base reward (correctness from agl.emit_reward)
                base_reward = self._extract_base_reward(attrs)
                
                # Apply multi-signal reward shaping
                final_reward = self._compute_multi_signal_reward(
                    base_reward=base_reward,
                    latency=float(attrs.get("gen_ai.response.latency_ms", 0)) / 1000.0,
                    total_tokens=int(attrs.get("gen_ai.usage.total_tokens", 1000)),
                )
                
                triplets.append(
                    agl.Triplet(prompt=prompt, response=response, reward=final_reward)
                )
        
        print(f"[MultiSignalAdapter] Extracted {len(triplets)} triplets")
        return triplets

    def _extract_base_reward(self, attrs: Dict[str, Any]) -> float:
        """Extract base correctness reward from span attributes."""
        # Reward is set by agl.emit_reward() in the agent
        return float(attrs.get("reward", 0.0))

    def _compute_multi_signal_reward(
        self, base_reward: float, latency: float, total_tokens: int
    ) -> float:
        """Compute final reward from multiple signals.
        
        Reward components:
        - Base: Correctness (0.0 or 1.0)
        - Penalty: Latency > 5s costs -0.1
        - Bonus: Tokens < 300 gains +0.05
        
        This encourages accurate, fast, and efficient responses.
        """
        reward = base_reward
        
        # Latency penalty - we want fast responses for production
        if latency > 5.0:
            reward -= 0.1
        
        # Token efficiency bonus - concise answers reduce API costs
        if total_tokens < 300:
            reward += 0.05
        
        print(f"[MultiSignalAdapter] Reward: {base_reward:.2f} (base) "
              f"→ {reward:.2f} (after latency={latency:.2f}s, tokens={total_tokens})")
        
        return reward


# simple agent


@agl.rollout
async def math_agent(task: Dict[str, str], llm: agl.LLM) -> None:
    """Simple agent that solves math problems.
    
    The agent emits rewards which trigger DataDogTracer.add_event().
    """
    client = AsyncOpenAI(
        base_url=llm.endpoint,
       
    )

    prompt = f"Solve this math problem and give only the numerical answer: {task['question']}"

    # Track latency for multi-signal reward
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
    except Exception as e:
        print(f" Failed to call LLM: {e}")
        print(f"   Endpoint: {llm.endpoint}")
        print(f"   Model: {llm.model}")
        agl.emit_reward(0.0)
        return
    
    latency = time.time() - start_time

    answer = response.choices[0].message.content or ""
    
    # Check correctness - math problems have exact answers
    correct = answer.strip() == task["answer"].strip()
    reward = 1.0 if correct else 0.0

    # Emit reward (triggers DataDogTracer.add_event)
    agl.emit_reward(reward)
    
    # Emit additional metrics for multi-signal adapter
    agl.emit_object({
        "latency_ms": latency * 1000,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    })

    print(f"Q: {task['question']} | A: {answer} | Expected: {task['answer']} | "
          f"R: {reward:.1f} | Latency: {latency:.2f}s")


# sample dataset


def create_dataset() -> List[Dict[str, str]]:
    """Create simple math dataset for demonstration."""
    return [
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "What is 5 * 3?", "answer": "15"},
        {"question": "What is 10 - 7?", "answer": "3"},
        {"question": "What is 12 / 4?", "answer": "3"},
        {"question": "What is 8 + 6?", "answer": "14"},
        {"question": "What is 9 * 2?", "answer": "18"},
    ]


# debug n training


async def debug_mode():
    """Debug mode: Test agent and see DataDog spans in real-time.
    
    This demonstrates the correct usage pattern:
    1. Create tracer
    2. Pass tracer to LitAgentRunner
    3. Runner manages tracer lifecycle during run_context()
    """
    
    print("DEBUG MODE: DataDog Integration Test")
   
    print("\nCheck DataDog APM at: https://app.datadoghq.com/apm/traces")

    # Initialize DataDog tracer
    tracer = DataDogTracer(service="agent-lightning-demo")
    
    # Pass tracer to Runner - Runner will manage tracer lifecycle
    runner = agl.LitAgentRunner[Dict[str, str]](tracer)
    store = agl.InMemoryLightningStore()

    # Get LLM configuration from environment
    endpoint = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    
    print(f"Using LLM endpoint: {endpoint}")
    print(f"Using model: {model}")
    print()

    llm = agl.LLM(
        endpoint=endpoint,
        model=model,
        sampling_parameters={"temperature": 0.7},
    )

    test_tasks = create_dataset()[:3]

    print("Running test tasks...")
    print("Watch for Runner calling tracer methods:\n")
    
    # Runner.run_context() manages the tracer lifecycle
    with runner.run_context(agent=math_agent, store=store):
        for i, task in enumerate(test_tasks, 1):
            print(f"--- Task {i}/{len(test_tasks)} ---")
            print("Runner will call:")
            print("  1. tracer.start_span()")
            print("  2. Agent executes, calls agl.emit_reward()")
            print("     → This triggers tracer.add_event()")
            print("  3. tracer.end_span()")
            print()
            
            # Runner.step() orchestrates tracer calls
            await runner.step(task, resources={"main_llm": llm})
            print()

    
    print(" Debug complete! Check DataDog APM for traces.")
    


def train_mode():
    """Training mode: Full RL training with DataDog observability."""
    
    print("TRAINING MODE: RL with DataDog Integration")
    
   
    print("\nMonitor training at: https://app.datadoghq.com/apm/traces")
    

    # Prepare data
    train_data = create_dataset()
    val_data = train_data[:3]

    # Get model from environment
    base_model = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    
    print(f"Training model: {base_model}")
    print()

    # VERL configuration
    config = {
        "algorithm": {"adv_estimator": "grpo"},
        "data": {"train_batch_size": 6},
        "actor_rollout_ref": {
            "model": {"path": base_model},
            "rollout": {"n": 2, "name": "vllm"},
        },
        "trainer": {
            "total_epochs": 1,
            "total_training_steps": 3,
            "project_name": "DataDogIntegration",
            "experiment_name": "math_agent_demo",
        },
    }

    algorithm = agl.VERL(config)
    
    # Use DataDog tracer and multi-signal adapter
    tracer = DataDogTracer(service="agent-lightning-training")
    adapter = MultiSignalAdapter()

    # Trainer will pass tracer to Runner internally
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=3,
        tracer=tracer,
        adapter=adapter,
    )

    print("Starting training...\n")
    trainer.fit(math_agent, train_data, val_dataset=val_data)

   
    print("Training complete. View results in DataDog APM dashboard")
    


# main


def check_prerequisites():
    """Check that required environment variables are set."""
    
    
    # Check DataDog API key
    if not os.getenv("DD_API_KEY"):
        print ( "  DD_API_KEY not set. DataDog tracing will be disabled.\n")
    
    # Check LLM endpoint for debug mode
    if not os.getenv("OLLAMA_BASE_URL"):
        print('set a LLM provider')
    
    


def main():
    parser = argparse.ArgumentParser(
        description="DataDog integration example for Agent-Lightning"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run full training with DataDog observability",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Test agent and DataDog integration without training",
    )
    
    args = parser.parse_args()

    # Check prerequisites
    check_prerequisites()

    if args.train:
        train_mode()
    elif args.debug:
        asyncio.run(debug_mode())
    else:
        print("Usage:")
        print("  python datadog_agent.py --debug   # Test integration")
        print("  python datadog_agent.py --train   # Full training")


if __name__ == "__main__":
    main()