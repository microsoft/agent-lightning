# Copyright (c) Microsoft. All rights reserved.

"""
Agent-OS + Agent-Lightning End-to-End Demo
==========================================

Demonstrates how to train an AI agent with kernel-level safety governance
using Agent-OS policy enforcement and Agent-Lightning RL training.

This script shows the full pipeline:
1. Define a governance policy (block dangerous SQL operations)
2. Create a governed runner that enforces the policy
3. Define a reward function that penalizes policy violations
4. Train the agent — it learns to avoid unsafe actions over time

Usage:
    python demo_governed_training.py

Requirements:
    pip install agentlightning agent-os-kernel
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("agentos-demo")


def build_kernel():
    """Create an Agent-OS kernel with SQL safety policy."""
    try:
        from agent_os import KernelSpace
    except ImportError:
        logger.warning("agent-os-kernel not installed – using stub kernel")
        return _StubKernel()

    kernel = KernelSpace(
        policies={
            "sql_safety": {
                "deny_patterns": ["DROP", "DELETE", "TRUNCATE", "ALTER"],
                "max_rows": 1000,
            }
        }
    )
    logger.info("Agent-OS kernel created with SQL safety policy")
    return kernel


def build_runner(kernel):
    """Wrap a runner with Agent-OS policy enforcement."""
    try:
        from agentlightning.contrib.runner.agentos import AgentOSRunner

        runner = AgentOSRunner(
            kernel,
            fail_on_violation=False,  # Continue but penalize
            emit_violations=True,  # Emit as spans for observability
        )
        logger.info("AgentOSRunner created (fail_on_violation=False)")
        return runner
    except ImportError:
        logger.warning("AgentOSRunner not available – using stub")
        return None


def build_reward(kernel, base_reward_fn=None):
    """Create a reward function that penalizes policy violations."""
    try:
        from agentlightning.contrib.reward.agentos import PolicyReward

        reward = PolicyReward(
            kernel,
            base_reward_fn=base_reward_fn,
            critical_penalty=-100.0,
            high_penalty=-50.0,
            medium_penalty=-10.0,
            low_penalty=-1.0,
            clean_bonus=5.0,
        )
        logger.info("PolicyReward created (critical=-100, clean_bonus=+5)")
        return reward
    except ImportError:
        logger.warning("PolicyReward not available – using stub")
        return base_reward_fn


def accuracy_reward(completions: list[str], references: list[str]) -> list[float]:
    """Simple accuracy reward: 1.0 for exact match, 0.0 otherwise."""
    return [1.0 if pred.strip().lower() == ref.strip().lower() else 0.0 for pred, ref in zip(completions, references)]


def demo_violation_detection(kernel):
    """Show that the kernel catches unsafe SQL operations."""
    logger.info("--- Violation Detection Demo ---")

    test_queries = [
        ("SELECT * FROM users WHERE id = 1", True),  # Safe
        ("DROP TABLE users", False),  # Blocked
        ("DELETE FROM orders WHERE id > 0", False),  # Blocked
        ("SELECT name FROM products LIMIT 10", True),  # Safe
        ("TRUNCATE TABLE logs", False),  # Blocked
    ]

    for query, should_pass in test_queries:
        try:
            result = kernel.check_action("sql_query", {"query": query})
            status = "✅ ALLOWED" if result.allowed else "🛑 BLOCKED"
        except Exception:
            # Stub kernel always allows
            status = "✅ ALLOWED (stub)"
            result = type("R", (), {"allowed": True})()

        expected = "should pass" if should_pass else "should block"
        logger.info(f"  {status}: {query!r}  ({expected})")


def demo_training_loop(runner, reward_fn):
    """Simulate a training loop with governed execution."""
    logger.info("--- Training Loop Demo ---")

    # Simulated training data
    prompts = [
        "Find all active users",
        "Remove all test data",
        "Show revenue by month",
        "Drop the staging table",
    ]

    for epoch in range(1, 3):
        logger.info(f"Epoch {epoch}/2")
        for prompt in prompts:
            logger.info(f"  Prompt: {prompt!r}")
            # In real training, the runner would execute the agent
            # and the reward function would score the output
            logger.info("    → Agent generates SQL → Runner enforces policy → Reward scored")

    logger.info("Training complete — agent learns to avoid policy violations over time")


def demo_audit_trail(kernel):
    """Show the audit trail captured by Agent-OS."""
    logger.info("--- Audit Trail Demo ---")

    try:
        from agentlightning.contrib.adapter.agentos import FlightRecorderAdapter

        recorder = getattr(kernel, "flight_recorder", None)
        if recorder:
            adapter = FlightRecorderAdapter(recorder)
            logger.info(f"Audit entries: {len(adapter.get_entries())}")
        else:
            logger.info("Flight recorder not available (stub kernel)")
    except ImportError:
        logger.info("FlightRecorderAdapter not available — skipping audit demo")


def main():
    """Run the full end-to-end demo."""
    logger.info("=" * 60)
    logger.info("Agent-OS + Agent-Lightning End-to-End Demo")
    logger.info("=" * 60)

    # 1. Build the governance kernel
    kernel = build_kernel()

    # 2. Demonstrate violation detection
    demo_violation_detection(kernel)

    # 3. Build governed runner + reward
    runner = build_runner(kernel)
    reward_fn = build_reward(kernel, base_reward_fn=accuracy_reward)

    # 4. Simulate training loop
    demo_training_loop(runner, reward_fn)

    # 5. Show audit trail
    demo_audit_trail(kernel)

    logger.info("=" * 60)
    logger.info("Demo complete! In production, replace stubs with:")
    logger.info("  pip install agentlightning agent-os-kernel")
    logger.info("=" * 60)


class _StubKernel:
    """Minimal stub for demo when agent-os is not installed."""

    def check_action(self, action_type, params):
        blocked = any(kw in params.get("query", "").upper() for kw in ("DROP", "DELETE", "TRUNCATE", "ALTER"))
        return type("Result", (), {"allowed": not blocked})()


if __name__ == "__main__":
    main()
