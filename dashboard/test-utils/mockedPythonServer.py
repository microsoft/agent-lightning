# Copyright (c) Microsoft. All rights reserved.

"""
Script to inject mock data directly into an InMemoryLightningStore.

This script creates sample rollouts, attempts, spans, and resources that match
the TypeScript mock data (from dashboard/src/components/RolloutTable.story.tsx)
and injects them directly into the store's private variables for testing.

Usage:
    from dashboard.test_utils.inject_mock_data import inject_mock_data

    store = InMemoryLightningStore()
    inject_mock_data(store)
"""

# pyright: reportPrivateUsage=false

import asyncio
import time
from typing import List

from agentlightning.store.client_server import LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import (
    Attempt,
    AttemptStatus,
    OtelResource,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    RolloutStatus,
    Span,
    TraceStatus,
)


def inject_mock_data(store: InMemoryLightningStore, now: float | None = None) -> None:
    """
    Inject mock data directly into the InMemoryLightningStore.

    Args:
        store: The InMemoryLightningStore instance to inject data into
        now: The current timestamp (defaults to current time)
    """
    if now is None:
        now = time.time()

    # Create rollouts matching the TypeScript mock data
    # Based on sampleRollouts from dashboard/src/components/RolloutTable.story.tsx

    # Rollout 1: Running
    rollout1 = Rollout(
        rollout_id="ro-story-001",
        input=dict(task="Generate onboarding summary"),
        start_time=now - 3200,
        end_time=None,
        mode="train",
        resources_id="rs-story-001",
        status="running",
        config=RolloutConfig(max_attempts=1),
        metadata={"owner": "alice"},
    )
    attempt1 = Attempt(
        rollout_id="ro-story-001",
        attempt_id="at-story-010",
        sequence_id=1,
        start_time=now - 3200,
        end_time=None,
        status="running",
        worker_id="worker-east",
        last_heartbeat_time=now - 45,
        metadata={"info": "Worker is processing"},
    )

    # Rollout 2: Succeeded
    rollout2 = Rollout(
        rollout_id="ro-story-002",
        # NOTE: input is a string here, not a dict
        input="Classify feedback tickets",
        start_time=now - 7200,
        end_time=now - 5400,
        mode="val",
        resources_id="rs-story-002",
        status="succeeded",
        config=RolloutConfig(max_attempts=2, timeout_seconds=10, unresponsive_seconds=20),
        metadata={"owner": "bob"},
    )
    attempt2_1 = Attempt(
        rollout_id="ro-story-002",
        attempt_id="at-story-021",
        sequence_id=1,
        start_time=now - 7200,
        end_time=now - 5400,
        status="timeout",
        last_heartbeat_time=None,
        worker_id=None,
        metadata=None,
    )
    attempt2_2 = Attempt(
        rollout_id="ro-story-002",
        attempt_id="at-story-022",
        sequence_id=2,
        start_time=now - 6200,
        end_time=now - 5400,
        status="succeeded",
        worker_id="worker-north",
        last_heartbeat_time=now - 5400,
        metadata={"previousAttempt": "at-story-010"},
    )

    # Rollout 3: Failed
    rollout3 = Rollout(
        rollout_id="ro-story-003",
        input=dict(task="Analyze experiment results"),
        start_time=now - 10800,
        end_time=now - 9600,
        mode="test",
        resources_id="rs-story-003",
        status="failed",
        config=RolloutConfig(max_attempts=3),
        metadata={"owner": "carol"},
    )
    attempt3_1 = Attempt(
        rollout_id="ro-story-003",
        attempt_id="at-story-031",
        sequence_id=3,
        start_time=now - 10200,
        end_time=now - 9600,
        status="failed",
        worker_id="worker-west",
        last_heartbeat_time=now - 9600,
        metadata={"reason": "Timeout"},
    )
    attempt3_2 = Attempt(
        rollout_id="ro-story-003",
        attempt_id="at-story-032",
        sequence_id=2,
        start_time=now - 9600,
        end_time=None,
        status="unresponsive",
        worker_id="worker-west",
        last_heartbeat_time=now - 3600,
        metadata={"reason": "Unresponsive"},
    )
    attempt3_3 = Attempt(
        rollout_id="ro-story-003",
        attempt_id="at-story-033",
        sequence_id=3,
        start_time=now - 9000,
        end_time=now - 3600,
        status="failed",
        worker_id="worker-west",
    )

    # Rollout 4: Preparing (no attempt yet)
    rollout4 = Rollout(
        rollout_id="ro-story-004",
        input=dict(task="Evaluate prompt variants"),
        start_time=now - 3600,
        end_time=None,
        mode="train",
        resources_id=None,
        status="preparing",
        config=RolloutConfig(max_attempts=5, retry_condition=["timeout"]),
        metadata={"owner": "dave"},
    )

    # Rollout 5: Running
    rollout5 = Rollout(
        rollout_id="ro-story-005",
        input=dict(task="Generate quick answers"),
        start_time=now - 1800,
        end_time=None,
        mode="val",
        resources_id="rs-story-004",
        status="running",
        config=RolloutConfig(max_attempts=0),
        metadata={"owner": "eva"},
    )
    attempt5 = Attempt(
        rollout_id="ro-story-005",
        attempt_id="at-story-013",
        sequence_id=1,
        start_time=now - 1800,
        end_time=None,
        status="running",
        worker_id=None,
        last_heartbeat_time=now - 75,
        metadata=None,
    )

    # Rollout 6: Cancelled
    rollout6 = Rollout(
        rollout_id="ro-story-006",
        input=dict(task="Compile release notes"),
        start_time=now - 9600,
        end_time=now - 9000,
        mode=None,
        resources_id="rs-story-005",
        status="cancelled",
        config=RolloutConfig(max_attempts=3),
        metadata=None,
    )
    attempt6 = Attempt(
        rollout_id="ro-story-006",
        attempt_id="at-story-014",
        sequence_id=1,
        start_time=now - 9600,
        end_time=now - 9000,
        status="timeout",
        worker_id="worker-south",
        last_heartbeat_time=now - 9000,
        metadata={"info": "Cancelled by operator"},
    )

    # Inject rollouts directly into store
    store._rollouts["ro-story-001"] = rollout1
    store._rollouts["ro-story-002"] = rollout2
    store._rollouts["ro-story-003"] = rollout3
    store._rollouts["ro-story-004"] = rollout4
    store._rollouts["ro-story-005"] = rollout5
    store._rollouts["ro-story-006"] = rollout6

    # Inject attempts directly into store
    store._attempts["ro-story-001"] = [attempt1]
    store._attempts["ro-story-002"] = [attempt2_1, attempt2_2]
    store._attempts["ro-story-003"] = [attempt3_1, attempt3_2, attempt3_3]
    store._attempts["ro-story-004"] = []  # No attempt for preparing rollout
    store._attempts["ro-story-005"] = [attempt5]
    store._attempts["ro-story-006"] = [attempt6]

    # Create and inject spans
    spans_ro1: List[Span] = [
        Span(
            rollout_id="ro-story-001",
            attempt_id="at-story-010",
            sequence_id=1,
            trace_id="trace-001",
            span_id="span-001",
            parent_id=None,
            name="main",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"component": "agent"},
            events=[],
            links=[],
            start_time=now - 3200,
            end_time=now - 3100,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-001",
            attempt_id="at-story-010",
            sequence_id=2,
            trace_id="trace-001",
            span_id="span-002",
            parent_id="span-001",
            name="llm_call",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"model": "gpt-4"},
            events=[],
            links=[],
            start_time=now - 3150,
            end_time=now - 3100,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
    ]

    spans_ro2: List[Span] = [
        Span(
            rollout_id="ro-story-002",
            attempt_id="at-story-011",
            sequence_id=1,
            trace_id="trace-002",
            span_id="span-003",
            parent_id=None,
            name="classify",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"type": "classification"},
            events=[],
            links=[],
            start_time=now - 6200,
            end_time=now - 5400,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
    ]

    store._spans["ro-story-001"] = spans_ro1
    store._spans["ro-story-002"] = spans_ro2

    # Create and inject resources
    resource1 = ResourcesUpdate(
        resources_id="rs-story-001",
        version=1,
        create_time=now - 86400,
        update_time=now - 86400,
        resources={
            "model": LLM(model="gpt-4", sampling_parameters={"temperature": 0.7}),
            "dataset": {"name": "train_v1", "size": 1000},
        },
    )

    resource2 = ResourcesUpdate(
        resources_id="rs-story-002",
        version=1,
        create_time=now - 86400,
        update_time=now - 43200,
        resources={
            "model": {"name": "gpt-3.5-turbo", "temperature": 0.5},
            "dataset": {"name": "val_v1", "size": 200},
        },
    )

    resource3 = ResourcesUpdate(
        resources_id="rs-story-003",
        version=2,
        create_time=now - 86400,
        update_time=now - 3600,
        resources={
            "model": {"name": "claude-3", "temperature": 0.8},
            "dataset": {"name": "test_v1", "size": 300},
        },
    )

    resource4 = ResourcesUpdate(
        resources_id="rs-story-004",
        version=1,
        create_time=now - 7200,
        update_time=now - 7200,
        resources={
            "model": {"name": "llama-3", "temperature": 0.6},
        },
    )

    resource5 = ResourcesUpdate(
        resources_id="rs-story-005",
        version=1,
        create_time=now - 14400,
        update_time=now - 14400,
        resources={
            "model": {"name": "mixtral", "temperature": 0.9},
        },
    )

    store._resources["rs-story-001"] = resource1
    store._resources["rs-story-002"] = resource2
    store._resources["rs-story-003"] = resource3
    store._resources["rs-story-004"] = resource4
    store._resources["rs-story-005"] = resource5
    store._latest_resources_id = "rs-story-005"


async def main():
    store = InMemoryLightningStore()
    inject_mock_data(store)

    # Start server
    server = LightningStoreServer(store, "127.0.0.1", 8765)
    await server.start()

    print(f"Server started at {server.endpoint}")
    print("Press Ctrl+C to stop...")

    try:
        # Keep server running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
