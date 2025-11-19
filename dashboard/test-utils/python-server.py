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

import argparse
import asyncio
import time
from typing import List

from agentlightning.store.client_server import LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import (
    LLM,
    Attempt,
    OtelResource,
    PromptTemplate,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    Span,
    TraceStatus,
    Worker,
)


async def inject_mock_data(store: InMemoryLightningStore, now: float | None = None) -> None:
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
        config=RolloutConfig(max_attempts=2, timeout_seconds=86400, unresponsive_seconds=86400 * 2),
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

    # Rollout 4: Queuing (no attempt yet)
    rollout4 = Rollout(
        rollout_id="ro-story-004",
        input=dict(task="Evaluate prompt variants"),
        start_time=now - 3600,
        end_time=None,
        mode="train",
        resources_id=None,
        status="requeuing",
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
        config=RolloutConfig(max_attempts=1),
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
    await store.collections.rollouts.insert([rollout1, rollout2, rollout3, rollout4, rollout5, rollout6])

    # Inject attempts directly into store
    await store.collections.attempts.insert(
        [attempt1, attempt2_1, attempt2_2, attempt3_1, attempt3_2, attempt3_3, attempt5, attempt6]
    )

    # Create and inject spans with diverse data
    # Spans for ro-story-001 (Running) - Multiple nested spans with ongoing execution
    spans_ro1: List[Span] = [
        Span(
            rollout_id="ro-story-001",
            attempt_id="at-story-010",
            sequence_id=1,
            trace_id="trace-001-main",
            span_id="span-001-root",
            parent_id=None,
            name="agent_execution",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"component": "agent", "framework": "autogen", "version": "0.2.0"},
            events=[],
            links=[],
            start_time=now - 3200,
            end_time=now - 3100,
            context=None,
            parent=None,
            resource=OtelResource(attributes={"service.name": "agent-service"}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-001",
            attempt_id="at-story-010",
            sequence_id=2,
            trace_id="trace-001-main",
            span_id="span-002-llm",
            parent_id="span-001-root",
            name="llm_call",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"model": "gpt-4", "temperature": 0.7, "tokens": 150},
            events=[],
            links=[],
            start_time=now - 3180,
            end_time=now - 3120,
            context=None,
            parent=None,
            resource=OtelResource(attributes={"service.name": "llm-service"}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-001",
            attempt_id="at-story-010",
            sequence_id=3,
            trace_id="trace-001-main",
            span_id="span-003-tool",
            parent_id="span-001-root",
            name="tool_execution",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"tool": "web_search", "query": "onboarding summary"},
            events=[],
            links=[],
            start_time=now - 3140,
            end_time=now - 3100,
            context=None,
            parent=None,
            resource=OtelResource(attributes={"service.name": "tool-service"}, schema_url=""),
        ),
    ]

    # Spans for ro-story-002 attempt 1 (timeout) - Failed workflow
    spans_ro2_a1: List[Span] = [
        Span(
            rollout_id="ro-story-002",
            attempt_id="at-story-021",
            sequence_id=1,
            trace_id="trace-002-attempt1",
            span_id="span-004-root",
            parent_id=None,
            name="classification_pipeline",
            status=TraceStatus(status_code="ERROR", description="Request timeout"),
            attributes={"type": "classification", "model": "bert-classifier", "timeout": True},
            events=[],
            links=[],
            start_time=now - 7200,
            end_time=now - 6800,
            context=None,
            parent=None,
            resource=OtelResource(attributes={"service.name": "classifier"}, schema_url=""),
        ),
    ]

    # Spans for ro-story-002 attempt 2 (succeeded) - Successful workflow
    spans_ro2_a2: List[Span] = [
        Span(
            rollout_id="ro-story-002",
            attempt_id="at-story-022",
            sequence_id=1,
            trace_id="trace-002-attempt2",
            span_id="span-005-root",
            parent_id=None,
            name="classification_pipeline",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"type": "classification", "model": "bert-classifier", "retry": True},
            events=[],
            links=[],
            start_time=now - 6200,
            end_time=now - 5400,
            context=None,
            parent=None,
            resource=OtelResource(attributes={"service.name": "classifier"}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-002",
            attempt_id="at-story-022",
            sequence_id=2,
            trace_id="trace-002-attempt2",
            span_id="span-006-preprocess",
            parent_id="span-005-root",
            name="preprocess_tickets",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"tickets_count": 50, "duration_ms": 120},
            events=[],
            links=[],
            start_time=now - 6200,
            end_time=now - 6100,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-002",
            attempt_id="at-story-022",
            sequence_id=3,
            trace_id="trace-002-attempt2",
            span_id="span-007-classify",
            parent_id="span-005-root",
            name="run_classifier",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"batch_size": 10, "accuracy": 0.95},
            events=[],
            links=[],
            start_time=now - 6000,
            end_time=now - 5400,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
    ]

    # Spans for ro-story-003 attempt 3 (failed) - Error scenario with nested failures
    spans_ro3_a3: List[Span] = [
        Span(
            rollout_id="ro-story-003",
            attempt_id="at-story-033",
            sequence_id=1,
            trace_id="trace-003-experiment",
            span_id="span-008-root",
            parent_id=None,
            name="experiment_analysis",
            status=TraceStatus(status_code="ERROR", description="Analysis failed"),
            attributes={"experiment_id": "exp-123", "timeout_seconds": 300},
            events=[],
            links=[],
            start_time=now - 9000,
            end_time=now - 3600,
            context=None,
            parent=None,
            resource=OtelResource(attributes={"service.name": "experiment-runner"}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-003",
            attempt_id="at-story-033",
            sequence_id=2,
            trace_id="trace-003-experiment",
            span_id="span-009-fetch",
            parent_id="span-008-root",
            name="fetch_experiment_data",
            status=TraceStatus(status_code="ERROR", description="Connection timeout"),
            attributes={"data_source": "database", "retry_count": 3},
            events=[],
            links=[],
            start_time=now - 9000,
            end_time=now - 7000,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-003",
            attempt_id="at-story-033",
            sequence_id=3,
            trace_id="trace-003-experiment",
            span_id="span-010-process",
            parent_id="span-008-root",
            name="process_results",
            status=TraceStatus(status_code="UNSET", description=None),
            attributes={"stage": "aggregation", "partial_results": True},
            events=[],
            links=[],
            start_time=now - 6000,
            end_time=now - 3600,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
    ]

    # Spans for ro-story-005 (Running) - In-progress with mixed statuses
    spans_ro5: List[Span] = [
        Span(
            rollout_id="ro-story-005",
            attempt_id="at-story-013",
            sequence_id=1,
            trace_id="trace-005-answers",
            span_id="span-011-root",
            parent_id=None,
            name="quick_answer_generation",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"batch_size": 10, "model": "llama-3"},
            events=[],
            links=[],
            start_time=now - 1800,
            end_time=now - 1500,
            context=None,
            parent=None,
            resource=OtelResource(attributes={"service.name": "qa-service"}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-005",
            attempt_id="at-story-013",
            sequence_id=2,
            trace_id="trace-005-answers",
            span_id="span-012-llm",
            parent_id="span-011-root",
            name="llm_inference",
            status=TraceStatus(status_code="UNSET", description=None),
            attributes={"model": "llama-3-70b", "batch_id": "batch-001"},
            events=[],
            links=[],
            start_time=now - 1700,
            end_time=None,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-005",
            attempt_id="at-story-013",
            sequence_id=3,
            trace_id="trace-005-answers",
            span_id="span-013-retrieve",
            parent_id="span-011-root",
            name="retrieve_context",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"documents_count": 5, "vector_db": "chroma"},
            events=[],
            links=[],
            start_time=now - 1780,
            end_time=now - 1700,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
    ]

    # Spans for ro-story-006 (Cancelled) - Cancelled workflow
    spans_ro6: List[Span] = [
        Span(
            rollout_id="ro-story-006",
            attempt_id="at-story-014",
            sequence_id=1,
            trace_id="trace-006-release",
            span_id="span-014-root",
            parent_id=None,
            name="compile_release_notes",
            status=TraceStatus(status_code="ERROR", description="Operation cancelled by user"),
            attributes={"release_version": "v2.0.0", "cancelled": True},
            events=[],
            links=[],
            start_time=now - 9600,
            end_time=now - 9000,
            context=None,
            parent=None,
            resource=OtelResource(attributes={"service.name": "release-automation"}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-006",
            attempt_id="at-story-014",
            sequence_id=2,
            trace_id="trace-006-release",
            span_id="span-015-git",
            parent_id="span-014-root",
            name="fetch_git_commits",
            status=TraceStatus(status_code="OK", description=None),
            attributes={"repo": "main", "commits_count": 45},
            events=[],
            links=[],
            start_time=now - 9600,
            end_time=now - 9400,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
        Span(
            rollout_id="ro-story-006",
            attempt_id="at-story-014",
            sequence_id=3,
            trace_id="trace-006-release",
            span_id="span-016-format",
            parent_id="span-014-root",
            name="format_changelog",
            status=TraceStatus(status_code="ERROR", description="Cancelled during formatting"),
            attributes={"format": "markdown", "cancelled_at": "50%"},
            events=[],
            links=[],
            start_time=now - 9300,
            end_time=now - 9000,
            context=None,
            parent=None,
            resource=OtelResource(attributes={}, schema_url=""),
        ),
    ]

    await store.collections.spans.insert(spans_ro1 + spans_ro2_a1 + spans_ro2_a2 + spans_ro3_a3 + spans_ro5 + spans_ro6)

    # Create and inject resources with diverse types
    resource1 = ResourcesUpdate(
        resources_id="rs-story-001",
        version=1,
        create_time=now - 86400,
        update_time=now - 86400,
        resources={
            "model": LLM(
                endpoint="https://api.openai.com/v1",
                model="gpt-4",
                sampling_parameters={"temperature": 0.7, "max_tokens": 1000},
            ),
        },
    )

    resource2 = ResourcesUpdate(
        resources_id="rs-story-002",
        version=1,
        create_time=now - 86400,
        update_time=now - 43200,
        resources={
            "prompt_main": PromptTemplate(
                template="Classify the following ticket: {ticket}\nCategory:",
                engine="f-string",
            ),
        },
    )

    resource3 = ResourcesUpdate(
        resources_id="rs-story-003",
        version=2,
        create_time=now - 86400,
        update_time=now - 3600,
        resources={
            "model": LLM(
                endpoint="https://api.anthropic.com/v1",
                model="claude-3-sonnet",
                sampling_parameters={"temperature": 0.8, "max_tokens": 2048},
            ),
            "prompt": PromptTemplate(
                template="Analyze experiment results:\n{% for result in results %}\n- {{ result }}\n{% endfor %}",
                engine="jinja",
            ),
        },
    )

    resource4 = ResourcesUpdate(
        resources_id="rs-story-004",
        version=1,
        create_time=now - 7200,
        update_time=now - 7200,
        resources={
            "system_prompt": PromptTemplate(
                template="You are a helpful assistant that generates quick, concise answers.",
                engine="f-string",
            ),
            "model": LLM(
                endpoint="http://localhost:8000/v1",
                model="llama-3-70b",
                sampling_parameters={"temperature": 0.6, "top_p": 0.95},
            ),
        },
    )

    resource5 = ResourcesUpdate(
        resources_id="rs-story-005",
        version=1,
        create_time=now - 14400,
        update_time=now - 14400,
        resources={
            "prompt": PromptTemplate(
                template="## Release Notes v{{ version }}\n\n### Changes\n{{ changes }}",
                engine="jinja",
            ),
        },
    )

    await store.collections.resources.insert([resource1, resource2, resource3, resource4, resource5])
    store._latest_resources_id = "rs-story-005"

    # Register workers with diverse states and activity windows.
    workers = [
        Worker(
            worker_id="worker-east",
            status="busy",
            heartbeat_stats={"queue_depth": 2, "gpu_utilization": 0.82},
            last_heartbeat_time=now - 20,
            last_dequeue_time=now - 60,
            last_busy_time=now - 120,
            last_idle_time=now - 600,
            current_rollout_id="ro-story-001",
            current_attempt_id="at-story-010",
        ),
        Worker(
            worker_id="worker-north",
            status="idle",
            heartbeat_stats={"queue_depth": 0, "gpu_utilization": 0.15},
            last_heartbeat_time=now - 90,
            last_dequeue_time=now - 3600,
            last_busy_time=now - 5400,
            last_idle_time=now - 5400,
            current_rollout_id=None,
            current_attempt_id=None,
        ),
        Worker(
            worker_id="worker-west",
            status="busy",
            heartbeat_stats={"queue_depth": 1, "gpu_utilization": 0.41},
            last_heartbeat_time=now - 45,
            last_dequeue_time=now - 300,
            last_busy_time=now - 200,
            last_idle_time=now - 4800,
            current_rollout_id="ro-story-003",
            current_attempt_id="at-story-033",
        ),
        Worker(
            worker_id="worker-south",
            status="idle",
            heartbeat_stats={"queue_depth": 0},
            last_heartbeat_time=now - 900,
            last_dequeue_time=now - 7200,
            last_busy_time=now - 8600,
            last_idle_time=now - 8600,
            current_rollout_id=None,
            current_attempt_id=None,
        ),
        Worker(
            worker_id="worker-observer",
            status="unknown",
            heartbeat_stats={"queue_depth": 0},
            last_heartbeat_time=now - 15,
            last_dequeue_time=now - 4000,
            last_busy_time=None,
            last_idle_time=None,
            current_rollout_id=None,
            current_attempt_id=None,
        ),
    ]

    await store.collections.workers.insert(workers)


async def main():
    parser = argparse.ArgumentParser(description="Run a Python server for the LightningStore")
    parser.add_argument("--now", type=float, default=time.time(), help="The current timestamp")
    args = parser.parse_args()

    store = InMemoryLightningStore()
    await inject_mock_data(store, now=args.now)

    # Start server
    server = LightningStoreServer(store, "127.0.0.1", 8765, "*")
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
