# Copyright (c) Microsoft. All rights reserved.

from types import SimpleNamespace

import pytest
from agentlightning.verl.trainer import _compute_reference_log_prob
class DummyWorker:
    """Mock worker for testing."""

    def __init__(self, name: str, result: str) -> None:
        self.name = name
        self.result = result
        self.calls: list[str] = []

    def compute_ref_log_prob(self, batch: str) -> str:
        """Simulate compute_ref_log_prob method."""
        self.calls.append(batch)
        return f"{self.result}:{batch}"


def make_trainer(**overrides: object) -> SimpleNamespace:
    """Create a mock trainer with specified attributes."""
    return SimpleNamespace(**overrides)


def test_compute_reference_log_prob_prefers_actor_worker_when_ref_in_actor_true() -> None:
    """Test LoRA scenario: ref_in_actor=True should use actor_rollout_wg (issue #383)."""
    actor = DummyWorker("actor", "actor-ref")
    ref = DummyWorker("ref", "ref-ref")
    trainer = make_trainer(
        ref_in_actor=True,
        actor_rollout_wg=actor,
        ref_policy_wg=ref,  # Even if ref_policy_wg exists, actor should be used
    )

    result = _compute_reference_log_prob(trainer, "batch")

    assert result == "actor-ref:batch"
    assert actor.calls == ["batch"], "actor_rollout_wg should be called when ref_in_actor=True"
    assert ref.calls == [], "ref_policy_wg should NOT be called when ref_in_actor=True"


def test_compute_reference_log_prob_uses_ref_worker_when_ref_in_actor_false() -> None:
    """Test standard scenario: ref_in_actor=False should use ref_policy_wg."""
    ref = DummyWorker("ref", "ref-ref")
    trainer = make_trainer(
        ref_in_actor=False,
        ref_policy_wg=ref,
    )

    result = _compute_reference_log_prob(trainer, "batch")

    assert result == "ref-ref:batch"
    assert ref.calls == ["batch"], "ref_policy_wg should be called when ref_in_actor=False"


def test_compute_reference_log_prob_raises_when_ref_worker_missing() -> None:
    """Test error handling: missing ref_policy_wg when ref_in_actor=False."""
    trainer = make_trainer(
        ref_in_actor=False,
        ref_policy_wg=None,
    )

    with pytest.raises(RuntimeError, match="Reference policy worker was not initialized"):
        _compute_reference_log_prob(trainer, "batch")


def test_compute_reference_log_prob_raises_when_actor_worker_missing_with_ref_in_actor() -> None:
    """Test error handling: missing actor_rollout_wg when ref_in_actor=True."""
    trainer = make_trainer(
        ref_in_actor=True,
        actor_rollout_wg=None,  # Missing actor worker
    )

    with pytest.raises(RuntimeError, match="actor_rollout_wg is required when ref_in_actor is True"):
        _compute_reference_log_prob(trainer, "batch")


def test_compute_reference_log_prob_handles_ref_in_actor_attribute_missing() -> None:
    """Test backward compatibility: missing ref_in_actor attribute defaults to False."""
    ref = DummyWorker("ref", "ref-ref")
    trainer = make_trainer(
        # ref_in_actor attribute is missing (simulates older verl versions)
        ref_policy_wg=ref,
    )

    result = _compute_reference_log_prob(trainer, "batch")

    assert result == "ref-ref:batch"
    assert ref.calls == ["batch"], "Should fall back to ref_policy_wg when ref_in_actor is missing"


def test_compute_reference_log_prob_handles_ref_in_actor_explicitly_false() -> None:
    """Test explicit ref_in_actor=False (not just missing)."""
    ref = DummyWorker("ref", "ref-ref")
    trainer = make_trainer(
        ref_in_actor=False,  # Explicitly False
        ref_policy_wg=ref,
    )

    result = _compute_reference_log_prob(trainer, "batch")

    assert result == "ref-ref:batch"
    assert ref.calls == ["batch"]


def test_compute_reference_log_prob_preserves_batch_data() -> None:
    """Test that batch data is passed correctly to worker methods."""
    batch_data = {"tokens": [1, 2, 3], "mask": [1, 1, 1]}
    actor = DummyWorker("actor", "result")

    trainer = make_trainer(
        ref_in_actor=True,
        actor_rollout_wg=actor,
    )

    result = _compute_reference_log_prob(trainer, batch_data)

    assert actor.calls == [batch_data], "Batch should be passed unchanged to worker"
    assert result == f"result:{batch_data}"


def test_compute_reference_log_prob_handles_multiple_calls() -> None:
    """Test that multiple calls work correctly (no state leakage)."""
    actor = DummyWorker("actor", "result")
    trainer = make_trainer(
        ref_in_actor=True,
        actor_rollout_wg=actor,
    )

    result1 = _compute_reference_log_prob(trainer, "batch1")
    result2 = _compute_reference_log_prob(trainer, "batch2")

    assert result1 == "result:batch1"
    assert result2 == "result:batch2"
    assert actor.calls == ["batch1", "batch2"], "Both calls should be recorded"
