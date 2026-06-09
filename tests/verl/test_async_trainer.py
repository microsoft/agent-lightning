"""Focused tests for async-rollout trainer glue."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from agl_lite.verl.trainer import AglLiteRayPPOTrainer


class _FakeBridge:
    def __init__(self) -> None:
        self.run_kwargs: dict[str, Any] | None = None

    def async_set_up_data_and_server(self, **kwargs: Any) -> None:
        pass

    def run_until_groups_finished(self, **kwargs: Any):
        self.run_kwargs = kwargs
        return {"r1"}, set(), {"training/async/n_selected_groups": 1}

    def commit_async_step_selection(self, **kwargs: Any) -> dict[str, Any]:
        return {"training/async/n_carry_over_out": 0}

    def async_get_train_data_batch(self, **kwargs: Any):
        batch = DataProto.from_single_dict({"input_ids": torch.zeros((1, 1), dtype=torch.long)})
        batch.batch["attention_mask"] = torch.ones((1, 1), dtype=torch.long)
        batch.batch["responses"] = torch.zeros((1, 1), dtype=torch.long)
        batch.batch["token_level_scores"] = torch.zeros((1, 1), dtype=torch.float32)
        batch.non_tensor_batch["data_id_list"] = ["d1"]
        return batch, {"training/n_placeholder_rows": 0}

class _RecordingBridge(_FakeBridge):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def async_set_up_data_and_server(self, **kwargs: Any) -> None:
        self.events.append("setup")


class _FakeLogger:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.logged: list[tuple[dict[str, Any], int]] = []

    def log(self, *, data: dict[str, Any], step: int) -> None:
        self.logged.append((data, step))


class _FakeProgressBar:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False

    def update(self, _: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeAdminClient:
    resumes = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeAdminClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def resume_gateway(self) -> dict[str, Any]:
        type(self).resumes += 1
        return {"paused": False, "retry_after_seconds": 5, "reason": None, "inflight": 0}


class _CarryOverBridge:
    """Small bridge double that preserves the async carry-over bookkeeping shape."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._carry_over_rids: set[str] = set()
        self._rid_to_data_id: dict[str, str] = {}
        self._data_id_to_rids: dict[str, set[str]] = {}
        self._enqueue_order: list[str] = []
        self._completed_rollouts: dict[str, object] = {}
        self._task_id_to_original_sample: dict[str, dict[str, str]] = {}
        self._rollout_status: dict[str, str] = {}
        self._rollout_error: dict[str, str] = {}
        self._rollout_start_time: dict[str, float] = {}
        self._rollout_end_time: dict[str, float] = {}
        self._raw_events_by_rollout: dict[str, list[dict[str, str]]] = {}
        self._triplet_events_by_rollout: dict[str, list[dict[str, str]]] = {}
        self._carry_over_birth_step: dict[str, int] = {}
        self._step_new_rids: set[str] = set()
        self._selected_rids: set[str] = set()
        self._group_finish_time: dict[str, float] = {}
        self.clear_calls = 0

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)
        self._loop.close()

    def n_carry_over_data_ids(self) -> int:
        return len({self._rid_to_data_id[r] for r in self._carry_over_rids if r in self._rid_to_data_id})

    def leave_one_carry_over(self) -> None:
        rollout = object()
        self._carry_over_rids = {"carry-r1"}
        self._rid_to_data_id = {"carry-r1": "carry-d1"}
        self._data_id_to_rids = {"carry-d1": {"carry-r1"}}
        self._enqueue_order = ["carry-r1"]
        self._completed_rollouts = {"carry-r1": rollout}
        self._task_id_to_original_sample = {"carry-r1": {"data_id": "carry-d1"}}
        self._rollout_status = {"carry-r1": "succeeded"}
        self._rollout_error = {"carry-r1": "kept-error"}
        self._rollout_start_time = {"carry-r1": 123.0}
        self._rollout_end_time = {"carry-r1": 456.0}
        self._raw_events_by_rollout = {"carry-r1": [{"event_type": "model_request"}]}
        self._triplet_events_by_rollout = {"carry-r1": [{"event_type": "triplet"}]}
        self._carry_over_birth_step = {"carry-r1": 7}
        self._step_new_rids = {"carry-r1"}
        self._selected_rids = {"carry-r1"}
        self._group_finish_time = {"carry-d1": 789.0}

    def saturate_carry_over(self) -> None:
        self._carry_over_rids = {"carry-r1", "carry-r2"}
        self._rid_to_data_id = {"carry-r1": "carry-d1", "carry-r2": "carry-d2"}
        self._data_id_to_rids = {"carry-d1": {"carry-r1"}, "carry-d2": {"carry-r2"}}

    def clear_data_and_server(self) -> None:
        # Mirrors the production bridge's legacy clear shape: it clears general
        # rollout bookkeeping but is not carry-over aware.
        self.clear_calls += 1
        self._enqueue_order.clear()
        self._completed_rollouts.clear()
        self._task_id_to_original_sample.clear()
        self._rollout_status.clear()
        self._rollout_error.clear()
        self._rollout_start_time.clear()
        self._rollout_end_time.clear()
        self._raw_events_by_rollout.clear()
        self._triplet_events_by_rollout.clear()
        self._carry_over_birth_step.clear()
        self._step_new_rids.clear()
        self._selected_rids.clear()


class _RestartableLoader:
    def __iter__(self) -> Iterator[dict[str, list[str]]]:
        yield {"prompt": ["p0", "p1"]}


def test_async_rollout_waits_for_train_batch_size_groups_not_active_pool() -> None:
    trainer = object.__new__(AglLiteRayPPOTrainer)
    bridge = _FakeBridge()
    trainer._rollout_bridge = cast(Any, bridge)
    trainer.async_rollout_manager = SimpleNamespace(server_addresses=["http://vllm:8000/v1"])
    trainer._resume_all_rollout_generation = lambda: None
    trainer.global_steps = 3
    trainer.config = OmegaConf.create(
        {
            "agentlightning": {
                "trace_aggregator": {"level": "transition"},
            },
            "data": {
                "train_batch_size": 2,
                "max_prompt_length": 128,
                "max_response_length": 64,
            },
            "actor_rollout_ref": {
                "rollout": {"n": 4},
                "actor": {"ppo_mini_batch_size": 2},
            },
        }
    )

    trainer._async_rollout(
        new_samples_dict={"prompt": ["a", "b", "c", "d", "e"]},
        async_train_batch_size=5,
        admin_base_url="http://agl",
        admin_key="admin-key",
        gateway_retry_after_seconds=9,
        gateway_drain_timeout_seconds=1.5,
        rollout_n=4,
    )

    assert bridge.run_kwargs is not None
    assert bridge.run_kwargs["target_groups"] == 2
    assert bridge.run_kwargs["rollout_n"] == 4
    assert bridge.run_kwargs["retry_after_seconds"] == 9
    assert bridge.run_kwargs["drain_timeout"] == 1.5


def test_async_rollout_resumes_generation_before_enqueueing() -> None:
    events: list[str] = []
    trainer = object.__new__(AglLiteRayPPOTrainer)
    bridge = _RecordingBridge(events)
    trainer._rollout_bridge = cast(Any, bridge)
    trainer.async_rollout_manager = SimpleNamespace(server_addresses=["http://vllm:8000/v1"])
    trainer.global_steps = 3
    trainer.config = OmegaConf.create(
        {
            "agentlightning": {
                "trace_aggregator": {"level": "transition"},
            },
            "data": {
                "train_batch_size": 2,
                "max_prompt_length": 128,
                "max_response_length": 64,
            },
            "actor_rollout_ref": {"rollout": {"n": 4}},
        }
    )

    def record_resume_generation() -> None:
        events.append("resume_generation")

    trainer._resume_all_rollout_generation = record_resume_generation

    trainer._async_rollout(
        new_samples_dict={"prompt": ["a", "b", "c", "d", "e"]},
        async_train_batch_size=5,
        admin_base_url="http://agl",
        admin_key="admin-key",
        gateway_retry_after_seconds=9,
        gateway_drain_timeout_seconds=1.5,
        rollout_n=4,
    )

    assert events[:2] == ["resume_generation", "setup"]


def test_async_fit_validation_test_freq_preserves_carry_over_bridge_state(monkeypatch) -> None:
    """Validation triggered by test_freq must not wipe async carry-over state.

    This drives async_fit far enough to hit the real test_freq validation branch:
    a training step leaves one rollout in carry-over, then validation runs. The
    carry-over rid must remain fully tracked for the next async step.
    """
    import agl_lite.client as client_module
    import agl_lite.verl.trainer as trainer_module

    bridge = _CarryOverBridge()
    trainer = object.__new__(AglLiteRayPPOTrainer)
    trainer._rollout_bridge = cast(Any, bridge)
    trainer.async_rollout_manager = SimpleNamespace(server_addresses=["http://vllm:8000/v1"])
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=lambda step: None,
        sleep_replicas=lambda: None,
    )
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    trainer.train_dataloader = _RestartableLoader()
    trainer.val_dataloader = [{"prompt": np.array(["v0"], dtype=object)}]
    trainer.total_training_steps = 1
    trainer.config = OmegaConf.create(
        {
            "agentlightning": {
                "agl_base_url": "http://agl",
                "agl_admin_key": "admin-key",
                "async_rollout": {
                    "enabled": True,
                    "async_train_batch_size": 2,
                    "gateway_retry_after_seconds": 5,
                    "gateway_drain_timeout_seconds": 1.0,
                    "allow_equal_batch_size_for_debug": False,
                },
            },
            "trainer": {
                "project_name": "unit",
                "experiment_name": "async-validation-carry-over",
                "logger": ["console", "wandb"],
                "val_before_train": False,
                "val_only": False,
                "test_freq": 1,
                "save_freq": -1,
            },
            "global_profiler": {"steps": None},
            "data": {"train_batch_size": 1},
            "actor_rollout_ref": {
                "rollout": {"n": 1, "val_kwargs": {"n": 1, "do_sample": False}},
            },
        }
    )

    monkeypatch.setattr(trainer_module, "Tracking", _FakeLogger)
    monkeypatch.setattr(trainer_module, "tqdm", _FakeProgressBar)
    monkeypatch.setattr(client_module, "AglLiteClient", _FakeAdminClient)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_ensure_rollout_bridge", lambda self: bridge)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_load_checkpoint", lambda self: None)
    monkeypatch.setattr(
        AglLiteRayPPOTrainer,
        "_async_train_step",
        lambda self, **kwargs: (bridge.leave_one_carry_over() or {"training/async/n_carry_over_out": 1}),
    )
    monkeypatch.setattr(
        AglLiteRayPPOTrainer,
        "_get_gen_batch",
        lambda self, batch: DataProto.from_single_dict({"prompt": np.array(["v0"], dtype=object)}),
    )

    def legacy_validation_rollout(self, gen_batch: DataProto, is_train: bool = False):
        assert not is_train
        bridge.clear_data_and_server()
        return DataProto(batch=None), {"val/reward": 0.0}

    monkeypatch.setattr(AglLiteRayPPOTrainer, "_rollout", legacy_validation_rollout)

    try:
        trainer.async_fit()
    finally:
        bridge.close()

    assert bridge._carry_over_rids == {"carry-r1"}
    assert bridge._rid_to_data_id == {"carry-r1": "carry-d1"}
    assert bridge._data_id_to_rids == {"carry-d1": {"carry-r1"}}
    assert bridge._enqueue_order == ["carry-r1"]
    assert set(bridge._completed_rollouts) == {"carry-r1"}
    assert bridge._task_id_to_original_sample == {"carry-r1": {"data_id": "carry-d1"}}
    assert bridge._rollout_status == {"carry-r1": "succeeded"}
    assert bridge._rollout_error == {"carry-r1": "kept-error"}
    assert bridge._rollout_start_time == {"carry-r1": 123.0}
    assert bridge._rollout_end_time == {"carry-r1": 456.0}
    assert bridge._raw_events_by_rollout == {"carry-r1": [{"event_type": "model_request"}]}
    assert bridge._triplet_events_by_rollout == {"carry-r1": [{"event_type": "triplet"}]}
    assert bridge._carry_over_birth_step == {"carry-r1": 7}
    assert bridge._step_new_rids == {"carry-r1"}
    assert bridge._selected_rids == {"carry-r1"}
    assert bridge._group_finish_time == {"carry-d1": 789.0}


def test_async_fit_validation_does_not_force_weight_sync(monkeypatch) -> None:
    """Validation should not force an extra weight sync.

    _rollout(is_train=False) may leave vLLM generation paused after
    abort_all_requests(). The async rollout path owns recovery by calling
    resume_generation() before enqueueing the next batch, so validation does not
    need an expensive update_weights() reset.
    """
    import agl_lite.client as client_module
    import agl_lite.verl.trainer as trainer_module

    bridge = _CarryOverBridge()
    trainer = object.__new__(AglLiteRayPPOTrainer)
    trainer._rollout_bridge = cast(Any, bridge)
    trainer.async_rollout_manager = SimpleNamespace(server_addresses=["http://vllm:8000/v1"])
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    trainer.train_dataloader = _RestartableLoader()
    trainer.val_dataloader = [{"prompt": np.array(["v0"], dtype=object)}]
    trainer.total_training_steps = 1

    events: list[tuple[str, int | None]] = []

    def record_update_weights(step: int) -> None:
        events.append(("update_weights", int(step)))

    def record_sleep_replicas() -> None:
        events.append(("sleep_replicas", None))

    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=record_update_weights,
        sleep_replicas=record_sleep_replicas,
    )

    trainer.config = OmegaConf.create(
        {
            "agentlightning": {
                "agl_base_url": "http://agl",
                "agl_admin_key": "admin-key",
                "async_rollout": {
                    "enabled": True,
                    "async_train_batch_size": 2,
                    "gateway_retry_after_seconds": 5,
                    "gateway_drain_timeout_seconds": 1.0,
                    "allow_equal_batch_size_for_debug": False,
                },
            },
            "trainer": {
                "project_name": "unit",
                "experiment_name": "async-val-to-train-engine-reset",
                "logger": ["console", "wandb"],
                "val_before_train": False,
                "val_only": False,
                "test_freq": 1,
                "save_freq": -1,
            },
            "global_profiler": {"steps": None},
            "data": {"train_batch_size": 1},
            "actor_rollout_ref": {
                "rollout": {"n": 1, "val_kwargs": {"n": 1, "do_sample": False}},
            },
        }
    )

    monkeypatch.setattr(trainer_module, "Tracking", _FakeLogger)
    monkeypatch.setattr(trainer_module, "tqdm", _FakeProgressBar)
    monkeypatch.setattr(client_module, "AglLiteClient", _FakeAdminClient)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_ensure_rollout_bridge", lambda self: bridge)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_load_checkpoint", lambda self: None)

    def record_async_step(self, **kwargs):
        events.append(("async_train_step", int(self.global_steps)))
        bridge.leave_one_carry_over()
        return {"training/async/n_carry_over_out": 1}

    monkeypatch.setattr(AglLiteRayPPOTrainer, "_async_train_step", record_async_step)
    monkeypatch.setattr(
        AglLiteRayPPOTrainer,
        "_get_gen_batch",
        lambda self, batch: DataProto.from_single_dict({"prompt": np.array(["v0"], dtype=object)}),
    )

    def legacy_validation_rollout(self, gen_batch: DataProto, is_train: bool = False):
        assert not is_train
        events.append(("val_rollout", int(self.global_steps)))
        bridge.clear_data_and_server()
        return DataProto(batch=None), {"val/reward": 0.0}

    monkeypatch.setattr(AglLiteRayPPOTrainer, "_rollout", legacy_validation_rollout)

    try:
        trainer.async_fit()
    finally:
        bridge.close()

    # async_fit() starts at global_steps=0 and increments to 1 before the first
    # iteration, so the train step records step=1 and the validation block fires
    # because global_steps % test_freq(=1) == 0.
    expected_prefix = [
        ("update_weights", 0),  # initial wake at top of async_fit
        ("async_train_step", 1),  # step 1 training
        ("val_rollout", 1),  # validation triggered by test_freq=1
    ]
    assert events[: len(expected_prefix)] == expected_prefix, events

    tail = events[len(expected_prefix) :]
    assert ("update_weights", 1) not in tail, events
    assert ("sleep_replicas", None) not in tail, events


def test_async_fit_skips_engine_reset_when_validation_not_triggered(monkeypatch) -> None:
    """The val→train engine reset must only run when validation actually ran.

    Steps where test_freq does not fire should leave the original train→train
    update_weights cadence untouched (a stray reset would double-pause the
    engine and double the per-step wake/sync cost).
    """
    import agl_lite.client as client_module
    import agl_lite.verl.trainer as trainer_module

    bridge = _CarryOverBridge()
    trainer = object.__new__(AglLiteRayPPOTrainer)
    trainer._rollout_bridge = cast(Any, bridge)
    trainer.async_rollout_manager = SimpleNamespace(server_addresses=["http://vllm:8000/v1"])
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    trainer.train_dataloader = _RestartableLoader()
    trainer.val_dataloader = [{"prompt": np.array(["v0"], dtype=object)}]
    trainer.total_training_steps = 1

    events: list[tuple[str, int | None]] = []

    def record_update_weights(step: int) -> None:
        events.append(("update_weights", int(step)))

    def record_sleep_replicas() -> None:
        events.append(("sleep_replicas", None))

    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=record_update_weights,
        sleep_replicas=record_sleep_replicas,
    )

    trainer.config = OmegaConf.create(
        {
            "agentlightning": {
                "agl_base_url": "http://agl",
                "agl_admin_key": "admin-key",
                "async_rollout": {
                    "enabled": True,
                    "async_train_batch_size": 2,
                    "gateway_retry_after_seconds": 5,
                    "gateway_drain_timeout_seconds": 1.0,
                    "allow_equal_batch_size_for_debug": False,
                },
            },
            "trainer": {
                "project_name": "unit",
                "experiment_name": "async-no-val-no-reset",
                "logger": ["console", "wandb"],
                "val_before_train": False,
                "val_only": False,
                # test_freq disabled: no validation, no engine reset
                "test_freq": -1,
                "save_freq": -1,
            },
            "global_profiler": {"steps": None},
            "data": {"train_batch_size": 1},
            "actor_rollout_ref": {
                "rollout": {"n": 1, "val_kwargs": {"n": 1, "do_sample": False}},
            },
        }
    )

    monkeypatch.setattr(trainer_module, "Tracking", _FakeLogger)
    monkeypatch.setattr(trainer_module, "tqdm", _FakeProgressBar)
    monkeypatch.setattr(client_module, "AglLiteClient", _FakeAdminClient)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_ensure_rollout_bridge", lambda self: bridge)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_load_checkpoint", lambda self: None)

    def record_async_step(self, **kwargs):
        events.append(("async_train_step", int(self.global_steps)))
        return {"training/async/n_carry_over_out": 0}

    monkeypatch.setattr(AglLiteRayPPOTrainer, "_async_train_step", record_async_step)

    def fail_validation(self, *args, **kwargs):
        pytest.fail("validation must not run when test_freq <= 0")

    monkeypatch.setattr(AglLiteRayPPOTrainer, "_rollout", fail_validation)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_validate_preserving_async_carry_over", fail_validation)

    try:
        trainer.async_fit()
    finally:
        bridge.close()

    # Only the initial-wake update_weights(0) at the top of async_fit; the
    # train step's update_weights would normally come from _async_train_step,
    # which is mocked out here. The key assertion: no second update_weights
    # call appears that would only be explained by the val→train engine reset.
    update_calls = [e for e in events if e[0] == "update_weights"]
    assert update_calls == [("update_weights", 0)], (
        f"unexpected update_weights calls without validation: {update_calls}"
    )

    # Same for sleep_replicas: the val→train engine reset is the only path that
    # calls it from async_fit. When validation does not fire there must be no
    # sleep_replicas call, otherwise vLLM gets paused without the matching
    # update_weights() that wakes it back up.
    sleep_calls = [e for e in events if e[0] == "sleep_replicas"]
    assert sleep_calls == [], (
        f"unexpected sleep_replicas calls without validation: {sleep_calls}"
    )


def test_async_fit_rejects_carry_over_saturation_before_sampling(monkeypatch) -> None:
    """Async rollout must fail fast instead of running a carry-over-only step."""
    import agl_lite.verl.trainer as trainer_module

    bridge = _CarryOverBridge()
    bridge.saturate_carry_over()
    trainer = object.__new__(AglLiteRayPPOTrainer)
    trainer._rollout_bridge = cast(Any, bridge)
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: None)
    trainer.train_dataloader = _RestartableLoader()
    trainer.total_training_steps = 1
    trainer.config = OmegaConf.create(
        {
            "agentlightning": {
                "agl_base_url": "http://agl",
                "agl_admin_key": "admin-key",
                "async_rollout": {
                    "enabled": True,
                    "async_train_batch_size": 2,
                    "allow_equal_batch_size_for_debug": False,
                },
            },
            "trainer": {
                "project_name": "unit",
                "experiment_name": "async-carry-over-saturation",
                "logger": ["console", "wandb"],
                "val_before_train": False,
                "test_freq": -1,
                "save_freq": -1,
            },
            "global_profiler": {"steps": None},
            "data": {"train_batch_size": 1},
        }
    )

    monkeypatch.setattr(trainer_module, "Tracking", _FakeLogger)
    monkeypatch.setattr(trainer_module, "tqdm", _FakeProgressBar)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_ensure_rollout_bridge", lambda self: bridge)
    monkeypatch.setattr(AglLiteRayPPOTrainer, "_load_checkpoint", lambda self: None)
    monkeypatch.setattr(
        AglLiteRayPPOTrainer,
        "_async_train_step",
        lambda self, **kwargs: pytest.fail("carry-over saturation should fail before training step"),
    )

    try:
        with pytest.raises(RuntimeError, match="carry-over saturated"):
            trainer.async_fit()
    finally:
        bridge.close()
