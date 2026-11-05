# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from verl.utils import tensordict_utils as tu

from agentlightning.credit.types import ProcessCreditConfig
from agentlightning.verl.process_batch import pad_process_batch
from agentlightning.verl.process_credit import ScheduledGroups, ScheduledRollout
from agentlightning.verl.process_loss import build_process_actor_payload
from agentlightning.verl.rollout_adapter import RolloutAdapter


def adapter_for(config, groups):
    return RolloutAdapter(
        max_prompt_length=20,
        max_response_length=20,
        pad_token_id=0,
        device=torch.device("cpu"),
        tokenizer=SimpleNamespace(eos_token_id=2),
        trace_aggregator_level="transition",
        process_credit_config=config,
        process_scheduled_groups=groups,
    )


def scheduled(ids, size=1):
    return ScheduledGroups(
        rollouts=tuple(
            ScheduledRollout(rollout_id=rid, group_id="task", group_size=size, policy_version=7) for rid in ids
        )
    )


@pytest.mark.asyncio
async def test_complete_batch_padding_and_native_actor_payload(completed_process_rollout, cpu_native_padding):
    config = ProcessCreditConfig(enabled=True, gamma=0.5)
    batch, _ = adapter_for(config, scheduled(("r",))).get_train_data_batch(
        [await completed_process_rollout(config=config)]
    )
    original_mass = batch.batch["advantages"].sum().item()
    padded = pad_process_batch(batch, dp_size=2)
    assert len(batch) == 3
    assert len(padded) == 4
    assert padded.meta_info["process_scheduled_count"] == 1
    assert padded.non_tensor_batch["process_padding"].tolist() == [False, False, False, True]
    assert padded.non_tensor_batch["process_token_origins"][-1] == []
    assert padded.batch["response_mask"][-1].sum() == 0
    assert padded.batch["attention_mask"][-1].sum() > 0
    assert padded.batch["advantages"].sum().item() == original_mass
    padded.batch["old_log_probs"] = padded.batch["rollout_log_probs"].clone()
    payload = build_process_actor_payload(padded, seed=123)
    assert payload["input_ids"].is_nested
    assert tu.get(payload, "mini_batch_size") == tu.get(payload, "global_batch_size") == 4
    assert tu.get(payload, "epochs") == 1
    assert tu.get(payload, "dataloader_kwargs") == {"shuffle": False}
    assert tu.get(payload, "compute_loss") is True
    assert tu.get(payload, "calculate_entropy") is False


@pytest.mark.asyncio
async def test_all_zero_action_collection_keeps_scheduling_in_audit(completed_process_rollout):
    config = ProcessCreditConfig(enabled=True, gamma=0.5)
    rollout = await completed_process_rollout(config=config, action_count=0, rewards=(0,))
    batch, metrics = adapter_for(config, scheduled(("r",))).get_train_data_batch([rollout])
    assert batch.batch is None
    assert batch.meta_info["process_zero_action"] is True
    assert metrics["process/zero_action_rollouts"] == 1
    assert len(rollout.process_credit.scheduled_groups) == 1


def test_dropout_and_checkpoint_configuration_are_explicit(tmp_path):
    from agentlightning.verl.process_credit import (
        read_process_checkpoint,
        validate_model_dropout,
        write_process_checkpoint,
    )

    config = ProcessCreditConfig(enabled=True)
    validate_model_dropout({"attention_dropout": 0, "text_config": {"hidden_dropout_prob": 0}})
    with pytest.raises(ValueError, match="dropout"):
        validate_model_dropout({"text_config": {"attention_dropout": 0.1}})
    path = tmp_path / "process_credit.json"
    metadata = {"policy_version": 7, "snapshot_hashes": ["a" * 64], "scheduled_count": 1}
    write_process_checkpoint(path, config, metadata)
    assert read_process_checkpoint(path, config) == metadata
    with pytest.raises(ValueError, match="configuration"):
        read_process_checkpoint(path, ProcessCreditConfig(enabled=True, gamma=0.5))


@pytest.mark.asyncio
async def test_native_trainer_bypasses_outcome_drops_and_updates_actor_once(
    completed_process_rollout, cpu_native_padding, tmp_path, monkeypatch
):
    import numpy as np
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from verl import DataProto

    from agentlightning.verl.trainer import AgentLightningRayPPOTrainer

    process = ProcessCreditConfig(enabled=True, gamma=0.5)
    rollout = await completed_process_rollout(config=process)
    batch, _ = adapter_for(process, scheduled(("r",))).get_train_data_batch([rollout])
    trainer = cast(Any, object.__new__(AgentLightningRayPPOTrainer))
    trainer._process_credit_config = process
    trainer.global_steps = 7
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 128, "data_loader_seed": 42},
                "rollout": {"n": 1, "temperature": 1, "multi_turn": {"enable": False}},
            },
            "algorithm": {"adv_estimator": "process_reinforce"},
            "agentlightning": {"max_ppo_update_times": None},
            "trainer": {"balance_batch": False, "default_local_dir": str(tmp_path)},
        }
    )
    trainer._next_train_batch_dict_for_rollout = lambda: {"data": np.array(["task"], dtype=object)}
    trainer._get_gen_batch = lambda value: value
    trainer._rollout = lambda *args, **kwargs: (batch, {"training/reward": 1})
    trainer._process_completed_rollouts = [rollout]
    trainer._process_rollout_manager = SimpleNamespace(_delete_rollout=lambda rid: None)
    trainer._compute_old_log_prob = lambda value: (
        DataProto(batch=TensorDict({"old_log_probs": value.batch["rollout_log_probs"].clone()}, batch_size=len(value))),
        0,
    )
    calls = []

    def update(payload):
        calls.append(payload)
        assert tu.get(payload, "mini_batch_size") == 4
        output = TensorDict({}, batch_size=[])
        tu.assign_non_tensor(output, metrics={"mfu": 0.0, "process_updates": 1.0})
        return output

    trainer.actor_rollout_wg = SimpleNamespace(world_size=2, update_actor=update)
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None, update_weights=lambda step: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("process mode must not call outcome selection/normalization")

    monkeypatch.setattr("agentlightning.verl.trainer._same_reward_uid_indices", forbidden)
    monkeypatch.setattr("agentlightning.verl.trainer.normalize_advantages_by_rollout", forbidden)
    result = trainer._train_step({}, False)
    assert result is not None
    metrics, trained = result
    assert len(calls) == 1
    assert metrics["process/actor_updates"] == 1
    assert metrics["training/n_sample_trained"] == 3
    assert len(trained) == 4
    assert len(list((tmp_path / "process_credit").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_zero_action_trainer_persists_and_skips_every_actor_operation(completed_process_rollout, tmp_path):
    from omegaconf import OmegaConf

    from agentlightning.verl.trainer import AgentLightningRayPPOTrainer

    config = ProcessCreditConfig(enabled=True)
    rollout = await completed_process_rollout(config=config, action_count=0, rewards=(1,))
    batch, _ = adapter_for(config, scheduled(("r",))).get_train_data_batch([rollout])
    trainer = cast(Any, object.__new__(AgentLightningRayPPOTrainer))
    trainer._process_credit_config = config
    trainer._process_completed_rollouts = [rollout]
    trainer.config = OmegaConf.create({"trainer": {"default_local_dir": str(tmp_path)}})
    trainer.global_steps = 7
    trainer._next_train_batch_dict_for_rollout = lambda: {}
    trainer._get_gen_batch = lambda value: value
    trainer._rollout = lambda *args, **kwargs: (batch, {"training/reward": 1})
    deleted = []
    trainer._process_rollout_manager = SimpleNamespace(_delete_rollout=deleted.append)
    # No actor/checkpoint manager is supplied: touching either would fail this native branch.
    assert trainer._train_step({}, False) is None
    assert trainer._process_skip_metrics["process/actor_updates"] == 0
    assert trainer._process_skip_metrics["process/zero_action_collection"] == 1
    assert deleted == ["r"]
    assert len(list((tmp_path / "process_credit").glob("*.json"))) == 1
