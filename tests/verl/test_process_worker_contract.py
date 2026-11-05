# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel
from verl.workers.config import ActorConfig

from agentlightning.credit.types import ProcessCreditConfig
from agentlightning.verl.process_batch import pad_process_batch
from agentlightning.verl.process_credit import ScheduledGroups, ScheduledRollout
from agentlightning.verl.process_loss import build_process_actor_payload
from agentlightning.verl.rollout_adapter import RolloutAdapter


class _ScalarLogProbability(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))

    def forward(self, template):
        return self.value.expand(template.shape)


def _distributed_loss_worker(rank, world_size, rendezvous, output_dir, coefficients, masks, expected_gradient, uneven):
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    import agentlightning.verl.process_loss  # noqa: F401 - actual registry registration in this process

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=25)
    )
    try:
        policy = DistributedDataParallel(_ScalarLogProbability())
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        optimizer.zero_grad()
        local = coefficients.chunk(world_size)[rank]
        local_mask = masks.chunk(world_size)[rank]
        config = ActorConfig(
            strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, global_batch_info={"dp_size": world_size}
        )
        loss_fn = get_policy_loss_fn("process_reinforce")
        split_sizes = [1, len(local) - 1] if uneven and rank == 0 and len(local) > 1 else [len(local)]
        offset = 0
        for index, size in enumerate(split_sizes):
            context = policy.no_sync() if index < len(split_sizes) - 1 else __import__("contextlib").nullcontext()
            with context:
                coef = local[offset : offset + size]
                mask = local_mask[offset : offset + size]
                log_prob = policy(coef)
                loss, _ = loss_fn(torch.zeros_like(log_prob), log_prob, coef, mask, "token-mean", config, None)
                loss.backward()
            offset += size
        gradient = policy.module.value.grad.item()
        assert gradient == pytest.approx(expected_gradient, rel=1e-10, abs=1e-10)
        optimizer.step()
        assert policy.module.value.item() == pytest.approx(-0.2 - 0.1 * expected_gradient, rel=1e-10, abs=1e-10)
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps({"gradient": gradient, "optimizer_steps": 1}))
    finally:
        dist.destroy_process_group()


@pytest.mark.asyncio
@pytest.mark.parametrize("action_count,uneven", [(3, True), (1, False)])
async def test_actual_two_rank_gloo_matches_global_gradient(completed_process_rollout, tmp_path, action_count, uneven):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("actual two-rank CPU normalization requires torch.distributed Gloo")
    config = ProcessCreditConfig(enabled=True, gamma=0.5)
    rollout = await completed_process_rollout(
        config=config, action_count=action_count, rewards=(0.2, -0.1, 1) if action_count == 3 else (1,)
    )
    scheduled = ScheduledGroups(
        rollouts=(ScheduledRollout(rollout_id="r", group_id="task", group_size=1, policy_version=7),)
    )
    adapter = RolloutAdapter(
        max_prompt_length=20,
        max_response_length=20,
        device=torch.device("cpu"),
        pad_token_id=0,
        tokenizer=SimpleNamespace(eos_token_id=2),
        trace_aggregator_level="transition",
        process_credit_config=config,
        process_scheduled_groups=scheduled,
    )
    batch, _ = adapter.get_train_data_batch([rollout])
    padded = pad_process_batch(batch, dp_size=2)
    coefficients = padded.batch["advantages"].double()
    masks = padded.batch["response_mask"]
    expected = -float((coefficients * masks).sum())
    spawn(
        _distributed_loss_worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path), coefficients, masks, expected, uneven),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result["optimizer_steps"] == 1
        assert result["gradient"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_real_native_ppo_loss_and_iterator_have_one_complete_minibatch(
    completed_process_rollout, cpu_native_padding
):
    from verl.utils import tensordict_utils as tu
    from verl.workers.config.actor import PolicyLossConfig
    from verl.workers.utils.losses import ppo_loss

    config = ProcessCreditConfig(enabled=True, gamma=0.5)
    rollout = await completed_process_rollout(config=config)
    scheduled = ScheduledGroups(
        rollouts=(ScheduledRollout(rollout_id="r", group_id="task", group_size=1, policy_version=7),)
    )
    adapter = RolloutAdapter(
        max_prompt_length=20,
        max_response_length=20,
        device=torch.device("cpu"),
        pad_token_id=0,
        tokenizer=SimpleNamespace(eos_token_id=2),
        trace_aggregator_level="transition",
        process_credit_config=config,
        process_scheduled_groups=scheduled,
    )
    batch, _ = adapter.get_train_data_batch([rollout])
    batch = pad_process_batch(batch, dp_size=2)
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"].clone()
    payload = build_process_actor_payload(batch, seed=12)
    minibatches = list(
        tu.make_iterator(payload, mini_batch_size=4, epochs=1, seed=12, dataloader_kwargs={"shuffle": False})
    )
    assert len(minibatches) == 1
    actor_config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="process_reinforce"),
    )
    parameter = torch.tensor(-0.2, requires_grad=True)
    losses = []
    # Unequal CPU microbatches execute the real native ppo_loss and registered function.
    for microbatch in (
        tu.index_select_tensor_dict(minibatches[0], [0]),
        tu.index_select_tensor_dict(minibatches[0], [1, 2, 3]),
    ):
        tu.assign_non_tensor(microbatch, dp_size=1, global_batch_size=4, batch_num_tokens=None)
        count = cast(torch.Tensor, microbatch["input_ids"]).values().numel()
        losses.append(ppo_loss(actor_config, {"log_probs": parameter.expand(count)}, microbatch)[0])
    torch.stack(losses).sum().backward()
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(-batch.batch["advantages"].sum().item())
