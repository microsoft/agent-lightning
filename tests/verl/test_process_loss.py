# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agentlightning.verl.process_loss import compute_policy_loss_process_reinforce


def test_native_registry_direct_loss_and_masked_nonfinite_values():
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    assert get_policy_loss_fn("process_reinforce") is compute_policy_loss_process_reinforce
    log_prob = torch.tensor([[-0.2, float("nan")], [-0.3, -0.4]], requires_grad=True)
    advantages = torch.tensor([[0.2, float("nan")], [-0.1, 0.3]], requires_grad=True)
    mask = torch.tensor([[1, 0], [1, 1]])
    loss, metrics = compute_policy_loss_process_reinforce(
        torch.zeros_like(log_prob), log_prob, advantages, mask, config=SimpleNamespace(global_batch_info={"dp_size": 2})
    )
    assert loss.item() == pytest.approx(0.26)
    loss.backward()
    torch.testing.assert_close(log_prob.grad, torch.tensor([[-0.4, 0], [0.2, -0.6]]))
    assert advantages.grad is None
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_empty_active_partition_has_differentiable_zero():
    log_prob = torch.tensor([[float("nan")]], requires_grad=True)
    loss, _ = compute_policy_loss_process_reinforce(
        log_prob, log_prob, log_prob, torch.zeros((1, 1)), config=SimpleNamespace(global_batch_info={"dp_size": 2})
    )
    loss.backward()
    assert loss.item() == 0
    assert log_prob.grad is not None
    assert log_prob.grad.item() == 0


@pytest.mark.parametrize("kind", ["active_nan", "weights", "missing_dp"])
def test_unsupported_loss_inputs_fail(kind):
    log_prob = torch.tensor([[-0.1]], requires_grad=True)
    if kind == "active_nan":
        log_prob = log_prob * float("nan")
    with pytest.raises(ValueError):
        compute_policy_loss_process_reinforce(
            torch.zeros_like(log_prob),
            log_prob,
            torch.ones_like(log_prob),
            torch.ones_like(log_prob),
            config=SimpleNamespace(global_batch_info={} if kind == "missing_dp" else {"dp_size": 1}),
            rollout_is_weights=torch.ones_like(log_prob) if kind == "weights" else None,
        )
