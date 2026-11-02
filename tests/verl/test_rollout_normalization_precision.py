# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("verl")

import torch

from agentlightning.verl.per_rollout_loss import normalize_advantages_by_rollout


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_normalization_preserves_long_rollout_mass(dtype: torch.dtype) -> None:
    mask = torch.ones(2, 1024, dtype=torch.long)
    advantages = torch.ones(2, 1024, dtype=dtype)

    scaled = normalize_advantages_by_rollout(advantages, mask, ["A", "A"], num_trained_rows=64)

    assert scaled.dtype == dtype
    assert scaled.device == advantages.device
    assert scaled.double().sum().item() == pytest.approx(1 / 64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_normalization_counts_tokens_before_dtype_conversion(dtype: torch.dtype) -> None:
    mask = torch.ones(2, 257, dtype=torch.long)
    advantages = torch.tensor([1.0, -1.0], dtype=dtype).unsqueeze(-1).expand_as(mask).clone()
    advantages.requires_grad_()

    scaled = normalize_advantages_by_rollout(advantages, mask, ["A", "A"], num_trained_rows=1)

    expected = (advantages.detach().double() / 514).to(dtype)
    torch.testing.assert_close(scaled, expected, rtol=0, atol=0)
    scaled.sum().backward()
    assert advantages.grad is not None
    torch.testing.assert_close(advantages.grad, torch.full_like(advantages, 1 / 514), rtol=0, atol=0)
