# Copyright (c) Microsoft. All rights reserved.

import torch

from agentlightning.verl.sc.int8_rollout import quantize_weights


def test_requantizes_linear_weights_and_preserves_other_parameters():
    weight = torch.tensor([[0.0, 0.0], [-2.0, 1.0]])
    norm = torch.ones(2)
    head = torch.ones(2, 2)
    result = dict(
        quantize_weights(
            [
                ("model.language_model.layers.0.self_attn.q_proj.weight", weight),
                ("model.language_model.layers.0.input_layernorm.weight", norm),
                ("lm_head.weight", head),
                ("model.visual.blocks.0.attn.qkv.weight", head),
            ]
        )
    )
    quantized = result["model.language_model.layers.0.self_attn.q_proj.weight"]
    scale = result["model.language_model.layers.0.self_attn.q_proj.weight_scale"]
    assert quantized.dtype == torch.int8
    assert torch.isfinite(scale).all() and (scale > 0).all()
    torch.testing.assert_close(quantized.float() * scale, weight, atol=2 / 127, rtol=0)
    assert result["model.language_model.layers.0.input_layernorm.weight"] is norm
    assert result["lm_head.weight"] is head
    assert "model.visual.blocks.0.attn.qkv.weight" not in result
