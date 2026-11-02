# Copyright (c) Microsoft. All rights reserved.

from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("torch")
pytest.importorskip("verl")

import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from agentlightning.verl.sc.qwen35_packing import apply_qwen35_packing_patch


def tiny_config():
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3.5 kernels require CUDA")
def test_packed_linear_attention_matches_independent_outputs_and_gradients(monkeypatch):
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    # VERL's registry decorator erases the concrete engine type.
    engine_cls = cast(Any, FSDPEngineWithLMHead)

    original = Qwen3_5GatedDeltaNet.forward
    monkeypatch.setattr(Qwen3_5GatedDeltaNet, "forward", original)
    monkeypatch.setattr(engine_cls, "prepare_model_inputs", engine_cls.prepare_model_inputs)
    torch.manual_seed(41)
    # Transformers annotates this layer with the multimodal config, but consumes text config fields.
    layer = Qwen3_5GatedDeltaNet(cast(Any, tiny_config()), 0).to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    separate = torch.cat([layer(inputs[:, :7]), layer(inputs[:, 7:11]), layer(inputs[:, 11:])], dim=1)
    unsafe = layer(inputs)
    assert (unsafe - separate).abs().max() > 1e-4
    apply_qwen35_packing_patch()
    packed = layer(inputs, cu_seq_lens_q=torch.tensor([0, 7, 11, 16], dtype=torch.int32))
    torch.testing.assert_close(packed, separate)
    parameters = (inputs, *layer.parameters())
    expected = torch.autograd.grad(separate.square().sum(), parameters, retain_graph=True)
    actual = torch.autograd.grad(packed.square().sum(), parameters)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)


def test_engine_passes_exact_packed_offsets_only_for_qwen35(monkeypatch):
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    # VERL's registry decorator erases the concrete engine type.
    engine_cls = cast(Any, FSDPEngineWithLMHead)

    original = Qwen3_5GatedDeltaNet.forward
    monkeypatch.setattr(Qwen3_5GatedDeltaNet, "forward", getattr(original, "__wrapped__", original))
    monkeypatch.setattr(engine_cls, "prepare_model_inputs", lambda self, data: ({}, {}))
    apply_qwen35_packing_patch()
    from tensordict import TensorDict

    tokens = torch.nested.nested_tensor([torch.arange(3), torch.arange(5)], layout=torch.jagged)
    data = TensorDict({"input_ids": tokens}, batch_size=[2])
    tu.assign_non_tensor(data, use_remove_padding=True)
    engine = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5")), use_ulysses_sp=False
    )
    inputs, _ = engine_cls.prepare_model_inputs(engine, data)
    assert inputs["cu_seq_lens_q"].tolist() == [0, 3, 8]
    assert inputs["cu_seq_lens_k"].dtype == torch.int32
    tu.assign_non_tensor(data, use_remove_padding=False)
    assert engine_cls.prepare_model_inputs(engine, data)[0] == {}
