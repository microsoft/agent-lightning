# Copyright (c) Microsoft. All rights reserved.

from functools import wraps
from itertools import pairwise
from typing import Any, cast

import torch


def apply_qwen35_packing_patch():
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    except ImportError:
        return
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    # VERL's registry decorator erases the concrete engine type to BaseEngine.
    engine_cls = cast(Any, FSDPEngineWithLMHead)
    original_linear = Qwen3_5GatedDeltaNet.forward
    if getattr(original_linear, "_agl_packed_sequences", False):
        return
    original_inputs = engine_cls.prepare_model_inputs

    @wraps(original_inputs)
    def prepare_model_inputs(self, micro_batch):
        inputs, output_args = original_inputs(self, micro_batch)
        if self.model_config.hf_config.model_type == "qwen3_5" and tu.get(micro_batch, "use_remove_padding", True):
            if self.use_ulysses_sp:
                raise ValueError("Qwen3.5 packed training currently requires sequence parallel size 1")
            offsets = micro_batch["input_ids"].offsets().to(torch.int32)
            inputs["cu_seq_lens_q"] = offsets
            inputs["cu_seq_lens_k"] = offsets
            max_length = int(offsets.diff().max())
            inputs["max_length_q"] = max_length
            inputs["max_length_k"] = max_length
        return inputs, output_args

    @wraps(original_linear)
    def linear_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        offsets = kwargs.pop("cu_seq_lens_q", None)
        kwargs.pop("cu_seq_lens_k", None)
        kwargs.pop("max_length_q", None)
        kwargs.pop("max_length_k", None)
        if offsets is None:
            return original_linear(
                self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
            )
        boundaries = offsets.tolist()
        return torch.cat(
            [
                original_linear(self, hidden_states[:, start:stop], cache_params=None, attention_mask=None, **kwargs)
                for start, stop in pairwise(boundaries)
            ],
            dim=1,
        )

    linear_forward._agl_packed_sequences = True  # pyright: ignore[reportAttributeAccessIssue]
    Qwen3_5GatedDeltaNet.forward = linear_forward
    engine_cls.prepare_model_inputs = prepare_model_inputs
