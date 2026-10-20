# Copyright (c) Microsoft. All rights reserved.

import json
from functools import wraps
from pathlib import Path

import torch


def quantize_weights(weights):
    result = []
    for name, weight in weights:
        if name.startswith("model.visual."):
            continue
        if ".layers." in name and name.endswith(".weight") and weight.ndim == 2:
            values = weight.float()
            scale = (values.abs().amax(dim=1, keepdim=True) / 127).clamp_min(torch.finfo(torch.float32).eps)
            result.append((name, (values / scale).round().clamp(-127, 127).to(torch.int8)))
            result.append((name.removesuffix(".weight") + ".weight_scale", scale))
        else:
            result.append((name, weight))
    return result


def apply_int8_rollout_patches():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    original = vLLMHttpServer._apply_quantization
    if getattr(original, "_agl_int8", False):
        return

    @wraps(original)
    def apply_quantization(self):
        if self.config.quantization != "int8":
            return original(self)
        config = json.loads(Path(self.config.quantization_config_file).read_text())["quantization_config"]
        return "compressed-tensors", {"quantization_config": config}

    preprocess = vLLMHttpServer._preprocess_engine_kwargs

    def preprocess_engine_kwargs(self, engine_kwargs):
        preprocess(self, engine_kwargs)
        if self.config.quantization == "int8":
            sampling = engine_kwargs.get("worker_extension_cls") == (
                "agentlightning.verl.sc.vllm_score_centering.ScoreCenteringWorkerExtension"
            )
            name = "Int8ScoreCenteringWorkerExtension" if sampling else "Int8ColocateWorkerExtension"
            engine_kwargs["worker_extension_cls"] = f"agentlightning.verl.sc.int8_worker.{name}"

    apply_quantization._agl_int8 = True  # pyright: ignore[reportAttributeAccessIssue]
    vLLMHttpServer._apply_quantization = apply_quantization
    vLLMHttpServer._preprocess_engine_kwargs = preprocess_engine_kwargs
