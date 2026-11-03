# Copyright (c) Microsoft. All rights reserved.

from typing import Any

from torch.utils._pytree import tree_leaves
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

# vLLM is supplied by the inference environment and is absent from CPU CI.
from vllm.model_executor.parameter import (  # pyright: ignore[reportMissingImports]
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)

from agentlightning.verl.sc.int8_rollout import quantize_weights
from agentlightning.verl.sc.vllm_score_centering import ScoreCenteringWorkerExtension


class Int8ColocateWorkerExtension(vLLMColocateWorkerExtension):
    # vLLM supplies this attribute when loading the worker extension.
    model_runner: Any

    def monkey_patch_model(self, vocab_size):
        super().monkey_patch_model(vocab_size)
        runner = self.model_runner
        initialize_scales = runner.init_fp8_kv_scales

        def init_scales():
            caches = runner.kv_caches
            runner.kv_caches = tree_leaves(caches)
            try:
                initialize_scales()
            finally:
                runner.kv_caches = caches

        runner.init_fp8_kv_scales = init_scales

    def update_weights_from_ipc(self, *args, **kwargs):
        for model in self._iter_all_models():
            if model is None:
                continue
            for module in model.modules():
                if getattr(module, "kv_cache_dtype", None) == "int4_per_token_head":
                    module.quant_method = None
                scheme = getattr(module, "scheme", None)
                if type(scheme).__name__ == "CompressedTensorsW8A8Int8":
                    module.weight = ModelWeightParameter(
                        data=module.weight.data.t(),
                        input_dim=1,
                        output_dim=0,
                        weight_loader=module.weight_loader_v2,
                    )
                    module.weight_scale = ChannelQuantScaleParameter(
                        data=module.weight_scale.data,
                        output_dim=0,
                        weight_loader=module.weight_loader_v2,
                    )
        return super().update_weights_from_ipc(*args, **kwargs)

    def _update_weights(self, weights, peft_config, base_sync_done):
        return super()._update_weights(quantize_weights(weights), peft_config, base_sync_done)


class Int8ScoreCenteringWorkerExtension(Int8ColocateWorkerExtension, ScoreCenteringWorkerExtension):
    pass
