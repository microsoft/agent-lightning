# Copyright (c) Microsoft. All rights reserved.

from functools import wraps


def register_in_worker():
    from ..per_rollout_loss import register_in_worker as register_base
    from .int8_rollout import apply_int8_rollout_patches
    from .qwen35_packing import apply_qwen35_packing_patch
    from .vllm_score_centering import install_server_support

    register_base()
    _patch_weight_sync_cache()
    apply_int8_rollout_patches()
    apply_qwen35_packing_patch()
    install_server_support()


def _patch_weight_sync_cache() -> None:
    from verl.utils.memory_utils import aggressive_empty_cache
    from verl.workers.engine_workers import ActorRolloutRefWorker

    current = ActorRolloutRefWorker.update_weights
    if getattr(current, "_agl_weight_sync_cache", False):
        return

    @wraps(current)
    async def update_weights(self, *args, **kwargs):
        aggressive_empty_cache(force_sync=True)
        return await current(self, *args, **kwargs)

    update_weights._agl_weight_sync_cache = True  # pyright: ignore[reportAttributeAccessIssue]
    ActorRolloutRefWorker.update_weights = update_weights
