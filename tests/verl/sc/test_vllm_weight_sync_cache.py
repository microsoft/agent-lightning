# Copyright (c) Microsoft. All rights reserved.

from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("verl")


@pytest.mark.asyncio
async def test_weight_sync_releases_cache_before_wake_and_preserves_dispatch(monkeypatch):
    from verl.utils import memory_utils
    from verl.workers.engine_workers import ActorRolloutRefWorker

    from agentlightning.verl.sc.patches import _patch_weight_sync_cache

    events = []

    async def update_weights(self, global_steps=None, mode="auto"):
        events.append(("wake", global_steps, mode))
        return 17

    dispatch = object()
    update_weights._dispatch = dispatch  # pyright: ignore[reportFunctionMemberAccess]
    monkeypatch.setattr(ActorRolloutRefWorker, "update_weights", update_weights)
    monkeypatch.setattr(memory_utils, "aggressive_empty_cache", lambda **kwargs: events.append(("cache", kwargs)))
    _patch_weight_sync_cache()
    patched = cast(Any, ActorRolloutRefWorker.update_weights)
    _patch_weight_sync_cache()
    assert ActorRolloutRefWorker.update_weights is patched
    assert patched._dispatch is dispatch
    result = await patched(SimpleNamespace(), global_steps=3, mode="naive")
    assert result == 17
    assert events == [("cache", {"force_sync": True}), ("wake", 3, "naive")]
