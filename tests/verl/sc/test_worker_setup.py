# Copyright (c) Microsoft. All rights reserved.

import builtins

import pytest

pytest.importorskip("verl")


def test_normal_worker_setup_does_not_import_sc_patches_or_require_vllm(monkeypatch):
    from verl.workers.engine_workers import ActorRolloutRefWorker

    from agentlightning.verl.per_rollout_loss import register_in_worker

    original_import = builtins.__import__
    original_update_weights = ActorRolloutRefWorker.update_weights

    def import_without_sc(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError("No module named 'vllm'")
        if name == "agentlightning.verl.sc" or name.startswith("agentlightning.verl.sc."):
            pytest.fail(f"Normal worker setup imported SC package: {name}")
        if name.rsplit(".", 1)[-1] in {
            "int8_rollout",
            "qwen35_packing",
            "vllm_score_centering",
            "engine_workers",
            "memory_utils",
        }:
            pytest.fail(f"Normal worker setup imported SC dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sc)
    register_in_worker()
    assert ActorRolloutRefWorker.update_weights is original_update_weights
