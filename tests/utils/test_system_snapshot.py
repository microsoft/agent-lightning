# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest

from agentlightning.utils import system_snapshot

try:
    import torch  # type: ignore

    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    GPU_AVAILABLE = False  # type: ignore


def _patch_system_snapshot(monkeypatch: pytest.MonkeyPatch, include_gpu: bool = False) -> Optional[SimpleNamespace]:
    monkeypatch.setattr(system_snapshot.platform, "processor", lambda: "test-cpu")
    monkeypatch.setattr(system_snapshot.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(system_snapshot.socket, "gethostname", lambda: "test-host")

    def fake_cpu_count(logical: bool = True) -> int:
        return 4 if logical else 2

    monkeypatch.setattr(system_snapshot.psutil, "cpu_count", fake_cpu_count)
    monkeypatch.setattr(system_snapshot.psutil, "cpu_percent", lambda _: 33.3)  # type: ignore

    vm = SimpleNamespace(used=5 * (2**30), total=10 * (2**30), percent=50.0)
    monkeypatch.setattr(system_snapshot.psutil, "virtual_memory", lambda: vm)

    du = SimpleNamespace(used=2 * (2**30), total=8 * (2**30), percent=25.0)
    monkeypatch.setattr(system_snapshot.psutil, "disk_usage", lambda _: du)  # type: ignore

    net = SimpleNamespace(bytes_sent=4 * (2**20), bytes_recv=6 * (2**20))
    monkeypatch.setattr(system_snapshot.psutil, "net_io_counters", lambda: net)

    if include_gpu:
        dummy_gpu = SimpleNamespace(
            name="Test GPU",
            utilization=37.5,
            memory_used=1024,
            memory_total=4096,
            temperature=68,
        )
        monkeypatch.setattr(
            system_snapshot.GPUStatCollection,
            "new_query",
            lambda *args, **kwargs: SimpleNamespace(gpus=[dummy_gpu]),  # type: ignore
        )

        return dummy_gpu
    return None


def test_system_snapshot_excludes_gpu_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_system_snapshot(monkeypatch)

    snapshot = system_snapshot.system_snapshot()

    assert snapshot["cpu_name"] == "test-cpu"
    assert snapshot["cpu_cores"] == 2
    assert snapshot["cpu_threads"] == 4
    assert snapshot["cpu_usage_pct"] == 33.3
    assert snapshot["mem_used_gb"] == 5.0
    assert snapshot["mem_total_gb"] == 10.0
    assert snapshot["mem_pct"] == 50.0
    assert snapshot["disk_used_gb"] == 2.0
    assert snapshot["disk_total_gb"] == 8.0
    assert snapshot["disk_pct"] == 25.0
    assert snapshot["bytes_sent_mb"] == 4.0
    assert snapshot["bytes_recv_mb"] == 6.0
    assert snapshot["host"] == "test-host"
    assert snapshot["os"] == "test-platform"
    assert "gpus" not in snapshot


def test_system_snapshot_includes_gpus_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_gpu = _patch_system_snapshot(monkeypatch, include_gpu=True)
    assert dummy_gpu is not None

    snapshot = system_snapshot.system_snapshot(include_gpu=True)

    assert snapshot["gpus"] == [
        {
            "gpu": dummy_gpu.name,
            "util_pct": dummy_gpu.utilization,
            "mem_used_mb": dummy_gpu.memory_used,
            "mem_total_mb": dummy_gpu.memory_total,
            "temp_c": dummy_gpu.temperature,
        }
    ]


def test_sanity_check() -> None:
    snapshot = system_snapshot.system_snapshot()
    assert snapshot is not None

    snapshot = system_snapshot.system_snapshot(include_gpu=True)
    assert snapshot is not None

    if GPU_AVAILABLE:
        assert snapshot["gpus"] is not None
        assert len(snapshot["gpus"]) > 0
