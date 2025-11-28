# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import platform
import socket
from contextlib import suppress
from datetime import datetime
from typing import Any, Dict, List, cast

import psutil
from gpustat import GPUStat, GPUStatCollection


def system_snapshot(include_gpu: bool = False) -> Dict[str, Any]:
    # CPU
    cpu = {
        "cpu_name": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "cpu_usage_pct": psutil.cpu_percent(0.05),
    }

    # Memory
    vm = psutil.virtual_memory()
    mem = {
        "mem_used_gb": round(vm.used / (2**30), 2),
        "mem_total_gb": round(vm.total / (2**30), 2),
        "mem_pct": vm.percent,
    }

    # Disk
    du = psutil.disk_usage("/")
    disk = {
        "disk_used_gb": round(du.used / (2**30), 2),
        "disk_total_gb": round(du.total / (2**30), 2),
        "disk_pct": du.percent,
    }

    # GPU
    gpus: List[Dict[str, Any]] = []
    with suppress(Exception):
        for g in GPUStatCollection.new_query().gpus:  # type: ignore
            g = cast(GPUStat, g)
            gpus.append(
                {
                    "gpu": g.name,  # type: ignore
                    "util_pct": g.utilization,
                    "mem_used_mb": g.memory_used,
                    "mem_total_mb": g.memory_total,
                    "temp_c": g.temperature,
                }
            )

    # Network
    net = psutil.net_io_counters()
    netinfo = {
        "bytes_sent_mb": round(net.bytes_sent / (2**20), 2),
        "bytes_recv_mb": round(net.bytes_recv / (2**20), 2),
    }

    # OS / meta
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "os": platform.platform(),
        **cpu,
        **mem,
        **disk,
        **netinfo,
        **({"gpus": gpus} if include_gpu else {}),
    }
