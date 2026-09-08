# Copyright (c) Microsoft. All rights reserved.

"""SHAPER integration for the official VLABench simulator and a frozen VLA actor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import VLABenchAgent, VLABenchRuntimeConfig
    from .dataset import load_reported_protocol_datasets


def __getattr__(name: str) -> Any:
    """Load simulator-facing exports only when downstream code requests them."""

    if name in {"VLABenchAgent", "VLABenchRuntimeConfig"}:
        from .agent import VLABenchAgent, VLABenchRuntimeConfig

        return {"VLABenchAgent": VLABenchAgent, "VLABenchRuntimeConfig": VLABenchRuntimeConfig}[name]
    if name == "load_reported_protocol_datasets":
        from .dataset import load_reported_protocol_datasets

        return load_reported_protocol_datasets
    raise AttributeError(name)


__all__ = [
    "VLABenchAgent",
    "VLABenchRuntimeConfig",
    "load_reported_protocol_datasets",
]
