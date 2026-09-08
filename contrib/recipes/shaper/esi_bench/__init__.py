# Copyright (c) Microsoft. All rights reserved.

"""SHAPER integration for the official ESI-Bench active-exploration runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import ESIBenchAgent, ESIBenchRuntimeConfig
    from .dataset import load_datasets


def __getattr__(name: str) -> Any:
    """Avoid importing Agent Lightning in the isolated simulator worker."""

    if name in {"ESIBenchAgent", "ESIBenchRuntimeConfig"}:
        from .agent import ESIBenchAgent, ESIBenchRuntimeConfig

        return {"ESIBenchAgent": ESIBenchAgent, "ESIBenchRuntimeConfig": ESIBenchRuntimeConfig}[name]
    if name == "load_datasets":
        from .dataset import load_datasets

        return load_datasets
    raise AttributeError(name)


__all__ = ["ESIBenchAgent", "ESIBenchRuntimeConfig", "load_datasets"]
