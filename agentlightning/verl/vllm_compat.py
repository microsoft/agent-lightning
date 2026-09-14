# Copyright (c) Microsoft. All rights reserved.

"""Runtime compatibility shims for the verl <-> vLLM version matrix."""

from __future__ import annotations

import inspect
import logging

__all__ = ["apply_vllm_compat_patches"]

logger = logging.getLogger(__name__)


def apply_vllm_compat_patches() -> None:
    """Apply all known verl <-> vLLM compatibility shims (idempotent)."""
    _patch_make_request_output_pooling_output()


def _patch_make_request_output_pooling_output() -> None:
    # verl's abort paths pass a `pooling_output` kwarg that only exists on
    # vLLM >= 0.9; on older vLLM the resulting TypeError is swallowed and
    # aborts silently do nothing. Drop the kwarg when the signature lacks it.
    try:
        from vllm.v1.engine.output_processor import RequestState  # pyright: ignore[reportMissingImports]
    except ImportError:
        return

    current = RequestState.make_request_output
    if getattr(current, "_agl_vllm_compat", False):
        return
    if "pooling_output" in inspect.signature(current).parameters:
        return

    def make_request_output(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("pooling_output", None)
        return current(self, *args, **kwargs)

    make_request_output._agl_vllm_compat = True  # type: ignore[attr-defined]
    RequestState.make_request_output = make_request_output
    logger.info("Applied vLLM < 0.9 compat shim for RequestState.make_request_output")
