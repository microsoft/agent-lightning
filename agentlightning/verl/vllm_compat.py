# Copyright (c) Microsoft. All rights reserved.

"""Runtime compatibility shims for the verl <-> vLLM version matrix.

verl 0.7.x/0.8.x support a wide vLLM range (``vllm>=0.8.5``) but call a few
vLLM APIs whose signatures changed across that range. The shims below bridge
known incompatibilities at runtime. They are applied in every Ray worker
process via the ``worker_process_setup_hook`` (see
:func:`agentlightning.verl.per_rollout_loss.register_in_worker`), and each
shim is a strict no-op when the installed vLLM does not need it.
"""

from __future__ import annotations

import inspect
import logging

__all__ = ["apply_vllm_compat_patches"]

logger = logging.getLogger(__name__)


def apply_vllm_compat_patches() -> None:
    """Apply all known verl <-> vLLM compatibility shims (idempotent)."""
    _patch_make_request_output_pooling_output()


def _patch_make_request_output_pooling_output() -> None:
    """Let ``RequestState.make_request_output`` ignore ``pooling_output`` on vLLM < 0.9.

    verl 0.7.x/0.8.x abort paths (``vLLMHttpServer.abort_all_requests`` /
    ``abort_request``) pass a ``pooling_output`` kwarg that only exists from
    vLLM 0.9 on. On vLLM 0.8.5 the resulting ``TypeError`` is swallowed by
    verl's blanket ``except Exception``, so aborts silently do nothing and
    orphaned requests keep generating concurrently with the training phase —
    observed to crash training with a CUDA illegal memory access (#589).
    """
    try:
        from vllm.v1.engine.output_processor import RequestState
    except ImportError:
        return

    current = RequestState.make_request_output
    if getattr(current, "_agl_vllm_compat", False):
        return
    if "pooling_output" in inspect.signature(current).parameters:
        return

    original = current

    def make_request_output(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("pooling_output", None)
        return original(self, *args, **kwargs)

    make_request_output._agl_vllm_compat = True  # type: ignore[attr-defined]
    RequestState.make_request_output = make_request_output
    logger.info("Applied vLLM < 0.9 compat shim: RequestState.make_request_output ignores pooling_output")
