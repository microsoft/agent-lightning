# Copyright (c) Microsoft. All rights reserved.

"""Tests for the verl <-> vLLM runtime compatibility shims."""

from __future__ import annotations

import sys
import types

import pytest

from agentlightning.verl.vllm_compat import apply_vllm_compat_patches


class _OldRequestState:
    """Mimics vLLM < 0.9: make_request_output has no pooling_output param."""

    def make_request_output(self, new_token_ids, finish_reason, stop_reason):
        return (new_token_ids, finish_reason, stop_reason)


class _NewRequestState:
    """Mimics vLLM >= 0.9: pooling_output is part of the signature."""

    def make_request_output(self, new_token_ids, pooling_output, finish_reason, stop_reason):
        return (new_token_ids, pooling_output, finish_reason, stop_reason)


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch):
    """Install a fake vllm.v1.engine.output_processor module and yield it."""
    module = types.ModuleType("vllm.v1.engine.output_processor")
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.engine",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.output_processor", module)
    return module


def test_old_vllm_gets_patched_and_ignores_pooling_output(fake_vllm) -> None:
    fake_vllm.RequestState = _OldRequestState

    apply_vllm_compat_patches()

    state = _OldRequestState()
    # verl's abort-path call style (kwargs incl. pooling_output) must now work.
    assert state.make_request_output([], pooling_output=None, finish_reason="abort", stop_reason=None) == (
        [],
        "abort",
        None,
    )
    # vLLM's own positional call style must pass through unchanged.
    assert state.make_request_output([1], "stop", 0) == ([1], "stop", 0)


def test_patch_is_idempotent(fake_vllm) -> None:
    fake_vllm.RequestState = _OldRequestState

    apply_vllm_compat_patches()
    patched_once = _OldRequestState.make_request_output
    apply_vllm_compat_patches()

    # The marker attribute must prevent a second wrap.
    assert _OldRequestState.make_request_output is patched_once
    assert getattr(patched_once, "_agl_vllm_compat", False)


def test_new_vllm_is_left_untouched(fake_vllm) -> None:
    fake_vllm.RequestState = _NewRequestState
    original = _NewRequestState.make_request_output

    apply_vllm_compat_patches()

    assert _NewRequestState.make_request_output is original


def test_missing_vllm_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", [])  # ensure the real vllm cannot be imported

    apply_vllm_compat_patches()  # must not raise
