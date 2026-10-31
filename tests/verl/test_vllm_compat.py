# Copyright (c) Microsoft. All rights reserved.

"""Compatibility tests at optional VERL/vLLM dependency interfaces, without an engine."""

from __future__ import annotations

import asyncio
import runpy
import sys
from types import SimpleNamespace

import pytest

from agentlightning.verl.vllm_compat import apply_vllm_compat_patches


@pytest.fixture
def server_boundary(monkeypatch, tmp_path):
    events = []
    registry = {"hermes"}

    class ToolParserManager:
        @classmethod
        def import_tool_parser(cls, path):
            events.append("import")
            runpy.run_path(path)

        @classmethod
        def register_module(cls, name):
            registry.add(name)

    class Server:
        async def run_server(self, args, *positional, **kwargs):
            events.append("delegate")
            if args.tool_call_parser not in registry:
                raise ValueError("unregistered tool parser")
            if kwargs.get("fail"):
                raise RuntimeError("delegate failed")
            return args, positional, kwargs

    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        SimpleNamespace(vLLMHttpServer=Server),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.tool_parsers",
        SimpleNamespace(ToolParserManager=ToolParserManager),
    )
    plugin = tmp_path / "custom_parser.py"
    plugin.write_text(
        "from vllm.entrypoints.openai.tool_parsers import ToolParserManager\n"
        "ToolParserManager.register_module('custom')\n"
    )
    return SimpleNamespace(server=Server(), plugin=str(plugin), events=events, manager=ToolParserManager)


def test_custom_parser_is_registered_before_server_start(server_boundary):
    args = SimpleNamespace(tool_parser_plugin=server_boundary.plugin, tool_call_parser="custom")
    apply_vllm_compat_patches()
    patched = server_boundary.server.run_server.__func__
    apply_vllm_compat_patches()
    assert server_boundary.server.run_server.__func__ is patched
    result = asyncio.run(server_boundary.server.run_server(args, "argument", option=True))
    assert result == (args, ("argument",), {"option": True})
    assert server_boundary.events == ["import", "delegate"]


@pytest.mark.parametrize("plugin", [None, ""])
def test_builtin_parser_without_plugin_is_unchanged(server_boundary, plugin):
    args = SimpleNamespace(tool_call_parser="hermes")
    if plugin is not None:
        args.tool_parser_plugin = plugin
    apply_vllm_compat_patches()
    assert asyncio.run(server_boundary.server.run_server(args)) == (args, (), {})
    assert server_boundary.events == ["delegate"]


def test_plugin_import_failure_prevents_server_start(server_boundary):
    args = SimpleNamespace(tool_parser_plugin=server_boundary.plugin + ".missing", tool_call_parser="custom")
    apply_vllm_compat_patches()
    with pytest.raises(FileNotFoundError):
        asyncio.run(server_boundary.server.run_server(args))
    assert server_boundary.events == ["import"]


def test_delegate_error_is_preserved(server_boundary):
    args = SimpleNamespace(tool_parser_plugin=server_boundary.plugin, tool_call_parser="custom")
    apply_vllm_compat_patches()
    with pytest.raises(RuntimeError, match="delegate failed"):
        asyncio.run(server_boundary.server.run_server(args, fail=True))
    assert server_boundary.events == ["import", "delegate"]


def test_pooling_output_shim_still_applies_without_verl(monkeypatch):
    class RequestState:
        def make_request_output(self, value):
            return value

    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.vllm_async_server", None)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.output_processor", SimpleNamespace(RequestState=RequestState))
    apply_vllm_compat_patches()
    assert RequestState().make_request_output("output", pooling_output=None) == "output"  # pyright: ignore[reportCallIssue]


def test_optional_dependencies_absent_is_a_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.vllm_async_server", None)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.output_processor", None)
    apply_vllm_compat_patches()
