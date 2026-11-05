# Copyright (c) Microsoft. All rights reserved.

"""Regression coverage for dynamically loaded rollout hooks."""

import pickle
import sys

from agentlightning.hooks import load_hooks
from agentlightning.schemas import RolloutCreate


def test_load_hooks_registers_dataclass_hook_module(tmp_path) -> None:
    hooks_path = tmp_path / "custom_hooks.py"
    hooks_path.write_text(
        """
from __future__ import annotations

from dataclasses import dataclass

from agentlightning.hooks import RolloutHooks
from agentlightning.schemas import RolloutCreate


@dataclass
class HookConfig:
    prefix: str = "checked:"


class ConfiguredHooks(RolloutHooks):
    def __init__(self) -> None:
        self.config = HookConfig()

    def on_enqueue(self, request: RolloutCreate) -> RolloutCreate:
        return request.model_copy(update={"input": f"{self.config.prefix}{request.input}"})
""",
        encoding="utf-8",
    )

    hooks = load_hooks(str(hooks_path))

    assert hooks.on_enqueue(RolloutCreate(input="task")).input == "checked:task"
    assert hooks.__class__.__module__ in sys.modules
    assert pickle.loads(pickle.dumps(hooks)).config.prefix == "checked:"
