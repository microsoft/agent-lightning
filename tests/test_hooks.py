# Copyright (c) Microsoft. All rights reserved.

"""Tests for loading user-defined rollout hooks from Python files."""

from pathlib import Path

import pytest

from agentlightning.hooks import load_hooks
from agentlightning.schemas import RolloutCreate

_PICKLE_HOOKS = """
from dataclasses import dataclass
import pickle

from agentlightning.hooks import RolloutHooks


@dataclass
class Input:
    label: str


class Hooks(RolloutHooks):
    def on_enqueue(self, request):
        restored = pickle.loads(pickle.dumps(Input(**request.input)))
        return request.model_copy(update={"input": {"label": restored.label}})
"""


@pytest.mark.parametrize("future_annotations", [False, True])
def test_load_hooks_with_dataclass_input(tmp_path: Path, future_annotations: bool) -> None:
    path = tmp_path / "custom_hooks.py"
    path.write_text(
        ("from __future__ import annotations\n" if future_annotations else "")
        + """
from dataclasses import asdict, dataclass
from typing import ClassVar

from agentlightning.hooks import RolloutHooks


@dataclass
class Input:
    label: str
    score: float = 0.0
    source: ClassVar[str] = "training"


class Hooks(RolloutHooks):
    def on_enqueue(self, request):
        return request.model_copy(update={"input": asdict(Input(**request.input))})
""",
        encoding="utf-8",
    )

    hooks = load_hooks(str(path))
    request = hooks.on_enqueue(RolloutCreate(input={"label": "example"}))

    assert request.input == {"label": "example", "score": 0.0}


def test_load_hooks_keeps_different_files_independent(tmp_path: Path) -> None:
    first_path = tmp_path / "first_hooks.py"
    second_path = tmp_path / "second_hooks.py"
    first_path.write_text(_PICKLE_HOOKS, encoding="utf-8")
    second_path.write_text(_PICKLE_HOOKS, encoding="utf-8")

    first = load_hooks(str(first_path))
    second = load_hooks(str(second_path))

    for hooks, label in [(first, "first"), (second, "second")]:
        request = hooks.on_enqueue(RolloutCreate(input={"label": label}))
        assert request.input == {"label": label}


@pytest.mark.parametrize(
    "invalid_source, error, message",
    [
        ('raise RuntimeError("invalid hook configuration")\n', RuntimeError, "invalid hook configuration"),
        ("VALUE = 123\n", ValueError, "No RolloutHooks subclass"),
        (
            "from agentlightning.hooks import RolloutHooks\n"
            "class First(RolloutHooks): pass\n"
            "class Second(RolloutHooks): pass\n",
            ValueError,
            "Multiple RolloutHooks subclasses",
        ),
        (
            "from agentlightning.hooks import RolloutHooks\n"
            "class Hooks(RolloutHooks):\n"
            "    def __init__(self): raise RuntimeError('constructor failed')\n",
            RuntimeError,
            "constructor failed",
        ),
    ],
)
def test_load_hooks_restores_previous_module_after_failure(
    tmp_path: Path, invalid_source: str, error: type[Exception], message: str
) -> None:
    path = tmp_path / "custom_hooks.py"
    path.write_text(_PICKLE_HOOKS, encoding="utf-8")
    hooks = load_hooks(str(path))

    path.write_text(invalid_source, encoding="utf-8")
    with pytest.raises(error, match=message):
        load_hooks(str(path))

    request = hooks.on_enqueue(RolloutCreate(input={"label": "original"}))
    assert request.input == {"label": "original"}


def test_load_hooks_removes_failed_module(tmp_path: Path) -> None:
    import sys

    path = tmp_path / "custom_hooks.py"
    path.write_text('raise RuntimeError("invalid hook configuration")\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid hook configuration"):
        load_hooks(str(path))

    assert not any(getattr(module, "__file__", None) == str(path) for module in sys.modules.copy().values())
