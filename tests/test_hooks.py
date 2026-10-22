# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for :func:`agentlightning.hooks.load_hooks`."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

from agentlightning.hooks import RolloutHooks, load_hooks


def _write_hooks(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "custom_hooks.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_hooks_supports_postponed_dataclass_annotations(tmp_path: Path) -> None:
    """A hook file using ``from __future__ import annotations`` plus ``@dataclass``
    must load, not die in ``dataclasses._is_type`` looking up a missing module."""
    path = _write_hooks(
        tmp_path,
        """\
from __future__ import annotations
from dataclasses import dataclass

from agentlightning.hooks import RolloutHooks


@dataclass
class Settings:
    label: str = "training"


class Hooks(RolloutHooks):
    settings = Settings()
""",
    )

    hooks = load_hooks(str(path))

    assert isinstance(hooks, RolloutHooks)
    assert type(hooks).__name__ == "Hooks"
    assert hooks.settings.label == "training"


def test_load_hooks_classes_are_picklable(tmp_path: Path) -> None:
    """Classes defined in a loaded hook file must be importable again, so workers
    can pickle them across process boundaries."""
    path = _write_hooks(
        tmp_path,
        """\
from agentlightning.hooks import RolloutHooks


class Hooks(RolloutHooks):
    def on_startup(self, store=None) -> None:
        pass
""",
    )

    hooks = load_hooks(str(path))

    # The class must reference a module that is registered in sys.modules and can
    # be re-imported; pickle round-trips the instance through that path.
    module_name = type(hooks).__module__
    assert module_name in sys.modules, "hook class module must be registered"
    assert pickle.loads(pickle.dumps(hooks)) is not None


def test_load_hooks_distinct_paths_get_distinct_module_identity(tmp_path: Path) -> None:
    """Two different hook files must not share one ``sys.modules`` entry, so loading
    one cannot leak into the other."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _write_hooks(
        first_dir, "from agentlightning.hooks import RolloutHooks\n\nclass Hooks(RolloutHooks):\n    pass\n"
    )
    second = _write_hooks(
        second_dir, "from agentlightning.hooks import RolloutHooks\n\nclass Hooks(RolloutHooks):\n    pass\n"
    )

    hooks_a = load_hooks(str(first))
    hooks_b = load_hooks(str(second))

    assert type(hooks_a).__module__ != type(hooks_b).__module__
    assert hooks_a is not hooks_b


def test_load_hooks_reuses_module_for_the_same_path(tmp_path: Path) -> None:
    """Loading one file twice must reuse its module, not replace its classes.

    Re-executing the file installs a new class object under the same module name,
    which makes instances returned by the first load unpicklable.
    """
    path = _write_hooks(
        tmp_path,
        """\
from agentlightning.hooks import RolloutHooks


class Hooks(RolloutHooks):
    pass
""",
    )

    first = load_hooks(str(path))
    assert pickle.loads(pickle.dumps(first)) is not None

    second = load_hooks(str(path))

    assert type(second) is type(first)
    assert pickle.loads(pickle.dumps(first)) is not None


def test_load_hooks_restores_sys_modules_on_failure(tmp_path: Path) -> None:
    """A hook file that raises during execution must not leave a half-imported
    module registered under its name."""
    path = _write_hooks(
        tmp_path,
        """\
raise RuntimeError("boom during import")
""",
    )

    # Snapshot module names before the attempt.
    before = set(sys.modules)

    with pytest.raises(RuntimeError, match="boom during import"):
        load_hooks(str(path))

    # Any module name this loader would have registered must not survive the error.
    after = set(sys.modules)
    assert after == before


def test_load_hooks_rejects_multiple_subclasses(tmp_path: Path) -> None:
    path = _write_hooks(
        tmp_path,
        """\
from agentlightning.hooks import RolloutHooks


class One(RolloutHooks):
    pass


class Two(RolloutHooks):
    pass
""",
    )
    with pytest.raises(ValueError, match="Multiple RolloutHooks subclasses"):
        load_hooks(str(path))
