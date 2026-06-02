"""Fake agent classes used by local reconciler worker tests."""

from __future__ import annotations

import json
import os
from typing import Any


class FakeAgent:
    """Sync agent that records the task it received into AGL_FAKE_AGENT_OUT."""

    def run(self, task: Any) -> None:
        out_path = os.environ.get("AGL_FAKE_AGENT_OUT")
        if out_path:
            with open(out_path, "w") as f:
                json.dump({"task": task}, f)


class FakeAsyncAgent:
    """Async agent that records the task it received."""

    async def run(self, task: Any) -> None:
        out_path = os.environ.get("AGL_FAKE_AGENT_OUT")
        if out_path:
            with open(out_path, "w") as f:
                json.dump({"task": task, "async": True}, f)


class FakeRaisingAgent:
    """Agent that raises an exception."""

    def run(self, _task: Any) -> None:
        raise RuntimeError("fake failure")


NOT_A_CLASS = "I am not a class"
