# Copyright (c) Microsoft. All rights reserved.

"""Exercise local workers with real importable agent classes."""

from pathlib import Path

import pytest

from agentlightning.controller.local_reconciler import _run_local_reconciler_worker


@pytest.mark.parametrize("result_kind", ["sync", "coroutine", "awaitable"])
def test_local_worker_completes_run_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result_kind: str) -> None:
    marker = tmp_path / "completed.txt"
    module_name = f"worker_result_{result_kind}"
    source = f"""import asyncio
from pathlib import Path

async def complete():
    await asyncio.sleep(0)
    Path({str(marker)!r}).write_text("completed")

class Deferred:
    def __await__(self):
        return complete().__await__()

class Agent:
    def run(self):
"""
    source += {
        "sync": f"        Path({str(marker)!r}).write_text('completed')\n",
        "coroutine": "        return complete()\n",
        "awaitable": "        return Deferred()\n",
    }[result_kind]
    (tmp_path / f"{module_name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))

    assert _run_local_reconciler_worker(f"{module_name}:Agent") == 0
    assert marker.read_text() == "completed"
