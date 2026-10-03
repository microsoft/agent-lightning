# Copyright (c) Microsoft. All rights reserved.

"""Agent Lightning entrypoint for the Vercel AI SDK example."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from pathlib import Path

_ENTRYPOINT = Path(__file__).with_name("dist") / "src" / "agent.js"
_CANCEL_WAIT_SECONDS = 5.0


class AISDKAgent:
    """Run the compiled TypeScript agent as one local rollout subprocess."""

    async def run(self) -> None:
        if not _ENTRYPOINT.is_file():
            raise FileNotFoundError(f"AI SDK build is missing: {_ENTRYPOINT}. Run `npm ci && npm run build` first.")
        node = os.environ.get("AI_SDK_NODE") or shutil.which("node")
        if node is None:
            raise FileNotFoundError("Node.js was not found; install Node.js 22 or newer or set AI_SDK_NODE")

        process = await asyncio.create_subprocess_exec(
            node,
            str(_ENTRYPOINT),
            env=os.environ.copy(),
        )
        try:
            returncode = await process.wait()
        except asyncio.CancelledError:
            await _stop_process(process)
            raise
        if returncode != 0:
            raise RuntimeError(f"AI SDK agent exited with status {returncode}")


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=_CANCEL_WAIT_SECONDS)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
