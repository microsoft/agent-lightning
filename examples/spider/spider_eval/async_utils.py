# Copyright (c) Microsoft. All rights reserved.

import asyncio
import queue
import threading
from typing import Any, Coroutine


def run_sync_ephemeral(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run an async coroutine from sync code.
    - If no loop in this thread: use asyncio.run() directly.
    - If already in an event loop: spawn a worker thread that calls asyncio.run()
      (which creates and closes a brand-new event loop per call).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread; safe to use asyncio.run
        return asyncio.run(coro)

    # Already in a running loop -> execute in a worker thread
    q = queue.Queue[Any]()

    def worker():
        try:
            result = asyncio.run(coro)  # creates & closes its own loop
            q.put((True, result))
        except BaseException as e:
            q.put((False, e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    ok, payload = q.get()
    t.join()
    if ok:
        return payload
    raise payload
