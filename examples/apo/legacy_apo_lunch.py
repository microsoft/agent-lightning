# Copyright (c) Microsoft. All rights reserved.

"""This launch script shows how to start the Legacy APO server and client.

You can run this in two modes:

1. Run this launch script to start both server and client together:
```bash
python legacy_apo_lunch.py
```

2. Run server and client separately if needed:
```bash
python legacy_apo_server.py   # start server only
python legacy_apo_client.py   # start client only
```

This script will:
Start server and client processes concurrently
Print their logs in real time
Save their logs into legacy_apo_server.log and legacy_apo_client.log
"""

import asyncio
from pathlib import Path
from typing import List

StreamBuffer = List[str]


async def stream_output(
    stream: asyncio.StreamReader,
    name: str,
    pipe_type: str,
    buffer: StreamBuffer,
) -> None:
    """Asynchronously read output stream, print to console, and store in buffer."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode().rstrip()
        print(f"[{name}] {text}")
        buffer.append(text)


async def run(cmd: list[str], name: str, buffer: StreamBuffer) -> asyncio.subprocess.Process:
    """Start a subprocess and launch stdout/stderr logging coroutines."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print(f"{name} started (pid={proc.pid})")

    # concurrently monitor stdout and stderr
    if proc.stdout:
        asyncio.create_task(stream_output(proc.stdout, name, "OUT", buffer))
    if proc.stderr:
        asyncio.create_task(stream_output(proc.stderr, name, "ERR", buffer))
    return proc


async def main() -> None:
    """Run server and client concurrently and save their logs."""
    server_buffer: StreamBuffer = []
    client_buffer: StreamBuffer = []

    server = await run(["python", "legacy_apo_server.py"], "legacy_apo_server", server_buffer)
    client = await run(["python", "legacy_apo_client.py"], "legacy_apo_client", client_buffer)

    # wait for both client and server to finish
    await client.wait()
    await server.wait()

    # write logs to files
    Path("legacy_apo_server.log").write_text("\n".join(server_buffer), encoding="utf-8")
    Path("legacy_apo_client.log").write_text("\n".join(client_buffer), encoding="utf-8")

    print("Logs saved to legacy_apo_server.log and legacy_apo_client.log")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user, cleaning up...")
