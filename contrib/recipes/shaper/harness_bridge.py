# Copyright (c) Microsoft. All rights reserved.

"""Private JSON bridge between simulator workers and the harness sandbox."""

from __future__ import annotations

import json
import os
import secrets
import socket
import struct
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, cast

_HEADER = struct.Struct("!Q")


class HarnessBridgeError(RuntimeError):
    """Raised when a harness bridge request cannot be completed."""


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise HarnessBridgeError("Harness bridge connection closed before the message completed.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_json(connection: socket.socket, max_bytes: int) -> dict[str, Any]:
    raw_size = _receive_exact(connection, _HEADER.size)
    size = _HEADER.unpack(raw_size)[0]
    if size > max_bytes:
        raise HarnessBridgeError(f"Harness bridge message is {size} bytes; limit is {max_bytes} bytes.")
    raw = _receive_exact(connection, size)
    try:
        value: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarnessBridgeError(f"Harness bridge received invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessBridgeError("Harness bridge message must be a JSON object.")
    return cast(dict[str, Any], value)


def _send_json(connection: socket.socket, value: Mapping[str, Any], max_bytes: int) -> None:
    raw = json.dumps(dict(value), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > max_bytes:
        raise HarnessBridgeError(f"Harness bridge response is {len(raw)} bytes; limit is {max_bytes} bytes.")
    connection.sendall(_HEADER.pack(len(raw)) + raw)


class HarnessBridgeClient:
    """Call a controller-owned harness runtime from an isolated worker."""

    def __init__(
        self,
        socket_path: Path,
        token: str,
        *,
        timeout_seconds: float = 10.0,
        max_request_bytes: int = 256_000_000,
        max_response_bytes: int = 32_000_000,
    ) -> None:
        self.socket_path = socket_path
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes

    def __call__(self, records: list[dict[str, Any]]) -> Any:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_seconds)
            try:
                connection.connect(str(self.socket_path))
                _send_json(
                    connection,
                    {"token": self.token, "records": records},
                    self.max_request_bytes,
                )
                response = _receive_json(connection, self.max_response_bytes)
            except (OSError, TimeoutError) as exc:
                raise HarnessBridgeError(f"Harness bridge request failed: {exc}") from exc
        if response.get("ok") is not True:
            raise HarnessBridgeError(str(response.get("error", "Harness bridge rejected the request.")))
        return response.get("output")


class HarnessBridgeServer:
    """Serve one restricted harness runtime on a private Unix socket."""

    def __init__(
        self,
        handler: Callable[[list[dict[str, Any]]], Any],
        *,
        max_request_bytes: int = 256_000_000,
        max_response_bytes: int = 32_000_000,
    ) -> None:
        self.handler = handler
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.token = secrets.token_hex(32)
        self.socket_path: Path | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    def __enter__(self) -> HarnessBridgeServer:
        self._directory = tempfile.TemporaryDirectory(prefix="shaper-harness-")
        self.socket_path = Path(self._directory.name) / "bridge.sock"
        self._thread = threading.Thread(target=self._serve, name="shaper-harness-bridge", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self.close()
            raise HarnessBridgeError("Harness bridge did not become ready within five seconds.")
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise HarnessBridgeError(f"Harness bridge failed to start: {error}") from error
        return self

    def _serve(self) -> None:
        assert self.socket_path is not None
        try:
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listener = listener
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(1)
            listener.settimeout(0.2)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        while not self._stop.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(15.0)
                try:
                    request = _receive_json(connection, self.max_request_bytes)
                    if not secrets.compare_digest(str(request.get("token", "")), self.token):
                        raise HarnessBridgeError("Harness bridge authentication failed.")
                    raw_records = request.get("records")
                    if not isinstance(raw_records, list) or not all(isinstance(item, dict) for item in raw_records):
                        raise HarnessBridgeError("Harness bridge records must be a list of JSON objects.")
                    records = cast(list[dict[str, Any]], raw_records)
                    response: dict[str, Any] = {"ok": True, "output": self.handler(records)}
                except BaseException as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                try:
                    _send_json(connection, response, self.max_response_bytes)
                except (HarnessBridgeError, OSError):
                    pass

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._directory is not None:
            self._directory.cleanup()
        self._listener = None
        self._thread = None
        self._directory = None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()
