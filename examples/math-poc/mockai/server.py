#!/usr/bin/env python3
"""Tiny OpenAI-compatible echo server for math-poc mock mode."""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else json.dumps(content)
    return ""


class MockAIHandler(BaseHTTPRequestHandler):
    server_version = "mockai/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _send_json(self, status_code: int, data: dict[str, Any]) -> None:
        payload = _json_bytes(data)
        self.send_response(status_code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            model = os.environ.get("AGL_MODEL_NAME", "mock-llm")
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "mockai",
                        }
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return

        length = int(self.headers.get("content-length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid json"}})
            return

        model = str(body.get("model") or os.environ.get("AGL_MODEL_NAME", "mock-llm"))
        content = _last_user_message(body.get("messages") or [])
        if body.get("stream"):
            self._send_stream(model, content)
            return

        now = int(time.time())
        self._send_json(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": now,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": None,
            },
        )

    def _send_stream(self, model: str, content: str) -> None:
        now = int(time.time())
        chunks = [
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(b"data: " + _json_bytes(chunk) + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    port = int(os.environ.get("SERVER_PORT", "5002"))
    server = ThreadingHTTPServer(("0.0.0.0", port), MockAIHandler)
    print(f"mockai listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
