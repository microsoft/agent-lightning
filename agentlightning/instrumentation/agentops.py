# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import multiprocessing
import signal
import socket
import time
from typing import Any, Callable

import flask
import setproctitle

logger = logging.getLogger(__name__)

# Module-level storage for originals
_original_handle_chat_attributes: Callable[..., Any] | None = None
_original_handle_response: Callable[..., Any] | None = None


def _patch_new_agentops():
    import agentops.instrumentation.providers.openai.stream_wrapper
    import agentops.instrumentation.providers.openai.wrappers.chat
    from agentops.instrumentation.providers.openai.wrappers.chat import handle_chat_attributes  # type: ignore

    global _original_handle_chat_attributes

    if _original_handle_chat_attributes is not None:
        logger.warning("AgentOps already patched. Skipping.")
        return True

    _original_handle_chat_attributes = handle_chat_attributes  # type: ignore

    def _handle_chat_attributes_with_tokens(args=None, kwargs=None, return_value=None, **kws):  # type: ignore
        attributes = _original_handle_chat_attributes(args=args, kwargs=kwargs, return_value=return_value, **kws)  # type: ignore
        if return_value is not None and hasattr(return_value, "prompt_token_ids"):  # type: ignore
            attributes["prompt_token_ids"] = list(return_value.prompt_token_ids)  # type: ignore
        if return_value is not None and hasattr(return_value, "response_token_ids"):  # type: ignore
            attributes["response_token_ids"] = list(return_value.response_token_ids[0])  # type: ignore

        # For LiteLLM, response is a openai._legacy_response.LegacyAPIResponse
        if (
            return_value is not None
            and hasattr(return_value, "http_response")  # type: ignore
            and return_value.http_response is not None  # type: ignore
            and hasattr(return_value.http_response, "json")  # type: ignore
        ):
            json_data = return_value.http_response.json()  # type: ignore
            if isinstance(json_data, dict):
                if "prompt_token_ids" in json_data:
                    attributes["prompt_token_ids"] = list(json_data["prompt_token_ids"])  # type: ignore
                if "response_token_ids" in json_data:
                    attributes["response_token_ids"] = list(json_data["response_token_ids"][0])  # type: ignore

        return attributes

    agentops.instrumentation.providers.openai.wrappers.chat.handle_chat_attributes = _handle_chat_attributes_with_tokens
    agentops.instrumentation.providers.openai.stream_wrapper.handle_chat_attributes = (
        _handle_chat_attributes_with_tokens
    )
    logger.info("Patched newer version of agentops using handle_chat_attributes")
    return True


def _unpatch_new_agentops():
    import agentops.instrumentation.providers.openai.stream_wrapper
    import agentops.instrumentation.providers.openai.wrappers.chat

    global _original_handle_chat_attributes
    if _original_handle_chat_attributes is not None:
        agentops.instrumentation.providers.openai.wrappers.chat.handle_chat_attributes = (
            _original_handle_chat_attributes
        )
        agentops.instrumentation.providers.openai.stream_wrapper.handle_chat_attributes = (
            _original_handle_chat_attributes
        )
        _original_handle_chat_attributes = None
        logger.info("Unpatched newer version of agentops using handle_chat_attributes")


def _patch_old_agentops():
    import opentelemetry.instrumentation.openai.shared.chat_wrappers  # type: ignore
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import _handle_response, dont_throw  # type: ignore

    global _original_handle_response
    _original_handle_response = _handle_response  # type: ignore

    @dont_throw  # type: ignore
    def _handle_response_with_tokens(response, span, *args, **kwargs):  # type: ignore
        _original_handle_response(response, span, *args, **kwargs)  # type: ignore
        if hasattr(response, "prompt_token_ids"):  # type: ignore
            span.set_attribute("prompt_token_ids", list(response.prompt_token_ids))  # type: ignore
        if hasattr(response, "response_token_ids"):  # type: ignore
            span.set_attribute("response_token_ids", list(response.response_token_ids[0]))  # type: ignore

        # For LiteLLM, response is a openai._legacy_response.LegacyAPIResponse
        if hasattr(response, "http_response") and hasattr(response.http_response, "json"):  # type: ignore
            json_data = response.http_response.json()  # type: ignore
            if isinstance(json_data, dict):
                if "prompt_token_ids" in json_data:
                    span.set_attribute("prompt_token_ids", list(json_data["prompt_token_ids"]))  # type: ignore
                if "response_token_ids" in json_data:
                    span.set_attribute("response_token_ids", list(json_data["response_token_ids"][0]))  # type: ignore

    opentelemetry.instrumentation.openai.shared.chat_wrappers._handle_response = _handle_response_with_tokens  # type: ignore
    logger.info("Patched earlier version of agentops using _handle_response")
    return True


def _unpatch_old_agentops():
    import opentelemetry.instrumentation.openai.shared.chat_wrappers  # type: ignore

    global _original_handle_response
    if _original_handle_response is not None:
        opentelemetry.instrumentation.openai.shared.chat_wrappers._handle_response = _original_handle_response  # type: ignore
        _original_handle_response = None
        logger.info("Unpatched earlier version of agentops using _handle_response")


def instrument_agentops():
    """
    Instrument agentops to capture token IDs.
    Automatically detects and uses the appropriate patching method based on the installed agentops version.
    """
    # Try newest version first (tested for 0.4.16)
    try:
        return _patch_new_agentops()
    except ImportError as e:
        logger.debug(f"Couldn't patch newer version of agentops: {str(e)}")

    # Note: 0.4.15 needs another patching method, but it's too shortlived to be worth handling separately.

    # Try older version (tested for 0.4.13)
    try:
        return _patch_old_agentops()
    except ImportError as e:
        logger.warning(f"Couldn't patch older version of agentops: {str(e)}")
        logger.error("Failed to instrument agentops - neither patching method was successful")
        return False


def uninstrument_agentops():
    try:
        _unpatch_new_agentops()
    except Exception:
        pass
    try:
        _unpatch_old_agentops()
    except Exception:
        pass


def agentops_local_server():
    """
    Returns a Flask app that can be used to test agentops integration.
    This server provides endpoints for token fetching and a catch-all endpoint.
    """
    app = flask.Flask(__name__)

    @app.route("/v3/auth/token", methods=["POST"])
    def fetch_token():  # type: ignore
        return {"token": "dummy", "project_id": "dummy"}

    @app.route("/", defaults={"path": ""}, methods=["GET", "POST"])
    @app.route("/<path:path>", methods=["GET", "POST"])
    def catch_all(path: str):  # type: ignore
        return {"path": path}

    return app


def _run_server(**kwargs: Any):  # type: ignore
    """
    Internal function to run the Flask server.
    This is used to avoid issues with multiprocessing and Flask's reloader.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore SIGINT in worker processes
    setproctitle.setproctitle(multiprocessing.current_process().name)
    app = agentops_local_server()
    app.run(**kwargs)


class AgentOpsServerManager:
    def __init__(self, daemon: bool = True, port: int | None = None):
        self.server_process: multiprocessing.Process | None = None
        self.server_port = port
        self.daemon = daemon
        logger.info("AgentOpsServerManager initialized.")

    def _find_available_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def start(self):
        if self.server_process and self.server_process.is_alive():
            logger.warning("AgentOps server process appears to be already running.")
            return

        if self.server_port is None:
            self.server_port = self._find_available_port()

        logger.info(f"Starting AgentOps local server on port {self.server_port}...")

        self.server_process = multiprocessing.Process(
            target=_run_server,
            kwargs={"host": "127.0.0.1", "port": self.server_port, "use_reloader": False, "debug": False},
            daemon=self.daemon,
            name="AgentLightning-AgentOpsServer",
        )
        self.server_process.start()
        logger.info(
            f"AgentOps local server process (PID: {self.server_process.pid}) started, targeting port {self.server_port}."
        )
        time.sleep(0.5)  # Brief wait for server to start up
        if not self.server_process.is_alive():
            logger.error(f"AgentOps local server failed to start or exited prematurely.")

    def is_alive(self) -> bool:
        if self.server_process and self.server_process.is_alive():
            return True
        return False

    def stop(self):
        if self.server_process is not None and self.server_process.is_alive():
            logger.info(f"Stopping AgentOps local server (PID: {self.server_process.pid})...")
            self.server_process.terminate()  # Send SIGTERM
            self.server_process.join(timeout=5)  # Wait for clean exit
            if self.server_process.is_alive():
                logger.warning(
                    f"AgentOps server (PID: {self.server_process.pid}) did not terminate gracefully, killing..."
                )
                self.server_process.kill()  # Force kill
                self.server_process.join(timeout=10)  # Wait for kill
            self.server_process = None
            logger.info(f"AgentOps local server stopped.")
        else:
            logger.info("AgentOps local server was not running or already stopped.")

    def get_port(self) -> int | None:
        # Check liveness again in case it died since start()
        if self.is_alive() and self.server_port is not None:
            return self.server_port
        # If called after server stopped or failed, port might be stale or None
        if self.server_port is not None and (self.server_process is None or not self.server_process.is_alive()):
            logger.warning(
                f"AgentOps server port {self.server_port} is stored, but server process is not alive. Returning stored port."
            )
        return self.server_port
