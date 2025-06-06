import logging
import multiprocessing
import signal
import socket
import time

import flask
import setproctitle
import opentelemetry.instrumentation.openai.shared.chat_wrappers
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _handle_response,
    dont_throw,
)

logger = logging.getLogger(__name__)

_original_handle_response = _handle_response


@dont_throw
def _handle_response_with_tokens(response, span, *args, **kwargs):
    _original_handle_response(response, span, *args, **kwargs)
    if hasattr(response, "prompt_token_ids"):
        span.set_attribute("prompt_token_ids", list(response.prompt_token_ids))
    if hasattr(response, "response_token_ids"):
        span.set_attribute("response_token_ids", list(response.response_token_ids[0]))

    # For LiteLLM, response is a openai._legacy_response.LegacyAPIResponse
    if hasattr(response, "http_response") and hasattr(response.http_response, "json"):
        json_data = response.http_response.json()
        if isinstance(json_data, dict):
            if "prompt_token_ids" in json_data:
                span.set_attribute("prompt_token_ids", list(json_data["prompt_token_ids"]))
            if "response_token_ids" in json_data:
                span.set_attribute("response_token_ids", list(json_data["response_token_ids"][0]))


def instrument_agentops():
    opentelemetry.instrumentation.openai.shared.chat_wrappers._handle_response = _handle_response_with_tokens


def agentops_local_server():
    """
    Returns a Flask app that can be used to test agentops integration.
    This server provides endpoints for token fetching and a catch-all endpoint.
    """
    app = flask.Flask(__name__)

    @app.route("/v3/auth/token", methods=["POST"])
    def fetch_token():
        return {"token": "dummy", "project_id": "dummy"}

    @app.route("/", defaults={"path": ""}, methods=["GET", "POST"])
    @app.route("/<path:path>", methods=["GET", "POST"])
    def catch_all(path):
        return {"path": path}

    return app


def _run_server(**kwargs):
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
        if self.is_alive():
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
