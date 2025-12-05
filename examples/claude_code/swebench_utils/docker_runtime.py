# Copyright (c) Microsoft. All rights reserved.

"""Docker runtime management for repository setup and command execution.

Provides containerized environment for repository testing with command execution,
file operations, and state management capabilities.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from docker.errors import DockerException, ImageNotFound
from docker.models.containers import Container
from swebench.harness.constants import SWEbenchInstance
from typing_extensions import Self

import docker

# This will log to the console for debugging purposes.
claude_code_logger = logging.getLogger("claude_code_agent.docker_runtime")

CMD_OUTPUT_PS1_BEGIN = "\n###PS1JSON###\n"
CMD_OUTPUT_PS1_END = "\n###PS1END###"
CMD_OUTPUT_METADATA_PS1_REGEX = re.compile(
    r"(?m)^\s*" + re.escape(CMD_OUTPUT_PS1_BEGIN.strip()) + r"\s*(.*?)\s*" + re.escape(CMD_OUTPUT_PS1_END.strip()),
    re.DOTALL,
)
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

TIMEOUT_EXIT_CODE = 124

MEM_LIMIT = "8g"
CPU_CORES = 4


VAR_PATTERNS = {
    "exit_code": re.compile(r'"exit_code":\s*(-?\d+)\s*(?:,|\})'),
    "username": re.compile(r'"username":\s*"([^"]*)"'),
    "hostname": re.compile(r'"hostname":\s*"([^"]*)"'),
    "working_dir": re.compile(r'"working_dir":\s*"([^"]*)"'),
    "py_interpreter_path": re.compile(r'"py_interpreter_path":\s*"([^"]*)"'),
}


@dataclass
class CmdOutputMetadata:
    """
    Additional metadata captured from PS1 shell prompt.

    Provides context about command execution environment including
    exit codes, user info, working directory, and Python interpreter.
    """

    exit_code: int = -1
    username: str | None = None
    hostname: str | None = None
    working_dir: str | None = None
    py_interpreter_path: str | None = None

    @classmethod
    def matches_ps1_metadata(cls, output: str) -> List[re.Match[str]]:
        matches: List[re.Match[str]] = []
        for match in CMD_OUTPUT_METADATA_PS1_REGEX.finditer(output):
            scope = match.group(1).strip()
            try:
                d = json.loads(scope)  # Try to parse as JSON
                matches.append(match)
            except json.JSONDecodeError:
                d = cls.best_effort_match(scope)
                if len(d) > 0:
                    matches.append(match)
        return matches

    @classmethod
    def best_effort_match(cls, scope: str) -> Dict[str, Any]:
        out: Dict[str, str] = {}
        for field, pattern in VAR_PATTERNS.items():
            m = pattern.search(scope)
            if m:
                out[field] = m.group(1)
            else:
                out[field] = ""
        return out

    @classmethod
    def from_ps1_match(cls, match: re.Match[str]) -> Self:
        """
        Extract metadata from a PS1 prompt regex match.

        Args:
            match (re.Match[str]): Regex match containing JSON metadata

        Returns:
            Self: CmdOutputMetadata instance with parsed values
        """
        try:
            metadata = json.loads(match.group(1))
        except:
            metadata = cls.best_effort_match(match.group(1))
        # Create a copy of metadata to avoid modifying the original
        processed = metadata.copy()
        # Convert numeric fields
        if "exit_code" in metadata:
            try:
                processed["exit_code"] = int(float(str(metadata["exit_code"])))
            except (ValueError, TypeError):
                processed["exit_code"] = -1
        return cls(**processed)


@dataclass
class CommandResult:
    """
    Result of a command execution with output and metadata.

    Attributes:
        output (str): Command output text
        metadata (Optional[CmdOutputMetadata]): Execution context metadata
    """

    output: str
    metadata: Optional[CmdOutputMetadata]

    def to_observation(self, strip: bool = True) -> str:
        """
        Convert command result to formatted observation string.

        Args:
            strip (bool): Whether to truncate long output

        Returns:
            str: Formatted observation with output and context
        """
        # compile regex once for efficiency
        ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        output = ANSI_ESCAPE.sub("", self.output).replace("\r", "")

        if len(output) > 1024 * 8 and strip:
            output = output[: 1024 * 4] + "....stripped due to length....\n" + output[-1024 * 4 :]

        if self.metadata is None:
            return f"\n{output}\n"
        return f"""{output}
{self.metadata.username}@{self.metadata.hostname}:{self.metadata.working_dir} $

exit code: {self.metadata.exit_code}
"""


class Runtime:
    """
    Docker container runtime for repository setup and testing.

    Manages a Docker container with persistent bash session, command execution,
    file operations, and container lifecycle management.
    """

    def __init__(self, container: Container, log_function: Callable[..., None]) -> None:
        """
        Initialize runtime with an existing Docker container.

        Args:
            container (Container): Docker container instance to manage
        """
        self.container = container
        self.logger = log_function  # Set logger early so it's available even if init fails later
        self.sock: Any = self.container.attach_socket(params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1})  # type: ignore
        self.output_queue: queue.Queue[bytes] = queue.Queue()
        self._start_output_thread()
        self._clear_initial_prompt()

        json_str = json.dumps(
            {
                "exit_code": "$?",
                "username": r"\u",
                "hostname": r"\h",
                "working_dir": r"$(pwd)",
                "py_interpreter_path": r'$(which python 2>/dev/null || echo "")',
            },
            indent=2,
        ).replace('"', r"\"")
        ps1 = CMD_OUTPUT_PS1_BEGIN + json_str + CMD_OUTPUT_PS1_END + "\n"
        self.send_command(f'export PROMPT_COMMAND=\'export PS1="{ps1}"\'; export PS2=""')
        self.send_command("apt update -qq && apt install -y -qq git")
        self.stopped = False

    def _stream_output(self):
        while True:
            try:
                output = self._recv_bytes(4096)
                if not output:
                    break
                self.output_queue.put(output)
            except (OSError, ConnectionError) as e:
                print(f"Connection error in _stream_output: {e}")
                break
            except Exception as e:
                # print(f"Unexpected error in _stream_output: {e}")
                break

    def _start_output_thread(self):
        self.output_thread = threading.Thread(target=self._stream_output, daemon=True)
        self.output_thread.start()
        # TODO: kill the thread if main thread is stopped

    def _clear_initial_prompt(self):
        time.sleep(0.5)
        while not self.output_queue.empty():
            self.output_queue.get()

    def _read_raw_output(self, timeout: float = 30) -> tuple[str, Optional[CmdOutputMetadata]]:
        accumulated_output = ""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                chunk = self.output_queue.get(timeout=0.1)
                accumulated_output += chunk.decode("utf-8", errors="ignore")
                # PSReadLine injects ANSI + cursor control; normalize before matching
                accumulated_clean = ANSI_ESCAPE.sub("", accumulated_output).replace("\r", "")
                ps1_matches = CmdOutputMetadata.matches_ps1_metadata(accumulated_clean)
                if ps1_matches:
                    break
            except queue.Empty:
                continue
        accumulated_output = ANSI_ESCAPE.sub("", accumulated_output).replace("\r", "")
        ps1_matches = CmdOutputMetadata.matches_ps1_metadata(accumulated_output)
        metadata = CmdOutputMetadata.from_ps1_match(ps1_matches[-1]) if ps1_matches else None
        output = self._combine_outputs_between_matches(
            accumulated_output,
            ps1_matches,
        )
        return output, metadata

    def _combine_outputs_between_matches(self, pane_content: str, ps1_matches: list[re.Match[str]]) -> str:
        if len(ps1_matches) == 1:
            return pane_content[: ps1_matches[0].start()]
        elif len(ps1_matches) == 0:
            return pane_content
        output_segments: List[str] = []
        for i in range(len(ps1_matches) - 1):
            output_segment = pane_content[ps1_matches[i].end() + 1 : ps1_matches[i + 1].start()]
            output_segments.append(output_segment)
        return "\n".join(output_segments) + "\n" if output_segments else ""

    def _recv_bytes(self, n: int = 4096) -> bytes:
        # Prefer the public API on whatever object the SDK returns
        for m in ("recv", "read"):
            if hasattr(self.sock, m):
                return getattr(self.sock, m)(n)
        # Last-resort fallback for odd wrappers that still expose ._sock
        if hasattr(self.sock, "_sock"):
            for m in ("recv", "read"):
                if hasattr(self.sock._sock, m):
                    return getattr(self.sock._sock, m)(n)
        raise TypeError(f"Don't know how to read from {type(self.sock).__name__}")

    def _send_bytes(self, data: bytes) -> None:
        if hasattr(self.sock, "_sock"):
            for m in ("send", "sendall", "write"):
                if hasattr(self.sock._sock, m):
                    getattr(self.sock._sock, m)(data)
                    return
        for m in ("send", "sendall", "write"):
            if hasattr(self.sock, m):
                getattr(self.sock, m)(data)
                return

        raise TypeError(f"Don't know how to write to {type(self.sock).__name__}")

    def _log_command_result(self, result: CommandResult) -> None:
        claude_code_logger.debug("Docker runtime command finished with metadata: %s", result.metadata)
        if len(result.output) > 2048:
            logged_output = result.output[:1024] + "\n(... stripped due to length ...)\n" + result.output[-1024:]
        else:
            logged_output = result.output
        claude_code_logger.debug(
            "Docker runtime command finished with output (length = %d):\n%s", len(result.output), logged_output
        )
        # Output to the evaluation logger simultaneously
        self.logger(text=logged_output)

    def send_command(self, command: str, timeout: float = 20 * 60) -> CommandResult:
        # Redact sensitive API keys from the command before logging
        redacted_command = command
        for sensitive_var in ["ANTHROPIC_AUTH_TOKEN", "API_KEY", "SECRET_KEY"]:
            pattern = rf"(export (.*?){re.escape(sensitive_var)}(.*?)=)[^\s]+"
            redacted_command = re.sub(pattern, rf"\1****REDACTED****", redacted_command)
        claude_code_logger.info("Docker runtime receiving command: %s", redacted_command)
        # Normalize newline semantics for interactive shells
        if not command.endswith("\n"):
            command += "\n"

        while not self.output_queue.empty():
            self.output_queue.get()

        self._send_bytes(command.encode())

        output, metadata = self._read_raw_output(timeout=timeout)
        # TODO: Check exit code of the command (claude code download fail will not be caught by this)
        if metadata is not None:
            result = CommandResult(output=output, metadata=metadata)
            self._log_command_result(result)
            return result

        # handle timeout
        self._send_bytes(b"\x03")

        kill_timeout = 5.0
        kill_output, kill_metadata = self._read_raw_output(timeout=kill_timeout)

        output = output + kill_output + "\n**Exited due to timeout**\n"
        if kill_metadata is not None:
            kill_metadata.exit_code = TIMEOUT_EXIT_CODE
            result = CommandResult(output=output, metadata=kill_metadata)
            self._log_command_result(result)
            return result

        fallback_metadata = CmdOutputMetadata(
            exit_code=TIMEOUT_EXIT_CODE,
        )
        result = CommandResult(output=output, metadata=fallback_metadata)
        self._log_command_result(result)
        return result

    def cleanup(self) -> None:
        if self.stopped:
            return
        try:
            claude_code_logger.info(f"Stopping container: {self.container.id}")
            self.container.stop()
            claude_code_logger.info(f"Removing container: {self.container.id}")
            self.container.remove(force=True)
            claude_code_logger.info(f"Container removed: {self.container.id}")
            self.stopped = True
        except Exception as e:
            print(f"Failed to stop container: {e}")

    def __del__(self):
        self.cleanup()

    @staticmethod
    def pull_image(image_name: str) -> bool:
        """
        Pull Docker image from registry.

        Args:
            image_name (str): Name of the Docker image to pull

        Returns:
            bool: True if successful, False if image not found
        """
        client = docker.from_env()
        try:
            client.images.pull(image_name)
            return True
        except ImageNotFound:
            return False

    @classmethod
    def start_session(
        cls,
        image_name: str,
        instance: SWEbenchInstance,
        log_function: Callable[..., None] = lambda: None,
    ) -> Runtime:
        """
        Start a Docker container session for repository testing.

        Args:
            image_name (str): Base Docker image name
            instance (dict): SWE-bench instance data with repo info

        Returns:
            SetupRuntime: Configured runtime session ready for command execution

        Raises:
            RuntimeError: If Docker is not available
        """
        try:
            docker.from_env().ping()  # type: ignore
        except DockerException:
            raise RuntimeError("Docker is not installed or not running.")

        _ = cls.pull_image(image_name)
        client = docker.from_env(timeout=600)
        container_id = instance["instance_id"]
        container_name = f"git-launch-{container_id}-{str(uuid.uuid4())[:4]}"
        info: Dict[str, str] = client.version()  # type: ignore
        engine_os: str = (info.get("Os") or info.get("OSType") or "").lower()  # type: ignore
        # which operating system this code is running on, note windows can run linux containers, so engine_os != (container) platform
        extra_hosts = {"host.docker.internal": "host-gateway"} if "linux" in engine_os else None

        shell_command = "/bin/bash"
        working_dir = "/testbed"

        claude_code_logger.info(
            f"Starting container {container_name} with image {image_name}. Shell command: {shell_command}"
        )
        container = client.containers.run(
            image_name,
            name=container_name,
            command=shell_command,
            stdin_open=True,
            tty=True,
            detach=True,
            environment={
                "TERM": "xterm-mono",
            },
            working_dir=working_dir,
            extra_hosts=extra_hosts,
            network_mode="host",
            cpu_quota=int(CPU_CORES * 100000),
            mem_limit=MEM_LIMIT,
        )
        claude_code_logger.info(f"Container {container_name} started with ID: {container.id}")

        session = cls(container, log_function=log_function)

        return session
