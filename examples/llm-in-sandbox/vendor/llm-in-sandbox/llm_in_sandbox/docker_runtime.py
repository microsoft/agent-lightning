"""
Docker Runtime - Execute commands in a Docker container
"""
import os
import re
import time
import uuid
import tarfile
import io
import datetime
import hashlib
import logging
import shlex
import docker
from typing import Dict, Tuple, Any, Optional

from . import CMD_TIMEOUT, DOCKER_PATH
import subprocess


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class LocalRuntime:

    def __init__(
        self,
        logger=None,
        repo_path: str = "/testbed",
    ):
        if logger is None:
            self.logger = get_logger("LocalRuntime")
        else:
            self.logger = logger

        self.repo_path = repo_path

    def start_container(self):
        self.logger.info("Skip starting container. Run in local environment.")

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str = None,
    ) -> Tuple[str, str]:
        """
        Execute a command locally.

        Returns:
            Tuple of (output, exit_code_or_error).
        """

        exec_workdir = self.repo_path if workdir is None else workdir

        env = os.environ.copy()
        env.update({
            "PATH": DOCKER_PATH,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_ROOT_USER_ACTION": "ignore",
            "PIP_NO_WARN_SCRIPT_LOCATION": "1",
        })

        try:
            result = subprocess.run(
                ["bash", "-c", code],
                cwd=exec_workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = (result.stdout or "") + (result.stderr or "")
            exit_code = result.returncode

            # Remove ANSI escape codes
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            if exit_code != 0:
                return output, f"Error: Exit code {exit_code}"

            return output, str(exit_code)

        except subprocess.TimeoutExpired as e:
            return f"The command took too long to execute (>{timeout}s)", "-1"

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str = None,
    ) -> Tuple[str, str, str]:
        """
        Execute a command locally with separate stdout and stderr.

        Returns:
            Tuple of (stdout, stderr, exit_code_or_error).
        """

        exec_workdir = self.repo_path if workdir is None else workdir

        env = os.environ.copy()
        env.update({
            "PATH": DOCKER_PATH,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_ROOT_USER_ACTION": "ignore",
            "PIP_NO_WARN_SCRIPT_LOCATION": "1",
        })

        try:
            result = subprocess.run(
                ["bash", "-c", code],
                cwd=exec_workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            exit_code = result.returncode

            # Remove ANSI escape codes
            stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
            stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)

            if exit_code != 0:
                return stdout, stderr, f"Error: Exit code {exit_code}"

            return stdout, stderr, str(exit_code)

        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ""
            stderr = e.stderr or ""
            return stdout, stderr + f"\nThe command took too long to execute (>{timeout}s)", "-1"

        except Exception as e:
            error_msg = f"Error: {repr(e)}"
            return "", error_msg, "-1"

    def close(self):
        self.logger.info("Skip closing container. Run in local environment.")

class DockerRuntime:
    """
    Docker runtime for executing commands in a Docker container.
    """

    def __init__(
        self,
        docker_image: str,
        repo_path: str = "/testbed",
        command: str = "sleep infinity",
        logger=None,
        **docker_kwargs,
    ):
        self.docker_image = docker_image
        self.repo_path = repo_path
        self.command = command
        self.docker_kwargs = docker_kwargs
        
        if logger is None:
            self.logger = get_logger("DockerRuntime")
        elif logger is False:
            # Quiet mode - only show warnings and errors
            self.logger = logging.getLogger("DockerRuntime.quiet")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger = logger

        self.client = docker.from_env(timeout=120)
        
        # Start the container
        self.container = None
        self.container_name = self._get_container_name(docker_image)
        self.start_container(docker_image, command, self.container_name, **docker_kwargs)
        
        # Initialize the environment
        self.setup_env()
        self.logger.info(f"Docker environment initialized")
        self.logger.info(f"Docker image: {self.docker_image}")
        self.logger.info(f"Container ID: {self.container.id}")

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Return name of container"""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-").replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def start_container(self, docker_image: str, command: str, container_name: str, **docker_kwargs):
        """Start a Docker container."""
        try:
            # Check if container already exists
            self.container = self.client.containers.get(container_name)
            self.logger.info(f"Found existing container: {container_name}")
            if self.container.status != "running":
                self.container.start()
            return
        except docker.errors.NotFound:
            pass

        # Pull the image if not present
        try:
            self.client.images.get(docker_image)
        except docker.errors.ImageNotFound:
            self.logger.info(f"Pulling Docker image: {docker_image}")
            self.client.images.pull(docker_image)

        # Create and start the container
        env_vars = {
            "PATH": DOCKER_PATH,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",  # Disable pip version check notice
            "PIP_ROOT_USER_ACTION": "ignore",  # Suppress root user warning
            "PIP_NO_WARN_SCRIPT_LOCATION": "1",  # Suppress script location warning
            **docker_kwargs.get("environment", {})
        }
        
        self.container = self.client.containers.run(
            docker_image,
            command=command,
            name=container_name,
            detach=True,
            stdin_open=True,
            tty=True,
            environment=env_vars,
            working_dir=self.repo_path,
            **{k: v for k, v in docker_kwargs.items() if k != "environment"},
        )
        self.logger.info(f"Started container: {container_name}")

    def setup_env(self):
        """Setup the container environment."""
        # Ensure working directory exists
        self.run(f"mkdir -p {self.repo_path}")
        # Create input/output directories
        self.run(f"mkdir -p {self.repo_path}/input {self.repo_path}/output")
        # Initialize git repo for tracking changes (optional)
        self.run(f"cd {self.repo_path} && git init 2>/dev/null || true")
        
        # Configure pip to use Tsinghua mirror (faster for users in China Mainland)
        self.run("mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'EOF'\n"
                 "[global]\n"
                 "index-url = https://pypi.tuna.tsinghua.edu.cn/simple\n"
                 "trusted-host = pypi.tuna.tsinghua.edu.cn\n"
                 "EOF")

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str = None,
    ) -> Tuple[str, str]:
        """
        Execute a command in the container.

        Returns:
            Tuple of (output, exit_code_or_error).
        """
        exec_workdir = self.repo_path if workdir is None else workdir
        
        # Wrap the entire command with timeout inside a bash -c
        # This ensures shell built-ins like 'cd' work correctly
        # Use shlex.quote to properly escape the command for shell execution
        command = ["bash", "-c", f"timeout {timeout} bash -c {shlex.quote(code)}"]

        try:
            exec_result = self.container.exec_run(
                command,
                workdir=exec_workdir,
                environment={
                    "PATH": DOCKER_PATH,
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",  # Disable pip version check notice
                    "PIP_ROOT_USER_ACTION": "ignore",  # Suppress root user warning
                    "PIP_NO_WARN_SCRIPT_LOCATION": "1",  # Suppress script location warning
                },
            )
            output = exec_result.output.decode("utf-8", errors="replace")
            exit_code = exec_result.exit_code

            if exit_code == 124:
                return f"The command took too long to execute (>{timeout}s)", "-1"

            # Remove ANSI escape codes
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            
            if exit_code != 0:
                return output, f"Error: Exit code {exit_code}"

            return output, str(exit_code)

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str = None,
    ) -> Tuple[str, str, str]:
        """
        Execute a command in the container with separate stdout and stderr.
        Uses demux=True to get separate stdout and stderr streams.

        Returns:
            Tuple of (stdout, stderr, exit_code_or_error).
        """
        exec_workdir = self.repo_path if workdir is None else workdir
        
        # Wrap the entire command with timeout inside a bash -c
        command = ["bash", "-c", f"timeout {timeout} bash -c {shlex.quote(code)}"]

        try:
            exec_result = self.container.exec_run(
                command,
                workdir=exec_workdir,
                demux=True,  # Key change: separate stdout and stderr
                environment={
                    "PATH": DOCKER_PATH,
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PIP_ROOT_USER_ACTION": "ignore",
                    "PIP_NO_WARN_SCRIPT_LOCATION": "1",
                },
            )
            
            # Unpack the result - when demux=True, output is a tuple of (stdout_data, stderr_data)
            stdout_data, stderr_data = exec_result.output
            exit_code = exec_result.exit_code

            # Handle None cases and decode the outputs
            stdout = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
            stderr = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""

            if exit_code == 124:
                return f"The command took too long to execute (>{timeout}s)", "", "-1"

            # Remove ANSI escape codes
            stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
            stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)
            
            if exit_code != 0:
                return stdout, stderr, f"Error: Exit code {exit_code}"

            return stdout, stderr, str(exit_code)

        except Exception as e:
            error_msg = f"Error: {repr(e)}"
            return error_msg, error_msg, "-1"

    def copy_to_container(self, src_path: str, dest_path: str):
        """Copy a file into the container."""
        try:
            # Ensure destination directory exists
            dest_dir = os.path.dirname(dest_path)
            if dest_dir:
                self.run(f"mkdir -p {dest_dir}")

            # Create a tar archive in memory
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tar.add(src_path, arcname=os.path.basename(dest_path))
            tar_stream.seek(0)

            # Put the archive into the container
            self.container.put_archive(dest_dir or "/", tar_stream)
            self.logger.info(f"Copied {src_path} to {dest_path}")
        except Exception as e:
            self.logger.error(f"Error copying file to container: {repr(e)}")
            raise

    def copy_dir_to_container(self, src_dir: str, dest_dir: str):
        """Copy a directory into the container."""
        try:
            # Ensure destination directory exists
            self.run(f"mkdir -p {dest_dir}")

            # Create a tar archive in memory
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                for item in os.listdir(src_dir):
                    item_path = os.path.join(src_dir, item)
                    tar.add(item_path, arcname=item)
            tar_stream.seek(0)

            # Put the archive into the container
            self.container.put_archive(dest_dir, tar_stream)
            self.logger.info(f"Copied directory {src_dir} to {dest_dir}")
        except Exception as e:
            self.logger.error(f"Error copying directory to container: {repr(e)}")
            raise

    def copy_from_container(self, container_path: str, local_path: str):
        """Copy files from container to local path."""
        try:
            # Get the archive from container
            bits, stat = self.container.get_archive(container_path)
            
            # Create local directory
            os.makedirs(local_path, exist_ok=True)
            
            # Extract the archive
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            
            # Get the base directory name from container_path (e.g., "output" from "/testbed/output")
            base_dir = os.path.basename(container_path.rstrip('/'))
            
            with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                # Process all members: strip base_dir prefix and extract
                members_to_extract = []
                for member in tar.getmembers():
                    # Skip the root directory itself
                    if member.name == base_dir:
                        continue
                    
                    # Strip the base directory from the path
                    if member.name.startswith(base_dir + '/'):
                        member.name = member.name[len(base_dir) + 1:]
                    
                    if not member.name:
                        continue
                    
                    members_to_extract.append(member)
                
                # Extract all members (handles nested dirs automatically)
                for member in members_to_extract:
                    target_path = os.path.join(local_path, member.name)
                    # Create parent directories for all types (files, dirs, symlinks)
                    parent_dir = os.path.dirname(target_path)
                    if parent_dir:
                        os.makedirs(parent_dir, exist_ok=True)
                    
                    tar.extract(member, path=local_path)
            
            self.logger.info(f"Copied {container_path} from container to {local_path}")
        except Exception as e:
            self.logger.error(f"Error copying from container: {repr(e)}")
            raise

    def get_task_instruction(self, problem_statement: str) -> str:
        """Returns the task instructions."""
        return problem_statement

    def close(self):
        """Stop and remove the container."""
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
                self.logger.info(f"Container {self.container_name} stopped and removed")
            except Exception as e:
                self.logger.warning(f"Error stopping container: {e}")
            finally:
                self.container = None  # Mark as closed
    
    def __del__(self):
        """Destructor - ensure container is cleaned up even if close() wasn't called."""
        if hasattr(self, 'container') and self.container is not None:
            try:
                self.container.stop(timeout=2)
                self.container.remove(force=True)
            except Exception:
                pass  # Best effort cleanup
