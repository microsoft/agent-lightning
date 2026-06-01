"""
LLM-in-Sandbox: A lightweight framework that connects LLMs to a virtual computer (Docker-based sandbox) to build general-purpose agents
"""

__version__ = "0.1.0"

# Default timeout for commands
CMD_TIMEOUT = 120

# Docker PATH
DOCKER_PATH = "/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Import main classes for easy access
from .agent import Agent, AgentArgs, get_logger
from .docker_runtime import DockerRuntime
from .trajectory import Trajectory, TrajectoryStep
from .action import Action

__all__ = [
    "Agent",
    "AgentArgs", 
    "DockerRuntime",
    "Trajectory",
    "TrajectoryStep",
    "Action",
    "get_logger",
    "__version__",
]
