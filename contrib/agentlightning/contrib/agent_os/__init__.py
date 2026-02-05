"""
Agent-OS Integration for Agent-Lightning
=========================================

Provides kernel-level safety during RL training.

Components:
- AgentOSRunner: Runner with policy enforcement
- PolicyReward: Convert violations to RL penalties
- FlightRecorderAdapter: Import audit logs

Example:
    >>> from agentlightning.contrib.agent_os import AgentOSRunner, PolicyReward
    >>> from agent_os import KernelSpace
    >>> 
    >>> kernel = KernelSpace(policy="safety-critical")
    >>> runner = AgentOSRunner(kernel)
    >>> reward_fn = PolicyReward(kernel)
"""

from .runner import AgentOSRunner
from .reward import PolicyReward
from .adapter import FlightRecorderAdapter

__all__ = [
    "AgentOSRunner",
    "PolicyReward",
    "FlightRecorderAdapter",
]
