# Copyright (c) Microsoft. All rights reserved.

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

from .adapter import FlightRecorderAdapter
from .reward import PolicyReward
from .runner import AgentOSRunner

__all__ = [
    "AgentOSRunner",
    "PolicyReward",
    "FlightRecorderAdapter",
]
