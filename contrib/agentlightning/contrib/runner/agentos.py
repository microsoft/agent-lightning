# Copyright (c) Microsoft. All rights reserved.

"""
AgentOSRunner - Agent-Lightning Runner with Kernel Safety
==========================================================

Wraps agent execution with Agent-OS kernel governance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T_task = TypeVar("T_task")


@dataclass
class PolicyViolation:
    """Record of a policy violation."""

    policy_name: str
    description: str
    severity: str
    blocked: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def penalty(self) -> float:
        """Calculate penalty based on severity.

        Returns:
            float: Negative penalty value, where more severe violations
                have larger negative magnitudes.
        """
        penalties = {
            "critical": -100.0,
            "high": -50.0,
            "medium": -10.0,
            "low": -1.0,
        }
        return penalties.get(self.severity, -10.0)


@dataclass
class GovernedRollout:
    """Rollout with governance metadata.

    This dataclass wraps execution results with governance information.
    It is compatible with Agent-Lightning's Rollout interface - the
    `task_input`, `task_output`, and `success` fields provide the core
    rollout data, while `violations` adds governance-specific metadata.
    """

    task_input: Any
    task_output: Any
    success: bool
    violations: list[PolicyViolation] = field(default_factory=list)

    @property
    def total_penalty(self) -> float:
        return sum(v.penalty for v in self.violations)


class AgentOSRunner(Generic[T_task]):
    """
    Agent-Lightning runner with Agent-OS kernel safety.

    This runner wraps agent execution in an Agent-OS kernel,
    enforcing policies and collecting violation data for RL training.

    Example:
        >>> from agent_os import KernelSpace
        >>> from agent_os.policies import SQLPolicy
        >>>
        >>> kernel = KernelSpace(policy=SQLPolicy())
        >>> runner = AgentOSRunner(kernel)
        >>>
        >>> rollout = await runner.step(task)
        >>> print(f"Violations: {len(rollout.violations)}")
    """

    def __init__(
        self,
        kernel: Any,
        *,
        fail_on_violation: bool = False,
        emit_violations: bool = True,
    ):
        """
        Initialize the governed runner.

        Args:
            kernel: Agent-OS KernelSpace with loaded policies
            fail_on_violation: Raise exception on violation
            emit_violations: Emit violations as spans
        """
        self.kernel = kernel
        self.fail_on_violation = fail_on_violation
        self.emit_violations = emit_violations

        self._violations: list[PolicyViolation] = []
        self._total_rollouts = 0
        self._total_violations = 0

        # Worker attributes (set by init_worker)
        self.worker_id: Optional[int] = None
        self.store: Optional[Any] = None

        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Set up kernel hooks."""
        on_violation = getattr(self.kernel, "on_policy_violation", None)
        if on_violation is None:
            logger.warning(
                "Kernel %r does not support policy violation hooks via 'on_policy_violation'.",
                self.kernel,
            )
            return
        if not callable(on_violation):
            logger.warning(
                "Kernel attribute 'on_policy_violation' is not callable: %r",
                on_violation,
            )
            return
        try:
            on_violation(self._handle_violation)
        except TypeError as exc:
            logger.warning(
                "Kernel.on_policy_violation has an incompatible signature: %s",
                exc,
            )

    def _handle_violation(
        self,
        policy_name: str,
        description: str,
        severity: str,
        blocked: bool,
    ) -> None:
        """Handle a policy violation."""
        violation = PolicyViolation(
            policy_name=policy_name,
            description=description,
            severity=severity,
            blocked=blocked,
        )
        self._violations.append(violation)
        self._total_violations += 1

        if self.emit_violations:
            self._emit_violation_span(violation)

        if self.fail_on_violation and blocked:
            raise PolicyViolationError(violation)

    def _emit_violation_span(self, violation: PolicyViolation) -> None:
        """Emit violation as Agent-Lightning span."""
        try:
            from agentlightning.emitter import emit_annotation

            emit_annotation(
                {
                    "agent_os.violation": True,
                    "agent_os.policy": violation.policy_name,
                    "agent_os.severity": violation.severity,
                    "agent_os.blocked": violation.blocked,
                }
            )
        except ImportError as exc:
            logger.debug(
                "agentlightning.emitter not available; skipping violation annotation: %s",
                exc,
            )

    @property
    def agent(self) -> Any:
        """
        Access the underlying agent.

        Raises:
            RuntimeError: If the agent has not been initialized via `init`.
        """
        if not hasattr(self, "_agent"):
            raise RuntimeError("AgentOSRunner.agent accessed before `init` has been called.")
        return self._agent

    @agent.setter
    def agent(self, value: Any) -> None:
        """Set the underlying agent instance."""
        self._agent = value

    def init(self, agent: Any, **kwargs: Any) -> None:
        """Initialize with agent."""
        self.agent = agent

    def init_worker(self, worker_id: int, store: Any, **kwargs: Any) -> None:
        """Initialize worker."""
        self.worker_id = worker_id
        self.store = store

    def teardown(self) -> None:
        """Release resources."""
        pass

    def teardown_worker(self, worker_id: int) -> None:
        """Release worker resources."""
        pass

    async def step(
        self,
        input: T_task,
        *,
        resources: Optional[Any] = None,
        mode: Optional[str] = None,
        event: Optional[Any] = None,
    ) -> GovernedRollout:
        """
        Execute task with governance.

        Args:
            input: Task input
            resources: Optional resources
            mode: Rollout mode
            event: Stop signal

        Returns:
            GovernedRollout with results and violations
        """
        self._violations = []

        try:
            if hasattr(self.kernel, "execute_async"):
                logger.debug("AgentOSRunner: executing task via kernel.execute_async")
                result = await self.kernel.execute_async(self.agent, input)
            elif hasattr(self.kernel, "execute"):
                logger.debug("AgentOSRunner: executing task via kernel.execute")
                result = self.kernel.execute(self.agent, input)
            else:
                logger.error(
                    "AgentOSRunner: kernel does not support 'execute_async' or 'execute'; "
                    "governed execution is not possible."
                )
                raise RuntimeError(
                    "Kernel does not support governed execution (missing 'execute_async' and 'execute')."
                )
            success = True
        except PolicyViolationError as e:
            # Record the policy violation and mark rollout as unsuccessful.
            self._violations.append(e.violation)
            result = None
            success = False

        self._total_rollouts += 1

        return GovernedRollout(
            task_input=input,
            task_output=result,
            success=success,
            violations=self._violations.copy(),
        )

    def get_stats(self) -> dict:
        """Get runner statistics."""
        return {
            "total_rollouts": self._total_rollouts,
            "total_violations": self._total_violations,
            "violation_rate": (self._total_violations / self._total_rollouts if self._total_rollouts > 0 else 0.0),
        }


class PolicyViolationError(Exception):
    """Raised when policy violation blocks execution."""

    def __init__(self, violation: PolicyViolation):
        self.violation = violation
        super().__init__(f"Policy violation: {violation.description}")
