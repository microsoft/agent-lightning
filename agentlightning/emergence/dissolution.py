# Copyright (c) Microsoft. All rights reserved.

"""Gap 4: Policy dissolution mechanism.

Manages resource lifecycle with TTL, validity conditions, and re-validation.
Wraps a LightningStore to intercept resource retrieval and check dissolution
conditions before returning resources.

Dissolution condition: if resource lifecycle management moves into
CollectionBasedLightningStore natively, the wrapper adds indirection
without benefit.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from agentlightning.store.base import LightningStore
from agentlightning.types import ResourcesUpdate

from .types import ConditionResult, DissolutionAction, DissolutionSignal

logger = logging.getLogger(__name__)


class DissolutionPolicy(str, Enum):
    """What to do when dissolution triggers."""

    REVALIDATE = "revalidate"
    """Re-run validation, keep if still good."""
    REGRESS = "regress"
    """Fall back to previous version."""
    EXPLORE = "explore"
    """Switch to exploration mode (no resource pinning)."""


class ValidityCondition(BaseModel):
    """A condition that must remain true for a resource to be valid."""

    name: str
    description: str = ""
    check_type: Literal["reward_threshold", "entropy_threshold", "custom"] = "custom"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ValidationRecord(BaseModel):
    """Record of a re-validation attempt."""

    timestamp: float
    trigger: str
    result: str
    """'valid' | 'invalid' | 'error'"""
    details: str = ""


class DissolutionMetadata(BaseModel):
    """Metadata attached to resource versions for dissolution tracking.

    Stored in ResourcesUpdate's metadata dict under the key
    'agentlightning.emergence.dissolution'.
    """

    ttl_seconds: Optional[int] = None
    """Time-to-live. After this duration, the resource should be
    re-validated before use. None = no temporal expiry."""

    created_at: float = 0.0
    """Timestamp when this resource version was created."""

    validity_conditions: List[ValidityCondition] = Field(default_factory=list)

    validation_history: List[ValidationRecord] = Field(default_factory=list)

    on_dissolution: DissolutionPolicy = DissolutionPolicy.REVALIDATE


class DissolutionEngine:
    """Manages resource lifecycle with TTL, validity conditions, and re-validation.

    Wraps a LightningStore to intercept resource retrieval and check
    dissolution conditions before returning resources.

    The engine NEVER automatically removes resources. It surfaces
    dissolution signals. The algorithm (or human operator) decides
    whether to act.
    """

    def __init__(
        self,
        store: LightningStore,
        default_ttl: Optional[int] = None,
        check_interval: int = 10,
    ):
        self._store = store
        self._default_ttl = default_ttl
        self._check_interval = check_interval
        self._dissolution_cache: Dict[str, DissolutionMetadata] = {}
        self._check_counter = 0
        self._condition_checkers: Dict[str, Callable[..., ConditionResult]] = {}

    def register_condition_checker(
        self,
        check_type: str,
        checker: Callable[..., ConditionResult],
    ) -> None:
        """Register a function that evaluates a specific condition type."""
        self._condition_checkers[check_type] = checker

    async def get_resources_with_dissolution_check(
        self,
        resources_id: Optional[str] = None,
    ) -> Tuple[Optional[ResourcesUpdate], Optional[DissolutionSignal]]:
        """Fetch resources, checking dissolution conditions.

        Returns the resources AND any dissolution signal. The caller
        decides what to do -- the engine does not block resource access.
        """
        if resources_id:
            resources = await self._store.get_resources_by_id(resources_id)
        else:
            resources = await self._store.get_latest_resources()

        if resources is None:
            return None, None

        self._check_counter += 1
        if self._check_counter % self._check_interval != 0:
            return resources, None

        signal = self._check_dissolution(resources)
        return resources, signal

    def _check_dissolution(self, resources: ResourcesUpdate) -> Optional[DissolutionSignal]:
        """Check TTL and validity conditions for a resource."""
        meta = self._dissolution_cache.get(resources.resources_id)
        if meta is None:
            return None

        now = time.time()

        # TTL check
        ttl = meta.ttl_seconds or self._default_ttl
        if ttl is not None and meta.created_at > 0:
            age = now - meta.created_at
            if age > ttl:
                severity = "warning" if age < ttl * 2 else "critical"
                return DissolutionSignal(
                    trigger="ttl_expired",
                    severity=severity,
                    recommendation=(
                        f"Resource version {resources.resources_id} has been active for "
                        f"{age / 3600:.1f}h (TTL: {ttl / 3600:.1f}h). "
                        f"Could indicate the environment has changed since training. "
                        f"Consider re-validation."
                    ),
                )

        # Validity condition checks
        for condition in meta.validity_conditions:
            result = self._evaluate_condition(condition)
            if result is not None and not result.passed:
                severity = "warning"
                return DissolutionSignal(
                    trigger=f"condition_failed:{condition.name}",
                    severity=severity,
                    recommendation=(
                        f"Validity condition '{condition.name}' failed for resource "
                        f"{resources.resources_id}: {result.description}. "
                        f"Could indicate the resource is no longer valid for the "
                        f"current environment."
                    ),
                )

        return None

    def _evaluate_condition(self, condition: ValidityCondition) -> Optional[ConditionResult]:
        """Evaluate a single validity condition."""
        checker = self._condition_checkers.get(condition.check_type)
        if checker is None:
            logger.debug(
                "No checker registered for condition type '%s'; skipping.",
                condition.check_type,
            )
            return None

        try:
            return checker(condition)
        except Exception:
            logger.debug("Condition check failed for '%s'.", condition.name, exc_info=True)
            return ConditionResult(
                condition_name=condition.name,
                passed=True,  # Best-effort: don't block on check failure
                description="Check failed; treated as passing (best-effort).",
            )

    async def attach_dissolution_metadata(
        self,
        resources_id: str,
        ttl_seconds: Optional[int] = None,
        validity_conditions: Optional[List[ValidityCondition]] = None,
        policy: DissolutionPolicy = DissolutionPolicy.REVALIDATE,
    ) -> None:
        """Attach dissolution metadata to a resource version."""
        self._dissolution_cache[resources_id] = DissolutionMetadata(
            ttl_seconds=ttl_seconds or self._default_ttl,
            created_at=time.time(),
            validity_conditions=validity_conditions or [],
            on_dissolution=policy,
        )

    async def check_conditions(
        self,
        resources_id: str,
    ) -> List[ConditionResult]:
        """Evaluate all validity conditions for a resource version."""
        meta = self._dissolution_cache.get(resources_id)
        if meta is None:
            return []

        results: List[ConditionResult] = []
        for condition in meta.validity_conditions:
            result = self._evaluate_condition(condition)
            if result is not None:
                results.append(result)
        return results

    async def dissolve(
        self,
        resources_id: str,
        trigger: str,
    ) -> DissolutionAction:
        """Execute dissolution policy for a resource version.

        REVALIDATE: signal that re-validation is needed
        REGRESS: find previous version, mark current as dissolved
        EXPLORE: clear resource pinning, let runners use no resource
        """
        meta = self._dissolution_cache.get(resources_id)
        policy = meta.on_dissolution if meta else DissolutionPolicy.REVALIDATE

        # Record in validation history
        if meta:
            meta.validation_history.append(
                ValidationRecord(
                    timestamp=time.time(),
                    trigger=trigger,
                    result="dissolved",
                    details=f"Policy: {policy.value}",
                )
            )

        if policy == DissolutionPolicy.REVALIDATE:
            return DissolutionAction(
                resources_id=resources_id,
                policy=policy.value,
                action_taken="revalidation_requested",
                description=(
                    f"Resource {resources_id} marked for re-validation due to: {trigger}. "
                    f"The algorithm should re-run validation rollouts before continuing."
                ),
            )

        elif policy == DissolutionPolicy.REGRESS:
            # Find the previous resource version
            all_resources = await self._store.query_resources(
                sort_by="version", sort_order="desc", limit=2
            )
            if len(all_resources) > 1:
                previous = all_resources[1]
                return DissolutionAction(
                    resources_id=resources_id,
                    policy=policy.value,
                    action_taken=f"regressed_to:{previous.resources_id}",
                    description=(
                        f"Resource {resources_id} dissolved. Regressed to previous "
                        f"version {previous.resources_id} (v{previous.version})."
                    ),
                )
            return DissolutionAction(
                resources_id=resources_id,
                policy=policy.value,
                action_taken="no_previous_version",
                description=(
                    f"Resource {resources_id} dissolution requested but no previous "
                    f"version exists. Could indicate this is the initial resource."
                ),
            )

        else:  # EXPLORE
            return DissolutionAction(
                resources_id=resources_id,
                policy=policy.value,
                action_taken="exploration_mode",
                description=(
                    f"Resource {resources_id} dissolved. Entering exploration mode — "
                    f"runners should proceed without resource pinning to allow "
                    f"behavioral diversity."
                ),
            )
