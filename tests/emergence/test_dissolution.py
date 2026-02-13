# Copyright (c) Microsoft. All rights reserved.

"""Tests for Gap 4: Dissolution engine."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentlightning.emergence.dissolution import (
    DissolutionEngine,
    DissolutionMetadata,
    DissolutionPolicy,
    ValidityCondition,
)
from agentlightning.emergence.types import ConditionResult, DissolutionSignal
from agentlightning.store.base import LightningStore
from agentlightning.types import ResourcesUpdate


def _make_resources(
    resources_id: str = "res-1",
    version: int = 1,
) -> ResourcesUpdate:
    return ResourcesUpdate(
        resources_id=resources_id,
        create_time=time.time(),
        update_time=time.time(),
        version=version,
        resources={},
    )


class FakeStore(LightningStore):
    """Minimal fake store for dissolution tests."""

    def __init__(self, resources: Optional[List[ResourcesUpdate]] = None):
        self._resources = resources if resources is not None else [_make_resources()]

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        return self._resources[-1] if self._resources else None

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        for r in self._resources:
            if r.resources_id == resources_id:
                return r
        return None

    async def query_resources(self, **kwargs: Any) -> Sequence[ResourcesUpdate]:
        result = list(self._resources)
        if kwargs.get("sort_order") == "desc":
            result.reverse()
        limit = kwargs.get("limit", -1)
        if limit > 0:
            result = result[:limit]
        return result


class TestDissolutionEngine:
    @pytest.mark.asyncio
    async def test_no_signal_without_metadata(self):
        store = FakeStore()
        engine = DissolutionEngine(store, check_interval=1)
        resources, signal = await engine.get_resources_with_dissolution_check()
        assert resources is not None
        assert signal is None

    @pytest.mark.asyncio
    async def test_ttl_expiry_signal(self):
        store = FakeStore()
        engine = DissolutionEngine(store, check_interval=1)

        # Attach metadata with expired TTL
        await engine.attach_dissolution_metadata(
            "res-1",
            ttl_seconds=1,
        )
        # Backdate the creation time
        engine._dissolution_cache["res-1"].created_at = time.time() - 100

        resources, signal = await engine.get_resources_with_dissolution_check("res-1")
        assert resources is not None
        assert signal is not None
        assert signal.trigger == "ttl_expired"
        assert "could indicate" in signal.recommendation.lower() or "Could indicate" in signal.recommendation

    @pytest.mark.asyncio
    async def test_condition_failure_signal(self):
        store = FakeStore()
        engine = DissolutionEngine(store, check_interval=1)

        condition = ValidityCondition(
            name="min_reward",
            check_type="reward_threshold",
            parameters={"threshold": 0.5},
        )
        await engine.attach_dissolution_metadata(
            "res-1",
            validity_conditions=[condition],
        )
        # Register a checker that fails
        engine.register_condition_checker(
            "reward_threshold",
            lambda cond: ConditionResult(
                condition_name=cond.name,
                passed=False,
                value=0.3,
                threshold=0.5,
                description="Reward below threshold.",
            ),
        )

        resources, signal = await engine.get_resources_with_dissolution_check("res-1")
        assert signal is not None
        assert "condition_failed" in signal.trigger

    @pytest.mark.asyncio
    async def test_check_interval_respected(self):
        store = FakeStore()
        engine = DissolutionEngine(store, check_interval=5)

        await engine.attach_dissolution_metadata("res-1", ttl_seconds=1)
        engine._dissolution_cache["res-1"].created_at = time.time() - 100

        # First 4 checks should not check dissolution
        for _ in range(4):
            _, signal = await engine.get_resources_with_dissolution_check("res-1")
            assert signal is None

        # 5th check should trigger
        _, signal = await engine.get_resources_with_dissolution_check("res-1")
        assert signal is not None

    @pytest.mark.asyncio
    async def test_dissolve_revalidate(self):
        store = FakeStore()
        engine = DissolutionEngine(store)
        await engine.attach_dissolution_metadata(
            "res-1",
            policy=DissolutionPolicy.REVALIDATE,
        )
        action = await engine.dissolve("res-1", "test_trigger")
        assert action.policy == "revalidate"
        assert "re-validation" in action.description.lower()

    @pytest.mark.asyncio
    async def test_dissolve_regress(self):
        resources = [_make_resources("res-1", 1), _make_resources("res-2", 2)]
        store = FakeStore(resources)
        engine = DissolutionEngine(store)
        await engine.attach_dissolution_metadata(
            "res-2",
            policy=DissolutionPolicy.REGRESS,
        )
        action = await engine.dissolve("res-2", "test_trigger")
        assert action.policy == "regress"
        assert "res-1" in action.action_taken

    @pytest.mark.asyncio
    async def test_dissolve_regress_no_previous(self):
        store = FakeStore([_make_resources("res-1")])
        engine = DissolutionEngine(store)
        await engine.attach_dissolution_metadata(
            "res-1",
            policy=DissolutionPolicy.REGRESS,
        )
        action = await engine.dissolve("res-1", "test_trigger")
        assert "no_previous_version" in action.action_taken

    @pytest.mark.asyncio
    async def test_dissolve_explore(self):
        store = FakeStore()
        engine = DissolutionEngine(store)
        await engine.attach_dissolution_metadata(
            "res-1",
            policy=DissolutionPolicy.EXPLORE,
        )
        action = await engine.dissolve("res-1", "test_trigger")
        assert action.policy == "explore"
        assert "exploration" in action.description.lower()

    @pytest.mark.asyncio
    async def test_check_conditions(self):
        store = FakeStore()
        engine = DissolutionEngine(store)
        condition = ValidityCondition(
            name="test_cond",
            check_type="custom",
        )
        await engine.attach_dissolution_metadata(
            "res-1",
            validity_conditions=[condition],
        )
        engine.register_condition_checker(
            "custom",
            lambda cond: ConditionResult(
                condition_name=cond.name,
                passed=True,
            ),
        )
        results = await engine.check_conditions("res-1")
        assert len(results) == 1
        assert results[0].passed

    @pytest.mark.asyncio
    async def test_check_conditions_empty(self):
        store = FakeStore()
        engine = DissolutionEngine(store)
        results = await engine.check_conditions("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_resources(self):
        store = FakeStore([])
        engine = DissolutionEngine(store)
        resources, signal = await engine.get_resources_with_dissolution_check()
        assert resources is None
        assert signal is None

    @pytest.mark.asyncio
    async def test_validation_history_recorded(self):
        store = FakeStore()
        engine = DissolutionEngine(store)
        await engine.attach_dissolution_metadata("res-1")
        await engine.dissolve("res-1", "test")
        meta = engine._dissolution_cache["res-1"]
        assert len(meta.validation_history) == 1
        assert meta.validation_history[0].trigger == "test"
