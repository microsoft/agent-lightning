# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar, Union

from agentlightning.types import Attempt, AttemptedRollout, Rollout
from agentlightning.utils.metrics import MetricsBackend

from .base import LightningStoreCapabilities, is_finished
from .collection.mongo import MongoClientPool, MongoLightningCollections
from .collection_based import CollectionBasedLightningStore, healthcheck_before, tracked

T_callable = TypeVar("T_callable", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def _generate_partition_id() -> str:
    return "pt-" + hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:12]


class MongoLightningStore(CollectionBasedLightningStore[MongoLightningCollections]):
    """
    MongoDB implementation of LightningStore using MongoDB collections.
    Data is persistent and can be shared between multiple processes.

    Args:
        mongo_uri: MongoDB connection string (defaults to local replica set).
        mongo_client_kwargs: Extra keyword arguments forwarded to `AsyncMongoClient`.
        database: The MongoDB database name. Defaults to ``agentlightning``.
        partition_id: The partition id. Useful when sharing the database among multiple Agent-lightning trainers.
        tracker: The metrics tracker to use.
        scan_debounce_seconds: The debounce time for the scan for unhealthy rollouts.
            Set to 0 to disable debouncing.
    """

    def __init__(
        self,
        *,
        mongo_uri: str = "mongodb://localhost:27017/?replicaSet=rs0",
        mongo_client_kwargs: Mapping[str, Any] | None = None,
        database_name: str | None = None,
        partition_id: str | None = None,
        tracker: MetricsBackend | None = None,
        scan_debounce_seconds: float = 10.0,
    ) -> None:
        self._mongo_uri = mongo_uri
        self._mongo_client_kwargs = dict(mongo_client_kwargs or {})

        if database_name is None:
            database_name = "agentlightning"
            logger.info("No database name provided, using default 'agentlightning'")

        if partition_id is None:
            partition_id = _generate_partition_id()
            logger.info("No partition id provided, generated a new one: %s", partition_id)

        self._client_pool = MongoClientPool[Mapping[str, Any]](
            mongo_uri=self._mongo_uri,
            mongo_client_kwargs=self._mongo_client_kwargs,
        )

        super().__init__(
            collections=MongoLightningCollections(
                self._client_pool,
                database_name,
                partition_id,
                tracker=tracker,
            ),
            tracker=tracker,
            scan_debounce_seconds=scan_debounce_seconds,
        )

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        return LightningStoreCapabilities(
            thread_safe=True,
            async_safe=True,
            zero_copy=True,
            otlp_traces=False,
        )

    async def close(self) -> None:
        """Close the store by closing the client pool."""
        await self._client_pool.close()

    @tracked("wait_for_rollouts")
    @healthcheck_before
    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        """Wait for specified rollouts to complete with a timeout.

        Concurrently wait for all rollouts to complete with a timeout.
        """
        start_time = time.time()
        current_time = start_time
        deadline = start_time + timeout if timeout is not None else None

        finished_rollouts: Dict[str, Rollout] = {}
        unfinished_rollout_ids = set(rollout_ids)

        while deadline is None or current_time <= deadline:
            async with self.collections.atomic(
                mode="r", snapshot=self._read_snapshot, labels=["rollouts"]
            ) as collections:
                # Query the rollouts that are not finished in a single query
                rollouts = await collections.rollouts.query(
                    filter={"rollout_id": {"within": list(unfinished_rollout_ids)}}
                )
            for rollout in rollouts.items:
                if is_finished(rollout):
                    finished_rollouts[rollout.rollout_id] = rollout
                    unfinished_rollout_ids.remove(rollout.rollout_id)

            if not unfinished_rollout_ids:
                break

            # Poll every 10 seconds by default
            # Minus 0.1 to make sure the time is still sufficient for another call
            rest_time = max(0.01, min(deadline - time.time() - 0.1, 10.0)) if deadline is not None else 10.0
            await asyncio.sleep(rest_time)
            current_time = time.time()

        # Logging will help debugging when there are stuck rollouts.
        logger.debug(
            "Waiting for rollouts. Number of finished rollouts: %d; number of unfinished rollouts: %d",
            len(finished_rollouts),
            len(unfinished_rollout_ids),
        )
        if len(unfinished_rollout_ids) < 30:
            logger.debug("Unfinished rollouts: %s", unfinished_rollout_ids)

        # Reorder the rollouts to match the input order
        return [finished_rollouts[rollout_id] for rollout_id in rollout_ids if rollout_id in finished_rollouts]

    @tracked("_unlocked_many_rollouts_to_attempted_rollouts")
    async def _unlocked_many_rollouts_to_attempted_rollouts(
        self, collections: MongoLightningCollections, rollouts: Sequence[Rollout]
    ) -> List[Union[Rollout, AttemptedRollout]]:
        """Query the latest attempts for the rollouts, and attach them to the rollout objects."""
        async with collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["attempts"]) as collections:
            attempts = await collections.attempts.query(
                filter={"rollout_id": {"within": [rollout.rollout_id for rollout in rollouts]}},
                sort={"name": "sequence_id", "order": "desc"},
            )
        latest_attempts: Dict[str, Attempt] = {}
        for attempt in attempts:
            if attempt.rollout_id not in latest_attempts:
                latest_attempts[attempt.rollout_id] = attempt
            # Otherwise we ignore the attempt because there's already a newer attempt

        return [
            (
                AttemptedRollout(**rollout.model_dump(), attempt=latest_attempts[rollout.rollout_id])
                if rollout.rollout_id in latest_attempts
                else rollout
            )
            for rollout in rollouts
        ]
