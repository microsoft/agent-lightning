# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    TypeVar,
)

from pymongo import AsyncMongoClient

from agentlightning.types import Rollout

from .base import LightningStoreCapabilities, is_finished
from .collection.mongo import MongoClientPool, MongoLightningCollections, MongoOperationPrometheusTracker
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
        client: The MongoDB client. Could be a string URI or an instance of AsyncMongoClient.
        database: The MongoDB database. Could be a string name or an instance of AsyncDatabase.
            You must provide at least one of client or database.
        partition_id: The partition id. Useful when sharing the database among multiple Agent-lightning trainers.
    """

    def __init__(
        self,
        *,
        client: AsyncMongoClient[Mapping[str, Any]] | str,
        database_name: str | None = None,
        partition_id: str | None = None,
        prometheus: bool = False,
    ) -> None:
        self._enable_prometheus = prometheus
        self._auto_created_client = False
        if isinstance(client, str):
            self._client = AsyncMongoClient[Mapping[str, Any]](client)
            self._auto_created_client = True
        else:
            self._client = client
        if database_name is None:
            database_name = "agentlightning"
            logger.info("No database name provided, using default 'agentlightning'")

        if partition_id is None:
            partition_id = _generate_partition_id()
            logger.info("No partition id provided, generated a new one: %s", partition_id)

        self._client_pool = MongoClientPool(self._client)

        super().__init__(
            collections=MongoLightningCollections(
                self._client_pool,
                database_name,
                partition_id,
                prometheus_tracker=MongoOperationPrometheusTracker(enabled=self._enable_prometheus),
            ),
            prometheus=self._enable_prometheus,
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
        # If I created the client, I should close it too.
        if self._auto_created_client:
            await self._client.close()

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
            # Query the rollouts that are not finished in a single query
            rollouts = await self.collections.rollouts.query(
                filter={"rollout_id": {"within": list(unfinished_rollout_ids)}}
            )
            for rollout in rollouts.items:
                if is_finished(rollout):
                    finished_rollouts[rollout.rollout_id] = rollout
                    unfinished_rollout_ids.remove(rollout.rollout_id)

            if not unfinished_rollout_ids:
                break

            # Poll every 10 seconds by default
            rest_time = max(0.01, min(deadline - time.time(), 10.0)) if deadline is not None else 10.0
            await asyncio.sleep(rest_time)
            current_time = time.time()

        # Reorder the rollouts to match the input order
        return [finished_rollouts[rollout_id] for rollout_id in rollout_ids if rollout_id in finished_rollouts]
