# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import (
    Any,
    Callable,
    Mapping,
    TypeVar,
)

from pymongo import AsyncMongoClient

from .base import LightningStoreCapabilities
from .collection.mongo import MongoClientPool, MongoLightningCollections
from .collection_based import CollectionBasedLightningStore

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
    ) -> None:
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

        super().__init__(collections=MongoLightningCollections(self._client_pool, database_name, partition_id))

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
