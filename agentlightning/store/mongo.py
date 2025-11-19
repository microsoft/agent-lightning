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
from pymongo.asynchronous.database import AsyncDatabase

from .base import LightningStoreCapabilities
from .collection.mongo import MongoLightningCollections
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
        client: AsyncMongoClient[Mapping[str, Any]] | str | None = None,
        database: AsyncDatabase[Mapping[str, Any]] | str | None = None,
        partition_id: str | None = None,
    ) -> None:
        if isinstance(client, str):
            client = AsyncMongoClient(client)
        if isinstance(database, str):
            if client is None:
                raise ValueError("You must provide a client when providing a database name")
            database = client[database]
        elif isinstance(database, AsyncDatabase):
            if client is not None:
                logger.warning("Ignoring client when database instance has been provided")
        else:
            if client is None:
                raise ValueError("You must provide either a client or a database")
            # database is None
            database_name = "agentlightning"
            logger.info("No database name provided, using default 'agentlightning'")
            database = client[database_name]

        if partition_id is None:
            partition_id = _generate_partition_id()
            logger.info("No partition id provided, generated a new one: %s", partition_id)

        super().__init__(collections=MongoLightningCollections(database, partition_id))

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        return LightningStoreCapabilities(
            thread_safe=True,
            async_safe=True,
            zero_copy=True,
            otlp_traces=False,
        )
