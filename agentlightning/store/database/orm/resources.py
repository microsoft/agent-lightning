# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations
from typing import Optional
import uuid
import hashlib

from agentlightning.types import NamedResources, ResourcesUpdate
from .base import SqlAlchemyBase, NamedDictBase
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


def _generate_resources_id() -> str:
    short_id = hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:12]
    return "rs-" + short_id


class NamedResourcesInDB(NamedDictBase):
    """Custom SQLAlchemy type to store NamedResources as JSON in the database."""

    target_alias = NamedResources


class ResourcesUpdateInDB(SqlAlchemyBase):
    __tablename__ = "resources"
    resources: Mapped[NamedResources] = mapped_column(NamedResourcesInDB, nullable=False)  # JSON serialized, convert to NamedResources when needed
    resources_id: Mapped[str] = mapped_column(primary_key=True, default_factory=_generate_resources_id)

    @classmethod
    async def get_resources_by_id(cls, session_factory: async_sessionmaker[AsyncSession], resources_id: str) -> Optional[ResourcesUpdate]:
        async with session_factory() as session:
            async with session.begin():
                obj = await session.get(cls, resources_id)
                if obj is None:
                    return None
                return ResourcesUpdate(
                    resources_id=obj.resources_id,
                    resources=obj.resources
                )

    def as_resources_update(self) -> ResourcesUpdate:
        return ResourcesUpdate(
            resources_id=self.resources_id,
            resources=self.resources
        )
