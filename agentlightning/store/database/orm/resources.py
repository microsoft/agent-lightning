# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from agentlightning.types import NamedResources, ResourcesUpdate

from .base import NamedDictBase, SqlAlchemyBase


def _generate_resources_id() -> str:
    short_id = hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:12]
    return "rs-" + short_id


class NamedResourcesInDB(NamedDictBase):
    """Custom SQLAlchemy type to store NamedResources as JSON in the database."""

    target_alias = NamedResources


class ResourcesUpdateInDB(SqlAlchemyBase):
    __tablename__ = "resources"
    resources: Mapped[NamedResources] = mapped_column(
        NamedResourcesInDB, nullable=False
    )  # JSON serialized, convert to NamedResources when needed
    resources_id: Mapped[str] = mapped_column(primary_key=True, default_factory=_generate_resources_id)
    create_time: Mapped[float] = mapped_column(nullable=False, default_factory=time.time)
    update_time: Mapped[float] = mapped_column(nullable=False, default_factory=time.time, onupdate=time.time)
    version: Mapped[int] = mapped_column(nullable=False, default=1)

    __mapper_args__ = {
        "version_id_col": version,
    }

    @classmethod
    async def get_resources_by_id(
        cls, session_factory: async_sessionmaker[AsyncSession], resources_id: str
    ) -> Optional[ResourcesUpdate]:
        async with session_factory() as session:
            async with session.begin():
                obj = await session.get(cls, resources_id)
                if obj is None:
                    return None
                return obj.as_resources_update()

    def as_resources_update(self) -> ResourcesUpdate:
        return ResourcesUpdate(**self.model_dump())
