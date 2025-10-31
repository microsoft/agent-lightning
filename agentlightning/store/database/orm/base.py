# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations
from pydantic import BaseModel, TypeAdapter
from typing import Any, Dict, List, Optional
import json
import logging

from sqlalchemy import JSON, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass
from sqlalchemy.ext.asyncio import AsyncAttrs



class SqlAlchemyBase(AsyncAttrs, MappedAsDataclass, DeclarativeBase):
    pass


class PydanticInDB(TypeDecorator):
    """Custom SQLAlchemy type to store pydantic.BaseModel as JSON in the database.
    Attributes:
        target_type: type[BaseModel], the type of the pydantic model to be stored.
    """

    impl = JSON
    target_type: type[BaseModel] | None = None

    def process_bind_param(self, value: BaseModel | None, dialect) -> Optional[str]:
        if value is None:
            return None
        if self.target_type is not None:
            return TypeAdapter(self.target_type).validate_python(value).model_dump_json()  # type: ignore
        return json.dumps(value)

    def process_result_value(self, value: Optional[str], dialect) -> Optional[BaseModel]:
        if value is None:
            return None
        if self.target_type is not None:
            return TypeAdapter(self.target_type).validate_json(value)  # type: ignore
        dic = json.loads(value)
        return dic  # type: ignore


class PydanticListInDB(TypeDecorator):
    """Custom SQLAlchemy type to store List[pydantic.BaseModel] as JSON in the database.
    Attributes:
        target_type: type[BaseModel], the type of the pydantic model to be stored in the list.
    """

    impl = JSON
    target_type: type[BaseModel] | None = None

    def process_bind_param(self, value: List[BaseModel] | None, dialect) -> Optional[str]:
        if value is None:
            return None
        if self.target_type is not None:
            lst = [TypeAdapter(self.target_type).validate_python(v).model_dump() for v in value]
            return json.dumps(lst)
        raise ValueError("target_type must be set for PydanticListInDB")

    def process_result_value(self, value: Optional[str], dialect) -> Optional[List[BaseModel]]:
        if value is None:
            return None
        if self.target_type is not None:
            dic = json.loads(value)
            return [
                TypeAdapter(self.target_type).validate_python(v)  # type: ignore
                for v in dic
            ]
        raise ValueError("target_type must be set for PydanticListInDB")


class NamedDictBase(TypeDecorator):
    """Custom SQLAlchemy type to store Dict[str, pydantic.BaseModel] as JSON in the database.
    Attributes:
        target_alias: type[Dict[str, BaseModel]], the alias type of the dict.
        value_type: type[BaseModel], the type of the values in the dict.

    For example, given NamedResources = Dict[str, ResourceUnion],
    we can define NamedDictBase with target_alias=NamedResources and target_type=ResourceUnion.
    """

    impl = JSON
    target_alias: type | None = None
    target_type: type[BaseModel] | None = None

    def process_bind_param(self, value: Dict[str, Any] | None, dialect) -> Optional[str]:
        if value is None:
            return None

        # ignore target_alias for when dumping because Dict is not a pydantic model
        if self.target_type is not None:
            dic = {k: TypeAdapter(self.target_type).validate_python(v).model_dump() if isinstance(v, BaseModel) else v for k, v in value.items()}
            return json.dumps(dic)
        dic = {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in value.items()}
        return json.dumps(dic)

    def process_result_value(self, value: Optional[str], dialect) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if self.target_alias is not None:
            return TypeAdapter(self.target_alias).validate_json(value)  # type: ignore
        if self.target_type is not None:
            dic = json.loads(value)
            return {
                k: TypeAdapter(self.target_type).validate_python(v)  # type: ignore
                for k, v in dic.items()
            }
        return json.loads(value)


class DatabaseRuntimeError(Exception):
    """Raised when a runtime error occurs during database operations.
    Particularly used when the execution of a query fails.
    """
    pass

class RaceConditionError(Exception):
    """Raised when a race condition is detected during database operations.
    """
    pass


class NoRolloutToDequeueError(Exception):
    """Raised when there is no rollout available to dequeue.
    """
    pass

