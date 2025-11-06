# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, TypeAdapter, computed_field

# from dataclasses import asdict
from sqlalchemy import JSON, TypeDecorator
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass


class SqlAlchemyBase(AsyncAttrs, MappedAsDataclass, DeclarativeBase):
    pass

    def model_dump(
        self,
        exclude: set[str] | None = None,
        mapper: Dict[str, Callable[["SqlAlchemyBase"], Any]] | None = None,
    ) -> Dict[str, Any]:
        """Dump the SQLAlchemy model to a dictionary.
        Args:
            exclude: set[str]
                The set of field names to exclude.
            mapper: Dict[str, Callable[[SqlAlchemyBase], Any]]
                A mapping from field names to functions that take the model instance and return the value to be used for that field.
                If the key is "*", the function should return a dictionary of additional fields to be added to the output.
        Returns:
            Dict[str, Any]: The dumped model as a dictionary.
        """
        exclude = exclude or set()
        mapper = mapper or {}
        dic = {k: getattr(self, k) for k in self.__table__.columns.keys() if k not in exclude}
        for k, func in mapper.items():
            if k == "*":
                dic.update(func(self))
            else:
                dic[k] = func(self)
        return dic


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
        value_type: type[BaseModel], the type of the pydantic model to be stored in the list.
    """

    impl = JSON
    value_type: type[BaseModel] | None = None

    def process_bind_param(self, value: List[BaseModel] | None, dialect) -> Optional[str]:
        if value is None:
            return None
        if self.value_type is not None:
            lst = [TypeAdapter(self.value_type).validate_python(v).model_dump() for v in value]
            return json.dumps(lst)
        raise ValueError("target_type must be set for PydanticListInDB")

    def process_result_value(self, value: Optional[str], dialect) -> Optional[List[BaseModel]]:
        if value is None:
            return None
        if self.value_type is not None:
            dic = json.loads(value)
            return [TypeAdapter(self.value_type).validate_python(v) for v in dic]  # type: ignore
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
    value_type: type[BaseModel] | None = None

    def process_bind_param(self, value: Dict[str, Any] | None, dialect) -> Optional[str]:
        if value is None:
            return None

        # ignore target_alias for when dumping because Dict is not a pydantic model
        if self.value_type is not None:
            dic = {
                k: TypeAdapter(self.value_type).validate_python(v).model_dump() if isinstance(v, BaseModel) else v
                for k, v in value.items()
            }
            return json.dumps(dic)
        dic = {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in value.items()}
        return json.dumps(dic)

    def process_result_value(self, value: Optional[str], dialect) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if self.target_alias is not None:
            return TypeAdapter(self.target_alias).validate_json(value)  # type: ignore
        if self.value_type is not None:
            dic = json.loads(value)
            return {k: TypeAdapter(self.value_type).validate_python(v) for k, v in dic.items()}  # type: ignore
        return json.loads(value)


class DatabaseRuntimeError(Exception):
    """Raised when a runtime error occurs during database operations.
    Particularly used when the execution of a query fails.
    """

    pass


class RaceConditionError(Exception):
    """Raised when a race condition is detected during database operations."""

    pass


class NoRolloutToDequeueError(Exception):
    """Raised when there is no rollout available to dequeue."""

    pass


class AttemptStatusUpdateMessage(BaseModel):
    attempt_id: str
    rollout_id: str
    timestamp: float = Field(default_factory=time.time)
    old_status: Optional[str] = None
    new_status: str

    @computed_field
    @property
    def event(self) -> str:
        return "attempt_status_update"

    @computed_field
    @property
    def is_failed(self) -> bool:
        return self.new_status in ["failed", "timeout", "unresponsive"]

    @computed_field
    @property
    def is_succeeded(self) -> bool:
        return self.new_status == "succeeded"

    @computed_field
    @property
    def is_finished(self) -> bool:
        return self.is_failed or self.is_succeeded

    @computed_field
    @property
    def is_running(self) -> bool:
        return self.new_status in ["running", "preparing"]
