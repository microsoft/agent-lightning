# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from typing import Any


class MappingMeta(type):
    """
    Metaclass to provide add_mapping functionality, allowing users to add
    custom attributes to derived classes at runtime.

    Using a metaclass allows us to share functionality between CrossSync
    and CrossSync._Sync_Impl, and it works better with mypy checks than
    monkypatching
    """

    # list of attributes that can be added to the derived class at runtime
    _runtime_replacements: dict[tuple[MappingMeta, str], Any] = {}

    def add_mapping(cls: MappingMeta, name: str, value: Any):
        """
        Add a new attribute to the class, for replacing library-level symbols

        Raises:
            - AttributeError if the attribute already exists with a different value
        """
        key = (cls, name)
        old_value = cls._runtime_replacements.get(key)
        if old_value is None:
            cls._runtime_replacements[key] = value
        elif old_value != value:
            raise AttributeError(f"Conflicting assignments for CrossSync.{name}")

    def add_mapping_decorator(cls: MappingMeta, name: str):
        """
        Exposes add_mapping as a class decorator
        """

        def decorator(wrapped_cls):
            cls.add_mapping(name, wrapped_cls)
            return wrapped_cls

        return decorator

    def __getattr__(cls: MappingMeta, name: str):
        """
        Retrieve custom attributes
        """
        key = (cls, name)
        found = cls._runtime_replacements.get(key)
        if found is not None:
            return found
        raise AttributeError(f"CrossSync has no attribute {name}")
