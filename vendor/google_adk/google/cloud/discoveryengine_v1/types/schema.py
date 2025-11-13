# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
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
#
from __future__ import annotations

from typing import MutableMapping, MutableSequence

from google.protobuf import struct_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "Schema",
    },
)


class Schema(proto.Message):
    r"""Defines the structure and layout of a type of document data.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        struct_schema (google.protobuf.struct_pb2.Struct):
            The structured representation of the schema.

            This field is a member of `oneof`_ ``schema``.
        json_schema (str):
            The JSON representation of the schema.

            This field is a member of `oneof`_ ``schema``.
        name (str):
            Immutable. The full resource name of the schema, in the
            format of
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/schemas/{schema}``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
    """

    struct_schema: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="schema",
        message=struct_pb2.Struct,
    )
    json_schema: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="schema",
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
