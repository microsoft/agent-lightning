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

import proto  # type: ignore

from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "Memory",
    },
)


class Memory(proto.Message):
    r"""A memory.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Optional. Timestamp of when this resource is considered
            expired. This is *always* provided on output, regardless of
            what ``expiration`` was sent on input.

            This field is a member of `oneof`_ ``expiration``.
        ttl (google.protobuf.duration_pb2.Duration):
            Optional. Input only. The TTL for this
            resource. The expiration time is computed: now +
            TTL.

            This field is a member of `oneof`_ ``expiration``.
        name (str):
            Identifier. The resource name of the Memory. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/memories/{memory}``
        display_name (str):
            Optional. Display name of the Memory.
        description (str):
            Optional. Description of the Memory.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Memory was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Memory was
            most recently updated.
        fact (str):
            Required. Semantic knowledge extracted from
            the source content.
        scope (MutableMapping[str, str]):
            Required. Immutable. The scope of the Memory. Memories are
            isolated within their scope. The scope is defined when
            creating or generating memories. Scope values cannot contain
            the wildcard character '\*'.
    """

    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="expiration",
        message=timestamp_pb2.Timestamp,
    )
    ttl: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="expiration",
        message=duration_pb2.Duration,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    fact: str = proto.Field(
        proto.STRING,
        number=10,
    )
    scope: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=11,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
