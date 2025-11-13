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

from google.cloud.aiplatform_v1beta1.types import value as gca_value
from google.protobuf import struct_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ArtifactTypeSchema",
        "RuntimeArtifact",
    },
)


class ArtifactTypeSchema(proto.Message):
    r"""The definition of a artifact type in MLMD.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        schema_title (str):
            The name of the type. The format of the title must be:
            ``<namespace>.<title>``. Examples:

            - ``aiplatform.Model``
            - ``acme.CustomModel`` When this field is set, the type must
              be pre-registered in the MLMD store.

            This field is a member of `oneof`_ ``kind``.
        schema_uri (str):
            Points to a YAML file stored on Cloud Storage describing the
            format. Deprecated. Use
            [PipelineArtifactTypeSchema.schema_title][] or
            [PipelineArtifactTypeSchema.instance_schema][] instead.

            This field is a member of `oneof`_ ``kind``.
        instance_schema (str):
            Contains a raw YAML string, describing the
            format of the properties of the type.

            This field is a member of `oneof`_ ``kind``.
        schema_version (str):
            The schema version of the artifact. If the
            value is not set, it defaults to the latest
            version in the system.
    """

    schema_title: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="kind",
    )
    schema_uri: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="kind",
    )
    instance_schema: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="kind",
    )
    schema_version: str = proto.Field(
        proto.STRING,
        number=4,
    )


class RuntimeArtifact(proto.Message):
    r"""The definition of a runtime artifact.

    Attributes:
        name (str):
            The name of an artifact.
        type_ (google.cloud.aiplatform_v1beta1.types.ArtifactTypeSchema):
            The type of the artifact.
        uri (str):
            The URI of the artifact.
        properties (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.Value]):
            The properties of the artifact. Deprecated. Use
            [RuntimeArtifact.metadata][google.cloud.aiplatform.v1beta1.RuntimeArtifact.metadata]
            instead.
        custom_properties (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.Value]):
            The custom properties of the artifact. Deprecated. Use
            [RuntimeArtifact.metadata][google.cloud.aiplatform.v1beta1.RuntimeArtifact.metadata]
            instead.
        metadata (google.protobuf.struct_pb2.Struct):
            Properties of the Artifact.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    type_: "ArtifactTypeSchema" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ArtifactTypeSchema",
    )
    uri: str = proto.Field(
        proto.STRING,
        number=3,
    )
    properties: MutableMapping[str, gca_value.Value] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=4,
        message=gca_value.Value,
    )
    custom_properties: MutableMapping[str, gca_value.Value] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=5,
        message=gca_value.Value,
    )
    metadata: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=6,
        message=struct_pb2.Struct,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
