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

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import schema as gcd_schema

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "GetSchemaRequest",
        "ListSchemasRequest",
        "ListSchemasResponse",
        "CreateSchemaRequest",
        "UpdateSchemaRequest",
        "DeleteSchemaRequest",
        "CreateSchemaMetadata",
        "UpdateSchemaMetadata",
        "DeleteSchemaMetadata",
    },
)


class GetSchemaRequest(proto.Message):
    r"""Request message for
    [SchemaService.GetSchema][google.cloud.discoveryengine.v1.SchemaService.GetSchema]
    method.

    Attributes:
        name (str):
            Required. The full resource name of the schema, in the
            format of
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/schemas/{schema}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListSchemasRequest(proto.Message):
    r"""Request message for
    [SchemaService.ListSchemas][google.cloud.discoveryengine.v1.SchemaService.ListSchemas]
    method.

    Attributes:
        parent (str):
            Required. The parent data store resource name, in the format
            of
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}``.
        page_size (int):
            The maximum number of
            [Schema][google.cloud.discoveryengine.v1.Schema]s to return.
            The service may return fewer than this value.

            If unspecified, at most 100
            [Schema][google.cloud.discoveryengine.v1.Schema]s are
            returned.

            The maximum value is 1000; values above 1000 are set to
            1000.
        page_token (str):
            A page token, received from a previous
            [SchemaService.ListSchemas][google.cloud.discoveryengine.v1.SchemaService.ListSchemas]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [SchemaService.ListSchemas][google.cloud.discoveryengine.v1.SchemaService.ListSchemas]
            must match the call that provided the page token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListSchemasResponse(proto.Message):
    r"""Response message for
    [SchemaService.ListSchemas][google.cloud.discoveryengine.v1.SchemaService.ListSchemas]
    method.

    Attributes:
        schemas (MutableSequence[google.cloud.discoveryengine_v1.types.Schema]):
            The [Schema][google.cloud.discoveryengine.v1.Schema]s.
        next_page_token (str):
            A token that can be sent as
            [ListSchemasRequest.page_token][google.cloud.discoveryengine.v1.ListSchemasRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    schemas: MutableSequence[gcd_schema.Schema] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_schema.Schema,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateSchemaRequest(proto.Message):
    r"""Request message for
    [SchemaService.CreateSchema][google.cloud.discoveryengine.v1.SchemaService.CreateSchema]
    method.

    Attributes:
        parent (str):
            Required. The parent data store resource name, in the format
            of
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}``.
        schema (google.cloud.discoveryengine_v1.types.Schema):
            Required. The
            [Schema][google.cloud.discoveryengine.v1.Schema] to create.
        schema_id (str):
            Required. The ID to use for the
            [Schema][google.cloud.discoveryengine.v1.Schema], which
            becomes the final component of the
            [Schema.name][google.cloud.discoveryengine.v1.Schema.name].

            This field should conform to
            `RFC-1034 <https://tools.ietf.org/html/rfc1034>`__ standard
            with a length limit of 63 characters.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    schema: gcd_schema.Schema = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_schema.Schema,
    )
    schema_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class UpdateSchemaRequest(proto.Message):
    r"""Request message for
    [SchemaService.UpdateSchema][google.cloud.discoveryengine.v1.SchemaService.UpdateSchema]
    method.

    Attributes:
        schema (google.cloud.discoveryengine_v1.types.Schema):
            Required. The
            [Schema][google.cloud.discoveryengine.v1.Schema] to update.
        allow_missing (bool):
            If set to true, and the
            [Schema][google.cloud.discoveryengine.v1.Schema] is not
            found, a new
            [Schema][google.cloud.discoveryengine.v1.Schema] is created.
            In this situation, ``update_mask`` is ignored.
    """

    schema: gcd_schema.Schema = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_schema.Schema,
    )
    allow_missing: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class DeleteSchemaRequest(proto.Message):
    r"""Request message for
    [SchemaService.DeleteSchema][google.cloud.discoveryengine.v1.SchemaService.DeleteSchema]
    method.

    Attributes:
        name (str):
            Required. The full resource name of the schema, in the
            format of
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/schemas/{schema}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateSchemaMetadata(proto.Message):
    r"""Metadata for Create Schema LRO.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class UpdateSchemaMetadata(proto.Message):
    r"""Metadata for UpdateSchema LRO.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class DeleteSchemaMetadata(proto.Message):
    r"""Metadata for DeleteSchema LRO.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
