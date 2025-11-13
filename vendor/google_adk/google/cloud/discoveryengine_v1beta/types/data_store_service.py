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

from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1beta.types import data_store as gcd_data_store

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "CreateDataStoreRequest",
        "GetDataStoreRequest",
        "CreateDataStoreMetadata",
        "ListDataStoresRequest",
        "ListDataStoresResponse",
        "DeleteDataStoreRequest",
        "UpdateDataStoreRequest",
        "DeleteDataStoreMetadata",
    },
)


class CreateDataStoreRequest(proto.Message):
    r"""Request for
    [DataStoreService.CreateDataStore][google.cloud.discoveryengine.v1beta.DataStoreService.CreateDataStore]
    method.

    Attributes:
        parent (str):
            Required. The parent resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection}``.
        data_store (google.cloud.discoveryengine_v1beta.types.DataStore):
            Required. The
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]
            to create.
        data_store_id (str):
            Required. The ID to use for the
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore],
            which will become the final component of the
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]'s
            resource name.

            This field must conform to
            `RFC-1034 <https://tools.ietf.org/html/rfc1034>`__ standard
            with a length limit of 63 characters. Otherwise, an
            INVALID_ARGUMENT error is returned.
        create_advanced_site_search (bool):
            A boolean flag indicating whether user want to directly
            create an advanced data store for site search. If the data
            store is not configured as site search (GENERIC vertical and
            PUBLIC_WEBSITE content_config), this flag will be ignored.
        skip_default_schema_creation (bool):
            A boolean flag indicating whether to skip the default schema
            creation for the data store. Only enable this flag if you
            are certain that the default schema is incompatible with
            your use case.

            If set to true, you must manually create a schema for the
            data store before any documents can be ingested.

            This flag cannot be specified if
            ``data_store.starting_schema`` is specified.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    data_store: gcd_data_store.DataStore = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_data_store.DataStore,
    )
    data_store_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    create_advanced_site_search: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    skip_default_schema_creation: bool = proto.Field(
        proto.BOOL,
        number=7,
    )


class GetDataStoreRequest(proto.Message):
    r"""Request message for
    [DataStoreService.GetDataStore][google.cloud.discoveryengine.v1beta.DataStoreService.GetDataStore]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore],
            such as
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}``.

            If the caller does not have permission to access the
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the requested
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]
            does not exist, a NOT_FOUND error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateDataStoreMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [DataStoreService.CreateDataStore][google.cloud.discoveryengine.v1beta.DataStoreService.CreateDataStore]
    operation. This will be returned by the
    google.longrunning.Operation.metadata field.

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


class ListDataStoresRequest(proto.Message):
    r"""Request message for
    [DataStoreService.ListDataStores][google.cloud.discoveryengine.v1beta.DataStoreService.ListDataStores]
    method.

    Attributes:
        parent (str):
            Required. The parent branch resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection_id}``.

            If the caller does not have permission to list
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]s
            under this location, regardless of whether or not this data
            store exists, a PERMISSION_DENIED error is returned.
        page_size (int):
            Maximum number of
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]s
            to return. If unspecified, defaults to 10. The maximum
            allowed value is 50. Values above 50 will be coerced to 50.

            If this field is negative, an INVALID_ARGUMENT is returned.
        page_token (str):
            A page token
            [ListDataStoresResponse.next_page_token][google.cloud.discoveryengine.v1beta.ListDataStoresResponse.next_page_token],
            received from a previous
            [DataStoreService.ListDataStores][google.cloud.discoveryengine.v1beta.DataStoreService.ListDataStores]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [DataStoreService.ListDataStores][google.cloud.discoveryengine.v1beta.DataStoreService.ListDataStores]
            must match the call that provided the page token. Otherwise,
            an INVALID_ARGUMENT error is returned.
        filter (str):
            Filter by solution type . For example:
            ``filter = 'solution_type:SOLUTION_TYPE_SEARCH'``
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
    filter: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListDataStoresResponse(proto.Message):
    r"""Response message for
    [DataStoreService.ListDataStores][google.cloud.discoveryengine.v1beta.DataStoreService.ListDataStores]
    method.

    Attributes:
        data_stores (MutableSequence[google.cloud.discoveryengine_v1beta.types.DataStore]):
            All the customer's
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]s.
        next_page_token (str):
            A token that can be sent as
            [ListDataStoresRequest.page_token][google.cloud.discoveryengine.v1beta.ListDataStoresRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    data_stores: MutableSequence[gcd_data_store.DataStore] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_data_store.DataStore,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteDataStoreRequest(proto.Message):
    r"""Request message for
    [DataStoreService.DeleteDataStore][google.cloud.discoveryengine.v1beta.DataStoreService.DeleteDataStore]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore],
            such as
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}``.

            If the caller does not have permission to delete the
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]
            to delete does not exist, a NOT_FOUND error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateDataStoreRequest(proto.Message):
    r"""Request message for
    [DataStoreService.UpdateDataStore][google.cloud.discoveryengine.v1beta.DataStoreService.UpdateDataStore]
    method.

    Attributes:
        data_store (google.cloud.discoveryengine_v1beta.types.DataStore):
            Required. The
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]
            to update.

            If the caller does not have permission to update the
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]
            to update does not exist, a NOT_FOUND error is returned.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            [DataStore][google.cloud.discoveryengine.v1beta.DataStore]
            to update.

            If an unsupported or unknown field is provided, an
            INVALID_ARGUMENT error is returned.
    """

    data_store: gcd_data_store.DataStore = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_data_store.DataStore,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteDataStoreMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [DataStoreService.DeleteDataStore][google.cloud.discoveryengine.v1beta.DataStoreService.DeleteDataStore]
    operation. This will be returned by the
    google.longrunning.Operation.metadata field.

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
