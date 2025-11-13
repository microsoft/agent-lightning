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
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import (
    identity_mapping_store as gcd_identity_mapping_store,
)

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "CreateIdentityMappingStoreRequest",
        "GetIdentityMappingStoreRequest",
        "DeleteIdentityMappingStoreRequest",
        "ImportIdentityMappingsRequest",
        "ImportIdentityMappingsResponse",
        "PurgeIdentityMappingsRequest",
        "ListIdentityMappingsRequest",
        "ListIdentityMappingsResponse",
        "ListIdentityMappingStoresRequest",
        "ListIdentityMappingStoresResponse",
        "IdentityMappingEntryOperationMetadata",
        "DeleteIdentityMappingStoreMetadata",
    },
)


class CreateIdentityMappingStoreRequest(proto.Message):
    r"""Request message for
    [IdentityMappingStoreService.CreateIdentityMappingStore][google.cloud.discoveryengine.v1.IdentityMappingStoreService.CreateIdentityMappingStore]

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        cmek_config_name (str):
            Resource name of the CmekConfig to use for
            protecting this Identity Mapping Store.

            This field is a member of `oneof`_ ``cmek_options``.
        disable_cmek (bool):
            Identity Mapping Store without CMEK
            protections. If a default CmekConfig is set for
            the project, setting this field will override
            the default CmekConfig as well.

            This field is a member of `oneof`_ ``cmek_options``.
        parent (str):
            Required. The parent collection resource name, such as
            ``projects/{project}/locations/{location}``.
        identity_mapping_store_id (str):
            Required. The ID of the Identity Mapping Store to create.

            The ID must contain only letters (a-z, A-Z), numbers (0-9),
            underscores (\_), and hyphens (-). The maximum length is 63
            characters.
        identity_mapping_store (google.cloud.discoveryengine_v1.types.IdentityMappingStore):
            Required. The Identity Mapping Store to
            create.
    """

    cmek_config_name: str = proto.Field(
        proto.STRING,
        number=5,
        oneof="cmek_options",
    )
    disable_cmek: bool = proto.Field(
        proto.BOOL,
        number=6,
        oneof="cmek_options",
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    identity_mapping_store_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    identity_mapping_store: gcd_identity_mapping_store.IdentityMappingStore = (
        proto.Field(
            proto.MESSAGE,
            number=3,
            message=gcd_identity_mapping_store.IdentityMappingStore,
        )
    )


class GetIdentityMappingStoreRequest(proto.Message):
    r"""Request message for
    [IdentityMappingStoreService.GetIdentityMappingStore][google.cloud.discoveryengine.v1.IdentityMappingStoreService.GetIdentityMappingStore]

    Attributes:
        name (str):
            Required. The name of the Identity Mapping Store to get.
            Format:
            ``projects/{project}/locations/{location}/identityMappingStores/{identityMappingStore}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteIdentityMappingStoreRequest(proto.Message):
    r"""Request message for
    [IdentityMappingStoreService.DeleteIdentityMappingStore][google.cloud.discoveryengine.v1.IdentityMappingStoreService.DeleteIdentityMappingStore]

    Attributes:
        name (str):
            Required. The name of the Identity Mapping Store to delete.
            Format:
            ``projects/{project}/locations/{location}/identityMappingStores/{identityMappingStore}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ImportIdentityMappingsRequest(proto.Message):
    r"""Request message for
    [IdentityMappingStoreService.ImportIdentityMappings][google.cloud.discoveryengine.v1.IdentityMappingStoreService.ImportIdentityMappings]


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        inline_source (google.cloud.discoveryengine_v1.types.ImportIdentityMappingsRequest.InlineSource):
            The inline source to import identity mapping
            entries from.

            This field is a member of `oneof`_ ``source``.
        identity_mapping_store (str):
            Required. The name of the Identity Mapping Store to import
            Identity Mapping Entries to. Format:
            ``projects/{project}/locations/{location}/identityMappingStores/{identityMappingStore}``
    """

    class InlineSource(proto.Message):
        r"""The inline source to import identity mapping entries from.

        Attributes:
            identity_mapping_entries (MutableSequence[google.cloud.discoveryengine_v1.types.IdentityMappingEntry]):
                A maximum of 10000 entries can be imported at
                one time
        """

        identity_mapping_entries: MutableSequence[
            gcd_identity_mapping_store.IdentityMappingEntry
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=gcd_identity_mapping_store.IdentityMappingEntry,
        )

    inline_source: InlineSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=InlineSource,
    )
    identity_mapping_store: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ImportIdentityMappingsResponse(proto.Message):
    r"""Response message for
    [IdentityMappingStoreService.ImportIdentityMappings][google.cloud.discoveryengine.v1.IdentityMappingStoreService.ImportIdentityMappings]

    Attributes:
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the request.
    """

    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )


class PurgeIdentityMappingsRequest(proto.Message):
    r"""Request message for
    [IdentityMappingStoreService.PurgeIdentityMappings][google.cloud.discoveryengine.v1.IdentityMappingStoreService.PurgeIdentityMappings]


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        inline_source (google.cloud.discoveryengine_v1.types.PurgeIdentityMappingsRequest.InlineSource):
            The inline source to purge identity mapping
            entries from.

            This field is a member of `oneof`_ ``source``.
        identity_mapping_store (str):
            Required. The name of the Identity Mapping Store to purge
            Identity Mapping Entries from. Format:
            ``projects/{project}/locations/{location}/identityMappingStores/{identityMappingStore}``
        filter (str):
            Filter matching identity mappings to purge. The eligible
            field for filtering is:

            - ``update_time``: in ISO 8601 "zulu" format.
            - ``external_id``

            Examples:

            - Deleting all identity mappings updated in a time range:
              ``update_time > "2012-04-23T18:25:43.511Z" AND update_time < "2012-04-23T18:30:43.511Z"``
            - Deleting all identity mappings for a given external_id:
              ``external_id = "id1"``
            - Deleting all identity mappings inside an identity mapping
              store: ``*``

            The filtering fields are assumed to have an implicit AND.
            Should not be used with source. An error will be thrown, if
            both are provided.
        force (bool):
            Actually performs the purge. If ``force`` is set to false,
            return the expected purge count without deleting any
            identity mappings. This field is only supported for purge
            with filter. For input source this field is ignored and data
            will be purged regardless of the value of this field.

            This field is a member of `oneof`_ ``_force``.
    """

    class InlineSource(proto.Message):
        r"""The inline source to purge identity mapping entries from.

        Attributes:
            identity_mapping_entries (MutableSequence[google.cloud.discoveryengine_v1.types.IdentityMappingEntry]):
                A maximum of 10000 entries can be purged at
                one time
        """

        identity_mapping_entries: MutableSequence[
            gcd_identity_mapping_store.IdentityMappingEntry
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=gcd_identity_mapping_store.IdentityMappingEntry,
        )

    inline_source: InlineSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=InlineSource,
    )
    identity_mapping_store: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=3,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=4,
        optional=True,
    )


class ListIdentityMappingsRequest(proto.Message):
    r"""Request message for
    [IdentityMappingStoreService.ListIdentityMappings][google.cloud.discoveryengine.v1.IdentityMappingStoreService.ListIdentityMappings]

    Attributes:
        identity_mapping_store (str):
            Required. The name of the Identity Mapping Store to list
            Identity Mapping Entries in. Format:
            ``projects/{project}/locations/{location}/identityMappingStores/{identityMappingStore}``
        page_size (int):
            Maximum number of IdentityMappings to return.
            If unspecified, defaults to 2000. The maximum
            allowed value is 10000. Values above 10000 will
            be coerced to 10000.
        page_token (str):
            A page token, received from a previous
            ``ListIdentityMappings`` call. Provide this to retrieve the
            subsequent page.

            When paginating, all other parameters provided to
            ``ListIdentityMappings`` must match the call that provided
            the page token.
    """

    identity_mapping_store: str = proto.Field(
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


class ListIdentityMappingsResponse(proto.Message):
    r"""Response message for
    [IdentityMappingStoreService.ListIdentityMappings][google.cloud.discoveryengine.v1.IdentityMappingStoreService.ListIdentityMappings]

    Attributes:
        identity_mapping_entries (MutableSequence[google.cloud.discoveryengine_v1.types.IdentityMappingEntry]):
            The Identity Mapping Entries.
        next_page_token (str):
            A token that can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    identity_mapping_entries: MutableSequence[
        gcd_identity_mapping_store.IdentityMappingEntry
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_identity_mapping_store.IdentityMappingEntry,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class ListIdentityMappingStoresRequest(proto.Message):
    r"""Request message for
    [IdentityMappingStoreService.ListIdentityMappingStores][google.cloud.discoveryengine.v1.IdentityMappingStoreService.ListIdentityMappingStores]

    Attributes:
        parent (str):
            Required. The parent of the Identity Mapping Stores to list.
            Format: ``projects/{project}/locations/{location}``.
        page_size (int):
            Maximum number of IdentityMappingStores to
            return. If unspecified, defaults to 100. The
            maximum allowed value is 1000. Values above 1000
            will be coerced to 1000.
        page_token (str):
            A page token, received from a previous
            ``ListIdentityMappingStores`` call. Provide this to retrieve
            the subsequent page.

            When paginating, all other parameters provided to
            ``ListIdentityMappingStores`` must match the call that
            provided the page token.
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


class ListIdentityMappingStoresResponse(proto.Message):
    r"""Response message for
    [IdentityMappingStoreService.ListIdentityMappingStores][google.cloud.discoveryengine.v1.IdentityMappingStoreService.ListIdentityMappingStores]

    Attributes:
        identity_mapping_stores (MutableSequence[google.cloud.discoveryengine_v1.types.IdentityMappingStore]):
            The Identity Mapping Stores.
        next_page_token (str):
            A token that can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages.
    """

    @property
    def raw_page(self):
        return self

    identity_mapping_stores: MutableSequence[
        gcd_identity_mapping_store.IdentityMappingStore
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_identity_mapping_store.IdentityMappingStore,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class IdentityMappingEntryOperationMetadata(proto.Message):
    r"""IdentityMappingEntry LongRunningOperation metadata for
    [IdentityMappingStoreService.ImportIdentityMappings][google.cloud.discoveryengine.v1.IdentityMappingStoreService.ImportIdentityMappings]
    and
    [IdentityMappingStoreService.PurgeIdentityMappings][google.cloud.discoveryengine.v1.IdentityMappingStoreService.PurgeIdentityMappings]

    Attributes:
        success_count (int):
            The number of IdentityMappingEntries that
            were successfully processed.
        failure_count (int):
            The number of IdentityMappingEntries that
            failed to be processed.
        total_count (int):
            The total number of IdentityMappingEntries
            that were processed.
    """

    success_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    failure_count: int = proto.Field(
        proto.INT64,
        number=2,
    )
    total_count: int = proto.Field(
        proto.INT64,
        number=3,
    )


class DeleteIdentityMappingStoreMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [IdentityMappingStoreService.DeleteIdentityMappingStore][google.cloud.discoveryengine.v1.IdentityMappingStoreService.DeleteIdentityMappingStore]
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
