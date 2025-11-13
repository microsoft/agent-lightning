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

from google.cloud.discoveryengine_v1.types import cmek_config_service

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "IdentityMappingStore",
        "IdentityMappingEntry",
    },
)


class IdentityMappingStore(proto.Message):
    r"""Identity Mapping Store which contains Identity Mapping
    Entries.

    Attributes:
        name (str):
            Immutable. The full resource name of the identity mapping
            store. Format:
            ``projects/{project}/locations/{location}/identityMappingStores/{identity_mapping_store}``.
            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        kms_key_name (str):
            Input only. The KMS key to be used to protect this Identity
            Mapping Store at creation time.

            Must be set for requests that need to comply with CMEK Org
            Policy protections.

            If this field is set and processed successfully, the
            Identity Mapping Store will be protected by the KMS key, as
            indicated in the cmek_config field.
        cmek_config (google.cloud.discoveryengine_v1.types.CmekConfig):
            Output only. CMEK-related information for the
            Identity Mapping Store.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    kms_key_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    cmek_config: cmek_config_service.CmekConfig = proto.Field(
        proto.MESSAGE,
        number=4,
        message=cmek_config_service.CmekConfig,
    )


class IdentityMappingEntry(proto.Message):
    r"""Identity Mapping Entry that maps an external identity to an
    internal identity.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        user_id (str):
            User identifier. For Google Workspace user account, user_id
            should be the google workspace user email. For non-google
            identity provider, user_id is the mapped user identifier
            configured during the workforcepool config.

            This field is a member of `oneof`_ ``identity_provider_id``.
        group_id (str):
            Group identifier. For Google Workspace user account,
            group_id should be the google workspace group email. For
            non-google identity provider, group_id is the mapped group
            identifier configured during the workforcepool config.

            This field is a member of `oneof`_ ``identity_provider_id``.
        external_identity (str):
            Required. Identity outside the customer
            identity provider. The length limit of external
            identity will be of 100 characters.
    """

    user_id: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="identity_provider_id",
    )
    group_id: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="identity_provider_id",
    )
    external_identity: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
