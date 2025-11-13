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

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "UpdateCmekConfigRequest",
        "GetCmekConfigRequest",
        "SingleRegionKey",
        "CmekConfig",
        "UpdateCmekConfigMetadata",
        "ListCmekConfigsRequest",
        "ListCmekConfigsResponse",
        "DeleteCmekConfigRequest",
        "DeleteCmekConfigMetadata",
    },
)


class UpdateCmekConfigRequest(proto.Message):
    r"""Request message for UpdateCmekConfig method.
    rpc.

    Attributes:
        config (google.cloud.discoveryengine_v1.types.CmekConfig):
            Required. The CmekConfig resource.
        set_default (bool):
            Set the following CmekConfig as the default
            to be used for child resources if one is not
            specified.
    """

    config: "CmekConfig" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CmekConfig",
    )
    set_default: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class GetCmekConfigRequest(proto.Message):
    r"""Request message for GetCmekConfigRequest method.

    Attributes:
        name (str):
            Required. Resource name of
            [CmekConfig][google.cloud.discoveryengine.v1.CmekConfig],
            such as ``projects/*/locations/*/cmekConfig`` or
            ``projects/*/locations/*/cmekConfigs/*``.

            If the caller does not have permission to access the
            [CmekConfig][google.cloud.discoveryengine.v1.CmekConfig],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class SingleRegionKey(proto.Message):
    r"""Metadata for single-regional CMEKs.

    Attributes:
        kms_key (str):
            Required. Single-regional kms key resource name which will
            be used to encrypt resources
            ``projects/{project}/locations/{location}/keyRings/{keyRing}/cryptoKeys/{keyId}``.
    """

    kms_key: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CmekConfig(proto.Message):
    r"""Configurations used to enable CMEK data encryption with Cloud
    KMS keys.

    Attributes:
        name (str):
            Required. The name of the CmekConfig of the form
            ``projects/{project}/locations/{location}/cmekConfig`` or
            ``projects/{project}/locations/{location}/cmekConfigs/{cmek_config}``.
        kms_key (str):
            KMS key resource name which will be used to encrypt
            resources
            ``projects/{project}/locations/{location}/keyRings/{keyRing}/cryptoKeys/{keyId}``.
        kms_key_version (str):
            KMS key version resource name which will be used to encrypt
            resources ``<kms_key>/cryptoKeyVersions/{keyVersion}``.
        state (google.cloud.discoveryengine_v1.types.CmekConfig.State):
            Output only. The states of the CmekConfig.
        is_default (bool):
            Output only. The default CmekConfig for the
            Customer.
        last_rotation_timestamp_micros (int):
            Output only. The timestamp of the last key
            rotation.
        single_region_keys (MutableSequence[google.cloud.discoveryengine_v1.types.SingleRegionKey]):
            Optional. Single-regional CMEKs that are
            required for some VAIS features.
        notebooklm_state (google.cloud.discoveryengine_v1.types.CmekConfig.NotebookLMState):
            Output only. Whether the NotebookLM Corpus is
            ready to be used.
    """

    class State(proto.Enum):
        r"""States of the CmekConfig.

        Values:
            STATE_UNSPECIFIED (0):
                The CmekConfig state is unknown.
            CREATING (1):
                The CmekConfig is creating.
            ACTIVE (2):
                The CmekConfig can be used with DataStores.
            KEY_ISSUE (3):
                The CmekConfig is unavailable, most likely
                due to the KMS Key being revoked.
            DELETING (4):
                The CmekConfig is deleting.
            DELETE_FAILED (7):
                The CmekConfig deletion process failed.
            UNUSABLE (5):
                The CmekConfig is not usable, most likely due
                to some internal issue.
            ACTIVE_ROTATING (6):
                The KMS key version is being rotated.
            DELETED (8):
                The KMS key is soft deleted. Some cleanup
                policy will eventually be applied.
        """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        KEY_ISSUE = 3
        DELETING = 4
        DELETE_FAILED = 7
        UNUSABLE = 5
        ACTIVE_ROTATING = 6
        DELETED = 8

    class NotebookLMState(proto.Enum):
        r"""States of NotebookLM.

        Values:
            NOTEBOOK_LM_STATE_UNSPECIFIED (0):
                The NotebookLM state is unknown.
            NOTEBOOK_LM_NOT_READY (1):
                The NotebookLM is not ready.
            NOTEBOOK_LM_READY (2):
                The NotebookLM is ready to be used.
            NOTEBOOK_LM_NOT_ENABLED (3):
                The NotebookLM is not enabled.
        """
        NOTEBOOK_LM_STATE_UNSPECIFIED = 0
        NOTEBOOK_LM_NOT_READY = 1
        NOTEBOOK_LM_READY = 2
        NOTEBOOK_LM_NOT_ENABLED = 3

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    kms_key: str = proto.Field(
        proto.STRING,
        number=2,
    )
    kms_key_version: str = proto.Field(
        proto.STRING,
        number=6,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=3,
        enum=State,
    )
    is_default: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    last_rotation_timestamp_micros: int = proto.Field(
        proto.INT64,
        number=5,
    )
    single_region_keys: MutableSequence["SingleRegionKey"] = proto.RepeatedField(
        proto.MESSAGE,
        number=7,
        message="SingleRegionKey",
    )
    notebooklm_state: NotebookLMState = proto.Field(
        proto.ENUM,
        number=8,
        enum=NotebookLMState,
    )


class UpdateCmekConfigMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [CmekConfigService.UpdateCmekConfig][google.cloud.discoveryengine.v1.CmekConfigService.UpdateCmekConfig]
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


class ListCmekConfigsRequest(proto.Message):
    r"""Request message for
    [CmekConfigService.ListCmekConfigs][google.cloud.discoveryengine.v1.CmekConfigService.ListCmekConfigs]
    method.

    Attributes:
        parent (str):
            Required. The parent location resource name, such as
            ``projects/{project}/locations/{location}``.

            If the caller does not have permission to list
            [CmekConfig][google.cloud.discoveryengine.v1.CmekConfig]s
            under this location, regardless of whether or not a
            CmekConfig exists, a PERMISSION_DENIED error is returned.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListCmekConfigsResponse(proto.Message):
    r"""Response message for
    [CmekConfigService.ListCmekConfigs][google.cloud.discoveryengine.v1.CmekConfigService.ListCmekConfigs]
    method.

    Attributes:
        cmek_configs (MutableSequence[google.cloud.discoveryengine_v1.types.CmekConfig]):
            All the customer's
            [CmekConfig][google.cloud.discoveryengine.v1.CmekConfig]s.
    """

    cmek_configs: MutableSequence["CmekConfig"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="CmekConfig",
    )


class DeleteCmekConfigRequest(proto.Message):
    r"""Request message for
    [CmekConfigService.DeleteCmekConfig][google.cloud.discoveryengine.v1.CmekConfigService.DeleteCmekConfig]
    method.

    Attributes:
        name (str):
            Required. The resource name of the
            [CmekConfig][google.cloud.discoveryengine.v1.CmekConfig] to
            delete, such as
            ``projects/{project}/locations/{location}/cmekConfigs/{cmek_config}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteCmekConfigMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [CmekConfigService.DeleteCmekConfig][google.cloud.discoveryengine.v1.CmekConfigService.DeleteCmekConfig]
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
