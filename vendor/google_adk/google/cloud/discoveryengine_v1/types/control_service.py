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
import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import control as gcd_control

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "CreateControlRequest",
        "UpdateControlRequest",
        "DeleteControlRequest",
        "GetControlRequest",
        "ListControlsRequest",
        "ListControlsResponse",
    },
)


class CreateControlRequest(proto.Message):
    r"""Request for CreateControl method.

    Attributes:
        parent (str):
            Required. Full resource name of parent data store. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}``
            or
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``.
        control (google.cloud.discoveryengine_v1.types.Control):
            Required. The Control to create.
        control_id (str):
            Required. The ID to use for the Control, which will become
            the final component of the Control's resource name.

            This value must be within 1-63 characters. Valid characters
            are /[a-z][0-9]-\_/.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    control: gcd_control.Control = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_control.Control,
    )
    control_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class UpdateControlRequest(proto.Message):
    r"""Request for UpdateControl method.

    Attributes:
        control (google.cloud.discoveryengine_v1.types.Control):
            Required. The Control to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Indicates which fields in the provided
            [Control][google.cloud.discoveryengine.v1.Control] to
            update. The following are NOT supported:

            - [Control.name][google.cloud.discoveryengine.v1.Control.name]
            - [Control.solution_type][google.cloud.discoveryengine.v1.Control.solution_type]

            If not set or empty, all supported fields are updated.
    """

    control: gcd_control.Control = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_control.Control,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteControlRequest(proto.Message):
    r"""Request for DeleteControl method.

    Attributes:
        name (str):
            Required. The resource name of the Control to delete.
            Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}/controls/{control_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetControlRequest(proto.Message):
    r"""Request for GetControl method.

    Attributes:
        name (str):
            Required. The resource name of the Control to get. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}/controls/{control_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListControlsRequest(proto.Message):
    r"""Request for ListControls method.

    Attributes:
        parent (str):
            Required. The data store resource name. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}``
            or
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``.
        page_size (int):
            Optional. Maximum number of results to
            return. If unspecified, defaults to 50. Max
            allowed value is 1000.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListControls`` call. Provide this to retrieve the
            subsequent page.
        filter (str):
            Optional. A filter to apply on the list results. Supported
            features:

            - List all the products under the parent branch if
              [filter][google.cloud.discoveryengine.v1.ListControlsRequest.filter]
              is unset. Currently this field is unsupported.
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


class ListControlsResponse(proto.Message):
    r"""Response for ListControls method.

    Attributes:
        controls (MutableSequence[google.cloud.discoveryengine_v1.types.Control]):
            All the Controls for a given data store.
        next_page_token (str):
            Pagination token, if not returned indicates
            the last page.
    """

    @property
    def raw_page(self):
        return self

    controls: MutableSequence[gcd_control.Control] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_control.Control,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
