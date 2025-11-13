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

from google.cloud.discoveryengine_v1beta.types import (
    serving_config as gcd_serving_config,
)

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "UpdateServingConfigRequest",
        "GetServingConfigRequest",
        "ListServingConfigsRequest",
        "ListServingConfigsResponse",
    },
)


class UpdateServingConfigRequest(proto.Message):
    r"""Request for UpdateServingConfig method.

    Attributes:
        serving_config (google.cloud.discoveryengine_v1beta.types.ServingConfig):
            Required. The ServingConfig to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            [ServingConfig][google.cloud.discoveryengine.v1beta.ServingConfig]
            to update. The following are NOT supported:

            - [ServingConfig.name][google.cloud.discoveryengine.v1beta.ServingConfig.name]

            If not set, all supported fields are updated.
    """

    serving_config: gcd_serving_config.ServingConfig = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_serving_config.ServingConfig,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class GetServingConfigRequest(proto.Message):
    r"""Request for GetServingConfig method.

    Attributes:
        name (str):
            Required. The resource name of the ServingConfig to get.
            Format:
            ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/servingConfigs/{serving_config_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListServingConfigsRequest(proto.Message):
    r"""Request for ListServingConfigs method.

    Attributes:
        parent (str):
            Required. Full resource name of the parent resource. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}``
        page_size (int):
            Optional. Maximum number of results to
            return. If unspecified, defaults to 100. If a
            value greater than 100 is provided, at most 100
            results are returned.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListServingConfigs`` call. Provide this to retrieve the
            subsequent page.
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


class ListServingConfigsResponse(proto.Message):
    r"""Response for ListServingConfigs method.

    Attributes:
        serving_configs (MutableSequence[google.cloud.discoveryengine_v1beta.types.ServingConfig]):
            All the ServingConfigs for a given dataStore.
        next_page_token (str):
            Pagination token, if not returned indicates
            the last page.
    """

    @property
    def raw_page(self):
        return self

    serving_configs: MutableSequence[
        gcd_serving_config.ServingConfig
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_serving_config.ServingConfig,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
