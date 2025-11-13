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

from google.cloud.discoveryengine_v1.types import serving_config as gcd_serving_config

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "UpdateServingConfigRequest",
    },
)


class UpdateServingConfigRequest(proto.Message):
    r"""Request for UpdateServingConfig method.

    Attributes:
        serving_config (google.cloud.discoveryengine_v1.types.ServingConfig):
            Required. The ServingConfig to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            [ServingConfig][google.cloud.discoveryengine.v1.ServingConfig]
            to update. The following are NOT supported:

            - [ServingConfig.name][google.cloud.discoveryengine.v1.ServingConfig.name]

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


__all__ = tuple(sorted(__protobuf__.manifest))
