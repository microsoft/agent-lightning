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

from google.cloud.discoveryengine_v1alpha.types import acl_config as gcd_acl_config

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "GetAclConfigRequest",
        "UpdateAclConfigRequest",
    },
)


class GetAclConfigRequest(proto.Message):
    r"""Request message for GetAclConfigRequest method.

    Attributes:
        name (str):
            Required. Resource name of
            [AclConfig][google.cloud.discoveryengine.v1alpha.AclConfig],
            such as ``projects/*/locations/*/aclConfig``.

            If the caller does not have permission to access the
            [AclConfig][google.cloud.discoveryengine.v1alpha.AclConfig],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateAclConfigRequest(proto.Message):
    r"""Request message for UpdateAclConfig method.

    Attributes:
        acl_config (google.cloud.discoveryengine_v1alpha.types.AclConfig):

    """

    acl_config: gcd_acl_config.AclConfig = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_acl_config.AclConfig,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
