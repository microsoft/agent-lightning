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

from google.cloud.discoveryengine_v1alpha.types import common

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "AclConfig",
    },
)


class AclConfig(proto.Message):
    r"""Access Control Configuration.

    Attributes:
        name (str):
            Immutable. The full resource name of the acl configuration.
            Format:
            ``projects/{project}/locations/{location}/aclConfig``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        idp_config (google.cloud.discoveryengine_v1alpha.types.IdpConfig):
            Identity provider config.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    idp_config: common.IdpConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        message=common.IdpConfig,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
