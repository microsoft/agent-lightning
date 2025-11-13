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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "ApiAuth",
    },
)


class ApiAuth(proto.Message):
    r"""The generic reusable api auth config.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        api_key_config (google.cloud.aiplatform_v1.types.ApiAuth.ApiKeyConfig):
            The API secret.

            This field is a member of `oneof`_ ``auth_config``.
    """

    class ApiKeyConfig(proto.Message):
        r"""The API secret.

        Attributes:
            api_key_secret_version (str):
                Required. The SecretManager secret version
                resource name storing API key. e.g.
                projects/{project}/secrets/{secret}/versions/{version}
        """

        api_key_secret_version: str = proto.Field(
            proto.STRING,
            number=1,
        )

    api_key_config: ApiKeyConfig = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="auth_config",
        message=ApiKeyConfig,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
