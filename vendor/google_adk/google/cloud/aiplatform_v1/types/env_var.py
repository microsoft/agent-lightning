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
        "EnvVar",
        "SecretRef",
        "SecretEnvVar",
    },
)


class EnvVar(proto.Message):
    r"""Represents an environment variable present in a Container or
    Python Module.

    Attributes:
        name (str):
            Required. Name of the environment variable.
            Must be a valid C identifier.
        value (str):
            Required. Variables that reference a $(VAR_NAME) are
            expanded using the previous defined environment variables in
            the container and any service environment variables. If a
            variable cannot be resolved, the reference in the input
            string will be unchanged. The $(VAR_NAME) syntax can be
            escaped with a double $$, ie: $$(VAR_NAME). Escaped
            references will never be expanded, regardless of whether the
            variable exists or not.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    value: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SecretRef(proto.Message):
    r"""Reference to a secret stored in the Cloud Secret Manager that
    will provide the value for this environment variable.

    Attributes:
        secret (str):
            Required. The name of the secret in Cloud Secret Manager.
            Format: {secret_name}.
        version (str):
            The Cloud Secret Manager secret version.
            Can be 'latest' for the latest version, an
            integer for a specific version, or a version
            alias.
    """

    secret: str = proto.Field(
        proto.STRING,
        number=1,
    )
    version: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SecretEnvVar(proto.Message):
    r"""Represents an environment variable where the value is a
    secret in Cloud Secret Manager.

    Attributes:
        name (str):
            Required. Name of the secret environment
            variable.
        secret_ref (google.cloud.aiplatform_v1.types.SecretRef):
            Required. Reference to a secret stored in the
            Cloud Secret Manager that will provide the value
            for this environment variable.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    secret_ref: "SecretRef" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SecretRef",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
