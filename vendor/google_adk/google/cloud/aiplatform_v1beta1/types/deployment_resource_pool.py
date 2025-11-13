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

from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1beta1.types import machine_resources
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "DeploymentResourcePool",
    },
)


class DeploymentResourcePool(proto.Message):
    r"""A description of resources that can be shared by multiple
    DeployedModels, whose underlying specification consists of a
    DedicatedResources.

    Attributes:
        name (str):
            Immutable. The resource name of the DeploymentResourcePool.
            Format:
            ``projects/{project}/locations/{location}/deploymentResourcePools/{deployment_resource_pool}``
        dedicated_resources (google.cloud.aiplatform_v1beta1.types.DedicatedResources):
            Required. The underlying DedicatedResources
            that the DeploymentResourcePool uses.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Customer-managed encryption key spec for a
            DeploymentResourcePool. If set, this
            DeploymentResourcePool will be secured by this
            key. Endpoints and the DeploymentResourcePool
            they deploy in need to have the same
            EncryptionSpec.
        service_account (str):
            The service account that the DeploymentResourcePool's
            container(s) run as. Specify the email address of the
            service account. If this service account is not specified,
            the container(s) run as a service account that doesn't have
            access to the resource project.

            Users deploying the Models to this DeploymentResourcePool
            must have the ``iam.serviceAccounts.actAs`` permission on
            this service account.
        disable_container_logging (bool):
            If the DeploymentResourcePool is deployed with
            custom-trained Models or AutoML Tabular Models, the
            container(s) of the DeploymentResourcePool will send
            ``stderr`` and ``stdout`` streams to Cloud Logging by
            default. Please note that the logs incur cost, which are
            subject to `Cloud Logging
            pricing <https://cloud.google.com/logging/pricing>`__.

            User can disable container logging by setting this flag to
            true.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            DeploymentResourcePool was created.
        satisfies_pzs (bool):
            Output only. Reserved for future use.
        satisfies_pzi (bool):
            Output only. Reserved for future use.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    dedicated_resources: machine_resources.DedicatedResources = proto.Field(
        proto.MESSAGE,
        number=2,
        message=machine_resources.DedicatedResources,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=5,
        message=gca_encryption_spec.EncryptionSpec,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=6,
    )
    disable_container_logging: bool = proto.Field(
        proto.BOOL,
        number=7,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    satisfies_pzs: bool = proto.Field(
        proto.BOOL,
        number=8,
    )
    satisfies_pzi: bool = proto.Field(
        proto.BOOL,
        number=9,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
