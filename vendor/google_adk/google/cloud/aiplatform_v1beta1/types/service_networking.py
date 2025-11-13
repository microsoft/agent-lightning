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
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "PSCAutomationState",
        "PSCAutomationConfig",
        "PrivateServiceConnectConfig",
        "PscAutomatedEndpoints",
        "PscInterfaceConfig",
        "DnsPeeringConfig",
    },
)


class PSCAutomationState(proto.Enum):
    r"""The state of the PSC service automation.

    Values:
        PSC_AUTOMATION_STATE_UNSPECIFIED (0):
            Should not be used.
        PSC_AUTOMATION_STATE_SUCCESSFUL (1):
            The PSC service automation is successful.
        PSC_AUTOMATION_STATE_FAILED (2):
            The PSC service automation has failed.
    """

    PSC_AUTOMATION_STATE_UNSPECIFIED = 0
    PSC_AUTOMATION_STATE_SUCCESSFUL = 1
    PSC_AUTOMATION_STATE_FAILED = 2


class PSCAutomationConfig(proto.Message):
    r"""PSC config that is used to automatically create PSC endpoints
    in the user projects.

    Attributes:
        project_id (str):
            Required. Project id used to create
            forwarding rule.
        network (str):
            Required. The full name of the Google Compute Engine
            `network <https://cloud.google.com/compute/docs/networks-and-firewalls#networks>`__.
            `Format <https://cloud.google.com/compute/docs/reference/rest/v1/networks/get>`__:
            ``projects/{project}/global/networks/{network}``.
        ip_address (str):
            Output only. IP address rule created by the
            PSC service automation.
        forwarding_rule (str):
            Output only. Forwarding rule created by the
            PSC service automation.
        state (google.cloud.aiplatform_v1beta1.types.PSCAutomationState):
            Output only. The state of the PSC service
            automation.
        error_message (str):
            Output only. Error message if the PSC service
            automation failed.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    network: str = proto.Field(
        proto.STRING,
        number=2,
    )
    ip_address: str = proto.Field(
        proto.STRING,
        number=3,
    )
    forwarding_rule: str = proto.Field(
        proto.STRING,
        number=4,
    )
    state: "PSCAutomationState" = proto.Field(
        proto.ENUM,
        number=5,
        enum="PSCAutomationState",
    )
    error_message: str = proto.Field(
        proto.STRING,
        number=6,
    )


class PrivateServiceConnectConfig(proto.Message):
    r"""Represents configuration for private service connect.

    Attributes:
        enable_private_service_connect (bool):
            Required. If true, expose the IndexEndpoint
            via private service connect.
        project_allowlist (MutableSequence[str]):
            A list of Projects from which the forwarding
            rule will target the service attachment.
        psc_automation_configs (MutableSequence[google.cloud.aiplatform_v1beta1.types.PSCAutomationConfig]):
            Optional. List of projects and networks where
            the PSC endpoints will be created. This field is
            used by Online Inference(Prediction) only.
        enable_secure_private_service_connect (bool):
            Optional. If set to true, enable secure
            private service connect with IAM authorization.
            Otherwise, private service connect will be done
            without authorization. Note latency will be
            slightly increased if authorization is enabled.
        service_attachment (str):
            Output only. The name of the generated
            service attachment resource. This is only
            populated if the endpoint is deployed with
            PrivateServiceConnect.
    """

    enable_private_service_connect: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    project_allowlist: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    psc_automation_configs: MutableSequence["PSCAutomationConfig"] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=3,
            message="PSCAutomationConfig",
        )
    )
    enable_secure_private_service_connect: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    service_attachment: str = proto.Field(
        proto.STRING,
        number=5,
    )


class PscAutomatedEndpoints(proto.Message):
    r"""PscAutomatedEndpoints defines the output of the forwarding
    rule automatically created by each PscAutomationConfig.

    Attributes:
        project_id (str):
            Corresponding project_id in pscAutomationConfigs
        network (str):
            Corresponding network in
            pscAutomationConfigs.
        match_address (str):
            Ip Address created by the automated
            forwarding rule.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    network: str = proto.Field(
        proto.STRING,
        number=2,
    )
    match_address: str = proto.Field(
        proto.STRING,
        number=3,
    )


class PscInterfaceConfig(proto.Message):
    r"""Configuration for PSC-I.

    Attributes:
        network_attachment (str):
            Optional. The name of the Compute Engine `network
            attachment <https://cloud.google.com/vpc/docs/about-network-attachments>`__
            to attach to the resource within the region and user
            project. To specify this field, you must have already
            [created a network attachment]
            (https://cloud.google.com/vpc/docs/create-manage-network-attachments#create-network-attachments).
            This field is only used for resources using PSC-I.
        dns_peering_configs (MutableSequence[google.cloud.aiplatform_v1beta1.types.DnsPeeringConfig]):
            Optional. DNS peering configurations. When
            specified, Vertex AI will attempt to configure
            DNS peering zones in the tenant project VPC to
            resolve the specified domains using the target
            network's Cloud DNS. The user must grant the
            dns.peer role to the Vertex AI Service Agent on
            the target project.
    """

    network_attachment: str = proto.Field(
        proto.STRING,
        number=1,
    )
    dns_peering_configs: MutableSequence["DnsPeeringConfig"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="DnsPeeringConfig",
    )


class DnsPeeringConfig(proto.Message):
    r"""DNS peering configuration. These configurations are used to
    create DNS peering zones in the Vertex tenant project VPC,
    enabling resolution of records within the specified domain
    hosted in the target network's Cloud DNS.

    Attributes:
        domain (str):
            Required. The DNS name suffix of the zone
            being peered to, e.g.,
            "my-internal-domain.corp.". Must end with a dot.
        target_project (str):
            Required. The project ID hosting the Cloud
            DNS managed zone that contains the 'domain'. The
            Vertex AI Service Agent requires the dns.peer
            role on this project.
        target_network (str):
            Required. The VPC network name in the target_project where
            the DNS zone specified by 'domain' is visible.
    """

    domain: str = proto.Field(
        proto.STRING,
        number=1,
    )
    target_project: str = proto.Field(
        proto.STRING,
        number=2,
    )
    target_network: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
