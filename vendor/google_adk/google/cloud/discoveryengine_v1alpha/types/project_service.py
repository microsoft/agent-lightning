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
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "GetProjectRequest",
        "ProvisionProjectRequest",
        "ProvisionProjectMetadata",
        "ReportConsentChangeRequest",
    },
)


class GetProjectRequest(proto.Message):
    r"""Request message for
    [ProjectService.GetProject][google.cloud.discoveryengine.v1alpha.ProjectService.GetProject]
    method.

    Attributes:
        name (str):
            Required. Full resource name of a
            [Project][google.cloud.discoveryengine.v1alpha.Project],
            such as ``projects/{project_id_or_number}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ProvisionProjectRequest(proto.Message):
    r"""Request for
    [ProjectService.ProvisionProject][google.cloud.discoveryengine.v1alpha.ProjectService.ProvisionProject]
    method.

    Attributes:
        name (str):
            Required. Full resource name of a
            [Project][google.cloud.discoveryengine.v1alpha.Project],
            such as ``projects/{project_id_or_number}``.
        accept_data_use_terms (bool):
            Required. Set to ``true`` to specify that caller has read
            and would like to give consent to the `Terms for data
            use <https://cloud.google.com/retail/data-use-terms>`__.
        data_use_terms_version (str):
            Required. The version of the `Terms for data
            use <https://cloud.google.com/retail/data-use-terms>`__ that
            caller has read and would like to give consent to.

            Acceptable version is ``2022-11-23``, and this may change
            over time.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    accept_data_use_terms: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    data_use_terms_version: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ProvisionProjectMetadata(proto.Message):
    r"""Metadata associated with a project provision operation."""


class ReportConsentChangeRequest(proto.Message):
    r"""Request for ReportConsentChange method.

    Attributes:
        consent_change_action (google.cloud.discoveryengine_v1alpha.types.ReportConsentChangeRequest.ConsentChangeAction):
            Required. Whether customer decides to accept
            or decline service term.
            At this moment, only accept action is supported.
        project (str):
            Required. Full resource name of a
            [Project][google.cloud.discoveryengine.v1alpha.Project],
            such as ``projects/{project_id_or_number}``.
        service_term_id (str):
            Required. The unique identifier of the terms of service to
            update. Available term ids:

            - ``GA_DATA_USE_TERMS``: `Terms for data
              use <https://cloud.google.com/retail/data-use-terms>`__.
              When using this service term id, the acceptable
              [service_term_version][google.cloud.discoveryengine.v1alpha.ReportConsentChangeRequest.service_term_version]
              to provide is ``2022-11-23``.
        service_term_version (str):
            Required. The version string of the terms of
            service to update.
    """

    class ConsentChangeAction(proto.Enum):
        r"""Type of consent acknowledgement (accept / reject).

        At this moment, only ``ACCEPT`` action is supported.

        Values:
            CONSENT_CHANGE_ACTION_UNSPECIFIED (0):
                Invalid action, user must specify
                accept/decline
            ACCEPT (1):
                User accepts service terms.
        """
        CONSENT_CHANGE_ACTION_UNSPECIFIED = 0
        ACCEPT = 1

    consent_change_action: ConsentChangeAction = proto.Field(
        proto.ENUM,
        number=1,
        enum=ConsentChangeAction,
    )
    project: str = proto.Field(
        proto.STRING,
        number=2,
    )
    service_term_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    service_term_version: str = proto.Field(
        proto.STRING,
        number=4,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
