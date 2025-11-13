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

from google.cloud.monitoring_v3.types import alert

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "CreateAlertPolicyRequest",
        "GetAlertPolicyRequest",
        "ListAlertPoliciesRequest",
        "ListAlertPoliciesResponse",
        "UpdateAlertPolicyRequest",
        "DeleteAlertPolicyRequest",
    },
)


class CreateAlertPolicyRequest(proto.Message):
    r"""The protocol for the ``CreateAlertPolicy`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            in which to create the alerting policy. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]

            Note that this field names the parent container in which the
            alerting policy will be written, not the name of the created
            policy. \|name\| must be a host project of a Metrics Scope,
            otherwise INVALID_ARGUMENT error will return. The alerting
            policy that is returned will have a name that contains a
            normalized representation of this name as a prefix but adds
            a suffix of the form ``/alertPolicies/[ALERT_POLICY_ID]``,
            identifying the policy in the container.
        alert_policy (google.cloud.monitoring_v3.types.AlertPolicy):
            Required. The requested alerting policy. You should omit the
            ``name`` field in this policy. The name will be returned in
            the new policy, including a new ``[ALERT_POLICY_ID]`` value.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    alert_policy: alert.AlertPolicy = proto.Field(
        proto.MESSAGE,
        number=2,
        message=alert.AlertPolicy,
    )


class GetAlertPolicyRequest(proto.Message):
    r"""The protocol for the ``GetAlertPolicy`` request.

    Attributes:
        name (str):
            Required. The alerting policy to retrieve. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/alertPolicies/[ALERT_POLICY_ID]
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListAlertPoliciesRequest(proto.Message):
    r"""The protocol for the ``ListAlertPolicies`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            whose alert policies are to be listed. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]

            Note that this field names the parent container in which the
            alerting policies to be listed are stored. To retrieve a
            single alerting policy by name, use the
            [GetAlertPolicy][google.monitoring.v3.AlertPolicyService.GetAlertPolicy]
            operation, instead.
        filter (str):
            Optional. If provided, this field specifies the criteria
            that must be met by alert policies to be included in the
            response.

            For more details, see `sorting and
            filtering <https://cloud.google.com/monitoring/api/v3/sorting-and-filtering>`__.
        order_by (str):
            Optional. A comma-separated list of fields by which to sort
            the result. Supports the same set of field references as the
            ``filter`` field. Entries can be prefixed with a minus sign
            to sort by the field in descending order.

            For more details, see `sorting and
            filtering <https://cloud.google.com/monitoring/api/v3/sorting-and-filtering>`__.
        page_size (int):
            Optional. The maximum number of results to
            return in a single response.
        page_token (str):
            Optional. If this field is not empty then it must contain
            the ``nextPageToken`` value returned by a previous call to
            this method. Using this field causes the method to return
            more results from the previous method call.
    """

    name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=5,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListAlertPoliciesResponse(proto.Message):
    r"""The protocol for the ``ListAlertPolicies`` response.

    Attributes:
        alert_policies (MutableSequence[google.cloud.monitoring_v3.types.AlertPolicy]):
            The returned alert policies.
        next_page_token (str):
            If there might be more results than were returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
        total_size (int):
            The total number of alert policies in all
            pages. This number is only an estimate, and may
            change in subsequent pages. https://aip.dev/158
    """

    @property
    def raw_page(self):
        return self

    alert_policies: MutableSequence[alert.AlertPolicy] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=alert.AlertPolicy,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    total_size: int = proto.Field(
        proto.INT32,
        number=4,
    )


class UpdateAlertPolicyRequest(proto.Message):
    r"""The protocol for the ``UpdateAlertPolicy`` request.

    Attributes:
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. A list of alerting policy field names. If this
            field is not empty, each listed field in the existing
            alerting policy is set to the value of the corresponding
            field in the supplied policy (``alert_policy``), or to the
            field's default value if the field is not in the supplied
            alerting policy. Fields not listed retain their previous
            value.

            Examples of valid field masks include ``display_name``,
            ``documentation``, ``documentation.content``,
            ``documentation.mime_type``, ``user_labels``,
            ``user_label.nameofkey``, ``enabled``, ``conditions``,
            ``combiner``, etc.

            If this field is empty, then the supplied alerting policy
            replaces the existing policy. It is the same as deleting the
            existing policy and adding the supplied policy, except for
            the following:

            - The new policy will have the same ``[ALERT_POLICY_ID]`` as
              the former policy. This gives you continuity with the
              former policy in your notifications and incidents.
            - Conditions in the new policy will keep their former
              ``[CONDITION_ID]`` if the supplied condition includes the
              ``name`` field with that ``[CONDITION_ID]``. If the
              supplied condition omits the ``name`` field, then a new
              ``[CONDITION_ID]`` is created.
        alert_policy (google.cloud.monitoring_v3.types.AlertPolicy):
            Required. The updated alerting policy or the updated values
            for the fields listed in ``update_mask``. If ``update_mask``
            is not empty, any fields in this policy that are not in
            ``update_mask`` are ignored.
    """

    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )
    alert_policy: alert.AlertPolicy = proto.Field(
        proto.MESSAGE,
        number=3,
        message=alert.AlertPolicy,
    )


class DeleteAlertPolicyRequest(proto.Message):
    r"""The protocol for the ``DeleteAlertPolicy`` request.

    Attributes:
        name (str):
            Required. The alerting policy to delete. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/alertPolicies/[ALERT_POLICY_ID]

            For more information, see
            [AlertPolicy][google.monitoring.v3.AlertPolicy].
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
