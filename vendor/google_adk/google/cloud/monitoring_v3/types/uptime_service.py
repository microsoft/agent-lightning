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

from google.cloud.monitoring_v3.types import uptime

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "ListUptimeCheckConfigsRequest",
        "ListUptimeCheckConfigsResponse",
        "GetUptimeCheckConfigRequest",
        "CreateUptimeCheckConfigRequest",
        "UpdateUptimeCheckConfigRequest",
        "DeleteUptimeCheckConfigRequest",
        "ListUptimeCheckIpsRequest",
        "ListUptimeCheckIpsResponse",
    },
)


class ListUptimeCheckConfigsRequest(proto.Message):
    r"""The protocol for the ``ListUptimeCheckConfigs`` request.

    Attributes:
        parent (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            whose Uptime check configurations are listed. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        filter (str):
            If provided, this field specifies the criteria that must be
            met by uptime checks to be included in the response.

            For more details, see `Filtering
            syntax <https://cloud.google.com/monitoring/api/v3/sorting-and-filtering#filter_syntax>`__.
        page_size (int):
            The maximum number of results to return in a single
            response. The server may further constrain the maximum
            number of results returned in a single page. If the
            page_size is <=0, the server will decide the number of
            results to be returned.
        page_token (str):
            If this field is not empty then it must contain the
            ``nextPageToken`` value returned by a previous call to this
            method. Using this field causes the method to return more
            results from the previous method call.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListUptimeCheckConfigsResponse(proto.Message):
    r"""The protocol for the ``ListUptimeCheckConfigs`` response.

    Attributes:
        uptime_check_configs (MutableSequence[google.cloud.monitoring_v3.types.UptimeCheckConfig]):
            The returned Uptime check configurations.
        next_page_token (str):
            This field represents the pagination token to retrieve the
            next page of results. If the value is empty, it means no
            further results for the request. To retrieve the next page
            of results, the value of the next_page_token is passed to
            the subsequent List method call (in the request message's
            page_token field).
        total_size (int):
            The total number of Uptime check
            configurations for the project, irrespective of
            any pagination.
    """

    @property
    def raw_page(self):
        return self

    uptime_check_configs: MutableSequence[
        uptime.UptimeCheckConfig
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=uptime.UptimeCheckConfig,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    total_size: int = proto.Field(
        proto.INT32,
        number=3,
    )


class GetUptimeCheckConfigRequest(proto.Message):
    r"""The protocol for the ``GetUptimeCheckConfig`` request.

    Attributes:
        name (str):
            Required. The Uptime check configuration to retrieve. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/uptimeCheckConfigs/[UPTIME_CHECK_ID]
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateUptimeCheckConfigRequest(proto.Message):
    r"""The protocol for the ``CreateUptimeCheckConfig`` request.

    Attributes:
        parent (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            in which to create the Uptime check. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        uptime_check_config (google.cloud.monitoring_v3.types.UptimeCheckConfig):
            Required. The new Uptime check configuration.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    uptime_check_config: uptime.UptimeCheckConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        message=uptime.UptimeCheckConfig,
    )


class UpdateUptimeCheckConfigRequest(proto.Message):
    r"""The protocol for the ``UpdateUptimeCheckConfig`` request.

    Attributes:
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. If present, only the listed fields
            in the current Uptime check configuration are
            updated with values from the new configuration.
            If this field is empty, then the current
            configuration is completely replaced with the
            new configuration.
        uptime_check_config (google.cloud.monitoring_v3.types.UptimeCheckConfig):
            Required. If an ``updateMask`` has been specified, this
            field gives the values for the set of fields mentioned in
            the ``updateMask``. If an ``updateMask`` has not been given,
            this Uptime check configuration replaces the current
            configuration. If a field is mentioned in ``updateMask`` but
            the corresponding field is omitted in this partial Uptime
            check configuration, it has the effect of deleting/clearing
            the field from the configuration on the server.

            The following fields can be updated: ``display_name``,
            ``http_check``, ``tcp_check``, ``timeout``,
            ``content_matchers``, and ``selected_regions``.
    """

    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )
    uptime_check_config: uptime.UptimeCheckConfig = proto.Field(
        proto.MESSAGE,
        number=3,
        message=uptime.UptimeCheckConfig,
    )


class DeleteUptimeCheckConfigRequest(proto.Message):
    r"""The protocol for the ``DeleteUptimeCheckConfig`` request.

    Attributes:
        name (str):
            Required. The Uptime check configuration to delete. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/uptimeCheckConfigs/[UPTIME_CHECK_ID]
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListUptimeCheckIpsRequest(proto.Message):
    r"""The protocol for the ``ListUptimeCheckIps`` request.

    Attributes:
        page_size (int):
            The maximum number of results to return in a single
            response. The server may further constrain the maximum
            number of results returned in a single page. If the
            page_size is <=0, the server will decide the number of
            results to be returned. NOTE: this field is not yet
            implemented
        page_token (str):
            If this field is not empty then it must contain the
            ``nextPageToken`` value returned by a previous call to this
            method. Using this field causes the method to return more
            results from the previous method call. NOTE: this field is
            not yet implemented
    """

    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListUptimeCheckIpsResponse(proto.Message):
    r"""The protocol for the ``ListUptimeCheckIps`` response.

    Attributes:
        uptime_check_ips (MutableSequence[google.cloud.monitoring_v3.types.UptimeCheckIp]):
            The returned list of IP addresses (including
            region and location) that the checkers run from.
        next_page_token (str):
            This field represents the pagination token to retrieve the
            next page of results. If the value is empty, it means no
            further results for the request. To retrieve the next page
            of results, the value of the next_page_token is passed to
            the subsequent List method call (in the request message's
            page_token field). NOTE: this field is not yet implemented
    """

    @property
    def raw_page(self):
        return self

    uptime_check_ips: MutableSequence[uptime.UptimeCheckIp] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=uptime.UptimeCheckIp,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
