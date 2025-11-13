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

from google.cloud.aiplatform_v1beta1.types import model_monitor as gca_model_monitor
from google.cloud.aiplatform_v1beta1.types import model_monitoring_alert
from google.cloud.aiplatform_v1beta1.types import (
    model_monitoring_job as gca_model_monitoring_job,
)
from google.cloud.aiplatform_v1beta1.types import model_monitoring_stats
from google.cloud.aiplatform_v1beta1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore
from google.type import interval_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateModelMonitorRequest",
        "CreateModelMonitorOperationMetadata",
        "UpdateModelMonitorRequest",
        "UpdateModelMonitorOperationMetadata",
        "GetModelMonitorRequest",
        "ListModelMonitorsRequest",
        "ListModelMonitorsResponse",
        "DeleteModelMonitorRequest",
        "CreateModelMonitoringJobRequest",
        "GetModelMonitoringJobRequest",
        "ListModelMonitoringJobsRequest",
        "ListModelMonitoringJobsResponse",
        "DeleteModelMonitoringJobRequest",
        "SearchModelMonitoringStatsRequest",
        "SearchModelMonitoringStatsResponse",
        "SearchModelMonitoringAlertsRequest",
        "SearchModelMonitoringAlertsResponse",
    },
)


class CreateModelMonitorRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.CreateModelMonitor][google.cloud.aiplatform.v1beta1.ModelMonitoringService.CreateModelMonitor].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            ModelMonitor in. Format:
            ``projects/{project}/locations/{location}``
        model_monitor (google.cloud.aiplatform_v1beta1.types.ModelMonitor):
            Required. The ModelMonitor to create.
        model_monitor_id (str):
            Optional. The ID to use for the Model Monitor, which will
            become the final component of the model monitor resource
            name.

            The maximum length is 63 characters, and valid characters
            are ``/^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    model_monitor: gca_model_monitor.ModelMonitor = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_model_monitor.ModelMonitor,
    )
    model_monitor_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CreateModelMonitorOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [ModelMonitoringService.CreateModelMonitor][google.cloud.aiplatform.v1beta1.ModelMonitoringService.CreateModelMonitor].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class UpdateModelMonitorRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.UpdateModelMonitor][google.cloud.aiplatform.v1beta1.ModelMonitoringService.UpdateModelMonitor].

    Attributes:
        model_monitor (google.cloud.aiplatform_v1beta1.types.ModelMonitor):
            Required. The model monitoring configuration
            which replaces the resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. Mask specifying which fields to
            update.
    """

    model_monitor: gca_model_monitor.ModelMonitor = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_model_monitor.ModelMonitor,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateModelMonitorOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [ModelMonitoringService.UpdateModelMonitor][google.cloud.aiplatform.v1beta1.ModelMonitoringService.UpdateModelMonitor].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The operation generic information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GetModelMonitorRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.GetModelMonitor][google.cloud.aiplatform.v1beta1.ModelMonitoringService.GetModelMonitor].

    Attributes:
        name (str):
            Required. The name of the ModelMonitor resource. Format:
            ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListModelMonitorsRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.ListModelMonitors][google.cloud.aiplatform.v1beta1.ModelMonitoringService.ListModelMonitors].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            ModelMonitors from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            The standard list filter. More detail in
            `AIP-160 <https://google.aip.dev/160>`__.
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read.
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListModelMonitorsResponse(proto.Message):
    r"""Response message for
    [ModelMonitoringService.ListModelMonitors][google.cloud.aiplatform.v1beta1.ModelMonitoringService.ListModelMonitors]

    Attributes:
        model_monitors (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitor]):
            List of ModelMonitor in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListModelMonitorsRequest.page_token][google.cloud.aiplatform.v1beta1.ListModelMonitorsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    model_monitors: MutableSequence[gca_model_monitor.ModelMonitor] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=gca_model_monitor.ModelMonitor,
        )
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteModelMonitorRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.DeleteModelMonitor][google.cloud.aiplatform.v1beta1.ModelMonitoringService.DeleteModelMonitor].

    Attributes:
        name (str):
            Required. The name of the ModelMonitor resource to be
            deleted. Format:
            ``projects/{project}/locations/{location}/modelMonitords/{model_monitor}``
        force (bool):
            Optional. Force delete the model monitor with
            schedules.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    force: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class CreateModelMonitoringJobRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.CreateModelMonitoringJob][google.cloud.aiplatform.v1beta1.ModelMonitoringService.CreateModelMonitoringJob].

    Attributes:
        parent (str):
            Required. The parent of the ModelMonitoringJob. Format:
            ``projects/{project}/locations/{location}/modelMoniitors/{model_monitor}``
        model_monitoring_job (google.cloud.aiplatform_v1beta1.types.ModelMonitoringJob):
            Required. The ModelMonitoringJob to create
        model_monitoring_job_id (str):
            Optional. The ID to use for the Model Monitoring Job, which
            will become the final component of the model monitoring job
            resource name.

            The maximum length is 63 characters, and valid characters
            are ``/^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    model_monitoring_job: gca_model_monitoring_job.ModelMonitoringJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_model_monitoring_job.ModelMonitoringJob,
    )
    model_monitoring_job_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class GetModelMonitoringJobRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.GetModelMonitoringJob][google.cloud.aiplatform.v1beta1.ModelMonitoringService.GetModelMonitoringJob].

    Attributes:
        name (str):
            Required. The resource name of the ModelMonitoringJob.
            Format:
            ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListModelMonitoringJobsRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.ListModelMonitoringJobs][google.cloud.aiplatform.v1beta1.ModelMonitoringService.ListModelMonitoringJobs].

    Attributes:
        parent (str):
            Required. The parent of the ModelMonitoringJob. Format:
            ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}``
        filter (str):
            The standard list filter. More detail in
            `AIP-160 <https://google.aip.dev/160>`__.
        page_size (int):
            The standard list page size.
        page_token (str):
            The standard list page token.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read
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
    read_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=5,
        message=field_mask_pb2.FieldMask,
    )


class ListModelMonitoringJobsResponse(proto.Message):
    r"""Response message for
    [ModelMonitoringService.ListModelMonitoringJobs][google.cloud.aiplatform.v1beta1.ModelMonitoringService.ListModelMonitoringJobs].

    Attributes:
        model_monitoring_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringJob]):
            A list of ModelMonitoringJobs that matches
            the specified filter in the request.
        next_page_token (str):
            The standard List next-page token.
    """

    @property
    def raw_page(self):
        return self

    model_monitoring_jobs: MutableSequence[
        gca_model_monitoring_job.ModelMonitoringJob
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_model_monitoring_job.ModelMonitoringJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteModelMonitoringJobRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.DeleteModelMonitoringJob][google.cloud.aiplatform.v1beta1.ModelMonitoringService.DeleteModelMonitoringJob].

    Attributes:
        name (str):
            Required. The resource name of the model monitoring job to
            delete. Format:
            ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class SearchModelMonitoringStatsRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.SearchModelMonitoringStats][google.cloud.aiplatform.v1beta1.ModelMonitoringService.SearchModelMonitoringStats].

    Attributes:
        model_monitor (str):
            Required. ModelMonitor resource name. Format:
            ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}``
        stats_filter (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsFilter):
            Filter for search different stats.
        time_interval (google.type.interval_pb2.Interval):
            The time interval for which results should be
            returned.
        page_size (int):
            The standard list page size.
        page_token (str):
            A page token received from a previous
            [ModelMonitoringService.SearchModelMonitoringStats][google.cloud.aiplatform.v1beta1.ModelMonitoringService.SearchModelMonitoringStats]
            call.
    """

    model_monitor: str = proto.Field(
        proto.STRING,
        number=1,
    )
    stats_filter: model_monitoring_stats.SearchModelMonitoringStatsFilter = proto.Field(
        proto.MESSAGE,
        number=2,
        message=model_monitoring_stats.SearchModelMonitoringStatsFilter,
    )
    time_interval: interval_pb2.Interval = proto.Field(
        proto.MESSAGE,
        number=3,
        message=interval_pb2.Interval,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=4,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=5,
    )


class SearchModelMonitoringStatsResponse(proto.Message):
    r"""Response message for
    [ModelMonitoringService.SearchModelMonitoringStats][google.cloud.aiplatform.v1beta1.ModelMonitoringService.SearchModelMonitoringStats].

    Attributes:
        monitoring_stats (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringStats]):
            Stats retrieved for requested objectives.
        next_page_token (str):
            The page token that can be used by the next
            [ModelMonitoringService.SearchModelMonitoringStats][google.cloud.aiplatform.v1beta1.ModelMonitoringService.SearchModelMonitoringStats]
            call.
    """

    @property
    def raw_page(self):
        return self

    monitoring_stats: MutableSequence[model_monitoring_stats.ModelMonitoringStats] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=model_monitoring_stats.ModelMonitoringStats,
        )
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class SearchModelMonitoringAlertsRequest(proto.Message):
    r"""Request message for
    [ModelMonitoringService.SearchModelMonitoringAlerts][google.cloud.aiplatform.v1beta1.ModelMonitoringService.SearchModelMonitoringAlerts].

    Attributes:
        model_monitor (str):
            Required. ModelMonitor resource name. Format:
            ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}``
        model_monitoring_job (str):
            If non-empty, returns the alerts of this
            model monitoring job.
        alert_time_interval (google.type.interval_pb2.Interval):
            If non-empty, returns the alerts in this time
            interval.
        stats_name (str):
            If non-empty, returns the alerts of this stats_name.
        objective_type (str):
            If non-empty, returns the alerts of this objective type.
            Supported monitoring objectives: ``raw-feature-drift``
            ``prediction-output-drift`` ``feature-attribution``
        page_size (int):
            The standard list page size.
        page_token (str):
            A page token received from a previous
            [ModelMonitoringService.SearchModelMonitoringAlerts][google.cloud.aiplatform.v1beta1.ModelMonitoringService.SearchModelMonitoringAlerts]
            call.
    """

    model_monitor: str = proto.Field(
        proto.STRING,
        number=1,
    )
    model_monitoring_job: str = proto.Field(
        proto.STRING,
        number=2,
    )
    alert_time_interval: interval_pb2.Interval = proto.Field(
        proto.MESSAGE,
        number=3,
        message=interval_pb2.Interval,
    )
    stats_name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    objective_type: str = proto.Field(
        proto.STRING,
        number=5,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=6,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=7,
    )


class SearchModelMonitoringAlertsResponse(proto.Message):
    r"""Response message for
    [ModelMonitoringService.SearchModelMonitoringAlerts][google.cloud.aiplatform.v1beta1.ModelMonitoringService.SearchModelMonitoringAlerts].

    Attributes:
        model_monitoring_alerts (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlert]):
            Alerts retrieved for the requested
            objectives. Sorted by alert time descendingly.
        total_number_alerts (int):
            The total number of alerts retrieved by the
            requested objectives.
        next_page_token (str):
            The page token that can be used by the next
            [ModelMonitoringService.SearchModelMonitoringAlerts][google.cloud.aiplatform.v1beta1.ModelMonitoringService.SearchModelMonitoringAlerts]
            call.
    """

    @property
    def raw_page(self):
        return self

    model_monitoring_alerts: MutableSequence[
        model_monitoring_alert.ModelMonitoringAlert
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=model_monitoring_alert.ModelMonitoringAlert,
    )
    total_number_alerts: int = proto.Field(
        proto.INT64,
        number=2,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
