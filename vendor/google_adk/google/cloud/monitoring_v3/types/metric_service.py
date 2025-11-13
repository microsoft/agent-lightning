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

from google.api import metric_pb2  # type: ignore
from google.api import monitored_resource_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.monitoring_v3.types import common
from google.cloud.monitoring_v3.types import metric as gm_metric

__protobuf__ = proto.module(
    package="google.monitoring.v3",
    manifest={
        "ListMonitoredResourceDescriptorsRequest",
        "ListMonitoredResourceDescriptorsResponse",
        "GetMonitoredResourceDescriptorRequest",
        "ListMetricDescriptorsRequest",
        "ListMetricDescriptorsResponse",
        "GetMetricDescriptorRequest",
        "CreateMetricDescriptorRequest",
        "DeleteMetricDescriptorRequest",
        "ListTimeSeriesRequest",
        "ListTimeSeriesResponse",
        "CreateTimeSeriesRequest",
        "CreateTimeSeriesError",
        "CreateTimeSeriesSummary",
        "QueryTimeSeriesRequest",
        "QueryTimeSeriesResponse",
        "QueryErrorList",
    },
)


class ListMonitoredResourceDescriptorsRequest(proto.Message):
    r"""The ``ListMonitoredResourceDescriptors`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            on which to execute the request. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        filter (str):
            An optional
            `filter <https://cloud.google.com/monitoring/api/v3/filters>`__
            describing the descriptors to be returned. The filter can
            reference the descriptor's type and labels. For example, the
            following filter returns only Google Compute Engine
            descriptors that have an ``id`` label:

            ::

                resource.type = starts_with("gce_") AND resource.label:id
        page_size (int):
            A positive number that is the maximum number
            of results to return.
        page_token (str):
            If this field is not empty then it must contain the
            ``nextPageToken`` value returned by a previous call to this
            method. Using this field causes the method to return
            additional results from the previous method call.
    """

    name: str = proto.Field(
        proto.STRING,
        number=5,
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


class ListMonitoredResourceDescriptorsResponse(proto.Message):
    r"""The ``ListMonitoredResourceDescriptors`` response.

    Attributes:
        resource_descriptors (MutableSequence[google.api.monitored_resource_pb2.MonitoredResourceDescriptor]):
            The monitored resource descriptors that are available to
            this project and that match ``filter``, if present.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
    """

    @property
    def raw_page(self):
        return self

    resource_descriptors: MutableSequence[
        monitored_resource_pb2.MonitoredResourceDescriptor
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=monitored_resource_pb2.MonitoredResourceDescriptor,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetMonitoredResourceDescriptorRequest(proto.Message):
    r"""The ``GetMonitoredResourceDescriptor`` request.

    Attributes:
        name (str):
            Required. The monitored resource descriptor to get. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/monitoredResourceDescriptors/[RESOURCE_TYPE]

            The ``[RESOURCE_TYPE]`` is a predefined type, such as
            ``cloudsql_database``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListMetricDescriptorsRequest(proto.Message):
    r"""The ``ListMetricDescriptors`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            on which to execute the request. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        filter (str):
            Optional. If this field is empty, all custom and
            system-defined metric descriptors are returned. Otherwise,
            the
            `filter <https://cloud.google.com/monitoring/api/v3/filters>`__
            specifies which metric descriptors are to be returned. For
            example, the following filter matches all `custom
            metrics <https://cloud.google.com/monitoring/custom-metrics>`__:

            ::

                metric.type = starts_with("custom.googleapis.com/")
        page_size (int):
            Optional. A positive number that is the maximum number of
            results to return. The default and maximum value is 10,000.
            If a page_size <= 0 or > 10,000 is submitted, will instead
            return a maximum of 10,000 results.
        page_token (str):
            Optional. If this field is not empty then it must contain
            the ``nextPageToken`` value returned by a previous call to
            this method. Using this field causes the method to return
            additional results from the previous method call.
        active_only (bool):
            Optional. If true, only metrics and monitored
            resource types that have recent data (within
            roughly 25 hours) will be included in the
            response.

            - If a metric descriptor enumerates monitored
              resource types, only the monitored resource
              types for which the metric type has recent
              data will be included in the returned
              metric descriptor, and if none of them have
              recent data, the metric descriptor will not be
              returned.
            - If a metric descriptor does not enumerate the
              compatible monitored resource types, it
              will be returned only if the metric type has
              recent data for some monitored resource
              type. The returned descriptor will not
              enumerate any monitored resource types.
    """

    name: str = proto.Field(
        proto.STRING,
        number=5,
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
    active_only: bool = proto.Field(
        proto.BOOL,
        number=6,
    )


class ListMetricDescriptorsResponse(proto.Message):
    r"""The ``ListMetricDescriptors`` response.

    Attributes:
        metric_descriptors (MutableSequence[google.api.metric_pb2.MetricDescriptor]):
            The metric descriptors that are available to the project and
            that match the value of ``filter``, if present.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
    """

    @property
    def raw_page(self):
        return self

    metric_descriptors: MutableSequence[
        metric_pb2.MetricDescriptor
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=metric_pb2.MetricDescriptor,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class GetMetricDescriptorRequest(proto.Message):
    r"""The ``GetMetricDescriptor`` request.

    Attributes:
        name (str):
            Required. The metric descriptor on which to execute the
            request. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/metricDescriptors/[METRIC_ID]

            An example value of ``[METRIC_ID]`` is
            ``"compute.googleapis.com/instance/disk/read_bytes_count"``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CreateMetricDescriptorRequest(proto.Message):
    r"""The ``CreateMetricDescriptor`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            on which to execute the request. The format is: 4
            projects/[PROJECT_ID_OR_NUMBER]
        metric_descriptor (google.api.metric_pb2.MetricDescriptor):
            Required. The new `custom
            metric <https://cloud.google.com/monitoring/custom-metrics>`__
            descriptor.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    metric_descriptor: metric_pb2.MetricDescriptor = proto.Field(
        proto.MESSAGE,
        number=2,
        message=metric_pb2.MetricDescriptor,
    )


class DeleteMetricDescriptorRequest(proto.Message):
    r"""The ``DeleteMetricDescriptor`` request.

    Attributes:
        name (str):
            Required. The metric descriptor on which to execute the
            request. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]/metricDescriptors/[METRIC_ID]

            An example of ``[METRIC_ID]`` is:
            ``"custom.googleapis.com/my_test_metric"``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListTimeSeriesRequest(proto.Message):
    r"""The ``ListTimeSeries`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__,
            organization or folder on which to execute the request. The
            format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
                organizations/[ORGANIZATION_ID]
                folders/[FOLDER_ID]
        filter (str):
            Required. A `monitoring
            filter <https://cloud.google.com/monitoring/api/v3/filters>`__
            that specifies which time series should be returned. The
            filter must specify a single metric type, and can
            additionally specify metric labels and other information.
            For example:

            ::

                metric.type = "compute.googleapis.com/instance/cpu/usage_time" AND
                    metric.labels.instance_name = "my-instance-name".
        interval (google.cloud.monitoring_v3.types.TimeInterval):
            Required. The time interval for which results
            should be returned. Only time series that
            contain data points in the specified interval
            are included in the response.
        aggregation (google.cloud.monitoring_v3.types.Aggregation):
            Specifies the alignment of data points in individual time
            series as well as how to combine the retrieved time series
            across specified labels.

            By default (if no ``aggregation`` is explicitly specified),
            the raw time series data is returned.
        secondary_aggregation (google.cloud.monitoring_v3.types.Aggregation):
            Apply a second aggregation after ``aggregation`` is applied.
            May only be specified if ``aggregation`` is specified.
        order_by (str):
            Unsupported: must be left blank. The points
            in each time series are currently returned in
            reverse time order (most recent to oldest).
        view (google.cloud.monitoring_v3.types.ListTimeSeriesRequest.TimeSeriesView):
            Required. Specifies which information is
            returned about the time series.
        page_size (int):
            A positive number that is the maximum number of results to
            return. If ``page_size`` is empty or more than 100,000
            results, the effective ``page_size`` is 100,000 results. If
            ``view`` is set to ``FULL``, this is the maximum number of
            ``Points`` returned. If ``view`` is set to ``HEADERS``, this
            is the maximum number of ``TimeSeries`` returned.
        page_token (str):
            If this field is not empty then it must contain the
            ``nextPageToken`` value returned by a previous call to this
            method. Using this field causes the method to return
            additional results from the previous method call.
    """

    class TimeSeriesView(proto.Enum):
        r"""Controls which fields are returned by ``ListTimeSeries*``.

        Values:
            FULL (0):
                Returns the identity of the metric(s), the
                time series, and the time series data.
            HEADERS (1):
                Returns the identity of the metric and the
                time series resource, but not the time series
                data.
        """
        FULL = 0
        HEADERS = 1

    name: str = proto.Field(
        proto.STRING,
        number=10,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    interval: common.TimeInterval = proto.Field(
        proto.MESSAGE,
        number=4,
        message=common.TimeInterval,
    )
    aggregation: common.Aggregation = proto.Field(
        proto.MESSAGE,
        number=5,
        message=common.Aggregation,
    )
    secondary_aggregation: common.Aggregation = proto.Field(
        proto.MESSAGE,
        number=11,
        message=common.Aggregation,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=6,
    )
    view: TimeSeriesView = proto.Field(
        proto.ENUM,
        number=7,
        enum=TimeSeriesView,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=8,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=9,
    )


class ListTimeSeriesResponse(proto.Message):
    r"""The ``ListTimeSeries`` response.

    Attributes:
        time_series (MutableSequence[google.cloud.monitoring_v3.types.TimeSeries]):
            One or more time series that match the filter
            included in the request.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
        execution_errors (MutableSequence[google.rpc.status_pb2.Status]):
            Query execution errors that may have caused
            the time series data returned to be incomplete.
        unit (str):
            The unit in which all ``time_series`` point values are
            reported. ``unit`` follows the UCUM format for units as seen
            in https://unitsofmeasure.org/ucum.html. If different
            ``time_series`` have different units (for example, because
            they come from different metric types, or a unit is absent),
            then ``unit`` will be "{not_a_unit}".
    """

    @property
    def raw_page(self):
        return self

    time_series: MutableSequence[gm_metric.TimeSeries] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gm_metric.TimeSeries,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    execution_errors: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=status_pb2.Status,
    )
    unit: str = proto.Field(
        proto.STRING,
        number=5,
    )


class CreateTimeSeriesRequest(proto.Message):
    r"""The ``CreateTimeSeries`` request.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            on which to execute the request. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        time_series (MutableSequence[google.cloud.monitoring_v3.types.TimeSeries]):
            Required. The new data to be added to a list of time series.
            Adds at most one data point to each of several time series.
            The new data point must be more recent than any other point
            in its time series. Each ``TimeSeries`` value must fully
            specify a unique time series by supplying all label values
            for the metric and the monitored resource.

            The maximum number of ``TimeSeries`` objects per ``Create``
            request is 200.
    """

    name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    time_series: MutableSequence[gm_metric.TimeSeries] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=gm_metric.TimeSeries,
    )


class CreateTimeSeriesError(proto.Message):
    r"""DEPRECATED. Used to hold per-time-series error status.

    Attributes:
        time_series (google.cloud.monitoring_v3.types.TimeSeries):
            DEPRECATED. Time series ID that resulted in the ``status``
            error.
        status (google.rpc.status_pb2.Status):
            DEPRECATED. The status of the requested write operation for
            ``time_series``.
    """

    time_series: gm_metric.TimeSeries = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gm_metric.TimeSeries,
    )
    status: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=2,
        message=status_pb2.Status,
    )


class CreateTimeSeriesSummary(proto.Message):
    r"""Summary of the result of a failed request to write data to a
    time series.

    Attributes:
        total_point_count (int):
            The number of points in the request.
        success_point_count (int):
            The number of points that were successfully
            written.
        errors (MutableSequence[google.cloud.monitoring_v3.types.CreateTimeSeriesSummary.Error]):
            The number of points that failed to be
            written. Order is not guaranteed.
    """

    class Error(proto.Message):
        r"""Detailed information about an error category.

        Attributes:
            status (google.rpc.status_pb2.Status):
                The status of the requested write operation.
            point_count (int):
                The number of points that couldn't be written because of
                ``status``.
        """

        status: status_pb2.Status = proto.Field(
            proto.MESSAGE,
            number=1,
            message=status_pb2.Status,
        )
        point_count: int = proto.Field(
            proto.INT32,
            number=2,
        )

    total_point_count: int = proto.Field(
        proto.INT32,
        number=1,
    )
    success_point_count: int = proto.Field(
        proto.INT32,
        number=2,
    )
    errors: MutableSequence[Error] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=Error,
    )


class QueryTimeSeriesRequest(proto.Message):
    r"""The ``QueryTimeSeries`` request. For information about the status of
    Monitoring Query Language (MQL), see the `MQL deprecation
    notice <https://cloud.google.com/stackdriver/docs/deprecations/mql>`__.

    Attributes:
        name (str):
            Required. The
            `project <https://cloud.google.com/monitoring/api/v3#project_name>`__
            on which to execute the request. The format is:

            ::

                projects/[PROJECT_ID_OR_NUMBER]
        query (str):
            Required. The query in the `Monitoring Query
            Language <https://cloud.google.com/monitoring/mql/reference>`__
            format. The default time zone is in UTC.
        page_size (int):
            A positive number that is the maximum number of
            time_series_data to return.
        page_token (str):
            If this field is not empty then it must contain the
            ``nextPageToken`` value returned by a previous call to this
            method. Using this field causes the method to return
            additional results from the previous method call.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: str = proto.Field(
        proto.STRING,
        number=7,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=9,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=10,
    )


class QueryTimeSeriesResponse(proto.Message):
    r"""The ``QueryTimeSeries`` response. For information about the status
    of Monitoring Query Language (MQL), see the `MQL deprecation
    notice <https://cloud.google.com/stackdriver/docs/deprecations/mql>`__.

    Attributes:
        time_series_descriptor (google.cloud.monitoring_v3.types.TimeSeriesDescriptor):
            The descriptor for the time series data.
        time_series_data (MutableSequence[google.cloud.monitoring_v3.types.TimeSeriesData]):
            The time series data.
        next_page_token (str):
            If there are more results than have been returned, then this
            field is set to a non-empty value. To see the additional
            results, use that value as ``page_token`` in the next call
            to this method.
        partial_errors (MutableSequence[google.rpc.status_pb2.Status]):
            Query execution errors that may have caused
            the time series data returned to be incomplete.
            The available data will be available in the
            response.
    """

    @property
    def raw_page(self):
        return self

    time_series_descriptor: gm_metric.TimeSeriesDescriptor = proto.Field(
        proto.MESSAGE,
        number=8,
        message=gm_metric.TimeSeriesDescriptor,
    )
    time_series_data: MutableSequence[gm_metric.TimeSeriesData] = proto.RepeatedField(
        proto.MESSAGE,
        number=9,
        message=gm_metric.TimeSeriesData,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=10,
    )
    partial_errors: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=11,
        message=status_pb2.Status,
    )


class QueryErrorList(proto.Message):
    r"""This is an error detail intended to be used with INVALID_ARGUMENT
    errors.

    Attributes:
        errors (MutableSequence[google.cloud.monitoring_v3.types.QueryError]):
            Errors in parsing the time series query
            language text. The number of errors in the
            response may be limited.
        error_summary (str):
            A summary of all the errors.
    """

    errors: MutableSequence[gm_metric.QueryError] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gm_metric.QueryError,
    )
    error_summary: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
