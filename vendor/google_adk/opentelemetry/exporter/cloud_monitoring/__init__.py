# Copyright 2021 Google LLC
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

import logging
import math
import random
from dataclasses import replace
from time import time_ns
from typing import Dict, List, NoReturn, Optional, Set, Union

import google.auth
from google.api.distribution_pb2 import (  # pylint: disable=no-name-in-module
    Distribution,
)
from google.api.label_pb2 import (  # pylint: disable=no-name-in-module
    LabelDescriptor,
)

# pylint: disable=no-name-in-module
from google.api.metric_pb2 import Metric as GMetric
from google.api.metric_pb2 import (  # pylint: disable=no-name-in-module
    MetricDescriptor,
)
from google.api.monitored_resource_pb2 import (  # pylint: disable=no-name-in-module
    MonitoredResource,
)
from google.cloud.monitoring_v3 import (
    CreateMetricDescriptorRequest,
    CreateTimeSeriesRequest,
    MetricServiceClient,
    Point,
    TimeInterval,
    TimeSeries,
    TypedValue,
)
from google.cloud.monitoring_v3.services.metric_service.transports.grpc import (
    MetricServiceGrpcTransport,
)

# pylint: disable=no-name-in-module
from google.protobuf.timestamp_pb2 import Timestamp
from opentelemetry.exporter.cloud_monitoring.version import __version__
from opentelemetry.resourcedetector.gcp_resource_detector._mapping import (
    get_monitored_resource,
)
from opentelemetry.sdk import version as opentelemetry_sdk_version
from opentelemetry.sdk.metrics.export import (
    ExponentialHistogram,
    ExponentialHistogramDataPoint,
    Gauge,
    Histogram,
    HistogramDataPoint,
    Metric,
    MetricExporter,
    MetricExportResult,
    MetricsData,
    NumberDataPoint,
    Sum,
)

logger = logging.getLogger(__name__)
MAX_BATCH_WRITE = 200
WRITE_INTERVAL = 10
UNIQUE_IDENTIFIER_KEY = "opentelemetry_id"
NANOS_PER_SECOND = 10**9

_OTEL_SDK_VERSION = opentelemetry_sdk_version.__version__
_USER_AGENT = f"opentelemetry-python {_OTEL_SDK_VERSION}; google-cloud-metric-exporter {__version__}"

# Set user-agent metadata, see https://github.com/grpc/grpc/issues/23644 and default options
# from
# https://github.com/googleapis/python-monitoring/blob/v2.11.3/google/cloud/monitoring_v3/services/metric_service/transports/grpc.py#L175-L178
_OPTIONS = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    ("grpc.primary_user_agent", _USER_AGENT),
]


# pylint is unable to resolve members of protobuf objects
# pylint: disable=no-member
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
class CloudMonitoringMetricsExporter(MetricExporter):
    """Implementation of Metrics Exporter to Google Cloud Monitoring

        You can manually pass in project_id and client, or else the
        Exporter will take that information from Application Default
        Credentials.

    Args:
        project_id: project id of your Google Cloud project.
        client: Client to upload metrics to Google Cloud Monitoring.
        add_unique_identifier: Add an identifier to each exporter metric. This
            must be used when there exist two (or more) exporters that may
            export to the same metric name within WRITE_INTERVAL seconds of
            each other.
        prefix: the prefix of the metric. It is "workload.googleapis.com" by
            default if not specified.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        client: Optional[MetricServiceClient] = None,
        add_unique_identifier: bool = False,
        prefix: Optional[str] = "workload.googleapis.com",
    ):
        # Default preferred_temporality is all CUMULATIVE so need to customize
        super().__init__()

        self.client = client or MetricServiceClient(
            transport=MetricServiceGrpcTransport(
                channel=MetricServiceGrpcTransport.create_channel(
                    options=_OPTIONS,
                )
            )
        )
        self.project_id: str
        if not project_id:
            _, default_project_id = google.auth.default()
            self.project_id = str(default_project_id)
        else:
            self.project_id = project_id
        self.project_name = self.client.common_project_path(self.project_id)
        self._metric_descriptors: Dict[str, MetricDescriptor] = {}
        self.unique_identifier = None
        if add_unique_identifier:
            self.unique_identifier = "{:08x}".format(
                random.randint(0, 16**8)
            )

        (
            self._exporter_start_time_seconds,
            self._exporter_start_time_nanos,
        ) = divmod(time_ns(), NANOS_PER_SECOND)
        self._prefix = prefix

    def _batch_write(self, series: List[TimeSeries]) -> None:
        """Cloud Monitoring allows writing up to 200 time series at once

        :param series: ProtoBuf TimeSeries
        :return:
        """
        write_ind = 0
        while write_ind < len(series):
            self.client.create_time_series(
                CreateTimeSeriesRequest(
                    name=self.project_name,
                    time_series=series[
                        write_ind : write_ind + MAX_BATCH_WRITE
                    ],
                ),
            )
            write_ind += MAX_BATCH_WRITE

    def _get_metric_descriptor(
        self, metric: Metric
    ) -> Optional[MetricDescriptor]:
        """We can map Metric to MetricDescriptor using Metric.name or
        MetricDescriptor.type. We create the MetricDescriptor if it doesn't
        exist already and cache it. Note that recreating MetricDescriptors is
        a no-op if it already exists.

        :param record:
        :return:
        """
        descriptor_type = f"{self._prefix}/{metric.name}"
        if descriptor_type in self._metric_descriptors:
            return self._metric_descriptors[descriptor_type]

        descriptor = MetricDescriptor(
            type=descriptor_type,
            display_name=metric.name,
            description=metric.description or "",
            unit=metric.unit or "",
        )
        seen_keys: Set[str] = set()
        for data_point in metric.data.data_points:
            for key in data_point.attributes or {}:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                descriptor.labels.append(
                    LabelDescriptor(key=_normalize_label_key(key))
                )

        if self.unique_identifier:
            descriptor.labels.append(
                LabelDescriptor(key=UNIQUE_IDENTIFIER_KEY)
            )

        data = metric.data
        if isinstance(data, Sum):
            descriptor.metric_kind = (
                MetricDescriptor.MetricKind.CUMULATIVE
                if data.is_monotonic
                else MetricDescriptor.MetricKind.GAUGE
            )
        elif isinstance(data, Gauge):
            descriptor.metric_kind = MetricDescriptor.MetricKind.GAUGE
        elif isinstance(data, Histogram):
            descriptor.metric_kind = MetricDescriptor.MetricKind.CUMULATIVE
        elif isinstance(data, ExponentialHistogram):
            descriptor.metric_kind = MetricDescriptor.MetricKind.CUMULATIVE
        else:
            # Exhaustive check
            _: NoReturn = data
            logger.warning(
                "Unsupported metric data type %s, ignoring it",
                type(data).__name__,
            )
            return None

        first_point = data.data_points[0] if len(data.data_points) else None
        if isinstance(first_point, NumberDataPoint):
            descriptor.value_type = (
                MetricDescriptor.ValueType.INT64
                if isinstance(first_point.value, int)
                else MetricDescriptor.ValueType.DOUBLE
            )
        elif isinstance(first_point, HistogramDataPoint):
            descriptor.value_type = MetricDescriptor.ValueType.DISTRIBUTION
        elif isinstance(first_point, ExponentialHistogramDataPoint):
            descriptor.value_type = MetricDescriptor.ValueType.DISTRIBUTION
        elif first_point is None:
            pass
        else:
            # Exhaustive check
            _ = first_point
            logger.warning(
                "Unsupported metric value type %s, ignoring it",
                type(first_point).__name__,
            )

        try:
            response_descriptor = self.client.create_metric_descriptor(
                CreateMetricDescriptorRequest(
                    name=self.project_name, metric_descriptor=descriptor
                )
            )
        # pylint: disable=broad-except
        except Exception as ex:
            logger.error(
                "Failed to create metric descriptor %s",
                descriptor,
                exc_info=ex,
            )
            return None
        self._metric_descriptors[descriptor_type] = response_descriptor
        return descriptor

    @staticmethod
    def _to_point(
        kind: "MetricDescriptor.MetricKind.V",
        data_point: Union[
            NumberDataPoint, HistogramDataPoint, ExponentialHistogramDataPoint
        ],
    ) -> Point:
        if isinstance(data_point, HistogramDataPoint):
            mean = (
                data_point.sum / data_point.count if data_point.count else 0.0
            )
            point_value = TypedValue(
                distribution_value=Distribution(
                    count=data_point.count,
                    mean=mean,
                    bucket_counts=data_point.bucket_counts,
                    bucket_options=Distribution.BucketOptions(
                        explicit_buckets=Distribution.BucketOptions.Explicit(
                            bounds=data_point.explicit_bounds,
                        )
                    ),
                )
            )
        elif isinstance(data_point, ExponentialHistogramDataPoint):
            # Adapted from https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/blob/v1.8.0/exporter/collector/metrics.go#L582

            # Calculate underflow bucket (zero count + negative buckets)
            underflow = data_point.zero_count
            if data_point.negative.bucket_counts:
                underflow += sum(data_point.negative.bucket_counts)

            # Create bucket counts array: [underflow, positive_buckets..., overflow=0]
            bucket_counts = [underflow]
            if data_point.positive.bucket_counts:
                bucket_counts.extend(data_point.positive.bucket_counts)
            bucket_counts.append(0)  # overflow bucket is always empty

            # Determine bucket options
            if not data_point.positive.bucket_counts:
                # If no positive buckets, use explicit buckets with bounds=[0]
                bucket_options = Distribution.BucketOptions(
                    explicit_buckets=Distribution.BucketOptions.Explicit(
                        bounds=[0.0],
                    )
                )
            else:
                # Use exponential bucket options
                # growth_factor = 2^(2^(-scale))
                growth_factor = math.pow(2, math.pow(2, -data_point.scale))
                # scale = growth_factor^(positive_bucket_offset)
                scale = math.pow(growth_factor, data_point.positive.offset)
                num_finite_buckets = len(bucket_counts) - 2

                bucket_options = Distribution.BucketOptions(
                    exponential_buckets=Distribution.BucketOptions.Exponential(
                        num_finite_buckets=num_finite_buckets,
                        growth_factor=growth_factor,
                        scale=scale,
                    )
                )

            mean = (
                data_point.sum / data_point.count if data_point.count else 0.0
            )
            point_value = TypedValue(
                distribution_value=Distribution(
                    count=data_point.count,
                    mean=mean,
                    bucket_counts=bucket_counts,
                    bucket_options=bucket_options,
                )
            )
        else:
            if isinstance(data_point.value, int):
                point_value = TypedValue(int64_value=data_point.value)
            else:
                point_value = TypedValue(double_value=data_point.value)

            if kind is MetricDescriptor.MetricKind.CUMULATIVE:
                pass

        # DELTA case should never happen but adding it to be future proof
        if (
            kind is MetricDescriptor.MetricKind.CUMULATIVE
            or kind is MetricDescriptor.MetricKind.DELTA
        ):
            interval = TimeInterval(
                start_time=_timestamp_from_nanos(
                    data_point.start_time_unix_nano
                ),
                end_time=_timestamp_from_nanos(data_point.time_unix_nano),
            )
        else:
            interval = TimeInterval(
                end_time=_timestamp_from_nanos(data_point.time_unix_nano),
            )
        return Point(interval=interval, value=point_value)

    def export(
        self,
        metrics_data: MetricsData,
        # TODO(aabmass): pass timeout to api calls
        timeout_millis: float = 10_000,
        **kwargs,
    ) -> MetricExportResult:
        all_series = []

        for resource_metric in metrics_data.resource_metrics:
            monitored_resource_data = get_monitored_resource(
                resource_metric.resource
            )
            # convert it to proto
            monitored_resource = (
                MonitoredResource(
                    type=monitored_resource_data.type,
                    labels=monitored_resource_data.labels,
                )
                if monitored_resource_data
                else None
            )

            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    # Convert all data_points to Sequences, see
                    # https://github.com/open-telemetry/opentelemetry-python/issues/3021.
                    # TODO(aabmass): remove once the issue is fixed upstream
                    metric = replace(
                        metric,
                        data=replace(
                            metric.data,
                            data_points=tuple(metric.data.data_points),
                        ),
                    )

                    descriptor = self._get_metric_descriptor(metric)
                    if not descriptor:
                        continue

                    for data_point in metric.data.data_points:
                        labels = {
                            _normalize_label_key(key): str(value)
                            for key, value in (
                                data_point.attributes or {}
                            ).items()
                        }
                        if self.unique_identifier:
                            labels[
                                UNIQUE_IDENTIFIER_KEY
                            ] = self.unique_identifier
                        point = self._to_point(
                            descriptor.metric_kind, data_point
                        )
                        series = TimeSeries(
                            resource=monitored_resource,
                            metric_kind=descriptor.metric_kind,
                            points=[point],
                            metric=GMetric(
                                type=descriptor.type,
                                labels=labels,
                            ),
                            unit=descriptor.unit,
                        )
                        all_series.append(series)

        try:
            self._batch_write(all_series)
        # pylint: disable=broad-except
        except Exception as ex:
            logger.error(
                "Error while writing to Cloud Monitoring", exc_info=ex
            )
            return MetricExportResult.FAILURE
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
        pass


def _timestamp_from_nanos(nanos: int) -> Timestamp:
    ts = Timestamp()
    ts.FromNanoseconds(nanos)
    return ts


def _normalize_label_key(key: str) -> str:
    """Makes the key into a valid GCM label key

    See reference impl
    https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/blob/e955c204f4f2bfdc92ff0ad52786232b975efcc2/exporter/metric/metric.go#L595-L604
    """
    sanitized = "".join(
        c if c.isalpha() or c.isnumeric() else "_" for c in key
    )
    if sanitized[0].isdigit():
        sanitized = "key_" + sanitized
    return sanitized
