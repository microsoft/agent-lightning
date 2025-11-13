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


from .constants import (
    BUILT_IN_METRICS_METER_NAME,
    NATIVE_METRICS_PREFIX,
    SPANNER_RESOURCE_TYPE,
    MONITORED_RESOURCE_LABELS,
    METRIC_LABELS,
    METRIC_NAMES,
)

import logging
from typing import Optional, List, Union, NoReturn, Tuple, Dict

import google.auth
from google.auth import credentials as ga_credentials
from google.api.distribution_pb2 import (  # pylint: disable=no-name-in-module
    Distribution,
)

# pylint: disable=no-name-in-module
from google.api.metric_pb2 import (  # pylint: disable=no-name-in-module
    Metric as GMetric,
    MetricDescriptor,
)
from google.api.monitored_resource_pb2 import (  # pylint: disable=no-name-in-module
    MonitoredResource,
)

# pylint: disable=no-name-in-module
from google.protobuf.timestamp_pb2 import Timestamp
from google.cloud.spanner_v1.gapic_version import __version__

try:
    from opentelemetry.sdk.metrics.export import (
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
    from opentelemetry.sdk.resources import Resource
    from google.cloud.monitoring_v3.services.metric_service.transports.grpc import (
        MetricServiceGrpcTransport,
    )
    from google.cloud.monitoring_v3 import (
        CreateTimeSeriesRequest,
        MetricServiceClient,
        Point,
        TimeInterval,
        TimeSeries,
        TypedValue,
    )

    HAS_OPENTELEMETRY_INSTALLED = True
except ImportError:  # pragma: NO COVER
    HAS_OPENTELEMETRY_INSTALLED = False
    MetricExporter = object

logger = logging.getLogger(__name__)
MAX_BATCH_WRITE = 200
MILLIS_PER_SECOND = 1000

_USER_AGENT = f"python-spanner; google-cloud-service-metric-exporter {__version__}"

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
    """Implementation of Metrics Exporter to Google Cloud Monitoring.

        You can manually pass in project_id and client, or else the
        Exporter will take that information from Application Default
        Credentials.

    Args:
        project_id: project id of your Google Cloud project.
        client: Client to upload metrics to Google Cloud Monitoring.
    """

    # Based on the cloud_monitoring exporter found here: https://github.com/GoogleCloudPlatform/opentelemetry-operations-python/blob/main/opentelemetry-exporter-gcp-monitoring/src/opentelemetry/exporter/cloud_monitoring/__init__.py

    def __init__(
        self,
        project_id: Optional[str] = None,
        client: Optional["MetricServiceClient"] = None,
        credentials: Optional[ga_credentials.Credentials] = None,
    ):
        """Initialize a custom exporter to send metrics for the Spanner Service Metrics."""
        # Default preferred_temporality is all CUMULATIVE so need to customize
        super().__init__()

        # Create a new GRPC Client for Google Cloud Monitoring if not provided
        self.client = client or MetricServiceClient(
            transport=MetricServiceGrpcTransport(
                channel=MetricServiceGrpcTransport.create_channel(
                    options=_OPTIONS,
                    credentials=credentials,
                )
            )
        )

        # Set project information
        self.project_id: str
        if not project_id:
            _, default_project_id = google.auth.default()
            self.project_id = str(default_project_id)
        else:
            self.project_id = project_id
        self.project_name = self.client.common_project_path(self.project_id)

    def _batch_write(self, series: List["TimeSeries"], timeout_millis: float) -> None:
        """Cloud Monitoring allows writing up to 200 time series at once.

        :param series: ProtoBuf TimeSeries
        :return:
        """
        write_ind = 0
        timeout = timeout_millis / MILLIS_PER_SECOND
        while write_ind < len(series):
            request = CreateTimeSeriesRequest(
                name=self.project_name,
                time_series=series[write_ind : write_ind + MAX_BATCH_WRITE],
            )

            self.client.create_service_time_series(
                request=request,
                timeout=timeout,
            )
            write_ind += MAX_BATCH_WRITE

    @staticmethod
    def _resource_to_monitored_resource_pb(
        resource: "Resource", labels: Dict[str, str]
    ) -> "MonitoredResource":
        """
        Convert the resource to a Google Cloud Monitoring monitored resource.

        :param resource: OpenTelemetry resource
        :param labels: labels to add to the monitored resource
        :return: Google Cloud Monitoring monitored resource
        """
        monitored_resource = MonitoredResource(
            type=SPANNER_RESOURCE_TYPE,
            labels=labels,
        )
        return monitored_resource

    @staticmethod
    def _to_metric_kind(metric: "Metric") -> MetricDescriptor.MetricKind:
        """
        Convert the metric to a Google Cloud Monitoring metric kind.

        :param metric: OpenTelemetry metric
        :return: Google Cloud Monitoring metric kind
        """
        data = metric.data
        if isinstance(data, Sum):
            if data.is_monotonic:
                return MetricDescriptor.MetricKind.CUMULATIVE
            else:
                return MetricDescriptor.MetricKind.GAUGE
        elif isinstance(data, Gauge):
            return MetricDescriptor.MetricKind.GAUGE
        elif isinstance(data, Histogram):
            return MetricDescriptor.MetricKind.CUMULATIVE
        else:
            # Exhaustive check
            _: NoReturn = data
            logger.warning(
                "Unsupported metric data type %s, ignoring it",
                type(data).__name__,
            )
            return None

    @staticmethod
    def _extract_metric_labels(
        data_point: Union["NumberDataPoint", "HistogramDataPoint"]
    ) -> Tuple[dict, dict]:
        """
        Extract the metric labels from the data point.

        :param data_point: OpenTelemetry data point
        :return: tuple of metric labels and monitored resource labels
        """
        metric_labels = {}
        monitored_resource_labels = {}
        for key, value in (data_point.attributes or {}).items():
            normalized_key = _normalize_label_key(key)
            val = str(value)
            if key in METRIC_LABELS:
                metric_labels[normalized_key] = val
            if key in MONITORED_RESOURCE_LABELS:
                monitored_resource_labels[normalized_key] = val
        return metric_labels, monitored_resource_labels

    # Unchanged from https://github.com/GoogleCloudPlatform/opentelemetry-operations-python/blob/main/opentelemetry-exporter-gcp-monitoring/src/opentelemetry/exporter/cloud_monitoring/__init__.py
    @staticmethod
    def _to_point(
        kind: "MetricDescriptor.MetricKind.V",
        data_point: Union["NumberDataPoint", "HistogramDataPoint"],
    ) -> "Point":
        # Create a Google Cloud Monitoring data point value based on the OpenTelemetry metric data point type
        ## For histograms, we need to calculate the mean and bucket counts
        if isinstance(data_point, HistogramDataPoint):
            mean = data_point.sum / data_point.count if data_point.count else 0.0
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
        else:
            # For other metric types, we can use the data point value directly
            if isinstance(data_point.value, int):
                point_value = TypedValue(int64_value=data_point.value)
            else:
                point_value = TypedValue(double_value=data_point.value)

        # DELTA case should never happen but adding it to be future proof
        if (
            kind is MetricDescriptor.MetricKind.CUMULATIVE
            or kind is MetricDescriptor.MetricKind.DELTA
        ):
            # Create a Google Cloud Monitoring time interval from the OpenTelemetry data point timestamps
            interval = TimeInterval(
                start_time=_timestamp_from_nanos(data_point.start_time_unix_nano),
                end_time=_timestamp_from_nanos(data_point.time_unix_nano),
            )
        else:
            # For non time ranged metrics, we only need the end time
            interval = TimeInterval(
                end_time=_timestamp_from_nanos(data_point.time_unix_nano),
            )
        return Point(interval=interval, value=point_value)

    @staticmethod
    def _data_point_to_timeseries_pb(
        data_point,
        metric,
        monitored_resource,
        labels,
    ) -> "TimeSeries":
        """
        Convert the data point to a Google Cloud Monitoring time series.

        :param data_point: OpenTelemetry data point
        :param metric: OpenTelemetry metric
        :param monitored_resource: Google Cloud Monitoring monitored resource
        :param labels: metric labels
        :return: Google Cloud Monitoring time series
        """
        if metric.name not in METRIC_NAMES:
            return None

        kind = CloudMonitoringMetricsExporter._to_metric_kind(metric)
        point = CloudMonitoringMetricsExporter._to_point(kind, data_point)
        type = f"{NATIVE_METRICS_PREFIX}/{metric.name}"
        series = TimeSeries(
            resource=monitored_resource,
            metric_kind=kind,
            points=[point],
            metric=GMetric(type=type, labels=labels),
            unit=metric.unit or "",
        )
        return series

    @staticmethod
    def _resource_metrics_to_timeseries_pb(
        metrics_data: "MetricsData",
    ) -> List["TimeSeries"]:
        """
        Convert the metrics data to a list of Google Cloud Monitoring time series.

        :param metrics_data: OpenTelemetry metrics data
        :return: list of Google Cloud Monitoring time series
        """
        timeseries_list = []
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                # Filter for spanner builtin metrics
                if scope_metric.scope.name != BUILT_IN_METRICS_METER_NAME:
                    continue

                for metric in scope_metric.metrics:
                    for data_point in metric.data.data_points:
                        (
                            metric_labels,
                            monitored_resource_labels,
                        ) = CloudMonitoringMetricsExporter._extract_metric_labels(
                            data_point
                        )
                        monitored_resource = CloudMonitoringMetricsExporter._resource_to_monitored_resource_pb(
                            resource_metric.resource, monitored_resource_labels
                        )
                        timeseries = (
                            CloudMonitoringMetricsExporter._data_point_to_timeseries_pb(
                                data_point, metric, monitored_resource, metric_labels
                            )
                        )
                        if timeseries is not None:
                            timeseries_list.append(timeseries)

        return timeseries_list

    def export(
        self,
        metrics_data: "MetricsData",
        timeout_millis: float = 10_000,
        **kwargs,
    ) -> "MetricExportResult":
        """
        Export the metrics data to Google Cloud Monitoring.

        :param metrics_data: OpenTelemetry metrics data
        :param timeout_millis: timeout in milliseconds
        :return: MetricExportResult
        """
        if not HAS_OPENTELEMETRY_INSTALLED:
            logger.warning("Metric exporter called without dependencies installed.")
            return False
        time_series_list = self._resource_metrics_to_timeseries_pb(metrics_data)
        self._batch_write(time_series_list, timeout_millis)
        return True

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Not implemented."""
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
        """Safely shuts down the exporter and closes all opened GRPC channels."""
        self.client.transport.close()


def _timestamp_from_nanos(nanos: int) -> Timestamp:
    ts = Timestamp()
    ts.FromNanoseconds(nanos)
    return ts


def _normalize_label_key(key: str) -> str:
    """Make the key into a valid Google Cloud Monitoring label key.

    See reference impl
    https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/blob/e955c204f4f2bfdc92ff0ad52786232b975efcc2/exporter/metric/metric.go#L595-L604
    """
    sanitized = "".join(c if c.isalpha() or c.isnumeric() else "_" for c in key)
    if sanitized[0].isdigit():
        sanitized = "key_" + sanitized
    return sanitized
