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

"""Factory for creating MetricTracer instances, facilitating metrics collection and tracing."""

from google.cloud.spanner_v1.metrics.metrics_tracer import MetricsTracer

from google.cloud.spanner_v1.metrics.constants import (
    METRIC_NAME_OPERATION_LATENCIES,
    MONITORED_RES_LABEL_KEY_PROJECT,
    METRIC_NAME_ATTEMPT_LATENCIES,
    METRIC_NAME_OPERATION_COUNT,
    METRIC_NAME_ATTEMPT_COUNT,
    MONITORED_RES_LABEL_KEY_INSTANCE,
    MONITORED_RES_LABEL_KEY_INSTANCE_CONFIG,
    MONITORED_RES_LABEL_KEY_LOCATION,
    MONITORED_RES_LABEL_KEY_CLIENT_HASH,
    METRIC_LABEL_KEY_CLIENT_UID,
    METRIC_LABEL_KEY_CLIENT_NAME,
    METRIC_LABEL_KEY_DATABASE,
    METRIC_LABEL_KEY_DIRECT_PATH_ENABLED,
    BUILT_IN_METRICS_METER_NAME,
    METRIC_NAME_GFE_LATENCY,
    METRIC_NAME_GFE_MISSING_HEADER_COUNT,
)

from typing import Dict

try:
    from opentelemetry.metrics import Counter, Histogram, get_meter_provider

    HAS_OPENTELEMETRY_INSTALLED = True
except ImportError:  # pragma: NO COVER
    HAS_OPENTELEMETRY_INSTALLED = False

from google.cloud.spanner_v1 import __version__


class MetricsTracerFactory:
    """Factory class for creating MetricTracer instances. This class facilitates the creation of MetricTracer objects, which are responsible for collecting and tracing metrics."""

    enabled: bool
    gfe_enabled: bool
    _instrument_attempt_latency: "Histogram"
    _instrument_attempt_counter: "Counter"
    _instrument_operation_latency: "Histogram"
    _instrument_operation_counter: "Counter"
    _instrument_gfe_latency: "Histogram"
    _instrument_gfe_missing_header_count: "Counter"
    _client_attributes: Dict[str, str]

    @property
    def instrument_attempt_latency(self) -> "Histogram":
        return self._instrument_attempt_latency

    @property
    def instrument_attempt_counter(self) -> "Counter":
        return self._instrument_attempt_counter

    @property
    def instrument_operation_latency(self) -> "Histogram":
        return self._instrument_operation_latency

    @property
    def instrument_operation_counter(self) -> "Counter":
        return self._instrument_operation_counter

    def __init__(self, enabled: bool, service_name: str):
        """Initialize a MetricsTracerFactory instance with the given parameters.

        This constructor initializes a MetricsTracerFactory instance with the provided service name, project, instance, instance configuration, location, client hash, client UID, client name, and database. It sets up the necessary metric instruments and client attributes for metrics tracing.

        Args:
            service_name (str): The name of the service for which metrics are being traced.
            project (str): The project ID for the monitored resource.
        """
        self.enabled = enabled
        self._create_metric_instruments(service_name)
        self._client_attributes = {}

    @property
    def client_attributes(self) -> Dict[str, str]:
        """Return a dictionary of client attributes used for metrics tracing.

        This property returns a dictionary containing client attributes such as project, instance,
        instance configuration, location, client hash, client UID, client name, and database.
        These attributes are used to provide context to the metrics being traced.

        Returns:
            dict[str, str]: A dictionary of client attributes.
        """
        return self._client_attributes

    def set_project(self, project: str) -> "MetricsTracerFactory":
        """Set the project attribute for metrics tracing.

        This method updates the client attributes dictionary with the provided project name.
        The project name is used to identify the project for which metrics are being traced
        and is passed to the created MetricsTracer.

        Args:
            project (str): The name of the project for metrics tracing.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[MONITORED_RES_LABEL_KEY_PROJECT] = project
        return self

    def set_instance(self, instance: str) -> "MetricsTracerFactory":
        """Set the instance attribute for metrics tracing.

        This method updates the client attributes dictionary with the provided instance name.
        The instance name is used to identify the instance for which metrics are being traced
        and is passed to the created MetricsTracer.

        Args:
            instance (str): The name of the instance for metrics tracing.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[MONITORED_RES_LABEL_KEY_INSTANCE] = instance
        return self

    def set_instance_config(self, instance_config: str) -> "MetricsTracerFactory":
        """Sets the instance configuration attribute for metrics tracing.

        This method updates the client attributes dictionary with the provided instance configuration.
        The instance configuration is used to identify the configuration of the instance for which
        metrics are being traced and is passed to the created MetricsTracer.

        Args:
            instance_config (str): The configuration of the instance for metrics tracing.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[
            MONITORED_RES_LABEL_KEY_INSTANCE_CONFIG
        ] = instance_config
        return self

    def set_location(self, location: str) -> "MetricsTracerFactory":
        """Set the location attribute for metrics tracing.

        This method updates the client attributes dictionary with the provided location.
        The location is used to identify the location for which metrics are being traced
        and is passed to the created MetricsTracer.

        Args:
            location (str): The location for metrics tracing.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[MONITORED_RES_LABEL_KEY_LOCATION] = location
        return self

    def set_client_hash(self, hash: str) -> "MetricsTracerFactory":
        """Set the client hash attribute for metrics tracing.

        This method updates the client attributes dictionary with the provided client hash.
        The client hash is used to identify the client for which metrics are being traced
        and is passed to the created MetricsTracer.

        Args:
            hash (str): The hash of the client for metrics tracing.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[MONITORED_RES_LABEL_KEY_CLIENT_HASH] = hash
        return self

    def set_client_uid(self, client_uid: str) -> "MetricsTracerFactory":
        """Set the client UID attribute for metrics tracing.

        This method updates the client attributes dictionary with the provided client UID.
        The client UID is used to identify the client for which metrics are being traced
        and is passed to the created MetricsTracer.

        Args:
            client_uid (str): The UID of the client for metrics tracing.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[METRIC_LABEL_KEY_CLIENT_UID] = client_uid
        return self

    def set_client_name(self, client_name: str) -> "MetricsTracerFactory":
        """Set the client name attribute for metrics tracing.

        This method updates the client attributes dictionary with the provided client name.
        The client name is used to identify the client for which metrics are being traced
        and is passed to the created MetricsTracer.

        Args:
            client_name (str): The name of the client for metrics tracing.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[METRIC_LABEL_KEY_CLIENT_NAME] = client_name
        return self

    def set_database(self, database: str) -> "MetricsTracerFactory":
        """Set the database attribute for metrics tracing.

        This method updates the client attributes dictionary with the provided database name.
        The database name is used to identify the database for which metrics are being traced
        and is passed to the created MetricsTracer.

        Args:
            database (str): The name of the database for metrics tracing.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[METRIC_LABEL_KEY_DATABASE] = database
        return self

    def enable_direct_path(self, enable: bool = False) -> "MetricsTracerFactory":
        """Enable or disable the direct path for metrics tracing.

        This method updates the client attributes dictionary with the provided enable status.
        The direct path enabled status is used to determine whether to use the direct path for metrics tracing
        and is passed to the created MetricsTracer.

        Args:
            enable (bool, optional): Whether to enable the direct path for metrics tracing. Defaults to False.

        Returns:
            MetricsTracerFactory: The current instance of MetricsTracerFactory to enable method chaining.
        """
        self._client_attributes[METRIC_LABEL_KEY_DIRECT_PATH_ENABLED] = enable
        return self

    def create_metrics_tracer(self) -> MetricsTracer:
        """
        Create and return a MetricsTracer instance with default settings and client attributes.

        This method initializes a MetricsTracer instance with default settings for metrics tracing,
        including metrics tracing enabled if OpenTelemetry is installed and the direct path disabled by default.
        It also sets the client attributes based on the factory's configuration.

        Returns:
            MetricsTracer: A MetricsTracer instance with default settings and client attributes.
        """
        if not HAS_OPENTELEMETRY_INSTALLED:
            return None

        metrics_tracer = MetricsTracer(
            enabled=self.enabled and HAS_OPENTELEMETRY_INSTALLED,
            instrument_attempt_latency=self._instrument_attempt_latency,
            instrument_attempt_counter=self._instrument_attempt_counter,
            instrument_operation_latency=self._instrument_operation_latency,
            instrument_operation_counter=self._instrument_operation_counter,
            client_attributes=self._client_attributes.copy(),
        )
        return metrics_tracer

    def _create_metric_instruments(self, service_name: str) -> None:
        """
        Creates and sets up metric instruments for the given service name.

        This method initializes and configures metric instruments for attempt latency, attempt counter,
        operation latency, and operation counter. These instruments are used to measure and track
        metrics related to attempts and operations within the service.

        Args:
            service_name (str): The name of the service for which metric instruments are being created.
        """
        if not HAS_OPENTELEMETRY_INSTALLED:  # pragma: NO COVER
            return

        meter_provider = get_meter_provider()
        meter = meter_provider.get_meter(
            name=BUILT_IN_METRICS_METER_NAME, version=__version__
        )

        self._instrument_attempt_latency = meter.create_histogram(
            name=METRIC_NAME_ATTEMPT_LATENCIES,
            unit="ms",
            description="Time an individual attempt took.",
        )

        self._instrument_attempt_counter = meter.create_counter(
            name=METRIC_NAME_ATTEMPT_COUNT,
            unit="1",
            description="Number of attempts.",
        )

        self._instrument_operation_latency = meter.create_histogram(
            name=METRIC_NAME_OPERATION_LATENCIES,
            unit="ms",
            description="Total time until final operation success or failure, including retries and backoff.",
        )

        self._instrument_operation_counter = meter.create_counter(
            name=METRIC_NAME_OPERATION_COUNT,
            unit="1",
            description="Number of operations.",
        )

        self._instrument_gfe_latency = meter.create_histogram(
            name=METRIC_NAME_GFE_LATENCY,
            unit="ms",
            description="GFE Latency.",
        )

        self._instrument_gfe_missing_header_count = meter.create_counter(
            name=METRIC_NAME_GFE_MISSING_HEADER_COUNT,
            unit="1",
            description="GFE missing header count.",
        )
