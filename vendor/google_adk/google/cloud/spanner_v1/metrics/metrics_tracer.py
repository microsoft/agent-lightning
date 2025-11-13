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

"""
This module contains the MetricTracer class and its related helper classes.

The MetricTracer class is responsible for collecting and tracing metrics,
while the helper classes provide additional functionality and context for the metrics being traced.
"""

from datetime import datetime
from typing import Dict
from grpc import StatusCode
from .constants import (
    METRIC_LABEL_KEY_CLIENT_NAME,
    METRIC_LABEL_KEY_CLIENT_UID,
    METRIC_LABEL_KEY_DATABASE,
    METRIC_LABEL_KEY_DIRECT_PATH_ENABLED,
    METRIC_LABEL_KEY_METHOD,
    METRIC_LABEL_KEY_STATUS,
    MONITORED_RES_LABEL_KEY_CLIENT_HASH,
    MONITORED_RES_LABEL_KEY_INSTANCE,
    MONITORED_RES_LABEL_KEY_INSTANCE_CONFIG,
    MONITORED_RES_LABEL_KEY_LOCATION,
    MONITORED_RES_LABEL_KEY_PROJECT,
)

try:
    from opentelemetry.metrics import Counter, Histogram

    HAS_OPENTELEMETRY_INSTALLED = True
except ImportError:  # pragma: NO COVER
    HAS_OPENTELEMETRY_INSTALLED = False


class MetricAttemptTracer:
    """
    This class is designed to hold information related to a metric attempt.

    It captures the start time of the attempt, whether the direct path was used, and the status of the attempt.
    """

    _start_time: datetime
    direct_path_used: bool
    status: str

    def __init__(self) -> None:
        """
        Initialize a MetricAttemptTracer instance with default values.

        This constructor sets the start time of the metric attempt to the current datetime, initializes the status as an empty string, and sets direct path used flag to False by default.
        """
        self._start_time = datetime.now()
        self.status = ""
        self.direct_path_used = False

    @property
    def start_time(self):
        """Getter method for the start_time property.

        This method returns the start time of the metric attempt.

        Returns:
            datetime: The start time of the metric attempt.
        """
        return self._start_time


class MetricOpTracer:
    """
    This class is designed to store and manage information related to metric operations.
    It captures the method name, start time, attempt count, current attempt, status, and direct path enabled status of a metric operation.
    """

    _attempt_count: int
    _start_time: datetime
    _current_attempt: MetricAttemptTracer
    status: str

    def __init__(self, is_direct_path_enabled: bool = False):
        """
        Initialize a MetricOpTracer instance with the given parameters.

        This constructor sets up a MetricOpTracer instance with the provided instrumentations for attempt latency,
        attempt counter, operation latency and operation counter.

        Args:
            instrument_attempt_latency (Histogram): The instrumentation for measuring attempt latency.
            instrument_attempt_counter (Counter): The instrumentation for counting attempts.
            instrument_operation_latency (Histogram): The instrumentation for measuring operation latency.
            instrument_operation_counter (Counter): The instrumentation for counting operations.
        """
        self._attempt_count = 0
        self._start_time = datetime.now()
        self._current_attempt = None
        self.status = ""

    @property
    def attempt_count(self):
        """
        Getter method for the attempt_count property.

        This method returns the current count of attempts made for the metric operation.

        Returns:
            int: The current count of attempts.
        """
        return self._attempt_count

    @property
    def current_attempt(self):
        """
        Getter method for the current_attempt property.

        This method returns the current MetricAttemptTracer instance associated with the metric operation.

        Returns:
            MetricAttemptTracer: The current MetricAttemptTracer instance.
        """
        return self._current_attempt

    @property
    def start_time(self):
        """
        Getter method for the start_time property.

        This method returns the start time of the metric operation.

        Returns:
            datetime: The start time of the metric operation.
        """
        return self._start_time

    def increment_attempt_count(self):
        """
        Increments the attempt count by 1.

        This method updates the attempt count by incrementing it by 1, indicating a new attempt has been made.
        """
        self._attempt_count += 1

    def start(self):
        """
        Set the start time of the metric operation to the current time.

        This method updates the start time of the metric operation to the current time, indicating the operation has started.
        """
        self._start_time = datetime.now()

    def new_attempt(self):
        """
        Initialize a new MetricAttemptTracer instance for the current metric operation.

        This method sets up a new MetricAttemptTracer instance, indicating a new attempt is being made within the metric operation.
        """
        self._current_attempt = MetricAttemptTracer()


class MetricsTracer:
    """
    This class computes generic metrics that can be observed in the lifecycle of an RPC operation.

    The responsibility of recording metrics should delegate to MetricsRecorder, hence this
    class should not have any knowledge about the observability framework used for metrics recording.
    """

    _client_attributes: Dict[str, str]
    _instrument_attempt_counter: "Counter"
    _instrument_attempt_latency: "Histogram"
    _instrument_operation_counter: "Counter"
    _instrument_operation_latency: "Histogram"
    _instrument_gfe_latency: "Histogram"
    _instrument_gfe_missing_header_count: "Counter"
    current_op: MetricOpTracer
    enabled: bool
    gfe_enabled: bool
    method: str

    def __init__(
        self,
        enabled: bool,
        instrument_attempt_latency: "Histogram",
        instrument_attempt_counter: "Counter",
        instrument_operation_latency: "Histogram",
        instrument_operation_counter: "Counter",
        client_attributes: Dict[str, str],
        gfe_enabled: bool = False,
    ):
        """
        Initialize a MetricsTracer instance with the given parameters.

        This constructor sets up a MetricsTracer instance with the specified parameters, including the enabled status,
        instruments for measuring and counting attempt and operation metrics, and client attributes. It prepares the
        infrastructure needed for recording metrics related to RPC operations.

        Args:
            enabled (bool): Indicates if metrics tracing is enabled.
            instrument_attempt_latency (Histogram): Instrument for measuring attempt latency.
            instrument_attempt_counter (Counter): Instrument for counting attempts.
            instrument_operation_latency (Histogram): Instrument for measuring operation latency.
            instrument_operation_counter (Counter): Instrument for counting operations.
            client_attributes (Dict[str, str]): Dictionary of client attributes used for metrics tracing.
            gfe_enabled (bool, optional): Indicates if GFE metrics are enabled. Defaults to False.
        """
        self.current_op = MetricOpTracer()
        self._client_attributes = client_attributes
        self._instrument_attempt_latency = instrument_attempt_latency
        self._instrument_attempt_counter = instrument_attempt_counter
        self._instrument_operation_latency = instrument_operation_latency
        self._instrument_operation_counter = instrument_operation_counter
        self.enabled = enabled
        self.gfe_enabled = gfe_enabled

    @staticmethod
    def _get_ms_time_diff(start: datetime, end: datetime) -> float:
        """
        Calculate the time difference in milliseconds between two datetime objects.

        This method calculates the time difference between two datetime objects and returns the result in milliseconds.
        This is useful for measuring the duration of operations or attempts for metrics tracing.
        Note: total_seconds() returns a float value of seconds.

        Args:
            start (datetime): The start datetime.
            end (datetime): The end datetime.

        Returns:
            float: The time difference in milliseconds.
        """
        time_delta = end - start
        return time_delta.total_seconds() * 1000

    @property
    def client_attributes(self) -> Dict[str, str]:
        """
        Return a dictionary of client attributes used for metrics tracing.

        This property returns a dictionary containing client attributes such as project, instance,
        instance configuration, location, client hash, client UID, client name, and database.
        These attributes are used to provide context to the metrics being traced.

        Returns:
            dict[str, str]: A dictionary of client attributes.
        """
        return self._client_attributes

    @property
    def instrument_attempt_counter(self) -> "Counter":
        """
        Return the instrument for counting attempts.

        This property returns the Counter instrument used to count the number of attempts made during RPC operations.
        This metric is useful for tracking the frequency of attempts and can help identify patterns or issues in the operation flow.

        Returns:
            Counter: The instrument for counting attempts.
        """
        return self._instrument_attempt_counter

    @property
    def instrument_attempt_latency(self) -> "Histogram":
        """
        Return the instrument for measuring attempt latency.

        This property returns the Histogram instrument used to measure the latency of individual attempts.
        This metric is useful for tracking the performance of attempts and can help identify bottlenecks or issues in the operation flow.

        Returns:
            Histogram: The instrument for measuring attempt latency.
        """
        return self._instrument_attempt_latency

    @property
    def instrument_operation_counter(self) -> "Counter":
        """
        Return the instrument for counting operations.

        This property returns the Counter instrument used to count the number of operations made during RPC operations.
        This metric is useful for tracking the frequency of operations and can help identify patterns or issues in the operation flow.

        Returns:
            Counter: The instrument for counting operations.
        """
        return self._instrument_operation_counter

    @property
    def instrument_operation_latency(self) -> "Histogram":
        """
        Return the instrument for measuring operation latency.

        This property returns the Histogram instrument used to measure the latency of operations.
        This metric is useful for tracking the performance of operations and can help identify bottlenecks or issues in the operation flow.

        Returns:
            Histogram: The instrument for measuring operation latency.
        """
        return self._instrument_operation_latency

    def record_attempt_start(self) -> None:
        """
        Record the start of a new attempt within the current operation.

        This method increments the attempt count for the current operation and marks the start of a new attempt.
        It is used to track the number of attempts made during an operation and to identify the start of each attempt for metrics and tracing purposes.
        """
        self.current_op.increment_attempt_count()
        self.current_op.new_attempt()

    def record_attempt_completion(self, status: str = StatusCode.OK.name) -> None:
        """
        Record the completion of an attempt within the current operation.

        This method updates the status of the current attempt to indicate its completion and records the latency of the attempt.
        It calculates the elapsed time since the attempt started and uses this value to record the attempt latency metric.
        This metric is useful for tracking the performance of individual attempts and can help identify bottlenecks or issues in the operation flow.

        If metrics tracing is not enabled, this method does not perform any operations.
        """
        if not self.enabled or not HAS_OPENTELEMETRY_INSTALLED:
            return
        self.current_op.current_attempt.status = status

        # Build Attributes
        attempt_attributes = self._create_attempt_otel_attributes()

        # Calculate elapsed time
        attempt_latency_ms = self._get_ms_time_diff(
            start=self.current_op.current_attempt.start_time, end=datetime.now()
        )

        # Record attempt latency
        self.instrument_attempt_latency.record(
            amount=attempt_latency_ms, attributes=attempt_attributes
        )

    def record_operation_start(self) -> None:
        """
        Record the start of a new operation.

        This method marks the beginning of a new operation and initializes the operation's metrics tracking.
        It is used to track the start time of an operation, which is essential for calculating operation latency and other metrics.
        If metrics tracing is not enabled, this method does not perform any operations.
        """
        if not self.enabled or not HAS_OPENTELEMETRY_INSTALLED:
            return
        self.current_op.start()

    def record_operation_completion(self) -> None:
        """
        Record the completion of an operation.

        This method marks the end of an operation and updates the metrics accordingly.
        It calculates the operation latency by measuring the time elapsed since the operation started and records this metric.
        Additionally, it increments the operation count and records the attempt count for the operation.
        If metrics tracing is not enabled, this method does not perform any operations.
        """
        if not self.enabled or not HAS_OPENTELEMETRY_INSTALLED:
            return
        end_time = datetime.now()
        # Build Attributes
        operation_attributes = self._create_operation_otel_attributes()
        attempt_attributes = self._create_attempt_otel_attributes()

        # Calculate elapsed time
        operation_latency_ms = self._get_ms_time_diff(
            start=self.current_op.start_time, end=end_time
        )

        # Increase operation count
        self.instrument_operation_counter.add(amount=1, attributes=operation_attributes)

        # Record operation latency
        self.instrument_operation_latency.record(
            amount=operation_latency_ms, attributes=operation_attributes
        )

        # Record Attempt Count
        self.instrument_attempt_counter.add(
            self.current_op.attempt_count, attributes=attempt_attributes
        )

    def record_gfe_latency(self, latency: int) -> None:
        """
        Records the GFE latency using the Histogram instrument.

        Args:
            latency (int): The latency duration to be recorded.
        """
        if not self.enabled or not HAS_OPENTELEMETRY_INSTALLED or not self.gfe_enabled:
            return
        self._instrument_gfe_latency.record(
            amount=latency, attributes=self.client_attributes
        )

    def record_gfe_missing_header_count(self) -> None:
        """
        Increments the counter for missing GFE headers.
        """
        if not self.enabled or not HAS_OPENTELEMETRY_INSTALLED or not self.gfe_enabled:
            return
        self._instrument_gfe_missing_header_count.add(
            amount=1, attributes=self.client_attributes
        )

    def _create_operation_otel_attributes(self) -> dict:
        """
        Create additional attributes for operation metrics tracing.

        This method populates the client attributes dictionary with the operation status if metrics tracing is enabled.
        It returns the updated client attributes dictionary.
        """
        if not self.enabled or not HAS_OPENTELEMETRY_INSTALLED:
            return {}
        attributes = self._client_attributes.copy()
        attributes[METRIC_LABEL_KEY_STATUS] = self.current_op.status
        return attributes

    def _create_attempt_otel_attributes(self) -> dict:
        """
        Create additional attributes for attempt metrics tracing.

        This method populates the attributes dictionary with the attempt status if metrics tracing is enabled and an attempt exists.
        It returns the updated attributes dictionary.
        """
        if not self.enabled or not HAS_OPENTELEMETRY_INSTALLED:
            return {}

        attributes = self._client_attributes.copy()

        # Short circuit out if we don't have an attempt
        if self.current_op.current_attempt is None:
            return attributes

        attributes[METRIC_LABEL_KEY_STATUS] = self.current_op.current_attempt.status
        return attributes

    def set_project(self, project: str) -> "MetricsTracer":
        """
        Set the project attribute for metrics tracing.

        This method updates the project attribute in the client attributes dictionary for metrics tracing purposes.
        If the project attribute already has a value, this method does nothing and returns.

        :param project: The project name to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if MONITORED_RES_LABEL_KEY_PROJECT not in self._client_attributes:
            self._client_attributes[MONITORED_RES_LABEL_KEY_PROJECT] = project
        return self

    def set_instance(self, instance: str) -> "MetricsTracer":
        """
        Set the instance attribute for metrics tracing.

        This method updates the instance attribute in the client attributes dictionary for metrics tracing purposes.
        If the instance attribute already has a value, this method does nothing and returns.

        :param instance: The instance name to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if MONITORED_RES_LABEL_KEY_INSTANCE not in self._client_attributes:
            self._client_attributes[MONITORED_RES_LABEL_KEY_INSTANCE] = instance
        return self

    def set_instance_config(self, instance_config: str) -> "MetricsTracer":
        """
        Set the instance configuration attribute for metrics tracing.

        This method updates the instance configuration attribute in the client attributes dictionary for metrics tracing purposes.
        If the instance configuration attribute already has a value, this method does nothing and returns.

        :param instance_config: The instance configuration name to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if MONITORED_RES_LABEL_KEY_INSTANCE_CONFIG not in self._client_attributes:
            self._client_attributes[
                MONITORED_RES_LABEL_KEY_INSTANCE_CONFIG
            ] = instance_config
        return self

    def set_location(self, location: str) -> "MetricsTracer":
        """
        Set the location attribute for metrics tracing.

        This method updates the location attribute in the client attributes dictionary for metrics tracing purposes.
        If the location attribute already has a value, this method does nothing and returns.

        :param location: The location name to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if MONITORED_RES_LABEL_KEY_LOCATION not in self._client_attributes:
            self._client_attributes[MONITORED_RES_LABEL_KEY_LOCATION] = location
        return self

    def set_client_hash(self, hash: str) -> "MetricsTracer":
        """
        Set the client hash attribute for metrics tracing.

        This method updates the client hash attribute in the client attributes dictionary for metrics tracing purposes.
        If the client hash attribute already has a value, this method does nothing and returns.

        :param hash: The client hash to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if MONITORED_RES_LABEL_KEY_CLIENT_HASH not in self._client_attributes:
            self._client_attributes[MONITORED_RES_LABEL_KEY_CLIENT_HASH] = hash
        return self

    def set_client_uid(self, client_uid: str) -> "MetricsTracer":
        """
        Set the client UID attribute for metrics tracing.

        This method updates the client UID attribute in the client attributes dictionary for metrics tracing purposes.
        If the client UID attribute already has a value, this method does nothing and returns.

        :param client_uid: The client UID to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if METRIC_LABEL_KEY_CLIENT_UID not in self._client_attributes:
            self._client_attributes[METRIC_LABEL_KEY_CLIENT_UID] = client_uid
        return self

    def set_client_name(self, client_name: str) -> "MetricsTracer":
        """
        Set the client name attribute for metrics tracing.

        This method updates the client name attribute in the client attributes dictionary for metrics tracing purposes.
        If the client name attribute already has a value, this method does nothing and returns.

        :param client_name: The client name to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if METRIC_LABEL_KEY_CLIENT_NAME not in self._client_attributes:
            self._client_attributes[METRIC_LABEL_KEY_CLIENT_NAME] = client_name
        return self

    def set_database(self, database: str) -> "MetricsTracer":
        """
        Set the database attribute for metrics tracing.

        This method updates the database attribute in the client attributes dictionary for metrics tracing purposes.
        If the database attribute already has a value, this method does nothing and returns.

        :param database: The database name to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if METRIC_LABEL_KEY_DATABASE not in self._client_attributes:
            self._client_attributes[METRIC_LABEL_KEY_DATABASE] = database
        return self

    def set_method(self, method: str) -> "MetricsTracer":
        """
        Set the method attribute for metrics tracing.

        This method updates the method attribute in the client attributes dictionary for metrics tracing purposes.
        If the database attribute already has a value, this method does nothing and returns.

        :param method: The method name to set.
        :return: This instance of MetricsTracer for method chaining.
        """
        if METRIC_LABEL_KEY_METHOD not in self._client_attributes:
            self.client_attributes[METRIC_LABEL_KEY_METHOD] = method
        return self

    def enable_direct_path(self, enable: bool = False) -> "MetricsTracer":
        """
        Enable or disable the direct path for metrics tracing.

        This method updates the direct path enabled attribute in the client attributes dictionary for metrics tracing purposes.
        If the direct path enabled attribute already has a value, this method does nothing and returns.

        :param enable: Boolean indicating whether to enable the direct path.
        :return: This instance of MetricsTracer for method chaining.
        """
        if METRIC_LABEL_KEY_DIRECT_PATH_ENABLED not in self._client_attributes:
            self._client_attributes[METRIC_LABEL_KEY_DIRECT_PATH_ENABLED] = str(enable)
        return self
