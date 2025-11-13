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


"""This module provides a singleton factory for creating SpannerMetricsTracer instances."""

from .metrics_tracer_factory import MetricsTracerFactory
import os
import logging
from .constants import (
    SPANNER_SERVICE_NAME,
    GOOGLE_CLOUD_REGION_KEY,
    GOOGLE_CLOUD_REGION_GLOBAL,
)

try:
    from opentelemetry.resourcedetector import gcp_resource_detector

    # Overwrite the requests timeout for the detector.
    # This is necessary as the client will wait the full timeout if the
    # code is not run in a GCP environment, with the location endpoints available.
    gcp_resource_detector._TIMEOUT_SEC = 0.2

    import mmh3

    logging.getLogger("opentelemetry.resourcedetector.gcp_resource_detector").setLevel(
        logging.ERROR
    )

    HAS_OPENTELEMETRY_INSTALLED = True
except ImportError:  # pragma: NO COVER
    HAS_OPENTELEMETRY_INSTALLED = False

from .metrics_tracer import MetricsTracer
from google.cloud.spanner_v1 import __version__
from uuid import uuid4

log = logging.getLogger(__name__)


class SpannerMetricsTracerFactory(MetricsTracerFactory):
    """A factory for creating SpannerMetricsTracer instances."""

    _metrics_tracer_factory: "SpannerMetricsTracerFactory" = None
    current_metrics_tracer: MetricsTracer = None

    def __new__(
        cls, enabled: bool = True, gfe_enabled: bool = False
    ) -> "SpannerMetricsTracerFactory":
        """
        Create a new instance of SpannerMetricsTracerFactory if it doesn't already exist.

        This method implements the singleton pattern for the SpannerMetricsTracerFactory class.
        It initializes the factory with the necessary client attributes and configuration settings
        if it hasn't been created yet.

        Args:
            enabled (bool): A flag indicating whether metrics tracing is enabled. Defaults to True.
            gfe_enabled (bool): A flag indicating whether GFE metrics are enabled. Defaults to False.

        Returns:
            SpannerMetricsTracerFactory: The singleton instance of SpannerMetricsTracerFactory.
        """
        if cls._metrics_tracer_factory is None:
            cls._metrics_tracer_factory = MetricsTracerFactory(
                enabled, SPANNER_SERVICE_NAME
            )
            if not HAS_OPENTELEMETRY_INSTALLED:
                return cls._metrics_tracer_factory

            client_uid = cls._generate_client_uid()
            cls._metrics_tracer_factory.set_client_uid(client_uid)
            cls._metrics_tracer_factory.set_instance_config(cls._get_instance_config())
            cls._metrics_tracer_factory.set_client_name(cls._get_client_name())
            cls._metrics_tracer_factory.set_client_hash(
                cls._generate_client_hash(client_uid)
            )
            cls._metrics_tracer_factory.set_location(cls._get_location())
            cls._metrics_tracer_factory.gfe_enabled = gfe_enabled

            if cls._metrics_tracer_factory.enabled != enabled:
                cls._metrics_tracer_factory.enabeld = enabled

        return cls._metrics_tracer_factory

    @staticmethod
    def _generate_client_uid() -> str:
        """Generate a client UID in the form of uuidv4@pid@hostname.

        This method generates a unique client identifier (UID) by combining a UUID version 4,
        the process ID (PID), and the hostname. The PID is limited to the first 10 characters.

        Returns:
            str: A string representing the client UID in the format uuidv4@pid@hostname.
        """
        try:
            hostname = os.uname()[1]
            pid = str(os.getpid())[0:10]  # Limit PID to 10 characters
            uuid = uuid4()
            return f"{uuid}@{pid}@{hostname}"
        except Exception:
            return ""

    @staticmethod
    def _get_instance_config() -> str:
        """Get the instance configuration."""
        # TODO: unknown until there's a good way to get it.
        return "unknown"

    @staticmethod
    def _get_client_name() -> str:
        """Get the client name."""
        return f"{SPANNER_SERVICE_NAME}/{__version__}"

    @staticmethod
    def _generate_client_hash(client_uid: str) -> str:
        """
        Generate a 6-digit zero-padded lowercase hexadecimal hash using the 10 most significant bits of a 64-bit hash value.

        The primary purpose of this function is to generate a hash value for the `client_hash`
        resource label using `client_uid` metric field. The range of values is chosen to be small
        enough to keep the cardinality of the Resource targets under control. Note: If at later time
        the range needs to be increased, it can be done by increasing the value of `kPrefixLength` to
        up to 24 bits without changing the format of the returned value.

        Args:
            client_uid (str): The client UID used to generate the hash.

        Returns:
            str: A 6-digit zero-padded lowercase hexadecimal hash.
        """
        if not client_uid:
            return "000000"
        hashed_client = mmh3.hash64(client_uid)

        # Join the hashes back together since mmh3 splits into high and low 32bits
        full_hash = (hashed_client[0] << 32) | (hashed_client[1] & 0xFFFFFFFF)
        unsigned_hash = full_hash & 0xFFFFFFFFFFFFFFFF

        k_prefix_length = 10
        sig_figs = unsigned_hash >> (64 - k_prefix_length)

        # Return as 6 digit zero padded hex string
        return f"{sig_figs:06x}"

    @staticmethod
    def _get_location() -> str:
        """Get the location of the resource.

        In case of any error during detection, this method will log a warning
        and default to the "global" location.

        Returns:
            str: The location of the resource. If OpenTelemetry is not installed, returns a global region.
        """
        if not HAS_OPENTELEMETRY_INSTALLED:
            return GOOGLE_CLOUD_REGION_GLOBAL
        try:
            detector = gcp_resource_detector.GoogleCloudResourceDetector()
            resources = detector.detect()

            if GOOGLE_CLOUD_REGION_KEY in resources.attributes:
                return resources.attributes[GOOGLE_CLOUD_REGION_KEY]
        except Exception as e:
            log.warning(
                "Failed to detect GCP resource location for Spanner metrics, defaulting to 'global'. Error: %s",
                e,
            )
        return GOOGLE_CLOUD_REGION_GLOBAL
