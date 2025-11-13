# Copyright 2020 Google LLC All rights reserved.
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

"""Manages OpenTelemetry trace creation and handling"""

from contextlib import contextmanager
from datetime import datetime
import os

from google.cloud.spanner_v1 import SpannerClient
from google.cloud.spanner_v1 import gapic_version
from google.cloud.spanner_v1._helpers import (
    _metadata_with_span_context,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace.status import Status, StatusCode
    from opentelemetry.semconv.attributes.otel_attributes import (
        OTEL_SCOPE_NAME,
        OTEL_SCOPE_VERSION,
    )

    HAS_OPENTELEMETRY_INSTALLED = True
except ImportError:
    HAS_OPENTELEMETRY_INSTALLED = False

from google.cloud.spanner_v1.metrics.metrics_capture import MetricsCapture

TRACER_NAME = "cloud.google.com/python/spanner"
TRACER_VERSION = gapic_version.__version__
extended_tracing_globally_disabled = (
    os.getenv("SPANNER_ENABLE_EXTENDED_TRACING", "").lower() == "false"
)
end_to_end_tracing_globally_enabled = (
    os.getenv("SPANNER_ENABLE_END_TO_END_TRACING", "").lower() == "true"
)


def get_tracer(tracer_provider=None):
    """
    get_tracer is a utility to unify and simplify retrieval of the tracer, without
    leaking implementation details given that retrieving a tracer requires providing
    the full qualified library name and version.
    When the tracer_provider is set, it'll retrieve the tracer from it, otherwise
    it'll fall back to the global tracer provider and use this library's specific semantics.
    """
    if not tracer_provider:
        # Acquire the global tracer provider.
        tracer_provider = trace.get_tracer_provider()

    return tracer_provider.get_tracer(TRACER_NAME, TRACER_VERSION)


@contextmanager
def trace_call(
    name, session=None, extra_attributes=None, observability_options=None, metadata=None
):
    if session:
        session._last_use_time = datetime.now()

    if not (HAS_OPENTELEMETRY_INSTALLED and name):
        # Empty context manager. Users will have to check if the generated value is None or a span
        yield None
        return

    tracer_provider = None

    # By default enable_extended_tracing=True because in a bid to minimize
    # breaking changes and preserve legacy behavior, we are keeping it turned
    # on by default.
    enable_extended_tracing = True

    enable_end_to_end_tracing = False

    db_name = ""
    if session and getattr(session, "_database", None):
        db_name = session._database.name

    if isinstance(observability_options, dict):  # Avoid false positives with mock.Mock
        tracer_provider = observability_options.get("tracer_provider", None)
        enable_extended_tracing = observability_options.get(
            "enable_extended_tracing", enable_extended_tracing
        )
        enable_end_to_end_tracing = observability_options.get(
            "enable_end_to_end_tracing", enable_end_to_end_tracing
        )
        db_name = observability_options.get("db_name", db_name)

    tracer = get_tracer(tracer_provider)

    # Set base attributes that we know for every trace created
    attributes = {
        "db.type": "spanner",
        "db.url": SpannerClient.DEFAULT_ENDPOINT,
        "db.instance": db_name,
        "net.host.name": SpannerClient.DEFAULT_ENDPOINT,
        OTEL_SCOPE_NAME: TRACER_NAME,
        OTEL_SCOPE_VERSION: TRACER_VERSION,
        # Standard GCP attributes for OTel, attributes are used for internal purpose and are subjected to change
        "gcp.client.service": "spanner",
        "gcp.client.version": TRACER_VERSION,
        "gcp.client.repo": "googleapis/python-spanner",
    }

    if extra_attributes:
        attributes.update(extra_attributes)

    if extended_tracing_globally_disabled:
        enable_extended_tracing = False

    if not enable_extended_tracing:
        attributes.pop("db.statement", False)

    if end_to_end_tracing_globally_enabled:
        enable_end_to_end_tracing = True

    with tracer.start_as_current_span(
        name, kind=trace.SpanKind.CLIENT, attributes=attributes
    ) as span:
        with MetricsCapture():
            try:
                if enable_end_to_end_tracing:
                    _metadata_with_span_context(metadata)
                yield span
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                # OpenTelemetry-Python imposes invoking span.record_exception on __exit__
                # on any exception. We should file a bug later on with them to only
                # invoke .record_exception if not already invoked, hence we should not
                # invoke .record_exception on our own else we shall have 2 exceptions.
                raise
            else:
                # All spans still have set_status available even if for example
                # NonRecordingSpan doesn't have "_status".
                absent_span_status = getattr(span, "_status", None) is None
                if absent_span_status or span._status.status_code == StatusCode.UNSET:
                    # OpenTelemetry-Python only allows a status change
                    # if the current code is UNSET or ERROR. At the end
                    # of the generator's consumption, only set it to OK
                    # it wasn't previously set otherwise.
                    # https://github.com/googleapis/python-spanner/issues/1246
                    span.set_status(Status(StatusCode.OK))


def get_current_span():
    if not HAS_OPENTELEMETRY_INSTALLED:
        return None
    return trace.get_current_span()


def add_span_event(span, event_name, event_attributes=None):
    if span:
        span.add_event(event_name, event_attributes)
