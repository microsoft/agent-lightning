# Copyright 2021 Google LLC All rights reserved.
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

import collections
import os

from contextlib import contextmanager

from google.api_core.exceptions import GoogleAPICallError
from google.cloud.spanner_v1 import SpannerClient

try:
    from opentelemetry import trace
    from opentelemetry.trace.status import Status, StatusCode

    HAS_OPENTELEMETRY_INSTALLED = True
except ImportError:
    HAS_OPENTELEMETRY_INSTALLED = False


@contextmanager
def trace_call(name, extra_attributes=None):
    if not HAS_OPENTELEMETRY_INSTALLED:
        # Empty context manager. Users will have to check if the generated value
        # is None or a span
        yield None
        return

    tracer = trace.get_tracer(__name__)
    # Set base attributes that we know for every trace created
    attributes = {
        "db.type": "spanner",
        "db.engine": "sqlalchemy_spanner",
        "db.url": SpannerClient.DEFAULT_ENDPOINT,
        "net.host.name": SpannerClient.DEFAULT_ENDPOINT,
    }

    if extra_attributes:
        if os.environ.get("SQLALCHEMY_SPANNER_TRACE_HIDE_QUERY_PARAMETERS"):
            extra_attributes.pop("db.params", None)

        # Stringify "db.params" sequence values before sending to OpenTelemetry,
        # otherwise OpenTelemetry may log a Warning if types differ.
        if isinstance(extra_attributes, dict):
            for k, v in extra_attributes.items():
                if k == "db.params" and isinstance(v, collections.abc.Sequence):
                    extra_attributes[k] = [str(e) for e in v]

        attributes.update(extra_attributes)

    with tracer.start_as_current_span(
        name, kind=trace.SpanKind.CLIENT, attributes=attributes
    ) as span:
        try:
            yield span
        except GoogleAPICallError as error:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(error)
            raise
        span.set_status(Status(StatusCode.OK))
