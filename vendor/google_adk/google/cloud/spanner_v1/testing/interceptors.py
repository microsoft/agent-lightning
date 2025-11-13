# Copyright 2023     Google LLC All rights reserved.
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

from collections import defaultdict
import threading

from grpc_interceptor import ClientInterceptor
from google.api_core.exceptions import Aborted
from google.cloud.spanner_v1.request_id_header import parse_request_id


class MethodCountInterceptor(ClientInterceptor):
    """Test interceptor that counts number of times a method is being called."""

    def __init__(self):
        self._counts = defaultdict(int)

    def intercept(self, method, request_or_iterator, call_details):
        """Count number of times a method is being called."""
        self._counts[call_details.method] += 1
        return method(request_or_iterator, call_details)

    def reset(self):
        self._counts = defaultdict(int)


class MethodAbortInterceptor(ClientInterceptor):
    """Test interceptor that throws Aborted exception for a specific method."""

    def __init__(self):
        self._method_to_abort = None
        self._count = 0
        self._max_raise_count = 1
        self._connection = None

    def intercept(self, method, request_or_iterator, call_details):
        if (
            self._count < self._max_raise_count
            and call_details.method == self._method_to_abort
        ):
            self._count += 1
            if self._connection is not None:
                self._connection._transaction.rollback()
            raise Aborted("Thrown from ClientInterceptor for testing")
        return method(request_or_iterator, call_details)

    def set_method_to_abort(self, method_to_abort, connection=None, max_raise_count=1):
        self._method_to_abort = method_to_abort
        self._count = 0
        self._max_raise_count = max_raise_count
        self._connection = connection

    def reset(self):
        """Reset the interceptor to the original state."""
        self._method_to_abort = None
        self._count = 0
        self._connection = None


X_GOOG_REQUEST_ID = "x-goog-spanner-request-id"


class XGoogRequestIDHeaderInterceptor(ClientInterceptor):
    def __init__(self):
        self._unary_req_segments = []
        self._stream_req_segments = []
        self.__lock = threading.Lock()

    def intercept(self, method, request_or_iterator, call_details):
        metadata = call_details.metadata
        x_goog_request_id = None
        for key, value in metadata:
            if key == X_GOOG_REQUEST_ID:
                x_goog_request_id = value
                break

        if not x_goog_request_id:
            raise Exception(
                f"Missing {X_GOOG_REQUEST_ID} header in {call_details.method}"
            )

        response_or_iterator = method(request_or_iterator, call_details)
        streaming = getattr(response_or_iterator, "__iter__", None) is not None

        with self.__lock:
            if streaming:
                self._stream_req_segments.append(
                    (call_details.method, parse_request_id(x_goog_request_id))
                )
            else:
                self._unary_req_segments.append(
                    (call_details.method, parse_request_id(x_goog_request_id))
                )

        return response_or_iterator

    @property
    def unary_request_ids(self):
        return self._unary_req_segments

    @property
    def stream_request_ids(self):
        return self._stream_req_segments

    def reset(self):
        self._stream_req_segments.clear()
        self._unary_req_segments.clear()
