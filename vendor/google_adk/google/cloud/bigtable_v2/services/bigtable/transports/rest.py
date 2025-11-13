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
import logging
import json  # type: ignore

from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import gapic_v1
import google.protobuf

from google.protobuf import json_format

from requests import __version__ as requests_version
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings


from google.cloud.bigtable_v2.types import bigtable


from .rest_base import _BaseBigtableRestTransport
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = logging.getLogger(__name__)

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=f"requests@{requests_version}",
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class BigtableRestInterceptor:
    """Interceptor for Bigtable.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the BigtableRestTransport.

    .. code-block:: python
        class MyCustomBigtableInterceptor(BigtableRestInterceptor):
            def pre_check_and_mutate_row(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_check_and_mutate_row(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_execute_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_execute_query(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_generate_initial_change_stream_partitions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_generate_initial_change_stream_partitions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_mutate_row(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_mutate_row(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_mutate_rows(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_mutate_rows(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_ping_and_warm(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_ping_and_warm(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_prepare_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_prepare_query(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_read_change_stream(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_read_change_stream(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_read_modify_write_row(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_read_modify_write_row(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_read_rows(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_read_rows(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_sample_row_keys(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_sample_row_keys(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = BigtableRestTransport(interceptor=MyCustomBigtableInterceptor())
        client = BigtableClient(transport=transport)


    """

    def pre_check_and_mutate_row(
        self,
        request: bigtable.CheckAndMutateRowRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable.CheckAndMutateRowRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for check_and_mutate_row

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_check_and_mutate_row(
        self, response: bigtable.CheckAndMutateRowResponse
    ) -> bigtable.CheckAndMutateRowResponse:
        """Post-rpc interceptor for check_and_mutate_row

        DEPRECATED. Please use the `post_check_and_mutate_row_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_check_and_mutate_row` interceptor runs
        before the `post_check_and_mutate_row_with_metadata` interceptor.
        """
        return response

    def post_check_and_mutate_row_with_metadata(
        self,
        response: bigtable.CheckAndMutateRowResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable.CheckAndMutateRowResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for check_and_mutate_row

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_check_and_mutate_row_with_metadata`
        interceptor in new development instead of the `post_check_and_mutate_row` interceptor.
        When both interceptors are used, this `post_check_and_mutate_row_with_metadata` interceptor runs after the
        `post_check_and_mutate_row` interceptor. The (possibly modified) response returned by
        `post_check_and_mutate_row` will be passed to
        `post_check_and_mutate_row_with_metadata`.
        """
        return response, metadata

    def pre_execute_query(
        self,
        request: bigtable.ExecuteQueryRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.ExecuteQueryRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for execute_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_execute_query(
        self, response: rest_streaming.ResponseIterator
    ) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for execute_query

        DEPRECATED. Please use the `post_execute_query_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_execute_query` interceptor runs
        before the `post_execute_query_with_metadata` interceptor.
        """
        return response

    def post_execute_query_with_metadata(
        self,
        response: rest_streaming.ResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming.ResponseIterator, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for execute_query

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_execute_query_with_metadata`
        interceptor in new development instead of the `post_execute_query` interceptor.
        When both interceptors are used, this `post_execute_query_with_metadata` interceptor runs after the
        `post_execute_query` interceptor. The (possibly modified) response returned by
        `post_execute_query` will be passed to
        `post_execute_query_with_metadata`.
        """
        return response, metadata

    def pre_generate_initial_change_stream_partitions(
        self,
        request: bigtable.GenerateInitialChangeStreamPartitionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable.GenerateInitialChangeStreamPartitionsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for generate_initial_change_stream_partitions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_generate_initial_change_stream_partitions(
        self, response: rest_streaming.ResponseIterator
    ) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for generate_initial_change_stream_partitions

        DEPRECATED. Please use the `post_generate_initial_change_stream_partitions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_generate_initial_change_stream_partitions` interceptor runs
        before the `post_generate_initial_change_stream_partitions_with_metadata` interceptor.
        """
        return response

    def post_generate_initial_change_stream_partitions_with_metadata(
        self,
        response: rest_streaming.ResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming.ResponseIterator, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for generate_initial_change_stream_partitions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_generate_initial_change_stream_partitions_with_metadata`
        interceptor in new development instead of the `post_generate_initial_change_stream_partitions` interceptor.
        When both interceptors are used, this `post_generate_initial_change_stream_partitions_with_metadata` interceptor runs after the
        `post_generate_initial_change_stream_partitions` interceptor. The (possibly modified) response returned by
        `post_generate_initial_change_stream_partitions` will be passed to
        `post_generate_initial_change_stream_partitions_with_metadata`.
        """
        return response, metadata

    def pre_mutate_row(
        self,
        request: bigtable.MutateRowRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.MutateRowRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for mutate_row

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_mutate_row(
        self, response: bigtable.MutateRowResponse
    ) -> bigtable.MutateRowResponse:
        """Post-rpc interceptor for mutate_row

        DEPRECATED. Please use the `post_mutate_row_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_mutate_row` interceptor runs
        before the `post_mutate_row_with_metadata` interceptor.
        """
        return response

    def post_mutate_row_with_metadata(
        self,
        response: bigtable.MutateRowResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.MutateRowResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for mutate_row

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_mutate_row_with_metadata`
        interceptor in new development instead of the `post_mutate_row` interceptor.
        When both interceptors are used, this `post_mutate_row_with_metadata` interceptor runs after the
        `post_mutate_row` interceptor. The (possibly modified) response returned by
        `post_mutate_row` will be passed to
        `post_mutate_row_with_metadata`.
        """
        return response, metadata

    def pre_mutate_rows(
        self,
        request: bigtable.MutateRowsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.MutateRowsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for mutate_rows

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_mutate_rows(
        self, response: rest_streaming.ResponseIterator
    ) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for mutate_rows

        DEPRECATED. Please use the `post_mutate_rows_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_mutate_rows` interceptor runs
        before the `post_mutate_rows_with_metadata` interceptor.
        """
        return response

    def post_mutate_rows_with_metadata(
        self,
        response: rest_streaming.ResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming.ResponseIterator, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for mutate_rows

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_mutate_rows_with_metadata`
        interceptor in new development instead of the `post_mutate_rows` interceptor.
        When both interceptors are used, this `post_mutate_rows_with_metadata` interceptor runs after the
        `post_mutate_rows` interceptor. The (possibly modified) response returned by
        `post_mutate_rows` will be passed to
        `post_mutate_rows_with_metadata`.
        """
        return response, metadata

    def pre_ping_and_warm(
        self,
        request: bigtable.PingAndWarmRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.PingAndWarmRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for ping_and_warm

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_ping_and_warm(
        self, response: bigtable.PingAndWarmResponse
    ) -> bigtable.PingAndWarmResponse:
        """Post-rpc interceptor for ping_and_warm

        DEPRECATED. Please use the `post_ping_and_warm_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_ping_and_warm` interceptor runs
        before the `post_ping_and_warm_with_metadata` interceptor.
        """
        return response

    def post_ping_and_warm_with_metadata(
        self,
        response: bigtable.PingAndWarmResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.PingAndWarmResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for ping_and_warm

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_ping_and_warm_with_metadata`
        interceptor in new development instead of the `post_ping_and_warm` interceptor.
        When both interceptors are used, this `post_ping_and_warm_with_metadata` interceptor runs after the
        `post_ping_and_warm` interceptor. The (possibly modified) response returned by
        `post_ping_and_warm` will be passed to
        `post_ping_and_warm_with_metadata`.
        """
        return response, metadata

    def pre_prepare_query(
        self,
        request: bigtable.PrepareQueryRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.PrepareQueryRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for prepare_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_prepare_query(
        self, response: bigtable.PrepareQueryResponse
    ) -> bigtable.PrepareQueryResponse:
        """Post-rpc interceptor for prepare_query

        DEPRECATED. Please use the `post_prepare_query_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_prepare_query` interceptor runs
        before the `post_prepare_query_with_metadata` interceptor.
        """
        return response

    def post_prepare_query_with_metadata(
        self,
        response: bigtable.PrepareQueryResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.PrepareQueryResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for prepare_query

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_prepare_query_with_metadata`
        interceptor in new development instead of the `post_prepare_query` interceptor.
        When both interceptors are used, this `post_prepare_query_with_metadata` interceptor runs after the
        `post_prepare_query` interceptor. The (possibly modified) response returned by
        `post_prepare_query` will be passed to
        `post_prepare_query_with_metadata`.
        """
        return response, metadata

    def pre_read_change_stream(
        self,
        request: bigtable.ReadChangeStreamRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable.ReadChangeStreamRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for read_change_stream

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_read_change_stream(
        self, response: rest_streaming.ResponseIterator
    ) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for read_change_stream

        DEPRECATED. Please use the `post_read_change_stream_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_read_change_stream` interceptor runs
        before the `post_read_change_stream_with_metadata` interceptor.
        """
        return response

    def post_read_change_stream_with_metadata(
        self,
        response: rest_streaming.ResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming.ResponseIterator, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for read_change_stream

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_read_change_stream_with_metadata`
        interceptor in new development instead of the `post_read_change_stream` interceptor.
        When both interceptors are used, this `post_read_change_stream_with_metadata` interceptor runs after the
        `post_read_change_stream` interceptor. The (possibly modified) response returned by
        `post_read_change_stream` will be passed to
        `post_read_change_stream_with_metadata`.
        """
        return response, metadata

    def pre_read_modify_write_row(
        self,
        request: bigtable.ReadModifyWriteRowRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable.ReadModifyWriteRowRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for read_modify_write_row

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_read_modify_write_row(
        self, response: bigtable.ReadModifyWriteRowResponse
    ) -> bigtable.ReadModifyWriteRowResponse:
        """Post-rpc interceptor for read_modify_write_row

        DEPRECATED. Please use the `post_read_modify_write_row_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_read_modify_write_row` interceptor runs
        before the `post_read_modify_write_row_with_metadata` interceptor.
        """
        return response

    def post_read_modify_write_row_with_metadata(
        self,
        response: bigtable.ReadModifyWriteRowResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        bigtable.ReadModifyWriteRowResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for read_modify_write_row

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_read_modify_write_row_with_metadata`
        interceptor in new development instead of the `post_read_modify_write_row` interceptor.
        When both interceptors are used, this `post_read_modify_write_row_with_metadata` interceptor runs after the
        `post_read_modify_write_row` interceptor. The (possibly modified) response returned by
        `post_read_modify_write_row` will be passed to
        `post_read_modify_write_row_with_metadata`.
        """
        return response, metadata

    def pre_read_rows(
        self,
        request: bigtable.ReadRowsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.ReadRowsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for read_rows

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_read_rows(
        self, response: rest_streaming.ResponseIterator
    ) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for read_rows

        DEPRECATED. Please use the `post_read_rows_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_read_rows` interceptor runs
        before the `post_read_rows_with_metadata` interceptor.
        """
        return response

    def post_read_rows_with_metadata(
        self,
        response: rest_streaming.ResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming.ResponseIterator, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for read_rows

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_read_rows_with_metadata`
        interceptor in new development instead of the `post_read_rows` interceptor.
        When both interceptors are used, this `post_read_rows_with_metadata` interceptor runs after the
        `post_read_rows` interceptor. The (possibly modified) response returned by
        `post_read_rows` will be passed to
        `post_read_rows_with_metadata`.
        """
        return response, metadata

    def pre_sample_row_keys(
        self,
        request: bigtable.SampleRowKeysRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[bigtable.SampleRowKeysRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for sample_row_keys

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Bigtable server.
        """
        return request, metadata

    def post_sample_row_keys(
        self, response: rest_streaming.ResponseIterator
    ) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for sample_row_keys

        DEPRECATED. Please use the `post_sample_row_keys_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Bigtable server but before
        it is returned to user code. This `post_sample_row_keys` interceptor runs
        before the `post_sample_row_keys_with_metadata` interceptor.
        """
        return response

    def post_sample_row_keys_with_metadata(
        self,
        response: rest_streaming.ResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming.ResponseIterator, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for sample_row_keys

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Bigtable server but before it is returned to user code.

        We recommend only using this `post_sample_row_keys_with_metadata`
        interceptor in new development instead of the `post_sample_row_keys` interceptor.
        When both interceptors are used, this `post_sample_row_keys_with_metadata` interceptor runs after the
        `post_sample_row_keys` interceptor. The (possibly modified) response returned by
        `post_sample_row_keys` will be passed to
        `post_sample_row_keys_with_metadata`.
        """
        return response, metadata


@dataclasses.dataclass
class BigtableRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: BigtableRestInterceptor


class BigtableRestTransport(_BaseBigtableRestTransport):
    """REST backend synchronous transport for Bigtable.

    Service for reading from and writing to existing Bigtable
    tables.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "bigtable.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[BigtableRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'bigtable.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        # Run the base constructor
        # TODO(yon-mg): resolve other ctor params i.e. scopes, quota, etc.
        # TODO: When custom host (api_endpoint) is set, `scopes` must *also* be set on the
        # credentials object
        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            url_scheme=url_scheme,
            api_audience=api_audience,
        )
        self._session = AuthorizedSession(
            self._credentials, default_host=self.DEFAULT_HOST
        )
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or BigtableRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CheckAndMutateRow(
        _BaseBigtableRestTransport._BaseCheckAndMutateRow, BigtableRestStub
    ):
        def __hash__(self):
            return hash("BigtableRestTransport.CheckAndMutateRow")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable.CheckAndMutateRowRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable.CheckAndMutateRowResponse:
            r"""Call the check and mutate row method over HTTP.

            Args:
                request (~.bigtable.CheckAndMutateRowRequest):
                    The request object. Request message for
                Bigtable.CheckAndMutateRow.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.CheckAndMutateRowResponse:
                    Response message for
                Bigtable.CheckAndMutateRow.

            """

            http_options = (
                _BaseBigtableRestTransport._BaseCheckAndMutateRow._get_http_options()
            )

            request, metadata = self._interceptor.pre_check_and_mutate_row(
                request, metadata
            )
            transcoded_request = _BaseBigtableRestTransport._BaseCheckAndMutateRow._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableRestTransport._BaseCheckAndMutateRow._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableRestTransport._BaseCheckAndMutateRow._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.CheckAndMutateRow",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "CheckAndMutateRow",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._CheckAndMutateRow._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable.CheckAndMutateRowResponse()
            pb_resp = bigtable.CheckAndMutateRowResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_check_and_mutate_row(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_check_and_mutate_row_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = bigtable.CheckAndMutateRowResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable_v2.BigtableClient.check_and_mutate_row",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "CheckAndMutateRow",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ExecuteQuery(_BaseBigtableRestTransport._BaseExecuteQuery, BigtableRestStub):
        def __hash__(self):
            return hash("BigtableRestTransport.ExecuteQuery")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                stream=True,
            )
            return response

        def __call__(
            self,
            request: bigtable.ExecuteQueryRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming.ResponseIterator:
            r"""Call the execute query method over HTTP.

            Args:
                request (~.bigtable.ExecuteQueryRequest):
                    The request object. Request message for
                Bigtable.ExecuteQuery
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.ExecuteQueryResponse:
                    Response message for
                Bigtable.ExecuteQuery

            """

            http_options = (
                _BaseBigtableRestTransport._BaseExecuteQuery._get_http_options()
            )

            request, metadata = self._interceptor.pre_execute_query(request, metadata)
            transcoded_request = (
                _BaseBigtableRestTransport._BaseExecuteQuery._get_transcoded_request(
                    http_options, request
                )
            )

            body = _BaseBigtableRestTransport._BaseExecuteQuery._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = (
                _BaseBigtableRestTransport._BaseExecuteQuery._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.ExecuteQuery",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "ExecuteQuery",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._ExecuteQuery._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = rest_streaming.ResponseIterator(
                response, bigtable.ExecuteQueryResponse
            )

            resp = self._interceptor.post_execute_query(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_execute_query_with_metadata(
                resp, response_metadata
            )
            return resp

    class _GenerateInitialChangeStreamPartitions(
        _BaseBigtableRestTransport._BaseGenerateInitialChangeStreamPartitions,
        BigtableRestStub,
    ):
        def __hash__(self):
            return hash("BigtableRestTransport.GenerateInitialChangeStreamPartitions")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                stream=True,
            )
            return response

        def __call__(
            self,
            request: bigtable.GenerateInitialChangeStreamPartitionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming.ResponseIterator:
            r"""Call the generate initial change
            stream partitions method over HTTP.

                Args:
                    request (~.bigtable.GenerateInitialChangeStreamPartitionsRequest):
                        The request object. NOTE: This API is intended to be used
                    by Apache Beam BigtableIO. Request
                    message for
                    Bigtable.GenerateInitialChangeStreamPartitions.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.bigtable.GenerateInitialChangeStreamPartitionsResponse:
                        NOTE: This API is intended to be used
                    by Apache Beam BigtableIO. Response
                    message for
                    Bigtable.GenerateInitialChangeStreamPartitions.

            """

            http_options = (
                _BaseBigtableRestTransport._BaseGenerateInitialChangeStreamPartitions._get_http_options()
            )

            (
                request,
                metadata,
            ) = self._interceptor.pre_generate_initial_change_stream_partitions(
                request, metadata
            )
            transcoded_request = _BaseBigtableRestTransport._BaseGenerateInitialChangeStreamPartitions._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableRestTransport._BaseGenerateInitialChangeStreamPartitions._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableRestTransport._BaseGenerateInitialChangeStreamPartitions._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.GenerateInitialChangeStreamPartitions",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "GenerateInitialChangeStreamPartitions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._GenerateInitialChangeStreamPartitions._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = rest_streaming.ResponseIterator(
                response, bigtable.GenerateInitialChangeStreamPartitionsResponse
            )

            resp = self._interceptor.post_generate_initial_change_stream_partitions(
                resp
            )
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_generate_initial_change_stream_partitions_with_metadata(
                resp, response_metadata
            )
            return resp

    class _MutateRow(_BaseBigtableRestTransport._BaseMutateRow, BigtableRestStub):
        def __hash__(self):
            return hash("BigtableRestTransport.MutateRow")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable.MutateRowRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable.MutateRowResponse:
            r"""Call the mutate row method over HTTP.

            Args:
                request (~.bigtable.MutateRowRequest):
                    The request object. Request message for
                Bigtable.MutateRow.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.MutateRowResponse:
                    Response message for
                Bigtable.MutateRow.

            """

            http_options = _BaseBigtableRestTransport._BaseMutateRow._get_http_options()

            request, metadata = self._interceptor.pre_mutate_row(request, metadata)
            transcoded_request = (
                _BaseBigtableRestTransport._BaseMutateRow._get_transcoded_request(
                    http_options, request
                )
            )

            body = _BaseBigtableRestTransport._BaseMutateRow._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = (
                _BaseBigtableRestTransport._BaseMutateRow._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.MutateRow",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "MutateRow",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._MutateRow._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable.MutateRowResponse()
            pb_resp = bigtable.MutateRowResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_mutate_row(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_mutate_row_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = bigtable.MutateRowResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable_v2.BigtableClient.mutate_row",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "MutateRow",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _MutateRows(_BaseBigtableRestTransport._BaseMutateRows, BigtableRestStub):
        def __hash__(self):
            return hash("BigtableRestTransport.MutateRows")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                stream=True,
            )
            return response

        def __call__(
            self,
            request: bigtable.MutateRowsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming.ResponseIterator:
            r"""Call the mutate rows method over HTTP.

            Args:
                request (~.bigtable.MutateRowsRequest):
                    The request object. Request message for
                BigtableService.MutateRows.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.MutateRowsResponse:
                    Response message for
                BigtableService.MutateRows.

            """

            http_options = (
                _BaseBigtableRestTransport._BaseMutateRows._get_http_options()
            )

            request, metadata = self._interceptor.pre_mutate_rows(request, metadata)
            transcoded_request = (
                _BaseBigtableRestTransport._BaseMutateRows._get_transcoded_request(
                    http_options, request
                )
            )

            body = _BaseBigtableRestTransport._BaseMutateRows._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = (
                _BaseBigtableRestTransport._BaseMutateRows._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.MutateRows",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "MutateRows",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._MutateRows._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = rest_streaming.ResponseIterator(
                response, bigtable.MutateRowsResponse
            )

            resp = self._interceptor.post_mutate_rows(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_mutate_rows_with_metadata(
                resp, response_metadata
            )
            return resp

    class _PingAndWarm(_BaseBigtableRestTransport._BasePingAndWarm, BigtableRestStub):
        def __hash__(self):
            return hash("BigtableRestTransport.PingAndWarm")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable.PingAndWarmRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable.PingAndWarmResponse:
            r"""Call the ping and warm method over HTTP.

            Args:
                request (~.bigtable.PingAndWarmRequest):
                    The request object. Request message for client connection
                keep-alive and warming.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.PingAndWarmResponse:
                    Response message for
                Bigtable.PingAndWarm connection
                keepalive and warming.

            """

            http_options = (
                _BaseBigtableRestTransport._BasePingAndWarm._get_http_options()
            )

            request, metadata = self._interceptor.pre_ping_and_warm(request, metadata)
            transcoded_request = (
                _BaseBigtableRestTransport._BasePingAndWarm._get_transcoded_request(
                    http_options, request
                )
            )

            body = _BaseBigtableRestTransport._BasePingAndWarm._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = (
                _BaseBigtableRestTransport._BasePingAndWarm._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.PingAndWarm",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "PingAndWarm",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._PingAndWarm._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable.PingAndWarmResponse()
            pb_resp = bigtable.PingAndWarmResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_ping_and_warm(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_ping_and_warm_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = bigtable.PingAndWarmResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable_v2.BigtableClient.ping_and_warm",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "PingAndWarm",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _PrepareQuery(_BaseBigtableRestTransport._BasePrepareQuery, BigtableRestStub):
        def __hash__(self):
            return hash("BigtableRestTransport.PrepareQuery")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable.PrepareQueryRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable.PrepareQueryResponse:
            r"""Call the prepare query method over HTTP.

            Args:
                request (~.bigtable.PrepareQueryRequest):
                    The request object. Request message for
                Bigtable.PrepareQuery
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.PrepareQueryResponse:
                    Response message for
                Bigtable.PrepareQueryResponse

            """

            http_options = (
                _BaseBigtableRestTransport._BasePrepareQuery._get_http_options()
            )

            request, metadata = self._interceptor.pre_prepare_query(request, metadata)
            transcoded_request = (
                _BaseBigtableRestTransport._BasePrepareQuery._get_transcoded_request(
                    http_options, request
                )
            )

            body = _BaseBigtableRestTransport._BasePrepareQuery._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = (
                _BaseBigtableRestTransport._BasePrepareQuery._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.PrepareQuery",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "PrepareQuery",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._PrepareQuery._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable.PrepareQueryResponse()
            pb_resp = bigtable.PrepareQueryResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_prepare_query(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_prepare_query_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = bigtable.PrepareQueryResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable_v2.BigtableClient.prepare_query",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "PrepareQuery",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ReadChangeStream(
        _BaseBigtableRestTransport._BaseReadChangeStream, BigtableRestStub
    ):
        def __hash__(self):
            return hash("BigtableRestTransport.ReadChangeStream")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                stream=True,
            )
            return response

        def __call__(
            self,
            request: bigtable.ReadChangeStreamRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming.ResponseIterator:
            r"""Call the read change stream method over HTTP.

            Args:
                request (~.bigtable.ReadChangeStreamRequest):
                    The request object. NOTE: This API is intended to be used
                by Apache Beam BigtableIO. Request
                message for Bigtable.ReadChangeStream.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.ReadChangeStreamResponse:
                    NOTE: This API is intended to be used
                by Apache Beam BigtableIO. Response
                message for Bigtable.ReadChangeStream.

            """

            http_options = (
                _BaseBigtableRestTransport._BaseReadChangeStream._get_http_options()
            )

            request, metadata = self._interceptor.pre_read_change_stream(
                request, metadata
            )
            transcoded_request = _BaseBigtableRestTransport._BaseReadChangeStream._get_transcoded_request(
                http_options, request
            )

            body = (
                _BaseBigtableRestTransport._BaseReadChangeStream._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseBigtableRestTransport._BaseReadChangeStream._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.ReadChangeStream",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "ReadChangeStream",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._ReadChangeStream._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = rest_streaming.ResponseIterator(
                response, bigtable.ReadChangeStreamResponse
            )

            resp = self._interceptor.post_read_change_stream(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_read_change_stream_with_metadata(
                resp, response_metadata
            )
            return resp

    class _ReadModifyWriteRow(
        _BaseBigtableRestTransport._BaseReadModifyWriteRow, BigtableRestStub
    ):
        def __hash__(self):
            return hash("BigtableRestTransport.ReadModifyWriteRow")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: bigtable.ReadModifyWriteRowRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> bigtable.ReadModifyWriteRowResponse:
            r"""Call the read modify write row method over HTTP.

            Args:
                request (~.bigtable.ReadModifyWriteRowRequest):
                    The request object. Request message for
                Bigtable.ReadModifyWriteRow.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.ReadModifyWriteRowResponse:
                    Response message for
                Bigtable.ReadModifyWriteRow.

            """

            http_options = (
                _BaseBigtableRestTransport._BaseReadModifyWriteRow._get_http_options()
            )

            request, metadata = self._interceptor.pre_read_modify_write_row(
                request, metadata
            )
            transcoded_request = _BaseBigtableRestTransport._BaseReadModifyWriteRow._get_transcoded_request(
                http_options, request
            )

            body = _BaseBigtableRestTransport._BaseReadModifyWriteRow._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseBigtableRestTransport._BaseReadModifyWriteRow._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.ReadModifyWriteRow",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "ReadModifyWriteRow",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._ReadModifyWriteRow._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = bigtable.ReadModifyWriteRowResponse()
            pb_resp = bigtable.ReadModifyWriteRowResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_read_modify_write_row(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_read_modify_write_row_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = bigtable.ReadModifyWriteRowResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.bigtable_v2.BigtableClient.read_modify_write_row",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "ReadModifyWriteRow",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ReadRows(_BaseBigtableRestTransport._BaseReadRows, BigtableRestStub):
        def __hash__(self):
            return hash("BigtableRestTransport.ReadRows")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                stream=True,
            )
            return response

        def __call__(
            self,
            request: bigtable.ReadRowsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming.ResponseIterator:
            r"""Call the read rows method over HTTP.

            Args:
                request (~.bigtable.ReadRowsRequest):
                    The request object. Request message for
                Bigtable.ReadRows.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.ReadRowsResponse:
                    Response message for
                Bigtable.ReadRows.

            """

            http_options = _BaseBigtableRestTransport._BaseReadRows._get_http_options()

            request, metadata = self._interceptor.pre_read_rows(request, metadata)
            transcoded_request = (
                _BaseBigtableRestTransport._BaseReadRows._get_transcoded_request(
                    http_options, request
                )
            )

            body = _BaseBigtableRestTransport._BaseReadRows._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = (
                _BaseBigtableRestTransport._BaseReadRows._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.ReadRows",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "ReadRows",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._ReadRows._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
                body,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = rest_streaming.ResponseIterator(response, bigtable.ReadRowsResponse)

            resp = self._interceptor.post_read_rows(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_read_rows_with_metadata(
                resp, response_metadata
            )
            return resp

    class _SampleRowKeys(
        _BaseBigtableRestTransport._BaseSampleRowKeys, BigtableRestStub
    ):
        def __hash__(self):
            return hash("BigtableRestTransport.SampleRowKeys")

        @staticmethod
        def _get_response(
            host,
            metadata,
            query_params,
            session,
            timeout,
            transcoded_request,
            body=None,
        ):
            uri = transcoded_request["uri"]
            method = transcoded_request["method"]
            headers = dict(metadata)
            headers["Content-Type"] = "application/json"
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                stream=True,
            )
            return response

        def __call__(
            self,
            request: bigtable.SampleRowKeysRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming.ResponseIterator:
            r"""Call the sample row keys method over HTTP.

            Args:
                request (~.bigtable.SampleRowKeysRequest):
                    The request object. Request message for
                Bigtable.SampleRowKeys.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.bigtable.SampleRowKeysResponse:
                    Response message for
                Bigtable.SampleRowKeys.

            """

            http_options = (
                _BaseBigtableRestTransport._BaseSampleRowKeys._get_http_options()
            )

            request, metadata = self._interceptor.pre_sample_row_keys(request, metadata)
            transcoded_request = (
                _BaseBigtableRestTransport._BaseSampleRowKeys._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseBigtableRestTransport._BaseSampleRowKeys._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.bigtable_v2.BigtableClient.SampleRowKeys",
                    extra={
                        "serviceName": "google.bigtable.v2.Bigtable",
                        "rpcName": "SampleRowKeys",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = BigtableRestTransport._SampleRowKeys._get_response(
                self._host,
                metadata,
                query_params,
                self._session,
                timeout,
                transcoded_request,
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = rest_streaming.ResponseIterator(
                response, bigtable.SampleRowKeysResponse
            )

            resp = self._interceptor.post_sample_row_keys(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_sample_row_keys_with_metadata(
                resp, response_metadata
            )
            return resp

    @property
    def check_and_mutate_row(
        self,
    ) -> Callable[
        [bigtable.CheckAndMutateRowRequest], bigtable.CheckAndMutateRowResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CheckAndMutateRow(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def execute_query(
        self,
    ) -> Callable[[bigtable.ExecuteQueryRequest], bigtable.ExecuteQueryResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ExecuteQuery(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def generate_initial_change_stream_partitions(
        self,
    ) -> Callable[
        [bigtable.GenerateInitialChangeStreamPartitionsRequest],
        bigtable.GenerateInitialChangeStreamPartitionsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GenerateInitialChangeStreamPartitions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def mutate_row(
        self,
    ) -> Callable[[bigtable.MutateRowRequest], bigtable.MutateRowResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._MutateRow(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def mutate_rows(
        self,
    ) -> Callable[[bigtable.MutateRowsRequest], bigtable.MutateRowsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._MutateRows(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def ping_and_warm(
        self,
    ) -> Callable[[bigtable.PingAndWarmRequest], bigtable.PingAndWarmResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._PingAndWarm(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def prepare_query(
        self,
    ) -> Callable[[bigtable.PrepareQueryRequest], bigtable.PrepareQueryResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._PrepareQuery(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def read_change_stream(
        self,
    ) -> Callable[
        [bigtable.ReadChangeStreamRequest], bigtable.ReadChangeStreamResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ReadChangeStream(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def read_modify_write_row(
        self,
    ) -> Callable[
        [bigtable.ReadModifyWriteRowRequest], bigtable.ReadModifyWriteRowResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ReadModifyWriteRow(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def read_rows(
        self,
    ) -> Callable[[bigtable.ReadRowsRequest], bigtable.ReadRowsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ReadRows(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def sample_row_keys(
        self,
    ) -> Callable[[bigtable.SampleRowKeysRequest], bigtable.SampleRowKeysResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SampleRowKeys(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("BigtableRestTransport",)
