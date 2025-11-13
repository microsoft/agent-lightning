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
import dataclasses
import json  # type: ignore
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from google.api_core import gapic_v1, operations_v1, rest_helpers, rest_streaming
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
import google.protobuf
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import json_format
from requests import __version__ as requests_version

from google.cloud.discoveryengine_v1beta.types import sample_query as gcd_sample_query
from google.cloud.discoveryengine_v1beta.types import import_config
from google.cloud.discoveryengine_v1beta.types import sample_query
from google.cloud.discoveryengine_v1beta.types import sample_query_service

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseSampleQueryServiceRestTransport

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


class SampleQueryServiceRestInterceptor:
    """Interceptor for SampleQueryService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the SampleQueryServiceRestTransport.

    .. code-block:: python
        class MyCustomSampleQueryServiceInterceptor(SampleQueryServiceRestInterceptor):
            def pre_create_sample_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_sample_query(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_sample_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_sample_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_sample_query(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_import_sample_queries(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_import_sample_queries(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_sample_queries(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_sample_queries(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_sample_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_sample_query(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = SampleQueryServiceRestTransport(interceptor=MyCustomSampleQueryServiceInterceptor())
        client = SampleQueryServiceClient(transport=transport)


    """

    def pre_create_sample_query(
        self,
        request: sample_query_service.CreateSampleQueryRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        sample_query_service.CreateSampleQueryRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_sample_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def post_create_sample_query(
        self, response: gcd_sample_query.SampleQuery
    ) -> gcd_sample_query.SampleQuery:
        """Post-rpc interceptor for create_sample_query

        DEPRECATED. Please use the `post_create_sample_query_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the SampleQueryService server but before
        it is returned to user code. This `post_create_sample_query` interceptor runs
        before the `post_create_sample_query_with_metadata` interceptor.
        """
        return response

    def post_create_sample_query_with_metadata(
        self,
        response: gcd_sample_query.SampleQuery,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_sample_query.SampleQuery, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_sample_query

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the SampleQueryService server but before it is returned to user code.

        We recommend only using this `post_create_sample_query_with_metadata`
        interceptor in new development instead of the `post_create_sample_query` interceptor.
        When both interceptors are used, this `post_create_sample_query_with_metadata` interceptor runs after the
        `post_create_sample_query` interceptor. The (possibly modified) response returned by
        `post_create_sample_query` will be passed to
        `post_create_sample_query_with_metadata`.
        """
        return response, metadata

    def pre_delete_sample_query(
        self,
        request: sample_query_service.DeleteSampleQueryRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        sample_query_service.DeleteSampleQueryRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_sample_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def pre_get_sample_query(
        self,
        request: sample_query_service.GetSampleQueryRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        sample_query_service.GetSampleQueryRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_sample_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def post_get_sample_query(
        self, response: sample_query.SampleQuery
    ) -> sample_query.SampleQuery:
        """Post-rpc interceptor for get_sample_query

        DEPRECATED. Please use the `post_get_sample_query_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the SampleQueryService server but before
        it is returned to user code. This `post_get_sample_query` interceptor runs
        before the `post_get_sample_query_with_metadata` interceptor.
        """
        return response

    def post_get_sample_query_with_metadata(
        self,
        response: sample_query.SampleQuery,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[sample_query.SampleQuery, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_sample_query

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the SampleQueryService server but before it is returned to user code.

        We recommend only using this `post_get_sample_query_with_metadata`
        interceptor in new development instead of the `post_get_sample_query` interceptor.
        When both interceptors are used, this `post_get_sample_query_with_metadata` interceptor runs after the
        `post_get_sample_query` interceptor. The (possibly modified) response returned by
        `post_get_sample_query` will be passed to
        `post_get_sample_query_with_metadata`.
        """
        return response, metadata

    def pre_import_sample_queries(
        self,
        request: import_config.ImportSampleQueriesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        import_config.ImportSampleQueriesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for import_sample_queries

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def post_import_sample_queries(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for import_sample_queries

        DEPRECATED. Please use the `post_import_sample_queries_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the SampleQueryService server but before
        it is returned to user code. This `post_import_sample_queries` interceptor runs
        before the `post_import_sample_queries_with_metadata` interceptor.
        """
        return response

    def post_import_sample_queries_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for import_sample_queries

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the SampleQueryService server but before it is returned to user code.

        We recommend only using this `post_import_sample_queries_with_metadata`
        interceptor in new development instead of the `post_import_sample_queries` interceptor.
        When both interceptors are used, this `post_import_sample_queries_with_metadata` interceptor runs after the
        `post_import_sample_queries` interceptor. The (possibly modified) response returned by
        `post_import_sample_queries` will be passed to
        `post_import_sample_queries_with_metadata`.
        """
        return response, metadata

    def pre_list_sample_queries(
        self,
        request: sample_query_service.ListSampleQueriesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        sample_query_service.ListSampleQueriesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_sample_queries

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def post_list_sample_queries(
        self, response: sample_query_service.ListSampleQueriesResponse
    ) -> sample_query_service.ListSampleQueriesResponse:
        """Post-rpc interceptor for list_sample_queries

        DEPRECATED. Please use the `post_list_sample_queries_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the SampleQueryService server but before
        it is returned to user code. This `post_list_sample_queries` interceptor runs
        before the `post_list_sample_queries_with_metadata` interceptor.
        """
        return response

    def post_list_sample_queries_with_metadata(
        self,
        response: sample_query_service.ListSampleQueriesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        sample_query_service.ListSampleQueriesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_sample_queries

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the SampleQueryService server but before it is returned to user code.

        We recommend only using this `post_list_sample_queries_with_metadata`
        interceptor in new development instead of the `post_list_sample_queries` interceptor.
        When both interceptors are used, this `post_list_sample_queries_with_metadata` interceptor runs after the
        `post_list_sample_queries` interceptor. The (possibly modified) response returned by
        `post_list_sample_queries` will be passed to
        `post_list_sample_queries_with_metadata`.
        """
        return response, metadata

    def pre_update_sample_query(
        self,
        request: sample_query_service.UpdateSampleQueryRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        sample_query_service.UpdateSampleQueryRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_sample_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def post_update_sample_query(
        self, response: gcd_sample_query.SampleQuery
    ) -> gcd_sample_query.SampleQuery:
        """Post-rpc interceptor for update_sample_query

        DEPRECATED. Please use the `post_update_sample_query_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the SampleQueryService server but before
        it is returned to user code. This `post_update_sample_query` interceptor runs
        before the `post_update_sample_query_with_metadata` interceptor.
        """
        return response

    def post_update_sample_query_with_metadata(
        self,
        response: gcd_sample_query.SampleQuery,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_sample_query.SampleQuery, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_sample_query

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the SampleQueryService server but before it is returned to user code.

        We recommend only using this `post_update_sample_query_with_metadata`
        interceptor in new development instead of the `post_update_sample_query` interceptor.
        When both interceptors are used, this `post_update_sample_query_with_metadata` interceptor runs after the
        `post_update_sample_query` interceptor. The (possibly modified) response returned by
        `post_update_sample_query` will be passed to
        `post_update_sample_query_with_metadata`.
        """
        return response, metadata

    def pre_cancel_operation(
        self,
        request: operations_pb2.CancelOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.CancelOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the SampleQueryService server but before
        it is returned to user code.
        """
        return response

    def pre_get_operation(
        self,
        request: operations_pb2.GetOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.GetOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the SampleQueryService server but before
        it is returned to user code.
        """
        return response

    def pre_list_operations(
        self,
        request: operations_pb2.ListOperationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.ListOperationsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SampleQueryService server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the SampleQueryService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class SampleQueryServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: SampleQueryServiceRestInterceptor


class SampleQueryServiceRestTransport(_BaseSampleQueryServiceRestTransport):
    """REST backend synchronous transport for SampleQueryService.

    Service for managing
    [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery]s,

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "discoveryengine.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[SampleQueryServiceRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'discoveryengine.googleapis.com').
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
        self._operations_client: Optional[operations_v1.AbstractOperationsClient] = None
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or SampleQueryServiceRestInterceptor()
        self._prep_wrapped_messages(client_info)

    @property
    def operations_client(self) -> operations_v1.AbstractOperationsClient:
        """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        # Only create a new client if we do not already have one.
        if self._operations_client is None:
            http_options: Dict[str, List[Dict[str, str]]] = {
                "google.longrunning.Operations.CancelOperation": [
                    {
                        "method": "post",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/branches/*/operations/*}:cancel",
                        "body": "*",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta/{name=projects/*/locations/*/dataStores/*/branches/*/operations/*}:cancel",
                        "body": "*",
                    },
                ],
                "google.longrunning.Operations.GetOperation": [
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataConnector/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/branches/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/models/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/schemas/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine/targetSites/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/engines/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/dataStores/*/branches/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/dataStores/*/models/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/dataStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/evaluations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/sampleQuerySets/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/operations/*}",
                    },
                ],
                "google.longrunning.Operations.ListOperations": [
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataConnector}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/branches/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/models/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/schemas/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine/targetSites}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/dataStores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*/engines/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/collections/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/dataStores/*/branches/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/dataStores/*/models/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*/dataStores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*/locations/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta/{name=projects/*}/operations",
                    },
                ],
            }

            rest_transport = operations_v1.OperationsRestTransport(
                host=self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                scopes=self._scopes,
                http_options=http_options,
                path_prefix="v1beta",
            )

            self._operations_client = operations_v1.AbstractOperationsClient(
                transport=rest_transport
            )

        # Return the client from cache.
        return self._operations_client

    class _CreateSampleQuery(
        _BaseSampleQueryServiceRestTransport._BaseCreateSampleQuery,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.CreateSampleQuery")

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
            request: sample_query_service.CreateSampleQueryRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_sample_query.SampleQuery:
            r"""Call the create sample query method over HTTP.

            Args:
                request (~.sample_query_service.CreateSampleQueryRequest):
                    The request object. Request message for
                [SampleQueryService.CreateSampleQuery][google.cloud.discoveryengine.v1beta.SampleQueryService.CreateSampleQuery]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_sample_query.SampleQuery:
                    Sample Query captures metadata to be
                used for evaluation.

            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseCreateSampleQuery._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_sample_query(
                request, metadata
            )
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseCreateSampleQuery._get_transcoded_request(
                http_options, request
            )

            body = _BaseSampleQueryServiceRestTransport._BaseCreateSampleQuery._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseCreateSampleQuery._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.CreateSampleQuery",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "CreateSampleQuery",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = SampleQueryServiceRestTransport._CreateSampleQuery._get_response(
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
            resp = gcd_sample_query.SampleQuery()
            pb_resp = gcd_sample_query.SampleQuery.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_sample_query(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_sample_query_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_sample_query.SampleQuery.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.create_sample_query",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "CreateSampleQuery",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteSampleQuery(
        _BaseSampleQueryServiceRestTransport._BaseDeleteSampleQuery,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.DeleteSampleQuery")

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
            )
            return response

        def __call__(
            self,
            request: sample_query_service.DeleteSampleQueryRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete sample query method over HTTP.

            Args:
                request (~.sample_query_service.DeleteSampleQueryRequest):
                    The request object. Request message for
                [SampleQueryService.DeleteSampleQuery][google.cloud.discoveryengine.v1beta.SampleQueryService.DeleteSampleQuery]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseDeleteSampleQuery._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_sample_query(
                request, metadata
            )
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseDeleteSampleQuery._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseDeleteSampleQuery._get_query_params_json(
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
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.DeleteSampleQuery",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "DeleteSampleQuery",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = SampleQueryServiceRestTransport._DeleteSampleQuery._get_response(
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

    class _GetSampleQuery(
        _BaseSampleQueryServiceRestTransport._BaseGetSampleQuery,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.GetSampleQuery")

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
            )
            return response

        def __call__(
            self,
            request: sample_query_service.GetSampleQueryRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> sample_query.SampleQuery:
            r"""Call the get sample query method over HTTP.

            Args:
                request (~.sample_query_service.GetSampleQueryRequest):
                    The request object. Request message for
                [SampleQueryService.GetSampleQuery][google.cloud.discoveryengine.v1beta.SampleQueryService.GetSampleQuery]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.sample_query.SampleQuery:
                    Sample Query captures metadata to be
                used for evaluation.

            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseGetSampleQuery._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_sample_query(
                request, metadata
            )
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseGetSampleQuery._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseGetSampleQuery._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.GetSampleQuery",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "GetSampleQuery",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = SampleQueryServiceRestTransport._GetSampleQuery._get_response(
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
            resp = sample_query.SampleQuery()
            pb_resp = sample_query.SampleQuery.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_sample_query(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_sample_query_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = sample_query.SampleQuery.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.get_sample_query",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "GetSampleQuery",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ImportSampleQueries(
        _BaseSampleQueryServiceRestTransport._BaseImportSampleQueries,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.ImportSampleQueries")

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
            request: import_config.ImportSampleQueriesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the import sample queries method over HTTP.

            Args:
                request (~.import_config.ImportSampleQueriesRequest):
                    The request object. Request message for
                [SampleQueryService.ImportSampleQueries][google.cloud.discoveryengine.v1beta.SampleQueryService.ImportSampleQueries]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseImportSampleQueries._get_http_options()
            )

            request, metadata = self._interceptor.pre_import_sample_queries(
                request, metadata
            )
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseImportSampleQueries._get_transcoded_request(
                http_options, request
            )

            body = _BaseSampleQueryServiceRestTransport._BaseImportSampleQueries._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseImportSampleQueries._get_query_params_json(
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
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.ImportSampleQueries",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "ImportSampleQueries",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                SampleQueryServiceRestTransport._ImportSampleQueries._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_import_sample_queries(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_import_sample_queries_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.import_sample_queries",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "ImportSampleQueries",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListSampleQueries(
        _BaseSampleQueryServiceRestTransport._BaseListSampleQueries,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.ListSampleQueries")

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
            )
            return response

        def __call__(
            self,
            request: sample_query_service.ListSampleQueriesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> sample_query_service.ListSampleQueriesResponse:
            r"""Call the list sample queries method over HTTP.

            Args:
                request (~.sample_query_service.ListSampleQueriesRequest):
                    The request object. Request message for
                [SampleQueryService.ListSampleQueries][google.cloud.discoveryengine.v1beta.SampleQueryService.ListSampleQueries]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.sample_query_service.ListSampleQueriesResponse:
                    Response message for
                [SampleQueryService.ListSampleQueries][google.cloud.discoveryengine.v1beta.SampleQueryService.ListSampleQueries]
                method.

            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseListSampleQueries._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_sample_queries(
                request, metadata
            )
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseListSampleQueries._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseListSampleQueries._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.ListSampleQueries",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "ListSampleQueries",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = SampleQueryServiceRestTransport._ListSampleQueries._get_response(
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
            resp = sample_query_service.ListSampleQueriesResponse()
            pb_resp = sample_query_service.ListSampleQueriesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_sample_queries(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_sample_queries_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        sample_query_service.ListSampleQueriesResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.list_sample_queries",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "ListSampleQueries",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateSampleQuery(
        _BaseSampleQueryServiceRestTransport._BaseUpdateSampleQuery,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.UpdateSampleQuery")

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
            request: sample_query_service.UpdateSampleQueryRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_sample_query.SampleQuery:
            r"""Call the update sample query method over HTTP.

            Args:
                request (~.sample_query_service.UpdateSampleQueryRequest):
                    The request object. Request message for
                [SampleQueryService.UpdateSampleQuery][google.cloud.discoveryengine.v1beta.SampleQueryService.UpdateSampleQuery]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_sample_query.SampleQuery:
                    Sample Query captures metadata to be
                used for evaluation.

            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseUpdateSampleQuery._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_sample_query(
                request, metadata
            )
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseUpdateSampleQuery._get_transcoded_request(
                http_options, request
            )

            body = _BaseSampleQueryServiceRestTransport._BaseUpdateSampleQuery._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseUpdateSampleQuery._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.UpdateSampleQuery",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "UpdateSampleQuery",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = SampleQueryServiceRestTransport._UpdateSampleQuery._get_response(
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
            resp = gcd_sample_query.SampleQuery()
            pb_resp = gcd_sample_query.SampleQuery.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_sample_query(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_sample_query_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_sample_query.SampleQuery.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.update_sample_query",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "UpdateSampleQuery",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def create_sample_query(
        self,
    ) -> Callable[
        [sample_query_service.CreateSampleQueryRequest], gcd_sample_query.SampleQuery
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateSampleQuery(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_sample_query(
        self,
    ) -> Callable[[sample_query_service.DeleteSampleQueryRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteSampleQuery(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_sample_query(
        self,
    ) -> Callable[
        [sample_query_service.GetSampleQueryRequest], sample_query.SampleQuery
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetSampleQuery(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def import_sample_queries(
        self,
    ) -> Callable[[import_config.ImportSampleQueriesRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ImportSampleQueries(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_sample_queries(
        self,
    ) -> Callable[
        [sample_query_service.ListSampleQueriesRequest],
        sample_query_service.ListSampleQueriesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListSampleQueries(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_sample_query(
        self,
    ) -> Callable[
        [sample_query_service.UpdateSampleQueryRequest], gcd_sample_query.SampleQuery
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateSampleQuery(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def cancel_operation(self):
        return self._CancelOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _CancelOperation(
        _BaseSampleQueryServiceRestTransport._BaseCancelOperation,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.CancelOperation")

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
            request: operations_pb2.CancelOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> None:
            r"""Call the cancel operation method over HTTP.

            Args:
                request (operations_pb2.CancelOperationRequest):
                    The request object for CancelOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            body = _BaseSampleQueryServiceRestTransport._BaseCancelOperation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseCancelOperation._get_query_params_json(
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
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.CancelOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = SampleQueryServiceRestTransport._CancelOperation._get_response(
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

            return self._interceptor.post_cancel_operation(None)

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BaseSampleQueryServiceRestTransport._BaseGetOperation,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.GetOperation")

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
            )
            return response

        def __call__(
            self,
            request: operations_pb2.GetOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the get operation method over HTTP.

            Args:
                request (operations_pb2.GetOperationRequest):
                    The request object for GetOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                operations_pb2.Operation: Response from GetOperation method.
            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseGetOperation._get_query_params_json(
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
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = SampleQueryServiceRestTransport._GetOperation._get_response(
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

            content = response.content.decode("utf-8")
            resp = operations_pb2.Operation()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_get_operation(resp)
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.SampleQueryServiceAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "GetOperation",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def list_operations(self):
        return self._ListOperations(self._session, self._host, self._interceptor)  # type: ignore

    class _ListOperations(
        _BaseSampleQueryServiceRestTransport._BaseListOperations,
        SampleQueryServiceRestStub,
    ):
        def __hash__(self):
            return hash("SampleQueryServiceRestTransport.ListOperations")

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
            )
            return response

        def __call__(
            self,
            request: operations_pb2.ListOperationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.ListOperationsResponse:
            r"""Call the list operations method over HTTP.

            Args:
                request (operations_pb2.ListOperationsRequest):
                    The request object for ListOperations method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                operations_pb2.ListOperationsResponse: Response from ListOperations method.
            """

            http_options = (
                _BaseSampleQueryServiceRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseSampleQueryServiceRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseSampleQueryServiceRestTransport._BaseListOperations._get_query_params_json(
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
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for google.cloud.discoveryengine_v1beta.SampleQueryServiceClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = SampleQueryServiceRestTransport._ListOperations._get_response(
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

            content = response.content.decode("utf-8")
            resp = operations_pb2.ListOperationsResponse()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_list_operations(resp)
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = json_format.MessageToJson(resp)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.SampleQueryServiceAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.SampleQueryService",
                        "rpcName": "ListOperations",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ("SampleQueryServiceRestTransport",)
