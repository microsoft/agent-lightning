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
from google.protobuf import json_format
from requests import __version__ as requests_version

from google.cloud.discoveryengine_v1alpha.types import data_store as gcd_data_store
from google.cloud.discoveryengine_v1alpha.types import document_processing_config
from google.cloud.discoveryengine_v1alpha.types import (
    document_processing_config as gcd_document_processing_config,
)
from google.cloud.discoveryengine_v1alpha.types import data_store
from google.cloud.discoveryengine_v1alpha.types import data_store_service

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseDataStoreServiceRestTransport

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


class DataStoreServiceRestInterceptor:
    """Interceptor for DataStoreService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the DataStoreServiceRestTransport.

    .. code-block:: python
        class MyCustomDataStoreServiceInterceptor(DataStoreServiceRestInterceptor):
            def pre_create_data_store(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_data_store(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_data_store(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_delete_data_store(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_data_store(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_data_store(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_document_processing_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_document_processing_config(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_data_stores(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_data_stores(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_data_store(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_data_store(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_document_processing_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_document_processing_config(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = DataStoreServiceRestTransport(interceptor=MyCustomDataStoreServiceInterceptor())
        client = DataStoreServiceClient(transport=transport)


    """

    def pre_create_data_store(
        self,
        request: data_store_service.CreateDataStoreRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        data_store_service.CreateDataStoreRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_data_store

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_create_data_store(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_data_store

        DEPRECATED. Please use the `post_create_data_store_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DataStoreService server but before
        it is returned to user code. This `post_create_data_store` interceptor runs
        before the `post_create_data_store_with_metadata` interceptor.
        """
        return response

    def post_create_data_store_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_data_store

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DataStoreService server but before it is returned to user code.

        We recommend only using this `post_create_data_store_with_metadata`
        interceptor in new development instead of the `post_create_data_store` interceptor.
        When both interceptors are used, this `post_create_data_store_with_metadata` interceptor runs after the
        `post_create_data_store` interceptor. The (possibly modified) response returned by
        `post_create_data_store` will be passed to
        `post_create_data_store_with_metadata`.
        """
        return response, metadata

    def pre_delete_data_store(
        self,
        request: data_store_service.DeleteDataStoreRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        data_store_service.DeleteDataStoreRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_data_store

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_delete_data_store(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_data_store

        DEPRECATED. Please use the `post_delete_data_store_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DataStoreService server but before
        it is returned to user code. This `post_delete_data_store` interceptor runs
        before the `post_delete_data_store_with_metadata` interceptor.
        """
        return response

    def post_delete_data_store_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_data_store

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DataStoreService server but before it is returned to user code.

        We recommend only using this `post_delete_data_store_with_metadata`
        interceptor in new development instead of the `post_delete_data_store` interceptor.
        When both interceptors are used, this `post_delete_data_store_with_metadata` interceptor runs after the
        `post_delete_data_store` interceptor. The (possibly modified) response returned by
        `post_delete_data_store` will be passed to
        `post_delete_data_store_with_metadata`.
        """
        return response, metadata

    def pre_get_data_store(
        self,
        request: data_store_service.GetDataStoreRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        data_store_service.GetDataStoreRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_data_store

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_get_data_store(
        self, response: data_store.DataStore
    ) -> data_store.DataStore:
        """Post-rpc interceptor for get_data_store

        DEPRECATED. Please use the `post_get_data_store_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DataStoreService server but before
        it is returned to user code. This `post_get_data_store` interceptor runs
        before the `post_get_data_store_with_metadata` interceptor.
        """
        return response

    def post_get_data_store_with_metadata(
        self,
        response: data_store.DataStore,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[data_store.DataStore, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_data_store

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DataStoreService server but before it is returned to user code.

        We recommend only using this `post_get_data_store_with_metadata`
        interceptor in new development instead of the `post_get_data_store` interceptor.
        When both interceptors are used, this `post_get_data_store_with_metadata` interceptor runs after the
        `post_get_data_store` interceptor. The (possibly modified) response returned by
        `post_get_data_store` will be passed to
        `post_get_data_store_with_metadata`.
        """
        return response, metadata

    def pre_get_document_processing_config(
        self,
        request: data_store_service.GetDocumentProcessingConfigRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        data_store_service.GetDocumentProcessingConfigRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_document_processing_config

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_get_document_processing_config(
        self, response: document_processing_config.DocumentProcessingConfig
    ) -> document_processing_config.DocumentProcessingConfig:
        """Post-rpc interceptor for get_document_processing_config

        DEPRECATED. Please use the `post_get_document_processing_config_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DataStoreService server but before
        it is returned to user code. This `post_get_document_processing_config` interceptor runs
        before the `post_get_document_processing_config_with_metadata` interceptor.
        """
        return response

    def post_get_document_processing_config_with_metadata(
        self,
        response: document_processing_config.DocumentProcessingConfig,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_processing_config.DocumentProcessingConfig,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for get_document_processing_config

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DataStoreService server but before it is returned to user code.

        We recommend only using this `post_get_document_processing_config_with_metadata`
        interceptor in new development instead of the `post_get_document_processing_config` interceptor.
        When both interceptors are used, this `post_get_document_processing_config_with_metadata` interceptor runs after the
        `post_get_document_processing_config` interceptor. The (possibly modified) response returned by
        `post_get_document_processing_config` will be passed to
        `post_get_document_processing_config_with_metadata`.
        """
        return response, metadata

    def pre_list_data_stores(
        self,
        request: data_store_service.ListDataStoresRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        data_store_service.ListDataStoresRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_data_stores

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_list_data_stores(
        self, response: data_store_service.ListDataStoresResponse
    ) -> data_store_service.ListDataStoresResponse:
        """Post-rpc interceptor for list_data_stores

        DEPRECATED. Please use the `post_list_data_stores_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DataStoreService server but before
        it is returned to user code. This `post_list_data_stores` interceptor runs
        before the `post_list_data_stores_with_metadata` interceptor.
        """
        return response

    def post_list_data_stores_with_metadata(
        self,
        response: data_store_service.ListDataStoresResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        data_store_service.ListDataStoresResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_data_stores

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DataStoreService server but before it is returned to user code.

        We recommend only using this `post_list_data_stores_with_metadata`
        interceptor in new development instead of the `post_list_data_stores` interceptor.
        When both interceptors are used, this `post_list_data_stores_with_metadata` interceptor runs after the
        `post_list_data_stores` interceptor. The (possibly modified) response returned by
        `post_list_data_stores` will be passed to
        `post_list_data_stores_with_metadata`.
        """
        return response, metadata

    def pre_update_data_store(
        self,
        request: data_store_service.UpdateDataStoreRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        data_store_service.UpdateDataStoreRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_data_store

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_update_data_store(
        self, response: gcd_data_store.DataStore
    ) -> gcd_data_store.DataStore:
        """Post-rpc interceptor for update_data_store

        DEPRECATED. Please use the `post_update_data_store_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DataStoreService server but before
        it is returned to user code. This `post_update_data_store` interceptor runs
        before the `post_update_data_store_with_metadata` interceptor.
        """
        return response

    def post_update_data_store_with_metadata(
        self,
        response: gcd_data_store.DataStore,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_data_store.DataStore, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_data_store

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DataStoreService server but before it is returned to user code.

        We recommend only using this `post_update_data_store_with_metadata`
        interceptor in new development instead of the `post_update_data_store` interceptor.
        When both interceptors are used, this `post_update_data_store_with_metadata` interceptor runs after the
        `post_update_data_store` interceptor. The (possibly modified) response returned by
        `post_update_data_store` will be passed to
        `post_update_data_store_with_metadata`.
        """
        return response, metadata

    def pre_update_document_processing_config(
        self,
        request: data_store_service.UpdateDocumentProcessingConfigRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        data_store_service.UpdateDocumentProcessingConfigRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_document_processing_config

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_update_document_processing_config(
        self, response: gcd_document_processing_config.DocumentProcessingConfig
    ) -> gcd_document_processing_config.DocumentProcessingConfig:
        """Post-rpc interceptor for update_document_processing_config

        DEPRECATED. Please use the `post_update_document_processing_config_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DataStoreService server but before
        it is returned to user code. This `post_update_document_processing_config` interceptor runs
        before the `post_update_document_processing_config_with_metadata` interceptor.
        """
        return response

    def post_update_document_processing_config_with_metadata(
        self,
        response: gcd_document_processing_config.DocumentProcessingConfig,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gcd_document_processing_config.DocumentProcessingConfig,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for update_document_processing_config

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DataStoreService server but before it is returned to user code.

        We recommend only using this `post_update_document_processing_config_with_metadata`
        interceptor in new development instead of the `post_update_document_processing_config` interceptor.
        When both interceptors are used, this `post_update_document_processing_config_with_metadata` interceptor runs after the
        `post_update_document_processing_config` interceptor. The (possibly modified) response returned by
        `post_update_document_processing_config` will be passed to
        `post_update_document_processing_config_with_metadata`.
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
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the DataStoreService server but before
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
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the DataStoreService server but before
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
        before they are sent to the DataStoreService server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the DataStoreService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class DataStoreServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: DataStoreServiceRestInterceptor


class DataStoreServiceRestTransport(_BaseDataStoreServiceRestTransport):
    """REST backend synchronous transport for DataStoreService.

    Service for managing
    [DataStore][google.cloud.discoveryengine.v1alpha.DataStore]
    configuration.

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
        interceptor: Optional[DataStoreServiceRestInterceptor] = None,
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
        self._interceptor = interceptor or DataStoreServiceRestInterceptor()
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
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/branches/*/operations/*}:cancel",
                        "body": "*",
                    },
                    {
                        "method": "post",
                        "uri": "/v1alpha/{name=projects/*/locations/*/dataStores/*/branches/*/operations/*}:cancel",
                        "body": "*",
                    },
                ],
                "google.longrunning.Operations.GetOperation": [
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataConnector/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/branches/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/models/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/schemas/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine/targetSites/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/engines/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/dataStores/*/branches/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/dataStores/*/models/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/dataStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/evaluations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/identity_mapping_stores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/sampleQuerySets/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/operations/*}",
                    },
                ],
                "google.longrunning.Operations.ListOperations": [
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataConnector}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/branches/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/models/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/schemas/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine/targetSites}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*/siteSearchEngine}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/dataStores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*/engines/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/collections/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/dataStores/*/branches/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/dataStores/*/models/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/dataStores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*/identity_mapping_stores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*/locations/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1alpha/{name=projects/*}/operations",
                    },
                ],
            }

            rest_transport = operations_v1.OperationsRestTransport(
                host=self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                scopes=self._scopes,
                http_options=http_options,
                path_prefix="v1alpha",
            )

            self._operations_client = operations_v1.AbstractOperationsClient(
                transport=rest_transport
            )

        # Return the client from cache.
        return self._operations_client

    class _CreateDataStore(
        _BaseDataStoreServiceRestTransport._BaseCreateDataStore,
        DataStoreServiceRestStub,
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.CreateDataStore")

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
            request: data_store_service.CreateDataStoreRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create data store method over HTTP.

            Args:
                request (~.data_store_service.CreateDataStoreRequest):
                    The request object. Request for
                [DataStoreService.CreateDataStore][google.cloud.discoveryengine.v1alpha.DataStoreService.CreateDataStore]
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
                _BaseDataStoreServiceRestTransport._BaseCreateDataStore._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_data_store(
                request, metadata
            )
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseCreateDataStore._get_transcoded_request(
                http_options, request
            )

            body = _BaseDataStoreServiceRestTransport._BaseCreateDataStore._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseCreateDataStore._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.CreateDataStore",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "CreateDataStore",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._CreateDataStore._get_response(
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
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_data_store(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_data_store_with_metadata(
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
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.create_data_store",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "CreateDataStore",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteDataStore(
        _BaseDataStoreServiceRestTransport._BaseDeleteDataStore,
        DataStoreServiceRestStub,
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.DeleteDataStore")

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
            request: data_store_service.DeleteDataStoreRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete data store method over HTTP.

            Args:
                request (~.data_store_service.DeleteDataStoreRequest):
                    The request object. Request message for
                [DataStoreService.DeleteDataStore][google.cloud.discoveryengine.v1alpha.DataStoreService.DeleteDataStore]
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
                _BaseDataStoreServiceRestTransport._BaseDeleteDataStore._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_data_store(
                request, metadata
            )
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseDeleteDataStore._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseDeleteDataStore._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.DeleteDataStore",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "DeleteDataStore",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._DeleteDataStore._get_response(
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
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_delete_data_store(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_delete_data_store_with_metadata(
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
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.delete_data_store",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "DeleteDataStore",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetDataStore(
        _BaseDataStoreServiceRestTransport._BaseGetDataStore, DataStoreServiceRestStub
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.GetDataStore")

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
            request: data_store_service.GetDataStoreRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> data_store.DataStore:
            r"""Call the get data store method over HTTP.

            Args:
                request (~.data_store_service.GetDataStoreRequest):
                    The request object. Request message for
                [DataStoreService.GetDataStore][google.cloud.discoveryengine.v1alpha.DataStoreService.GetDataStore]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.data_store.DataStore:
                    DataStore captures global settings
                and configs at the DataStore level.

            """

            http_options = (
                _BaseDataStoreServiceRestTransport._BaseGetDataStore._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_data_store(request, metadata)
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseGetDataStore._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseGetDataStore._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.GetDataStore",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "GetDataStore",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._GetDataStore._get_response(
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
            resp = data_store.DataStore()
            pb_resp = data_store.DataStore.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_data_store(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_data_store_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = data_store.DataStore.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.get_data_store",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "GetDataStore",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetDocumentProcessingConfig(
        _BaseDataStoreServiceRestTransport._BaseGetDocumentProcessingConfig,
        DataStoreServiceRestStub,
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.GetDocumentProcessingConfig")

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
            request: data_store_service.GetDocumentProcessingConfigRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> document_processing_config.DocumentProcessingConfig:
            r"""Call the get document processing
            config method over HTTP.

                Args:
                    request (~.data_store_service.GetDocumentProcessingConfigRequest):
                        The request object. Request for
                    [DataStoreService.GetDocumentProcessingConfig][google.cloud.discoveryengine.v1alpha.DataStoreService.GetDocumentProcessingConfig]
                    method.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.document_processing_config.DocumentProcessingConfig:
                        A singleton resource of
                    [DataStore][google.cloud.discoveryengine.v1alpha.DataStore].
                    It's empty when
                    [DataStore][google.cloud.discoveryengine.v1alpha.DataStore]
                    is created, which defaults to digital parser. The first
                    call to
                    [DataStoreService.UpdateDocumentProcessingConfig][google.cloud.discoveryengine.v1alpha.DataStoreService.UpdateDocumentProcessingConfig]
                    method will initialize the config.

            """

            http_options = (
                _BaseDataStoreServiceRestTransport._BaseGetDocumentProcessingConfig._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_document_processing_config(
                request, metadata
            )
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseGetDocumentProcessingConfig._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseGetDocumentProcessingConfig._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.GetDocumentProcessingConfig",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "GetDocumentProcessingConfig",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._GetDocumentProcessingConfig._get_response(
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
            resp = document_processing_config.DocumentProcessingConfig()
            pb_resp = document_processing_config.DocumentProcessingConfig.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_document_processing_config(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_get_document_processing_config_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        document_processing_config.DocumentProcessingConfig.to_json(
                            response
                        )
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.get_document_processing_config",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "GetDocumentProcessingConfig",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListDataStores(
        _BaseDataStoreServiceRestTransport._BaseListDataStores, DataStoreServiceRestStub
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.ListDataStores")

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
            request: data_store_service.ListDataStoresRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> data_store_service.ListDataStoresResponse:
            r"""Call the list data stores method over HTTP.

            Args:
                request (~.data_store_service.ListDataStoresRequest):
                    The request object. Request message for
                [DataStoreService.ListDataStores][google.cloud.discoveryengine.v1alpha.DataStoreService.ListDataStores]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.data_store_service.ListDataStoresResponse:
                    Response message for
                [DataStoreService.ListDataStores][google.cloud.discoveryengine.v1alpha.DataStoreService.ListDataStores]
                method.

            """

            http_options = (
                _BaseDataStoreServiceRestTransport._BaseListDataStores._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_data_stores(
                request, metadata
            )
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseListDataStores._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseListDataStores._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.ListDataStores",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "ListDataStores",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._ListDataStores._get_response(
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
            resp = data_store_service.ListDataStoresResponse()
            pb_resp = data_store_service.ListDataStoresResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_data_stores(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_data_stores_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        data_store_service.ListDataStoresResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.list_data_stores",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "ListDataStores",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateDataStore(
        _BaseDataStoreServiceRestTransport._BaseUpdateDataStore,
        DataStoreServiceRestStub,
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.UpdateDataStore")

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
            request: data_store_service.UpdateDataStoreRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_data_store.DataStore:
            r"""Call the update data store method over HTTP.

            Args:
                request (~.data_store_service.UpdateDataStoreRequest):
                    The request object. Request message for
                [DataStoreService.UpdateDataStore][google.cloud.discoveryengine.v1alpha.DataStoreService.UpdateDataStore]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_data_store.DataStore:
                    DataStore captures global settings
                and configs at the DataStore level.

            """

            http_options = (
                _BaseDataStoreServiceRestTransport._BaseUpdateDataStore._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_data_store(
                request, metadata
            )
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseUpdateDataStore._get_transcoded_request(
                http_options, request
            )

            body = _BaseDataStoreServiceRestTransport._BaseUpdateDataStore._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseUpdateDataStore._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.UpdateDataStore",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "UpdateDataStore",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._UpdateDataStore._get_response(
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
            resp = gcd_data_store.DataStore()
            pb_resp = gcd_data_store.DataStore.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_data_store(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_data_store_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_data_store.DataStore.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.update_data_store",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "UpdateDataStore",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateDocumentProcessingConfig(
        _BaseDataStoreServiceRestTransport._BaseUpdateDocumentProcessingConfig,
        DataStoreServiceRestStub,
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.UpdateDocumentProcessingConfig")

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
            request: data_store_service.UpdateDocumentProcessingConfigRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_document_processing_config.DocumentProcessingConfig:
            r"""Call the update document
            processing config method over HTTP.

                Args:
                    request (~.data_store_service.UpdateDocumentProcessingConfigRequest):
                        The request object. Request for
                    [DataStoreService.UpdateDocumentProcessingConfig][google.cloud.discoveryengine.v1alpha.DataStoreService.UpdateDocumentProcessingConfig]
                    method.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.gcd_document_processing_config.DocumentProcessingConfig:
                        A singleton resource of
                    [DataStore][google.cloud.discoveryengine.v1alpha.DataStore].
                    It's empty when
                    [DataStore][google.cloud.discoveryengine.v1alpha.DataStore]
                    is created, which defaults to digital parser. The first
                    call to
                    [DataStoreService.UpdateDocumentProcessingConfig][google.cloud.discoveryengine.v1alpha.DataStoreService.UpdateDocumentProcessingConfig]
                    method will initialize the config.

            """

            http_options = (
                _BaseDataStoreServiceRestTransport._BaseUpdateDocumentProcessingConfig._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_document_processing_config(
                request, metadata
            )
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseUpdateDocumentProcessingConfig._get_transcoded_request(
                http_options, request
            )

            body = _BaseDataStoreServiceRestTransport._BaseUpdateDocumentProcessingConfig._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseUpdateDocumentProcessingConfig._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.UpdateDocumentProcessingConfig",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "UpdateDocumentProcessingConfig",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._UpdateDocumentProcessingConfig._get_response(
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
            resp = gcd_document_processing_config.DocumentProcessingConfig()
            pb_resp = gcd_document_processing_config.DocumentProcessingConfig.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_document_processing_config(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_update_document_processing_config_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        gcd_document_processing_config.DocumentProcessingConfig.to_json(
                            response
                        )
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.update_document_processing_config",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "UpdateDocumentProcessingConfig",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def create_data_store(
        self,
    ) -> Callable[
        [data_store_service.CreateDataStoreRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateDataStore(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_data_store(
        self,
    ) -> Callable[
        [data_store_service.DeleteDataStoreRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteDataStore(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_data_store(
        self,
    ) -> Callable[[data_store_service.GetDataStoreRequest], data_store.DataStore]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetDataStore(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_document_processing_config(
        self,
    ) -> Callable[
        [data_store_service.GetDocumentProcessingConfigRequest],
        document_processing_config.DocumentProcessingConfig,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetDocumentProcessingConfig(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_data_stores(
        self,
    ) -> Callable[
        [data_store_service.ListDataStoresRequest],
        data_store_service.ListDataStoresResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListDataStores(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_data_store(
        self,
    ) -> Callable[
        [data_store_service.UpdateDataStoreRequest], gcd_data_store.DataStore
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateDataStore(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_document_processing_config(
        self,
    ) -> Callable[
        [data_store_service.UpdateDocumentProcessingConfigRequest],
        gcd_document_processing_config.DocumentProcessingConfig,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateDocumentProcessingConfig(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def cancel_operation(self):
        return self._CancelOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _CancelOperation(
        _BaseDataStoreServiceRestTransport._BaseCancelOperation,
        DataStoreServiceRestStub,
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.CancelOperation")

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
                _BaseDataStoreServiceRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            body = _BaseDataStoreServiceRestTransport._BaseCancelOperation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseCancelOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.CancelOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._CancelOperation._get_response(
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
        _BaseDataStoreServiceRestTransport._BaseGetOperation, DataStoreServiceRestStub
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.GetOperation")

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
                _BaseDataStoreServiceRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._GetOperation._get_response(
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
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
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
        _BaseDataStoreServiceRestTransport._BaseListOperations, DataStoreServiceRestStub
    ):
        def __hash__(self):
            return hash("DataStoreServiceRestTransport.ListOperations")

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
                _BaseDataStoreServiceRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseDataStoreServiceRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDataStoreServiceRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.DataStoreServiceClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DataStoreServiceRestTransport._ListOperations._get_response(
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
                    "Received response for google.cloud.discoveryengine_v1alpha.DataStoreServiceAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.DataStoreService",
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


__all__ = ("DataStoreServiceRestTransport",)
