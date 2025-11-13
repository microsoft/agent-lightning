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

from google.cloud.discoveryengine_v1beta.types import (
    document_service,
    import_config,
    purge_config,
)
from google.cloud.discoveryengine_v1beta.types import document
from google.cloud.discoveryengine_v1beta.types import document as gcd_document

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseDocumentServiceRestTransport

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


class DocumentServiceRestInterceptor:
    """Interceptor for DocumentService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the DocumentServiceRestTransport.

    .. code-block:: python
        class MyCustomDocumentServiceInterceptor(DocumentServiceRestInterceptor):
            def pre_batch_get_documents_metadata(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_batch_get_documents_metadata(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_document(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_document(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_import_documents(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_import_documents(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_documents(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_documents(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_purge_documents(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_purge_documents(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_document(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_document(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = DocumentServiceRestTransport(interceptor=MyCustomDocumentServiceInterceptor())
        client = DocumentServiceClient(transport=transport)


    """

    def pre_batch_get_documents_metadata(
        self,
        request: document_service.BatchGetDocumentsMetadataRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_service.BatchGetDocumentsMetadataRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for batch_get_documents_metadata

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_batch_get_documents_metadata(
        self, response: document_service.BatchGetDocumentsMetadataResponse
    ) -> document_service.BatchGetDocumentsMetadataResponse:
        """Post-rpc interceptor for batch_get_documents_metadata

        DEPRECATED. Please use the `post_batch_get_documents_metadata_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DocumentService server but before
        it is returned to user code. This `post_batch_get_documents_metadata` interceptor runs
        before the `post_batch_get_documents_metadata_with_metadata` interceptor.
        """
        return response

    def post_batch_get_documents_metadata_with_metadata(
        self,
        response: document_service.BatchGetDocumentsMetadataResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_service.BatchGetDocumentsMetadataResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for batch_get_documents_metadata

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DocumentService server but before it is returned to user code.

        We recommend only using this `post_batch_get_documents_metadata_with_metadata`
        interceptor in new development instead of the `post_batch_get_documents_metadata` interceptor.
        When both interceptors are used, this `post_batch_get_documents_metadata_with_metadata` interceptor runs after the
        `post_batch_get_documents_metadata` interceptor. The (possibly modified) response returned by
        `post_batch_get_documents_metadata` will be passed to
        `post_batch_get_documents_metadata_with_metadata`.
        """
        return response, metadata

    def pre_create_document(
        self,
        request: document_service.CreateDocumentRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_service.CreateDocumentRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for create_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_create_document(
        self, response: gcd_document.Document
    ) -> gcd_document.Document:
        """Post-rpc interceptor for create_document

        DEPRECATED. Please use the `post_create_document_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DocumentService server but before
        it is returned to user code. This `post_create_document` interceptor runs
        before the `post_create_document_with_metadata` interceptor.
        """
        return response

    def post_create_document_with_metadata(
        self,
        response: gcd_document.Document,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_document.Document, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_document

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DocumentService server but before it is returned to user code.

        We recommend only using this `post_create_document_with_metadata`
        interceptor in new development instead of the `post_create_document` interceptor.
        When both interceptors are used, this `post_create_document_with_metadata` interceptor runs after the
        `post_create_document` interceptor. The (possibly modified) response returned by
        `post_create_document` will be passed to
        `post_create_document_with_metadata`.
        """
        return response, metadata

    def pre_delete_document(
        self,
        request: document_service.DeleteDocumentRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_service.DeleteDocumentRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def pre_get_document(
        self,
        request: document_service.GetDocumentRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_service.GetDocumentRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_get_document(self, response: document.Document) -> document.Document:
        """Post-rpc interceptor for get_document

        DEPRECATED. Please use the `post_get_document_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DocumentService server but before
        it is returned to user code. This `post_get_document` interceptor runs
        before the `post_get_document_with_metadata` interceptor.
        """
        return response

    def post_get_document_with_metadata(
        self,
        response: document.Document,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[document.Document, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_document

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DocumentService server but before it is returned to user code.

        We recommend only using this `post_get_document_with_metadata`
        interceptor in new development instead of the `post_get_document` interceptor.
        When both interceptors are used, this `post_get_document_with_metadata` interceptor runs after the
        `post_get_document` interceptor. The (possibly modified) response returned by
        `post_get_document` will be passed to
        `post_get_document_with_metadata`.
        """
        return response, metadata

    def pre_import_documents(
        self,
        request: import_config.ImportDocumentsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        import_config.ImportDocumentsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for import_documents

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_import_documents(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for import_documents

        DEPRECATED. Please use the `post_import_documents_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DocumentService server but before
        it is returned to user code. This `post_import_documents` interceptor runs
        before the `post_import_documents_with_metadata` interceptor.
        """
        return response

    def post_import_documents_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for import_documents

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DocumentService server but before it is returned to user code.

        We recommend only using this `post_import_documents_with_metadata`
        interceptor in new development instead of the `post_import_documents` interceptor.
        When both interceptors are used, this `post_import_documents_with_metadata` interceptor runs after the
        `post_import_documents` interceptor. The (possibly modified) response returned by
        `post_import_documents` will be passed to
        `post_import_documents_with_metadata`.
        """
        return response, metadata

    def pre_list_documents(
        self,
        request: document_service.ListDocumentsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_service.ListDocumentsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_documents

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_list_documents(
        self, response: document_service.ListDocumentsResponse
    ) -> document_service.ListDocumentsResponse:
        """Post-rpc interceptor for list_documents

        DEPRECATED. Please use the `post_list_documents_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DocumentService server but before
        it is returned to user code. This `post_list_documents` interceptor runs
        before the `post_list_documents_with_metadata` interceptor.
        """
        return response

    def post_list_documents_with_metadata(
        self,
        response: document_service.ListDocumentsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_service.ListDocumentsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_documents

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DocumentService server but before it is returned to user code.

        We recommend only using this `post_list_documents_with_metadata`
        interceptor in new development instead of the `post_list_documents` interceptor.
        When both interceptors are used, this `post_list_documents_with_metadata` interceptor runs after the
        `post_list_documents` interceptor. The (possibly modified) response returned by
        `post_list_documents` will be passed to
        `post_list_documents_with_metadata`.
        """
        return response, metadata

    def pre_purge_documents(
        self,
        request: purge_config.PurgeDocumentsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        purge_config.PurgeDocumentsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for purge_documents

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_purge_documents(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for purge_documents

        DEPRECATED. Please use the `post_purge_documents_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DocumentService server but before
        it is returned to user code. This `post_purge_documents` interceptor runs
        before the `post_purge_documents_with_metadata` interceptor.
        """
        return response

    def post_purge_documents_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for purge_documents

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DocumentService server but before it is returned to user code.

        We recommend only using this `post_purge_documents_with_metadata`
        interceptor in new development instead of the `post_purge_documents` interceptor.
        When both interceptors are used, this `post_purge_documents_with_metadata` interceptor runs after the
        `post_purge_documents` interceptor. The (possibly modified) response returned by
        `post_purge_documents` will be passed to
        `post_purge_documents_with_metadata`.
        """
        return response, metadata

    def pre_update_document(
        self,
        request: document_service.UpdateDocumentRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        document_service.UpdateDocumentRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for update_document

        Override in a subclass to manipulate the request or metadata
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_update_document(
        self, response: gcd_document.Document
    ) -> gcd_document.Document:
        """Post-rpc interceptor for update_document

        DEPRECATED. Please use the `post_update_document_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the DocumentService server but before
        it is returned to user code. This `post_update_document` interceptor runs
        before the `post_update_document_with_metadata` interceptor.
        """
        return response

    def post_update_document_with_metadata(
        self,
        response: gcd_document.Document,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_document.Document, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_document

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the DocumentService server but before it is returned to user code.

        We recommend only using this `post_update_document_with_metadata`
        interceptor in new development instead of the `post_update_document` interceptor.
        When both interceptors are used, this `post_update_document_with_metadata` interceptor runs after the
        `post_update_document` interceptor. The (possibly modified) response returned by
        `post_update_document` will be passed to
        `post_update_document_with_metadata`.
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
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the DocumentService server but before
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
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the DocumentService server but before
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
        before they are sent to the DocumentService server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the DocumentService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class DocumentServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: DocumentServiceRestInterceptor


class DocumentServiceRestTransport(_BaseDocumentServiceRestTransport):
    """REST backend synchronous transport for DocumentService.

    Service for ingesting
    [Document][google.cloud.discoveryengine.v1beta.Document] information
    of the customer's website.

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
        interceptor: Optional[DocumentServiceRestInterceptor] = None,
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
        self._interceptor = interceptor or DocumentServiceRestInterceptor()
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

    class _BatchGetDocumentsMetadata(
        _BaseDocumentServiceRestTransport._BaseBatchGetDocumentsMetadata,
        DocumentServiceRestStub,
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.BatchGetDocumentsMetadata")

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
            request: document_service.BatchGetDocumentsMetadataRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> document_service.BatchGetDocumentsMetadataResponse:
            r"""Call the batch get documents
            metadata method over HTTP.

                Args:
                    request (~.document_service.BatchGetDocumentsMetadataRequest):
                        The request object. Request message for
                    [DocumentService.BatchGetDocumentsMetadata][google.cloud.discoveryengine.v1beta.DocumentService.BatchGetDocumentsMetadata]
                    method.
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.document_service.BatchGetDocumentsMetadataResponse:
                        Response message for
                    [DocumentService.BatchGetDocumentsMetadata][google.cloud.discoveryengine.v1beta.DocumentService.BatchGetDocumentsMetadata]
                    method.

            """

            http_options = (
                _BaseDocumentServiceRestTransport._BaseBatchGetDocumentsMetadata._get_http_options()
            )

            request, metadata = self._interceptor.pre_batch_get_documents_metadata(
                request, metadata
            )
            transcoded_request = _BaseDocumentServiceRestTransport._BaseBatchGetDocumentsMetadata._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseBatchGetDocumentsMetadata._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.BatchGetDocumentsMetadata",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "BatchGetDocumentsMetadata",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                DocumentServiceRestTransport._BatchGetDocumentsMetadata._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                )
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = document_service.BatchGetDocumentsMetadataResponse()
            pb_resp = document_service.BatchGetDocumentsMetadataResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_batch_get_documents_metadata(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_batch_get_documents_metadata_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        document_service.BatchGetDocumentsMetadataResponse.to_json(
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
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceClient.batch_get_documents_metadata",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "BatchGetDocumentsMetadata",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateDocument(
        _BaseDocumentServiceRestTransport._BaseCreateDocument, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.CreateDocument")

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
            request: document_service.CreateDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_document.Document:
            r"""Call the create document method over HTTP.

            Args:
                request (~.document_service.CreateDocumentRequest):
                    The request object. Request message for
                [DocumentService.CreateDocument][google.cloud.discoveryengine.v1beta.DocumentService.CreateDocument]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_document.Document:
                    Document captures all raw metadata
                information of items to be recommended
                or searched.

            """

            http_options = (
                _BaseDocumentServiceRestTransport._BaseCreateDocument._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_document(request, metadata)
            transcoded_request = _BaseDocumentServiceRestTransport._BaseCreateDocument._get_transcoded_request(
                http_options, request
            )

            body = _BaseDocumentServiceRestTransport._BaseCreateDocument._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseCreateDocument._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.CreateDocument",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "CreateDocument",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._CreateDocument._get_response(
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
            resp = gcd_document.Document()
            pb_resp = gcd_document.Document.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_document(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_document_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_document.Document.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceClient.create_document",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "CreateDocument",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteDocument(
        _BaseDocumentServiceRestTransport._BaseDeleteDocument, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.DeleteDocument")

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
            request: document_service.DeleteDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete document method over HTTP.

            Args:
                request (~.document_service.DeleteDocumentRequest):
                    The request object. Request message for
                [DocumentService.DeleteDocument][google.cloud.discoveryengine.v1beta.DocumentService.DeleteDocument]
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
                _BaseDocumentServiceRestTransport._BaseDeleteDocument._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_document(request, metadata)
            transcoded_request = _BaseDocumentServiceRestTransport._BaseDeleteDocument._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseDeleteDocument._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.DeleteDocument",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "DeleteDocument",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._DeleteDocument._get_response(
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

    class _GetDocument(
        _BaseDocumentServiceRestTransport._BaseGetDocument, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.GetDocument")

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
            request: document_service.GetDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> document.Document:
            r"""Call the get document method over HTTP.

            Args:
                request (~.document_service.GetDocumentRequest):
                    The request object. Request message for
                [DocumentService.GetDocument][google.cloud.discoveryengine.v1beta.DocumentService.GetDocument]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.document.Document:
                    Document captures all raw metadata
                information of items to be recommended
                or searched.

            """

            http_options = (
                _BaseDocumentServiceRestTransport._BaseGetDocument._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_document(request, metadata)
            transcoded_request = _BaseDocumentServiceRestTransport._BaseGetDocument._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseGetDocument._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.GetDocument",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "GetDocument",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._GetDocument._get_response(
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
            resp = document.Document()
            pb_resp = document.Document.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_document(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_document_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = document.Document.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceClient.get_document",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "GetDocument",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ImportDocuments(
        _BaseDocumentServiceRestTransport._BaseImportDocuments, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.ImportDocuments")

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
            request: import_config.ImportDocumentsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the import documents method over HTTP.

            Args:
                request (~.import_config.ImportDocumentsRequest):
                    The request object. Request message for Import methods.
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
                _BaseDocumentServiceRestTransport._BaseImportDocuments._get_http_options()
            )

            request, metadata = self._interceptor.pre_import_documents(
                request, metadata
            )
            transcoded_request = _BaseDocumentServiceRestTransport._BaseImportDocuments._get_transcoded_request(
                http_options, request
            )

            body = _BaseDocumentServiceRestTransport._BaseImportDocuments._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseImportDocuments._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.ImportDocuments",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "ImportDocuments",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._ImportDocuments._get_response(
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

            resp = self._interceptor.post_import_documents(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_import_documents_with_metadata(
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
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceClient.import_documents",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "ImportDocuments",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListDocuments(
        _BaseDocumentServiceRestTransport._BaseListDocuments, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.ListDocuments")

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
            request: document_service.ListDocumentsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> document_service.ListDocumentsResponse:
            r"""Call the list documents method over HTTP.

            Args:
                request (~.document_service.ListDocumentsRequest):
                    The request object. Request message for
                [DocumentService.ListDocuments][google.cloud.discoveryengine.v1beta.DocumentService.ListDocuments]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.document_service.ListDocumentsResponse:
                    Response message for
                [DocumentService.ListDocuments][google.cloud.discoveryengine.v1beta.DocumentService.ListDocuments]
                method.

            """

            http_options = (
                _BaseDocumentServiceRestTransport._BaseListDocuments._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_documents(request, metadata)
            transcoded_request = _BaseDocumentServiceRestTransport._BaseListDocuments._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseListDocuments._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.ListDocuments",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "ListDocuments",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._ListDocuments._get_response(
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
            resp = document_service.ListDocumentsResponse()
            pb_resp = document_service.ListDocumentsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_documents(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_documents_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = document_service.ListDocumentsResponse.to_json(
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
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceClient.list_documents",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "ListDocuments",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _PurgeDocuments(
        _BaseDocumentServiceRestTransport._BasePurgeDocuments, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.PurgeDocuments")

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
            request: purge_config.PurgeDocumentsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the purge documents method over HTTP.

            Args:
                request (~.purge_config.PurgeDocumentsRequest):
                    The request object. Request message for
                [DocumentService.PurgeDocuments][google.cloud.discoveryengine.v1beta.DocumentService.PurgeDocuments]
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
                _BaseDocumentServiceRestTransport._BasePurgeDocuments._get_http_options()
            )

            request, metadata = self._interceptor.pre_purge_documents(request, metadata)
            transcoded_request = _BaseDocumentServiceRestTransport._BasePurgeDocuments._get_transcoded_request(
                http_options, request
            )

            body = _BaseDocumentServiceRestTransport._BasePurgeDocuments._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BasePurgeDocuments._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.PurgeDocuments",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "PurgeDocuments",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._PurgeDocuments._get_response(
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

            resp = self._interceptor.post_purge_documents(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_purge_documents_with_metadata(
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
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceClient.purge_documents",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "PurgeDocuments",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateDocument(
        _BaseDocumentServiceRestTransport._BaseUpdateDocument, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.UpdateDocument")

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
            request: document_service.UpdateDocumentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_document.Document:
            r"""Call the update document method over HTTP.

            Args:
                request (~.document_service.UpdateDocumentRequest):
                    The request object. Request message for
                [DocumentService.UpdateDocument][google.cloud.discoveryengine.v1beta.DocumentService.UpdateDocument]
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_document.Document:
                    Document captures all raw metadata
                information of items to be recommended
                or searched.

            """

            http_options = (
                _BaseDocumentServiceRestTransport._BaseUpdateDocument._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_document(request, metadata)
            transcoded_request = _BaseDocumentServiceRestTransport._BaseUpdateDocument._get_transcoded_request(
                http_options, request
            )

            body = _BaseDocumentServiceRestTransport._BaseUpdateDocument._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseUpdateDocument._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.UpdateDocument",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "UpdateDocument",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._UpdateDocument._get_response(
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
            resp = gcd_document.Document()
            pb_resp = gcd_document.Document.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_document(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_document_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_document.Document.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceClient.update_document",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "UpdateDocument",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def batch_get_documents_metadata(
        self,
    ) -> Callable[
        [document_service.BatchGetDocumentsMetadataRequest],
        document_service.BatchGetDocumentsMetadataResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._BatchGetDocumentsMetadata(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_document(
        self,
    ) -> Callable[[document_service.CreateDocumentRequest], gcd_document.Document]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_document(
        self,
    ) -> Callable[[document_service.DeleteDocumentRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_document(
        self,
    ) -> Callable[[document_service.GetDocumentRequest], document.Document]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def import_documents(
        self,
    ) -> Callable[[import_config.ImportDocumentsRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ImportDocuments(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_documents(
        self,
    ) -> Callable[
        [document_service.ListDocumentsRequest], document_service.ListDocumentsResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListDocuments(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def purge_documents(
        self,
    ) -> Callable[[purge_config.PurgeDocumentsRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._PurgeDocuments(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_document(
        self,
    ) -> Callable[[document_service.UpdateDocumentRequest], gcd_document.Document]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateDocument(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def cancel_operation(self):
        return self._CancelOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _CancelOperation(
        _BaseDocumentServiceRestTransport._BaseCancelOperation, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.CancelOperation")

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
                _BaseDocumentServiceRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseDocumentServiceRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            body = _BaseDocumentServiceRestTransport._BaseCancelOperation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseCancelOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.CancelOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._CancelOperation._get_response(
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
        _BaseDocumentServiceRestTransport._BaseGetOperation, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.GetOperation")

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
                _BaseDocumentServiceRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BaseDocumentServiceRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._GetOperation._get_response(
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
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
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
        _BaseDocumentServiceRestTransport._BaseListOperations, DocumentServiceRestStub
    ):
        def __hash__(self):
            return hash("DocumentServiceRestTransport.ListOperations")

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
                _BaseDocumentServiceRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseDocumentServiceRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseDocumentServiceRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1beta.DocumentServiceClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = DocumentServiceRestTransport._ListOperations._get_response(
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
                    "Received response for google.cloud.discoveryengine_v1beta.DocumentServiceAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1beta.DocumentService",
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


__all__ = ("DocumentServiceRestTransport",)
