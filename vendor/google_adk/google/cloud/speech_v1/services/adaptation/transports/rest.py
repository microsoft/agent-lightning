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

from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1, rest_helpers, rest_streaming
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
import google.protobuf
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import json_format
from requests import __version__ as requests_version

from google.cloud.speech_v1.types import cloud_speech_adaptation, resource

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseAdaptationRestTransport

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


class AdaptationRestInterceptor:
    """Interceptor for Adaptation.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the AdaptationRestTransport.

    .. code-block:: python
        class MyCustomAdaptationInterceptor(AdaptationRestInterceptor):
            def pre_create_custom_class(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_custom_class(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_phrase_set(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_custom_class(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_custom_class(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_custom_class(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_phrase_set(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_custom_classes(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_custom_classes(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_phrase_set(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_custom_class(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_custom_class(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_phrase_set(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = AdaptationRestTransport(interceptor=MyCustomAdaptationInterceptor())
        client = AdaptationClient(transport=transport)


    """

    def pre_create_custom_class(
        self,
        request: cloud_speech_adaptation.CreateCustomClassRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.CreateCustomClassRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_custom_class

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_create_custom_class(
        self, response: resource.CustomClass
    ) -> resource.CustomClass:
        """Post-rpc interceptor for create_custom_class

        DEPRECATED. Please use the `post_create_custom_class_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code. This `post_create_custom_class` interceptor runs
        before the `post_create_custom_class_with_metadata` interceptor.
        """
        return response

    def post_create_custom_class_with_metadata(
        self,
        response: resource.CustomClass,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[resource.CustomClass, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_custom_class

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Adaptation server but before it is returned to user code.

        We recommend only using this `post_create_custom_class_with_metadata`
        interceptor in new development instead of the `post_create_custom_class` interceptor.
        When both interceptors are used, this `post_create_custom_class_with_metadata` interceptor runs after the
        `post_create_custom_class` interceptor. The (possibly modified) response returned by
        `post_create_custom_class` will be passed to
        `post_create_custom_class_with_metadata`.
        """
        return response, metadata

    def pre_create_phrase_set(
        self,
        request: cloud_speech_adaptation.CreatePhraseSetRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.CreatePhraseSetRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_create_phrase_set(
        self, response: resource.PhraseSet
    ) -> resource.PhraseSet:
        """Post-rpc interceptor for create_phrase_set

        DEPRECATED. Please use the `post_create_phrase_set_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code. This `post_create_phrase_set` interceptor runs
        before the `post_create_phrase_set_with_metadata` interceptor.
        """
        return response

    def post_create_phrase_set_with_metadata(
        self,
        response: resource.PhraseSet,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[resource.PhraseSet, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_phrase_set

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Adaptation server but before it is returned to user code.

        We recommend only using this `post_create_phrase_set_with_metadata`
        interceptor in new development instead of the `post_create_phrase_set` interceptor.
        When both interceptors are used, this `post_create_phrase_set_with_metadata` interceptor runs after the
        `post_create_phrase_set` interceptor. The (possibly modified) response returned by
        `post_create_phrase_set` will be passed to
        `post_create_phrase_set_with_metadata`.
        """
        return response, metadata

    def pre_delete_custom_class(
        self,
        request: cloud_speech_adaptation.DeleteCustomClassRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.DeleteCustomClassRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_custom_class

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def pre_delete_phrase_set(
        self,
        request: cloud_speech_adaptation.DeletePhraseSetRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.DeletePhraseSetRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def pre_get_custom_class(
        self,
        request: cloud_speech_adaptation.GetCustomClassRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.GetCustomClassRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_custom_class

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_get_custom_class(
        self, response: resource.CustomClass
    ) -> resource.CustomClass:
        """Post-rpc interceptor for get_custom_class

        DEPRECATED. Please use the `post_get_custom_class_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code. This `post_get_custom_class` interceptor runs
        before the `post_get_custom_class_with_metadata` interceptor.
        """
        return response

    def post_get_custom_class_with_metadata(
        self,
        response: resource.CustomClass,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[resource.CustomClass, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_custom_class

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Adaptation server but before it is returned to user code.

        We recommend only using this `post_get_custom_class_with_metadata`
        interceptor in new development instead of the `post_get_custom_class` interceptor.
        When both interceptors are used, this `post_get_custom_class_with_metadata` interceptor runs after the
        `post_get_custom_class` interceptor. The (possibly modified) response returned by
        `post_get_custom_class` will be passed to
        `post_get_custom_class_with_metadata`.
        """
        return response, metadata

    def pre_get_phrase_set(
        self,
        request: cloud_speech_adaptation.GetPhraseSetRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.GetPhraseSetRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_get_phrase_set(self, response: resource.PhraseSet) -> resource.PhraseSet:
        """Post-rpc interceptor for get_phrase_set

        DEPRECATED. Please use the `post_get_phrase_set_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code. This `post_get_phrase_set` interceptor runs
        before the `post_get_phrase_set_with_metadata` interceptor.
        """
        return response

    def post_get_phrase_set_with_metadata(
        self,
        response: resource.PhraseSet,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[resource.PhraseSet, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_phrase_set

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Adaptation server but before it is returned to user code.

        We recommend only using this `post_get_phrase_set_with_metadata`
        interceptor in new development instead of the `post_get_phrase_set` interceptor.
        When both interceptors are used, this `post_get_phrase_set_with_metadata` interceptor runs after the
        `post_get_phrase_set` interceptor. The (possibly modified) response returned by
        `post_get_phrase_set` will be passed to
        `post_get_phrase_set_with_metadata`.
        """
        return response, metadata

    def pre_list_custom_classes(
        self,
        request: cloud_speech_adaptation.ListCustomClassesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.ListCustomClassesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_custom_classes

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_list_custom_classes(
        self, response: cloud_speech_adaptation.ListCustomClassesResponse
    ) -> cloud_speech_adaptation.ListCustomClassesResponse:
        """Post-rpc interceptor for list_custom_classes

        DEPRECATED. Please use the `post_list_custom_classes_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code. This `post_list_custom_classes` interceptor runs
        before the `post_list_custom_classes_with_metadata` interceptor.
        """
        return response

    def post_list_custom_classes_with_metadata(
        self,
        response: cloud_speech_adaptation.ListCustomClassesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.ListCustomClassesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_custom_classes

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Adaptation server but before it is returned to user code.

        We recommend only using this `post_list_custom_classes_with_metadata`
        interceptor in new development instead of the `post_list_custom_classes` interceptor.
        When both interceptors are used, this `post_list_custom_classes_with_metadata` interceptor runs after the
        `post_list_custom_classes` interceptor. The (possibly modified) response returned by
        `post_list_custom_classes` will be passed to
        `post_list_custom_classes_with_metadata`.
        """
        return response, metadata

    def pre_list_phrase_set(
        self,
        request: cloud_speech_adaptation.ListPhraseSetRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.ListPhraseSetRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_list_phrase_set(
        self, response: cloud_speech_adaptation.ListPhraseSetResponse
    ) -> cloud_speech_adaptation.ListPhraseSetResponse:
        """Post-rpc interceptor for list_phrase_set

        DEPRECATED. Please use the `post_list_phrase_set_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code. This `post_list_phrase_set` interceptor runs
        before the `post_list_phrase_set_with_metadata` interceptor.
        """
        return response

    def post_list_phrase_set_with_metadata(
        self,
        response: cloud_speech_adaptation.ListPhraseSetResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.ListPhraseSetResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_phrase_set

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Adaptation server but before it is returned to user code.

        We recommend only using this `post_list_phrase_set_with_metadata`
        interceptor in new development instead of the `post_list_phrase_set` interceptor.
        When both interceptors are used, this `post_list_phrase_set_with_metadata` interceptor runs after the
        `post_list_phrase_set` interceptor. The (possibly modified) response returned by
        `post_list_phrase_set` will be passed to
        `post_list_phrase_set_with_metadata`.
        """
        return response, metadata

    def pre_update_custom_class(
        self,
        request: cloud_speech_adaptation.UpdateCustomClassRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.UpdateCustomClassRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_custom_class

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_update_custom_class(
        self, response: resource.CustomClass
    ) -> resource.CustomClass:
        """Post-rpc interceptor for update_custom_class

        DEPRECATED. Please use the `post_update_custom_class_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code. This `post_update_custom_class` interceptor runs
        before the `post_update_custom_class_with_metadata` interceptor.
        """
        return response

    def post_update_custom_class_with_metadata(
        self,
        response: resource.CustomClass,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[resource.CustomClass, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_custom_class

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Adaptation server but before it is returned to user code.

        We recommend only using this `post_update_custom_class_with_metadata`
        interceptor in new development instead of the `post_update_custom_class` interceptor.
        When both interceptors are used, this `post_update_custom_class_with_metadata` interceptor runs after the
        `post_update_custom_class` interceptor. The (possibly modified) response returned by
        `post_update_custom_class` will be passed to
        `post_update_custom_class_with_metadata`.
        """
        return response, metadata

    def pre_update_phrase_set(
        self,
        request: cloud_speech_adaptation.UpdatePhraseSetRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        cloud_speech_adaptation.UpdatePhraseSetRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_update_phrase_set(
        self, response: resource.PhraseSet
    ) -> resource.PhraseSet:
        """Post-rpc interceptor for update_phrase_set

        DEPRECATED. Please use the `post_update_phrase_set_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code. This `post_update_phrase_set` interceptor runs
        before the `post_update_phrase_set_with_metadata` interceptor.
        """
        return response

    def post_update_phrase_set_with_metadata(
        self,
        response: resource.PhraseSet,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[resource.PhraseSet, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_phrase_set

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the Adaptation server but before it is returned to user code.

        We recommend only using this `post_update_phrase_set_with_metadata`
        interceptor in new development instead of the `post_update_phrase_set` interceptor.
        When both interceptors are used, this `post_update_phrase_set_with_metadata` interceptor runs after the
        `post_update_phrase_set` interceptor. The (possibly modified) response returned by
        `post_update_phrase_set` will be passed to
        `post_update_phrase_set_with_metadata`.
        """
        return response, metadata

    def pre_get_operation(
        self,
        request: operations_pb2.GetOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.GetOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
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
        before they are sent to the Adaptation server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class AdaptationRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: AdaptationRestInterceptor


class AdaptationRestTransport(_BaseAdaptationRestTransport):
    """REST backend synchronous transport for Adaptation.

    Service that implements Google Cloud Speech Adaptation API.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "speech.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[AdaptationRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'speech.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): Deprecated. A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided. This argument will be
                removed in the next major version of this library.
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
        self._interceptor = interceptor or AdaptationRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CreateCustomClass(
        _BaseAdaptationRestTransport._BaseCreateCustomClass, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.CreateCustomClass")

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
            request: cloud_speech_adaptation.CreateCustomClassRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> resource.CustomClass:
            r"""Call the create custom class method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.CreateCustomClassRequest):
                    The request object. Message sent by the client for the ``CreateCustomClass``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.resource.CustomClass:
                    A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

            """

            http_options = (
                _BaseAdaptationRestTransport._BaseCreateCustomClass._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_custom_class(
                request, metadata
            )
            transcoded_request = _BaseAdaptationRestTransport._BaseCreateCustomClass._get_transcoded_request(
                http_options, request
            )

            body = _BaseAdaptationRestTransport._BaseCreateCustomClass._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseAdaptationRestTransport._BaseCreateCustomClass._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.CreateCustomClass",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "CreateCustomClass",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._CreateCustomClass._get_response(
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
            resp = resource.CustomClass()
            pb_resp = resource.CustomClass.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_custom_class(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_custom_class_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = resource.CustomClass.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.speech_v1.AdaptationClient.create_custom_class",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "CreateCustomClass",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreatePhraseSet(
        _BaseAdaptationRestTransport._BaseCreatePhraseSet, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.CreatePhraseSet")

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
            request: cloud_speech_adaptation.CreatePhraseSetRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> resource.PhraseSet:
            r"""Call the create phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.CreatePhraseSetRequest):
                    The request object. Message sent by the client for the ``CreatePhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.resource.PhraseSet:
                    Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

            """

            http_options = (
                _BaseAdaptationRestTransport._BaseCreatePhraseSet._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_phrase_set(
                request, metadata
            )
            transcoded_request = _BaseAdaptationRestTransport._BaseCreatePhraseSet._get_transcoded_request(
                http_options, request
            )

            body = _BaseAdaptationRestTransport._BaseCreatePhraseSet._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseAdaptationRestTransport._BaseCreatePhraseSet._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.CreatePhraseSet",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "CreatePhraseSet",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._CreatePhraseSet._get_response(
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
            resp = resource.PhraseSet()
            pb_resp = resource.PhraseSet.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_phrase_set(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_phrase_set_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = resource.PhraseSet.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.speech_v1.AdaptationClient.create_phrase_set",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "CreatePhraseSet",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteCustomClass(
        _BaseAdaptationRestTransport._BaseDeleteCustomClass, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.DeleteCustomClass")

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
            request: cloud_speech_adaptation.DeleteCustomClassRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete custom class method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.DeleteCustomClassRequest):
                    The request object. Message sent by the client for the ``DeleteCustomClass``
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
                _BaseAdaptationRestTransport._BaseDeleteCustomClass._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_custom_class(
                request, metadata
            )
            transcoded_request = _BaseAdaptationRestTransport._BaseDeleteCustomClass._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseAdaptationRestTransport._BaseDeleteCustomClass._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.DeleteCustomClass",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "DeleteCustomClass",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._DeleteCustomClass._get_response(
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

    class _DeletePhraseSet(
        _BaseAdaptationRestTransport._BaseDeletePhraseSet, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.DeletePhraseSet")

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
            request: cloud_speech_adaptation.DeletePhraseSetRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.DeletePhraseSetRequest):
                    The request object. Message sent by the client for the ``DeletePhraseSet``
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
                _BaseAdaptationRestTransport._BaseDeletePhraseSet._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_phrase_set(
                request, metadata
            )
            transcoded_request = _BaseAdaptationRestTransport._BaseDeletePhraseSet._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseAdaptationRestTransport._BaseDeletePhraseSet._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.DeletePhraseSet",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "DeletePhraseSet",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._DeletePhraseSet._get_response(
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

    class _GetCustomClass(
        _BaseAdaptationRestTransport._BaseGetCustomClass, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.GetCustomClass")

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
            request: cloud_speech_adaptation.GetCustomClassRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> resource.CustomClass:
            r"""Call the get custom class method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.GetCustomClassRequest):
                    The request object. Message sent by the client for the ``GetCustomClass``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.resource.CustomClass:
                    A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

            """

            http_options = (
                _BaseAdaptationRestTransport._BaseGetCustomClass._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_custom_class(
                request, metadata
            )
            transcoded_request = _BaseAdaptationRestTransport._BaseGetCustomClass._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseAdaptationRestTransport._BaseGetCustomClass._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.GetCustomClass",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "GetCustomClass",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._GetCustomClass._get_response(
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
            resp = resource.CustomClass()
            pb_resp = resource.CustomClass.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_custom_class(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_custom_class_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = resource.CustomClass.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.speech_v1.AdaptationClient.get_custom_class",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "GetCustomClass",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetPhraseSet(
        _BaseAdaptationRestTransport._BaseGetPhraseSet, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.GetPhraseSet")

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
            request: cloud_speech_adaptation.GetPhraseSetRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> resource.PhraseSet:
            r"""Call the get phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.GetPhraseSetRequest):
                    The request object. Message sent by the client for the ``GetPhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.resource.PhraseSet:
                    Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

            """

            http_options = (
                _BaseAdaptationRestTransport._BaseGetPhraseSet._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_phrase_set(request, metadata)
            transcoded_request = (
                _BaseAdaptationRestTransport._BaseGetPhraseSet._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseAdaptationRestTransport._BaseGetPhraseSet._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.GetPhraseSet",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "GetPhraseSet",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._GetPhraseSet._get_response(
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
            resp = resource.PhraseSet()
            pb_resp = resource.PhraseSet.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_phrase_set(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_phrase_set_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = resource.PhraseSet.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.speech_v1.AdaptationClient.get_phrase_set",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "GetPhraseSet",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListCustomClasses(
        _BaseAdaptationRestTransport._BaseListCustomClasses, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.ListCustomClasses")

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
            request: cloud_speech_adaptation.ListCustomClassesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> cloud_speech_adaptation.ListCustomClassesResponse:
            r"""Call the list custom classes method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.ListCustomClassesRequest):
                    The request object. Message sent by the client for the ``ListCustomClasses``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.cloud_speech_adaptation.ListCustomClassesResponse:
                    Message returned to the client by the
                ``ListCustomClasses`` method.

            """

            http_options = (
                _BaseAdaptationRestTransport._BaseListCustomClasses._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_custom_classes(
                request, metadata
            )
            transcoded_request = _BaseAdaptationRestTransport._BaseListCustomClasses._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseAdaptationRestTransport._BaseListCustomClasses._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.ListCustomClasses",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "ListCustomClasses",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._ListCustomClasses._get_response(
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
            resp = cloud_speech_adaptation.ListCustomClassesResponse()
            pb_resp = cloud_speech_adaptation.ListCustomClassesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_custom_classes(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_custom_classes_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        cloud_speech_adaptation.ListCustomClassesResponse.to_json(
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
                    "Received response for google.cloud.speech_v1.AdaptationClient.list_custom_classes",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "ListCustomClasses",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListPhraseSet(
        _BaseAdaptationRestTransport._BaseListPhraseSet, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.ListPhraseSet")

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
            request: cloud_speech_adaptation.ListPhraseSetRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> cloud_speech_adaptation.ListPhraseSetResponse:
            r"""Call the list phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.ListPhraseSetRequest):
                    The request object. Message sent by the client for the ``ListPhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.cloud_speech_adaptation.ListPhraseSetResponse:
                    Message returned to the client by the ``ListPhraseSet``
                method.

            """

            http_options = (
                _BaseAdaptationRestTransport._BaseListPhraseSet._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_phrase_set(request, metadata)
            transcoded_request = (
                _BaseAdaptationRestTransport._BaseListPhraseSet._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseAdaptationRestTransport._BaseListPhraseSet._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.ListPhraseSet",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "ListPhraseSet",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._ListPhraseSet._get_response(
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
            resp = cloud_speech_adaptation.ListPhraseSetResponse()
            pb_resp = cloud_speech_adaptation.ListPhraseSetResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_phrase_set(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_phrase_set_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        cloud_speech_adaptation.ListPhraseSetResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.speech_v1.AdaptationClient.list_phrase_set",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "ListPhraseSet",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateCustomClass(
        _BaseAdaptationRestTransport._BaseUpdateCustomClass, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.UpdateCustomClass")

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
            request: cloud_speech_adaptation.UpdateCustomClassRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> resource.CustomClass:
            r"""Call the update custom class method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.UpdateCustomClassRequest):
                    The request object. Message sent by the client for the ``UpdateCustomClass``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.resource.CustomClass:
                    A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

            """

            http_options = (
                _BaseAdaptationRestTransport._BaseUpdateCustomClass._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_custom_class(
                request, metadata
            )
            transcoded_request = _BaseAdaptationRestTransport._BaseUpdateCustomClass._get_transcoded_request(
                http_options, request
            )

            body = _BaseAdaptationRestTransport._BaseUpdateCustomClass._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseAdaptationRestTransport._BaseUpdateCustomClass._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.UpdateCustomClass",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "UpdateCustomClass",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._UpdateCustomClass._get_response(
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
            resp = resource.CustomClass()
            pb_resp = resource.CustomClass.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_custom_class(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_custom_class_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = resource.CustomClass.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.speech_v1.AdaptationClient.update_custom_class",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "UpdateCustomClass",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdatePhraseSet(
        _BaseAdaptationRestTransport._BaseUpdatePhraseSet, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.UpdatePhraseSet")

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
            request: cloud_speech_adaptation.UpdatePhraseSetRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> resource.PhraseSet:
            r"""Call the update phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.UpdatePhraseSetRequest):
                    The request object. Message sent by the client for the ``UpdatePhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.resource.PhraseSet:
                    Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

            """

            http_options = (
                _BaseAdaptationRestTransport._BaseUpdatePhraseSet._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_phrase_set(
                request, metadata
            )
            transcoded_request = _BaseAdaptationRestTransport._BaseUpdatePhraseSet._get_transcoded_request(
                http_options, request
            )

            body = _BaseAdaptationRestTransport._BaseUpdatePhraseSet._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseAdaptationRestTransport._BaseUpdatePhraseSet._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.UpdatePhraseSet",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "UpdatePhraseSet",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._UpdatePhraseSet._get_response(
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
            resp = resource.PhraseSet()
            pb_resp = resource.PhraseSet.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_phrase_set(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_phrase_set_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = resource.PhraseSet.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.speech_v1.AdaptationClient.update_phrase_set",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "UpdatePhraseSet",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def create_custom_class(
        self,
    ) -> Callable[
        [cloud_speech_adaptation.CreateCustomClassRequest], resource.CustomClass
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateCustomClass(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_phrase_set(
        self,
    ) -> Callable[[cloud_speech_adaptation.CreatePhraseSetRequest], resource.PhraseSet]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreatePhraseSet(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_custom_class(
        self,
    ) -> Callable[[cloud_speech_adaptation.DeleteCustomClassRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteCustomClass(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_phrase_set(
        self,
    ) -> Callable[[cloud_speech_adaptation.DeletePhraseSetRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeletePhraseSet(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_custom_class(
        self,
    ) -> Callable[
        [cloud_speech_adaptation.GetCustomClassRequest], resource.CustomClass
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetCustomClass(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_phrase_set(
        self,
    ) -> Callable[[cloud_speech_adaptation.GetPhraseSetRequest], resource.PhraseSet]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetPhraseSet(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_custom_classes(
        self,
    ) -> Callable[
        [cloud_speech_adaptation.ListCustomClassesRequest],
        cloud_speech_adaptation.ListCustomClassesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListCustomClasses(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_phrase_set(
        self,
    ) -> Callable[
        [cloud_speech_adaptation.ListPhraseSetRequest],
        cloud_speech_adaptation.ListPhraseSetResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListPhraseSet(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_custom_class(
        self,
    ) -> Callable[
        [cloud_speech_adaptation.UpdateCustomClassRequest], resource.CustomClass
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateCustomClass(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_phrase_set(
        self,
    ) -> Callable[[cloud_speech_adaptation.UpdatePhraseSetRequest], resource.PhraseSet]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdatePhraseSet(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BaseAdaptationRestTransport._BaseGetOperation, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.GetOperation")

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
                _BaseAdaptationRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = (
                _BaseAdaptationRestTransport._BaseGetOperation._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseAdaptationRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._GetOperation._get_response(
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
                    "Received response for google.cloud.speech_v1.AdaptationAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
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
        _BaseAdaptationRestTransport._BaseListOperations, AdaptationRestStub
    ):
        def __hash__(self):
            return hash("AdaptationRestTransport.ListOperations")

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
                _BaseAdaptationRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseAdaptationRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseAdaptationRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.cloud.speech_v1.AdaptationClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = AdaptationRestTransport._ListOperations._get_response(
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
                    "Received response for google.cloud.speech_v1.AdaptationAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.speech.v1.Adaptation",
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


__all__ = ("AdaptationRestTransport",)
