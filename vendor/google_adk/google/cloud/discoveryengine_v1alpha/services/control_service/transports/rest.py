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
from google.cloud.location import locations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
import google.protobuf
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import json_format
from requests import __version__ as requests_version

from google.cloud.discoveryengine_v1alpha.types import control
from google.cloud.discoveryengine_v1alpha.types import control as gcd_control
from google.cloud.discoveryengine_v1alpha.types import control_service

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseControlServiceRestTransport

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


class ControlServiceRestInterceptor:
    """Interceptor for ControlService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the ControlServiceRestTransport.

    .. code-block:: python
        class MyCustomControlServiceInterceptor(ControlServiceRestInterceptor):
            def pre_create_control(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_control(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_control(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_control(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_control(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_controls(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_controls(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_control(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_control(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = ControlServiceRestTransport(interceptor=MyCustomControlServiceInterceptor())
        client = ControlServiceClient(transport=transport)


    """

    def pre_create_control(
        self,
        request: control_service.CreateControlRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        control_service.CreateControlRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for create_control

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ControlService server.
        """
        return request, metadata

    def post_create_control(self, response: gcd_control.Control) -> gcd_control.Control:
        """Post-rpc interceptor for create_control

        DEPRECATED. Please use the `post_create_control_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ControlService server but before
        it is returned to user code. This `post_create_control` interceptor runs
        before the `post_create_control_with_metadata` interceptor.
        """
        return response

    def post_create_control_with_metadata(
        self,
        response: gcd_control.Control,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_control.Control, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_control

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ControlService server but before it is returned to user code.

        We recommend only using this `post_create_control_with_metadata`
        interceptor in new development instead of the `post_create_control` interceptor.
        When both interceptors are used, this `post_create_control_with_metadata` interceptor runs after the
        `post_create_control` interceptor. The (possibly modified) response returned by
        `post_create_control` will be passed to
        `post_create_control_with_metadata`.
        """
        return response, metadata

    def pre_delete_control(
        self,
        request: control_service.DeleteControlRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        control_service.DeleteControlRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_control

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ControlService server.
        """
        return request, metadata

    def pre_get_control(
        self,
        request: control_service.GetControlRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        control_service.GetControlRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_control

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ControlService server.
        """
        return request, metadata

    def post_get_control(self, response: control.Control) -> control.Control:
        """Post-rpc interceptor for get_control

        DEPRECATED. Please use the `post_get_control_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ControlService server but before
        it is returned to user code. This `post_get_control` interceptor runs
        before the `post_get_control_with_metadata` interceptor.
        """
        return response

    def post_get_control_with_metadata(
        self,
        response: control.Control,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[control.Control, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_control

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ControlService server but before it is returned to user code.

        We recommend only using this `post_get_control_with_metadata`
        interceptor in new development instead of the `post_get_control` interceptor.
        When both interceptors are used, this `post_get_control_with_metadata` interceptor runs after the
        `post_get_control` interceptor. The (possibly modified) response returned by
        `post_get_control` will be passed to
        `post_get_control_with_metadata`.
        """
        return response, metadata

    def pre_list_controls(
        self,
        request: control_service.ListControlsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        control_service.ListControlsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_controls

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ControlService server.
        """
        return request, metadata

    def post_list_controls(
        self, response: control_service.ListControlsResponse
    ) -> control_service.ListControlsResponse:
        """Post-rpc interceptor for list_controls

        DEPRECATED. Please use the `post_list_controls_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ControlService server but before
        it is returned to user code. This `post_list_controls` interceptor runs
        before the `post_list_controls_with_metadata` interceptor.
        """
        return response

    def post_list_controls_with_metadata(
        self,
        response: control_service.ListControlsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        control_service.ListControlsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_controls

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ControlService server but before it is returned to user code.

        We recommend only using this `post_list_controls_with_metadata`
        interceptor in new development instead of the `post_list_controls` interceptor.
        When both interceptors are used, this `post_list_controls_with_metadata` interceptor runs after the
        `post_list_controls` interceptor. The (possibly modified) response returned by
        `post_list_controls` will be passed to
        `post_list_controls_with_metadata`.
        """
        return response, metadata

    def pre_update_control(
        self,
        request: control_service.UpdateControlRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        control_service.UpdateControlRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for update_control

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ControlService server.
        """
        return request, metadata

    def post_update_control(self, response: gcd_control.Control) -> gcd_control.Control:
        """Post-rpc interceptor for update_control

        DEPRECATED. Please use the `post_update_control_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ControlService server but before
        it is returned to user code. This `post_update_control` interceptor runs
        before the `post_update_control_with_metadata` interceptor.
        """
        return response

    def post_update_control_with_metadata(
        self,
        response: gcd_control.Control,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gcd_control.Control, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_control

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ControlService server but before it is returned to user code.

        We recommend only using this `post_update_control_with_metadata`
        interceptor in new development instead of the `post_update_control` interceptor.
        When both interceptors are used, this `post_update_control_with_metadata` interceptor runs after the
        `post_update_control` interceptor. The (possibly modified) response returned by
        `post_update_control` will be passed to
        `post_update_control_with_metadata`.
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
        before they are sent to the ControlService server.
        """
        return request, metadata

    def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the ControlService server but before
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
        before they are sent to the ControlService server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the ControlService server but before
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
        before they are sent to the ControlService server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the ControlService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class ControlServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: ControlServiceRestInterceptor


class ControlServiceRestTransport(_BaseControlServiceRestTransport):
    """REST backend synchronous transport for ControlService.

    Service for performing CRUD operations on Controls.
    Controls allow for custom logic to be implemented in the serving
    path. Controls need to be attached to a Serving Config to be
    considered during a request.

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
        interceptor: Optional[ControlServiceRestInterceptor] = None,
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
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or ControlServiceRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CreateControl(
        _BaseControlServiceRestTransport._BaseCreateControl, ControlServiceRestStub
    ):
        def __hash__(self):
            return hash("ControlServiceRestTransport.CreateControl")

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
            request: control_service.CreateControlRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_control.Control:
            r"""Call the create control method over HTTP.

            Args:
                request (~.control_service.CreateControlRequest):
                    The request object. Request for CreateControl method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_control.Control:
                    Defines a conditioned behavior to employ during serving.
                Must be attached to a
                [ServingConfig][google.cloud.discoveryengine.v1alpha.ServingConfig]
                to be considered at serving time. Permitted actions
                dependent on ``SolutionType``.

            """

            http_options = (
                _BaseControlServiceRestTransport._BaseCreateControl._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_control(request, metadata)
            transcoded_request = _BaseControlServiceRestTransport._BaseCreateControl._get_transcoded_request(
                http_options, request
            )

            body = _BaseControlServiceRestTransport._BaseCreateControl._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseControlServiceRestTransport._BaseCreateControl._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.ControlServiceClient.CreateControl",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "CreateControl",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ControlServiceRestTransport._CreateControl._get_response(
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
            resp = gcd_control.Control()
            pb_resp = gcd_control.Control.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_create_control(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_control_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_control.Control.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1alpha.ControlServiceClient.create_control",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "CreateControl",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteControl(
        _BaseControlServiceRestTransport._BaseDeleteControl, ControlServiceRestStub
    ):
        def __hash__(self):
            return hash("ControlServiceRestTransport.DeleteControl")

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
            request: control_service.DeleteControlRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete control method over HTTP.

            Args:
                request (~.control_service.DeleteControlRequest):
                    The request object. Request for DeleteControl method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseControlServiceRestTransport._BaseDeleteControl._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_control(request, metadata)
            transcoded_request = _BaseControlServiceRestTransport._BaseDeleteControl._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseControlServiceRestTransport._BaseDeleteControl._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.ControlServiceClient.DeleteControl",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "DeleteControl",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ControlServiceRestTransport._DeleteControl._get_response(
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

    class _GetControl(
        _BaseControlServiceRestTransport._BaseGetControl, ControlServiceRestStub
    ):
        def __hash__(self):
            return hash("ControlServiceRestTransport.GetControl")

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
            request: control_service.GetControlRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> control.Control:
            r"""Call the get control method over HTTP.

            Args:
                request (~.control_service.GetControlRequest):
                    The request object. Request for GetControl method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.control.Control:
                    Defines a conditioned behavior to employ during serving.
                Must be attached to a
                [ServingConfig][google.cloud.discoveryengine.v1alpha.ServingConfig]
                to be considered at serving time. Permitted actions
                dependent on ``SolutionType``.

            """

            http_options = (
                _BaseControlServiceRestTransport._BaseGetControl._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_control(request, metadata)
            transcoded_request = _BaseControlServiceRestTransport._BaseGetControl._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseControlServiceRestTransport._BaseGetControl._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.ControlServiceClient.GetControl",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "GetControl",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ControlServiceRestTransport._GetControl._get_response(
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
            resp = control.Control()
            pb_resp = control.Control.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_control(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_control_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = control.Control.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1alpha.ControlServiceClient.get_control",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "GetControl",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListControls(
        _BaseControlServiceRestTransport._BaseListControls, ControlServiceRestStub
    ):
        def __hash__(self):
            return hash("ControlServiceRestTransport.ListControls")

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
            request: control_service.ListControlsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> control_service.ListControlsResponse:
            r"""Call the list controls method over HTTP.

            Args:
                request (~.control_service.ListControlsRequest):
                    The request object. Request for ListControls method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.control_service.ListControlsResponse:
                    Response for ListControls method.
            """

            http_options = (
                _BaseControlServiceRestTransport._BaseListControls._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_controls(request, metadata)
            transcoded_request = _BaseControlServiceRestTransport._BaseListControls._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseControlServiceRestTransport._BaseListControls._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.ControlServiceClient.ListControls",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "ListControls",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ControlServiceRestTransport._ListControls._get_response(
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
            resp = control_service.ListControlsResponse()
            pb_resp = control_service.ListControlsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_controls(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_controls_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = control_service.ListControlsResponse.to_json(
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
                    "Received response for google.cloud.discoveryengine_v1alpha.ControlServiceClient.list_controls",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "ListControls",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateControl(
        _BaseControlServiceRestTransport._BaseUpdateControl, ControlServiceRestStub
    ):
        def __hash__(self):
            return hash("ControlServiceRestTransport.UpdateControl")

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
            request: control_service.UpdateControlRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gcd_control.Control:
            r"""Call the update control method over HTTP.

            Args:
                request (~.control_service.UpdateControlRequest):
                    The request object. Request for UpdateControl method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gcd_control.Control:
                    Defines a conditioned behavior to employ during serving.
                Must be attached to a
                [ServingConfig][google.cloud.discoveryengine.v1alpha.ServingConfig]
                to be considered at serving time. Permitted actions
                dependent on ``SolutionType``.

            """

            http_options = (
                _BaseControlServiceRestTransport._BaseUpdateControl._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_control(request, metadata)
            transcoded_request = _BaseControlServiceRestTransport._BaseUpdateControl._get_transcoded_request(
                http_options, request
            )

            body = _BaseControlServiceRestTransport._BaseUpdateControl._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseControlServiceRestTransport._BaseUpdateControl._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.ControlServiceClient.UpdateControl",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "UpdateControl",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ControlServiceRestTransport._UpdateControl._get_response(
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
            resp = gcd_control.Control()
            pb_resp = gcd_control.Control.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_control(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_control_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gcd_control.Control.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.discoveryengine_v1alpha.ControlServiceClient.update_control",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "UpdateControl",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def create_control(
        self,
    ) -> Callable[[control_service.CreateControlRequest], gcd_control.Control]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateControl(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_control(
        self,
    ) -> Callable[[control_service.DeleteControlRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteControl(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_control(
        self,
    ) -> Callable[[control_service.GetControlRequest], control.Control]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetControl(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_controls(
        self,
    ) -> Callable[
        [control_service.ListControlsRequest], control_service.ListControlsResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListControls(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_control(
        self,
    ) -> Callable[[control_service.UpdateControlRequest], gcd_control.Control]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateControl(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def cancel_operation(self):
        return self._CancelOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _CancelOperation(
        _BaseControlServiceRestTransport._BaseCancelOperation, ControlServiceRestStub
    ):
        def __hash__(self):
            return hash("ControlServiceRestTransport.CancelOperation")

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
                _BaseControlServiceRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseControlServiceRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            body = _BaseControlServiceRestTransport._BaseCancelOperation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseControlServiceRestTransport._BaseCancelOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.ControlServiceClient.CancelOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ControlServiceRestTransport._CancelOperation._get_response(
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
        _BaseControlServiceRestTransport._BaseGetOperation, ControlServiceRestStub
    ):
        def __hash__(self):
            return hash("ControlServiceRestTransport.GetOperation")

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
                _BaseControlServiceRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BaseControlServiceRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseControlServiceRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.ControlServiceClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ControlServiceRestTransport._GetOperation._get_response(
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
                    "Received response for google.cloud.discoveryengine_v1alpha.ControlServiceAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
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
        _BaseControlServiceRestTransport._BaseListOperations, ControlServiceRestStub
    ):
        def __hash__(self):
            return hash("ControlServiceRestTransport.ListOperations")

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
                _BaseControlServiceRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseControlServiceRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseControlServiceRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.cloud.discoveryengine_v1alpha.ControlServiceClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ControlServiceRestTransport._ListOperations._get_response(
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
                    "Received response for google.cloud.discoveryengine_v1alpha.ControlServiceAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.discoveryengine.v1alpha.ControlService",
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


__all__ = ("ControlServiceRestTransport",)
