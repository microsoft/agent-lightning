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

import google.auth

try:
    import aiohttp  # type: ignore
    from google.auth.aio.transport.sessions import AsyncAuthorizedSession  # type: ignore
    from google.api_core import rest_streaming_async  # type: ignore
    from google.api_core.operations_v1 import AsyncOperationsRestClient  # type: ignore
except ImportError as e:  # pragma: NO COVER
    raise ImportError(
        "`rest_asyncio` transport requires the library to be installed with the `async_rest` extra. Install the library with the `async_rest` extra using `pip install google-cloud-aiplatform[async_rest]`"
    ) from e

from google.auth.aio import credentials as ga_credentials_async  # type: ignore

from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.api_core import retry_async as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming_async  # type: ignore
import google.protobuf

from google.protobuf import json_format
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore

import json  # type: ignore
import dataclasses
from typing import Any, Dict, List, Callable, Tuple, Optional, Sequence, Union


from google.api import httpbody_pb2  # type: ignore
from google.cloud.aiplatform_v1.types import prediction_service
from google.longrunning import operations_pb2  # type: ignore


from .rest_base import _BasePredictionServiceRestTransport

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO


import logging

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = logging.getLogger(__name__)

try:
    OptionalRetry = Union[retries.AsyncRetry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.AsyncRetry, object, None]  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=f"google-auth@{google.auth.__version__}",
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


class AsyncPredictionServiceRestInterceptor:
    """Asynchronous Interceptor for PredictionService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the AsyncPredictionServiceRestTransport.

    .. code-block:: python
        class MyCustomPredictionServiceInterceptor(PredictionServiceRestInterceptor):
            async def pre_direct_predict(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_direct_predict(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_direct_raw_predict(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_direct_raw_predict(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_explain(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_explain(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_generate_content(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_generate_content(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_predict(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_predict(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_raw_predict(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_raw_predict(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_server_streaming_predict(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_server_streaming_predict(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_stream_generate_content(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_stream_generate_content(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_stream_raw_predict(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_stream_raw_predict(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = AsyncPredictionServiceRestTransport(interceptor=MyCustomPredictionServiceInterceptor())
        client = async PredictionServiceClient(transport=transport)


    """

    async def pre_direct_predict(
        self,
        request: prediction_service.DirectPredictRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.DirectPredictRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for direct_predict

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_direct_predict(
        self, response: prediction_service.DirectPredictResponse
    ) -> prediction_service.DirectPredictResponse:
        """Post-rpc interceptor for direct_predict

        DEPRECATED. Please use the `post_direct_predict_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_direct_predict` interceptor runs
        before the `post_direct_predict_with_metadata` interceptor.
        """
        return response

    async def post_direct_predict_with_metadata(
        self,
        response: prediction_service.DirectPredictResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.DirectPredictResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for direct_predict

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_direct_predict_with_metadata`
        interceptor in new development instead of the `post_direct_predict` interceptor.
        When both interceptors are used, this `post_direct_predict_with_metadata` interceptor runs after the
        `post_direct_predict` interceptor. The (possibly modified) response returned by
        `post_direct_predict` will be passed to
        `post_direct_predict_with_metadata`.
        """
        return response, metadata

    async def pre_direct_raw_predict(
        self,
        request: prediction_service.DirectRawPredictRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.DirectRawPredictRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for direct_raw_predict

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_direct_raw_predict(
        self, response: prediction_service.DirectRawPredictResponse
    ) -> prediction_service.DirectRawPredictResponse:
        """Post-rpc interceptor for direct_raw_predict

        DEPRECATED. Please use the `post_direct_raw_predict_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_direct_raw_predict` interceptor runs
        before the `post_direct_raw_predict_with_metadata` interceptor.
        """
        return response

    async def post_direct_raw_predict_with_metadata(
        self,
        response: prediction_service.DirectRawPredictResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.DirectRawPredictResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for direct_raw_predict

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_direct_raw_predict_with_metadata`
        interceptor in new development instead of the `post_direct_raw_predict` interceptor.
        When both interceptors are used, this `post_direct_raw_predict_with_metadata` interceptor runs after the
        `post_direct_raw_predict` interceptor. The (possibly modified) response returned by
        `post_direct_raw_predict` will be passed to
        `post_direct_raw_predict_with_metadata`.
        """
        return response, metadata

    async def pre_explain(
        self,
        request: prediction_service.ExplainRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.ExplainRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for explain

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_explain(
        self, response: prediction_service.ExplainResponse
    ) -> prediction_service.ExplainResponse:
        """Post-rpc interceptor for explain

        DEPRECATED. Please use the `post_explain_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_explain` interceptor runs
        before the `post_explain_with_metadata` interceptor.
        """
        return response

    async def post_explain_with_metadata(
        self,
        response: prediction_service.ExplainResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.ExplainResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for explain

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_explain_with_metadata`
        interceptor in new development instead of the `post_explain` interceptor.
        When both interceptors are used, this `post_explain_with_metadata` interceptor runs after the
        `post_explain` interceptor. The (possibly modified) response returned by
        `post_explain` will be passed to
        `post_explain_with_metadata`.
        """
        return response, metadata

    async def pre_generate_content(
        self,
        request: prediction_service.GenerateContentRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.GenerateContentRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for generate_content

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_generate_content(
        self, response: prediction_service.GenerateContentResponse
    ) -> prediction_service.GenerateContentResponse:
        """Post-rpc interceptor for generate_content

        DEPRECATED. Please use the `post_generate_content_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_generate_content` interceptor runs
        before the `post_generate_content_with_metadata` interceptor.
        """
        return response

    async def post_generate_content_with_metadata(
        self,
        response: prediction_service.GenerateContentResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.GenerateContentResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for generate_content

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_generate_content_with_metadata`
        interceptor in new development instead of the `post_generate_content` interceptor.
        When both interceptors are used, this `post_generate_content_with_metadata` interceptor runs after the
        `post_generate_content` interceptor. The (possibly modified) response returned by
        `post_generate_content` will be passed to
        `post_generate_content_with_metadata`.
        """
        return response, metadata

    async def pre_predict(
        self,
        request: prediction_service.PredictRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.PredictRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for predict

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_predict(
        self, response: prediction_service.PredictResponse
    ) -> prediction_service.PredictResponse:
        """Post-rpc interceptor for predict

        DEPRECATED. Please use the `post_predict_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_predict` interceptor runs
        before the `post_predict_with_metadata` interceptor.
        """
        return response

    async def post_predict_with_metadata(
        self,
        response: prediction_service.PredictResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.PredictResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for predict

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_predict_with_metadata`
        interceptor in new development instead of the `post_predict` interceptor.
        When both interceptors are used, this `post_predict_with_metadata` interceptor runs after the
        `post_predict` interceptor. The (possibly modified) response returned by
        `post_predict` will be passed to
        `post_predict_with_metadata`.
        """
        return response, metadata

    async def pre_raw_predict(
        self,
        request: prediction_service.RawPredictRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.RawPredictRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for raw_predict

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_raw_predict(
        self, response: httpbody_pb2.HttpBody
    ) -> httpbody_pb2.HttpBody:
        """Post-rpc interceptor for raw_predict

        DEPRECATED. Please use the `post_raw_predict_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_raw_predict` interceptor runs
        before the `post_raw_predict_with_metadata` interceptor.
        """
        return response

    async def post_raw_predict_with_metadata(
        self,
        response: httpbody_pb2.HttpBody,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[httpbody_pb2.HttpBody, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for raw_predict

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_raw_predict_with_metadata`
        interceptor in new development instead of the `post_raw_predict` interceptor.
        When both interceptors are used, this `post_raw_predict_with_metadata` interceptor runs after the
        `post_raw_predict` interceptor. The (possibly modified) response returned by
        `post_raw_predict` will be passed to
        `post_raw_predict_with_metadata`.
        """
        return response, metadata

    async def pre_server_streaming_predict(
        self,
        request: prediction_service.StreamingPredictRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.StreamingPredictRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for server_streaming_predict

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_server_streaming_predict(
        self, response: rest_streaming_async.AsyncResponseIterator
    ) -> rest_streaming_async.AsyncResponseIterator:
        """Post-rpc interceptor for server_streaming_predict

        DEPRECATED. Please use the `post_server_streaming_predict_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_server_streaming_predict` interceptor runs
        before the `post_server_streaming_predict_with_metadata` interceptor.
        """
        return response

    async def post_server_streaming_predict_with_metadata(
        self,
        response: rest_streaming_async.AsyncResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming_async.AsyncResponseIterator,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for server_streaming_predict

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_server_streaming_predict_with_metadata`
        interceptor in new development instead of the `post_server_streaming_predict` interceptor.
        When both interceptors are used, this `post_server_streaming_predict_with_metadata` interceptor runs after the
        `post_server_streaming_predict` interceptor. The (possibly modified) response returned by
        `post_server_streaming_predict` will be passed to
        `post_server_streaming_predict_with_metadata`.
        """
        return response, metadata

    async def pre_stream_generate_content(
        self,
        request: prediction_service.GenerateContentRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.GenerateContentRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for stream_generate_content

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_stream_generate_content(
        self, response: rest_streaming_async.AsyncResponseIterator
    ) -> rest_streaming_async.AsyncResponseIterator:
        """Post-rpc interceptor for stream_generate_content

        DEPRECATED. Please use the `post_stream_generate_content_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_stream_generate_content` interceptor runs
        before the `post_stream_generate_content_with_metadata` interceptor.
        """
        return response

    async def post_stream_generate_content_with_metadata(
        self,
        response: rest_streaming_async.AsyncResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming_async.AsyncResponseIterator,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for stream_generate_content

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_stream_generate_content_with_metadata`
        interceptor in new development instead of the `post_stream_generate_content` interceptor.
        When both interceptors are used, this `post_stream_generate_content_with_metadata` interceptor runs after the
        `post_stream_generate_content` interceptor. The (possibly modified) response returned by
        `post_stream_generate_content` will be passed to
        `post_stream_generate_content_with_metadata`.
        """
        return response, metadata

    async def pre_stream_raw_predict(
        self,
        request: prediction_service.StreamRawPredictRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        prediction_service.StreamRawPredictRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for stream_raw_predict

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_stream_raw_predict(
        self, response: rest_streaming_async.AsyncResponseIterator
    ) -> rest_streaming_async.AsyncResponseIterator:
        """Post-rpc interceptor for stream_raw_predict

        DEPRECATED. Please use the `post_stream_raw_predict_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code. This `post_stream_raw_predict` interceptor runs
        before the `post_stream_raw_predict_with_metadata` interceptor.
        """
        return response

    async def post_stream_raw_predict_with_metadata(
        self,
        response: rest_streaming_async.AsyncResponseIterator,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        rest_streaming_async.AsyncResponseIterator,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for stream_raw_predict

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the PredictionService server but before it is returned to user code.

        We recommend only using this `post_stream_raw_predict_with_metadata`
        interceptor in new development instead of the `post_stream_raw_predict` interceptor.
        When both interceptors are used, this `post_stream_raw_predict_with_metadata` interceptor runs after the
        `post_stream_raw_predict` interceptor. The (possibly modified) response returned by
        `post_stream_raw_predict` will be passed to
        `post_stream_raw_predict_with_metadata`.
        """
        return response, metadata

    async def pre_get_location(
        self,
        request: locations_pb2.GetLocationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        locations_pb2.GetLocationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_location

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_get_location(
        self, response: locations_pb2.Location
    ) -> locations_pb2.Location:
        """Post-rpc interceptor for get_location

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_list_locations(
        self,
        request: locations_pb2.ListLocationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        locations_pb2.ListLocationsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_locations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_list_locations(
        self, response: locations_pb2.ListLocationsResponse
    ) -> locations_pb2.ListLocationsResponse:
        """Post-rpc interceptor for list_locations

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_get_iam_policy(
        self,
        request: iam_policy_pb2.GetIamPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.GetIamPolicyRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_get_iam_policy(
        self, response: policy_pb2.Policy
    ) -> policy_pb2.Policy:
        """Post-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_set_iam_policy(
        self,
        request: iam_policy_pb2.SetIamPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.SetIamPolicyRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_set_iam_policy(
        self, response: policy_pb2.Policy
    ) -> policy_pb2.Policy:
        """Post-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_test_iam_permissions(
        self,
        request: iam_policy_pb2.TestIamPermissionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.TestIamPermissionsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_test_iam_permissions(
        self, response: iam_policy_pb2.TestIamPermissionsResponse
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        """Post-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_cancel_operation(
        self,
        request: operations_pb2.CancelOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.CancelOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_delete_operation(
        self,
        request: operations_pb2.DeleteOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.DeleteOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_delete_operation(self, response: None) -> None:
        """Post-rpc interceptor for delete_operation

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_get_operation(
        self,
        request: operations_pb2.GetOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.GetOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_list_operations(
        self,
        request: operations_pb2.ListOperationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.ListOperationsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response

    async def pre_wait_operation(
        self,
        request: operations_pb2.WaitOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.WaitOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for wait_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        return request, metadata

    async def post_wait_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for wait_operation

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class AsyncPredictionServiceRestStub:
    _session: AsyncAuthorizedSession
    _host: str
    _interceptor: AsyncPredictionServiceRestInterceptor


class AsyncPredictionServiceRestTransport(_BasePredictionServiceRestTransport):
    """Asynchronous REST backend transport for PredictionService.

    A service for online predictions and explanations.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "aiplatform.googleapis.com",
        credentials: Optional[ga_credentials_async.Credentials] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        url_scheme: str = "https",
        interceptor: Optional[AsyncPredictionServiceRestInterceptor] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'aiplatform.googleapis.com').
            credentials (Optional[google.auth.aio.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            url_scheme (str): the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        # Run the base constructor
        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=False,
            url_scheme=url_scheme,
            api_audience=None,
        )
        self._session = AsyncAuthorizedSession(self._credentials)  # type: ignore
        self._interceptor = interceptor or AsyncPredictionServiceRestInterceptor()
        self._wrap_with_kind = True
        self._prep_wrapped_messages(client_info)

    def _prep_wrapped_messages(self, client_info):
        """Precompute the wrapped methods, overriding the base class method to use async wrappers."""
        self._wrapped_methods = {
            self.predict: self._wrap_method(
                self.predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.raw_predict: self._wrap_method(
                self.raw_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.stream_raw_predict: self._wrap_method(
                self.stream_raw_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.direct_predict: self._wrap_method(
                self.direct_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.direct_raw_predict: self._wrap_method(
                self.direct_raw_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.stream_direct_predict: self._wrap_method(
                self.stream_direct_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.stream_direct_raw_predict: self._wrap_method(
                self.stream_direct_raw_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.streaming_predict: self._wrap_method(
                self.streaming_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.server_streaming_predict: self._wrap_method(
                self.server_streaming_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.streaming_raw_predict: self._wrap_method(
                self.streaming_raw_predict,
                default_timeout=None,
                client_info=client_info,
            ),
            self.explain: self._wrap_method(
                self.explain,
                default_timeout=None,
                client_info=client_info,
            ),
            self.generate_content: self._wrap_method(
                self.generate_content,
                default_timeout=None,
                client_info=client_info,
            ),
            self.stream_generate_content: self._wrap_method(
                self.stream_generate_content,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_location: self._wrap_method(
                self.get_location,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_locations: self._wrap_method(
                self.list_locations,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_iam_policy: self._wrap_method(
                self.get_iam_policy,
                default_timeout=None,
                client_info=client_info,
            ),
            self.set_iam_policy: self._wrap_method(
                self.set_iam_policy,
                default_timeout=None,
                client_info=client_info,
            ),
            self.test_iam_permissions: self._wrap_method(
                self.test_iam_permissions,
                default_timeout=None,
                client_info=client_info,
            ),
            self.cancel_operation: self._wrap_method(
                self.cancel_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_operation: self._wrap_method(
                self.delete_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_operation: self._wrap_method(
                self.get_operation,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_operations: self._wrap_method(
                self.list_operations,
                default_timeout=None,
                client_info=client_info,
            ),
            self.wait_operation: self._wrap_method(
                self.wait_operation,
                default_timeout=None,
                client_info=client_info,
            ),
        }

    def _wrap_method(self, func, *args, **kwargs):
        if self._wrap_with_kind:  # pragma: NO COVER
            kwargs["kind"] = self.kind
        return gapic_v1.method_async.wrap_method(func, *args, **kwargs)

    class _DirectPredict(
        _BasePredictionServiceRestTransport._BaseDirectPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.DirectPredict")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.DirectPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> prediction_service.DirectPredictResponse:
            r"""Call the direct predict method over HTTP.

            Args:
                request (~.prediction_service.DirectPredictRequest):
                    The request object. Request message for
                [PredictionService.DirectPredict][google.cloud.aiplatform.v1.PredictionService.DirectPredict].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.prediction_service.DirectPredictResponse:
                    Response message for
                [PredictionService.DirectPredict][google.cloud.aiplatform.v1.PredictionService.DirectPredict].

            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseDirectPredict._get_http_options()
            )

            request, metadata = await self._interceptor.pre_direct_predict(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseDirectPredict._get_transcoded_request(
                http_options, request
            )

            body = _BasePredictionServiceRestTransport._BaseDirectPredict._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseDirectPredict._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.DirectPredict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "DirectPredict",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._DirectPredict._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = prediction_service.DirectPredictResponse()
            pb_resp = prediction_service.DirectPredictResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_direct_predict(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_direct_predict_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = prediction_service.DirectPredictResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.direct_predict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "DirectPredict",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _DirectRawPredict(
        _BasePredictionServiceRestTransport._BaseDirectRawPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.DirectRawPredict")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.DirectRawPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> prediction_service.DirectRawPredictResponse:
            r"""Call the direct raw predict method over HTTP.

            Args:
                request (~.prediction_service.DirectRawPredictRequest):
                    The request object. Request message for
                [PredictionService.DirectRawPredict][google.cloud.aiplatform.v1.PredictionService.DirectRawPredict].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.prediction_service.DirectRawPredictResponse:
                    Response message for
                [PredictionService.DirectRawPredict][google.cloud.aiplatform.v1.PredictionService.DirectRawPredict].

            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseDirectRawPredict._get_http_options()
            )

            request, metadata = await self._interceptor.pre_direct_raw_predict(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseDirectRawPredict._get_transcoded_request(
                http_options, request
            )

            body = _BasePredictionServiceRestTransport._BaseDirectRawPredict._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseDirectRawPredict._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.DirectRawPredict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "DirectRawPredict",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._DirectRawPredict._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = prediction_service.DirectRawPredictResponse()
            pb_resp = prediction_service.DirectRawPredictResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_direct_raw_predict(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_direct_raw_predict_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        prediction_service.DirectRawPredictResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.direct_raw_predict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "DirectRawPredict",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _Explain(
        _BasePredictionServiceRestTransport._BaseExplain, AsyncPredictionServiceRestStub
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.Explain")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.ExplainRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> prediction_service.ExplainResponse:
            r"""Call the explain method over HTTP.

            Args:
                request (~.prediction_service.ExplainRequest):
                    The request object. Request message for
                [PredictionService.Explain][google.cloud.aiplatform.v1.PredictionService.Explain].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.prediction_service.ExplainResponse:
                    Response message for
                [PredictionService.Explain][google.cloud.aiplatform.v1.PredictionService.Explain].

            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseExplain._get_http_options()
            )

            request, metadata = await self._interceptor.pre_explain(request, metadata)
            transcoded_request = _BasePredictionServiceRestTransport._BaseExplain._get_transcoded_request(
                http_options, request
            )

            body = (
                _BasePredictionServiceRestTransport._BaseExplain._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BasePredictionServiceRestTransport._BaseExplain._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.Explain",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "Explain",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._Explain._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = prediction_service.ExplainResponse()
            pb_resp = prediction_service.ExplainResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_explain(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_explain_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = prediction_service.ExplainResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.explain",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "Explain",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _GenerateContent(
        _BasePredictionServiceRestTransport._BaseGenerateContent,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.GenerateContent")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.GenerateContentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> prediction_service.GenerateContentResponse:
            r"""Call the generate content method over HTTP.

            Args:
                request (~.prediction_service.GenerateContentRequest):
                    The request object. Request message for [PredictionService.GenerateContent].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.prediction_service.GenerateContentResponse:
                    Response message for
                [PredictionService.GenerateContent].

            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseGenerateContent._get_http_options()
            )

            request, metadata = await self._interceptor.pre_generate_content(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseGenerateContent._get_transcoded_request(
                http_options, request
            )

            body = _BasePredictionServiceRestTransport._BaseGenerateContent._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseGenerateContent._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.GenerateContent",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "GenerateContent",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._GenerateContent._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = prediction_service.GenerateContentResponse()
            pb_resp = prediction_service.GenerateContentResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_generate_content(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_generate_content_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        prediction_service.GenerateContentResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.generate_content",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "GenerateContent",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _Predict(
        _BasePredictionServiceRestTransport._BasePredict, AsyncPredictionServiceRestStub
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.Predict")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.PredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> prediction_service.PredictResponse:
            r"""Call the predict method over HTTP.

            Args:
                request (~.prediction_service.PredictRequest):
                    The request object. Request message for
                [PredictionService.Predict][google.cloud.aiplatform.v1.PredictionService.Predict].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.prediction_service.PredictResponse:
                    Response message for
                [PredictionService.Predict][google.cloud.aiplatform.v1.PredictionService.Predict].

            """

            http_options = (
                _BasePredictionServiceRestTransport._BasePredict._get_http_options()
            )

            request, metadata = await self._interceptor.pre_predict(request, metadata)
            transcoded_request = _BasePredictionServiceRestTransport._BasePredict._get_transcoded_request(
                http_options, request
            )

            body = (
                _BasePredictionServiceRestTransport._BasePredict._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BasePredictionServiceRestTransport._BasePredict._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.Predict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "Predict",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._Predict._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = prediction_service.PredictResponse()
            pb_resp = prediction_service.PredictResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_predict(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_predict_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = prediction_service.PredictResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.predict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "Predict",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _RawPredict(
        _BasePredictionServiceRestTransport._BaseRawPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.RawPredict")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.RawPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> httpbody_pb2.HttpBody:
            r"""Call the raw predict method over HTTP.

            Args:
                request (~.prediction_service.RawPredictRequest):
                    The request object. Request message for
                [PredictionService.RawPredict][google.cloud.aiplatform.v1.PredictionService.RawPredict].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.httpbody_pb2.HttpBody:
                    Message that represents an arbitrary HTTP body. It
                should only be used for payload formats that can't be
                represented as JSON, such as raw binary or an HTML page.

                This message can be used both in streaming and
                non-streaming API methods in the request as well as the
                response.

                It can be used as a top-level request field, which is
                convenient if one wants to extract parameters from
                either the URL or HTTP template into the request fields
                and also want access to the raw HTTP body.

                Example:

                ::

                    message GetResourceRequest {
                      // A unique request id.
                      string request_id = 1;

                      // The raw HTTP body is bound to this field.
                      google.api.HttpBody http_body = 2;

                    }

                    service ResourceService {
                      rpc GetResource(GetResourceRequest)
                        returns (google.api.HttpBody);
                      rpc UpdateResource(google.api.HttpBody)
                        returns (google.protobuf.Empty);

                    }

                Example with streaming methods:

                ::

                    service CaldavService {
                      rpc GetCalendar(stream google.api.HttpBody)
                        returns (stream google.api.HttpBody);
                      rpc UpdateCalendar(stream google.api.HttpBody)
                        returns (stream google.api.HttpBody);

                    }

                Use of this type only changes how the request and
                response bodies are handled, all other features will
                continue to work unchanged.

            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseRawPredict._get_http_options()
            )

            request, metadata = await self._interceptor.pre_raw_predict(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseRawPredict._get_transcoded_request(
                http_options, request
            )

            body = _BasePredictionServiceRestTransport._BaseRawPredict._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseRawPredict._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.RawPredict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "RawPredict",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._RawPredict._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = httpbody_pb2.HttpBody()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_raw_predict(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_raw_predict_with_metadata(
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
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.raw_predict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "RawPredict",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _ServerStreamingPredict(
        _BasePredictionServiceRestTransport._BaseServerStreamingPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.ServerStreamingPredict")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.StreamingPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming_async.AsyncResponseIterator:
            r"""Call the server streaming predict method over HTTP.

            Args:
                request (~.prediction_service.StreamingPredictRequest):
                    The request object. Request message for
                [PredictionService.StreamingPredict][google.cloud.aiplatform.v1.PredictionService.StreamingPredict].

                The first message must contain
                [endpoint][google.cloud.aiplatform.v1.StreamingPredictRequest.endpoint]
                field and optionally [input][]. The subsequent messages
                must contain [input][].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.prediction_service.StreamingPredictResponse:
                    Response message for
                [PredictionService.StreamingPredict][google.cloud.aiplatform.v1.PredictionService.StreamingPredict].

            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseServerStreamingPredict._get_http_options()
            )

            request, metadata = await self._interceptor.pre_server_streaming_predict(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseServerStreamingPredict._get_transcoded_request(
                http_options, request
            )

            body = _BasePredictionServiceRestTransport._BaseServerStreamingPredict._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseServerStreamingPredict._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.ServerStreamingPredict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "ServerStreamingPredict",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._ServerStreamingPredict._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = rest_streaming_async.AsyncResponseIterator(
                response, prediction_service.StreamingPredictResponse
            )
            resp = await self._interceptor.post_server_streaming_predict(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                await self._interceptor.post_server_streaming_predict_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                http_response = {
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.server_streaming_predict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "ServerStreamingPredict",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _StreamDirectPredict(
        _BasePredictionServiceRestTransport._BaseStreamDirectPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.StreamDirectPredict")

        async def __call__(
            self,
            request: prediction_service.StreamDirectPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming_async.AsyncResponseIterator:
            raise NotImplementedError(
                "Method StreamDirectPredict is not available over REST transport"
            )

    class _StreamDirectRawPredict(
        _BasePredictionServiceRestTransport._BaseStreamDirectRawPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.StreamDirectRawPredict")

        async def __call__(
            self,
            request: prediction_service.StreamDirectRawPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming_async.AsyncResponseIterator:
            raise NotImplementedError(
                "Method StreamDirectRawPredict is not available over REST transport"
            )

    class _StreamGenerateContent(
        _BasePredictionServiceRestTransport._BaseStreamGenerateContent,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.StreamGenerateContent")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.GenerateContentRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming_async.AsyncResponseIterator:
            r"""Call the stream generate content method over HTTP.

            Args:
                request (~.prediction_service.GenerateContentRequest):
                    The request object. Request message for [PredictionService.GenerateContent].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.prediction_service.GenerateContentResponse:
                    Response message for
                [PredictionService.GenerateContent].

            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseStreamGenerateContent._get_http_options()
            )

            request, metadata = await self._interceptor.pre_stream_generate_content(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseStreamGenerateContent._get_transcoded_request(
                http_options, request
            )

            body = _BasePredictionServiceRestTransport._BaseStreamGenerateContent._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseStreamGenerateContent._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.StreamGenerateContent",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "StreamGenerateContent",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._StreamGenerateContent._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = rest_streaming_async.AsyncResponseIterator(
                response, prediction_service.GenerateContentResponse
            )
            resp = await self._interceptor.post_stream_generate_content(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                await self._interceptor.post_stream_generate_content_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                http_response = {
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.stream_generate_content",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "StreamGenerateContent",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _StreamingPredict(
        _BasePredictionServiceRestTransport._BaseStreamingPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.StreamingPredict")

        async def __call__(
            self,
            request: prediction_service.StreamingPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming_async.AsyncResponseIterator:
            raise NotImplementedError(
                "Method StreamingPredict is not available over REST transport"
            )

    class _StreamingRawPredict(
        _BasePredictionServiceRestTransport._BaseStreamingRawPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.StreamingRawPredict")

        async def __call__(
            self,
            request: prediction_service.StreamingRawPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming_async.AsyncResponseIterator:
            raise NotImplementedError(
                "Method StreamingRawPredict is not available over REST transport"
            )

    class _StreamRawPredict(
        _BasePredictionServiceRestTransport._BaseStreamRawPredict,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.StreamRawPredict")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: prediction_service.StreamRawPredictRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming_async.AsyncResponseIterator:
            r"""Call the stream raw predict method over HTTP.

            Args:
                request (~.prediction_service.StreamRawPredictRequest):
                    The request object. Request message for
                [PredictionService.StreamRawPredict][google.cloud.aiplatform.v1.PredictionService.StreamRawPredict].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.httpbody_pb2.HttpBody:
                    Message that represents an arbitrary HTTP body. It
                should only be used for payload formats that can't be
                represented as JSON, such as raw binary or an HTML page.

                This message can be used both in streaming and
                non-streaming API methods in the request as well as the
                response.

                It can be used as a top-level request field, which is
                convenient if one wants to extract parameters from
                either the URL or HTTP template into the request fields
                and also want access to the raw HTTP body.

                Example:

                ::

                    message GetResourceRequest {
                      // A unique request id.
                      string request_id = 1;

                      // The raw HTTP body is bound to this field.
                      google.api.HttpBody http_body = 2;

                    }

                    service ResourceService {
                      rpc GetResource(GetResourceRequest)
                        returns (google.api.HttpBody);
                      rpc UpdateResource(google.api.HttpBody)
                        returns (google.protobuf.Empty);

                    }

                Example with streaming methods:

                ::

                    service CaldavService {
                      rpc GetCalendar(stream google.api.HttpBody)
                        returns (stream google.api.HttpBody);
                      rpc UpdateCalendar(stream google.api.HttpBody)
                        returns (stream google.api.HttpBody);

                    }

                Use of this type only changes how the request and
                response bodies are handled, all other features will
                continue to work unchanged.

            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseStreamRawPredict._get_http_options()
            )

            request, metadata = await self._interceptor.pre_stream_raw_predict(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseStreamRawPredict._get_transcoded_request(
                http_options, request
            )

            body = _BasePredictionServiceRestTransport._BaseStreamRawPredict._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseStreamRawPredict._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.StreamRawPredict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "StreamRawPredict",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._StreamRawPredict._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            # Return the response
            resp = rest_streaming_async.AsyncResponseIterator(
                response, httpbody_pb2.HttpBody
            )
            resp = await self._interceptor.post_stream_raw_predict(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_stream_raw_predict_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                http_response = {
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.stream_raw_predict",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "StreamRawPredict",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    @property
    def direct_predict(
        self,
    ) -> Callable[
        [prediction_service.DirectPredictRequest],
        prediction_service.DirectPredictResponse,
    ]:
        return self._DirectPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def direct_raw_predict(
        self,
    ) -> Callable[
        [prediction_service.DirectRawPredictRequest],
        prediction_service.DirectRawPredictResponse,
    ]:
        return self._DirectRawPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def explain(
        self,
    ) -> Callable[
        [prediction_service.ExplainRequest], prediction_service.ExplainResponse
    ]:
        return self._Explain(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def generate_content(
        self,
    ) -> Callable[
        [prediction_service.GenerateContentRequest],
        prediction_service.GenerateContentResponse,
    ]:
        return self._GenerateContent(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def predict(
        self,
    ) -> Callable[
        [prediction_service.PredictRequest], prediction_service.PredictResponse
    ]:
        return self._Predict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def raw_predict(
        self,
    ) -> Callable[[prediction_service.RawPredictRequest], httpbody_pb2.HttpBody]:
        return self._RawPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def server_streaming_predict(
        self,
    ) -> Callable[
        [prediction_service.StreamingPredictRequest],
        prediction_service.StreamingPredictResponse,
    ]:
        return self._ServerStreamingPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def stream_direct_predict(
        self,
    ) -> Callable[
        [prediction_service.StreamDirectPredictRequest],
        prediction_service.StreamDirectPredictResponse,
    ]:
        return self._StreamDirectPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def stream_direct_raw_predict(
        self,
    ) -> Callable[
        [prediction_service.StreamDirectRawPredictRequest],
        prediction_service.StreamDirectRawPredictResponse,
    ]:
        return self._StreamDirectRawPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def stream_generate_content(
        self,
    ) -> Callable[
        [prediction_service.GenerateContentRequest],
        prediction_service.GenerateContentResponse,
    ]:
        return self._StreamGenerateContent(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def streaming_predict(
        self,
    ) -> Callable[
        [prediction_service.StreamingPredictRequest],
        prediction_service.StreamingPredictResponse,
    ]:
        return self._StreamingPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def streaming_raw_predict(
        self,
    ) -> Callable[
        [prediction_service.StreamingRawPredictRequest],
        prediction_service.StreamingRawPredictResponse,
    ]:
        return self._StreamingRawPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def stream_raw_predict(
        self,
    ) -> Callable[[prediction_service.StreamRawPredictRequest], httpbody_pb2.HttpBody]:
        return self._StreamRawPredict(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_location(self):
        return self._GetLocation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetLocation(
        _BasePredictionServiceRestTransport._BaseGetLocation,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.GetLocation")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
            self,
            request: locations_pb2.GetLocationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> locations_pb2.Location:
            r"""Call the get location method over HTTP.

            Args:
                request (locations_pb2.GetLocationRequest):
                    The request object for GetLocation method.
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                locations_pb2.Location: Response from GetLocation method.
            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseGetLocation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_location(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseGetLocation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseGetLocation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.GetLocation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "GetLocation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._GetLocation._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            content = await response.read()
            resp = locations_pb2.Location()
            resp = json_format.Parse(content, resp)
            resp = await self._interceptor.post_get_location(resp)
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
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.GetLocation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "GetLocation",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def list_locations(self):
        return self._ListLocations(self._session, self._host, self._interceptor)  # type: ignore

    class _ListLocations(
        _BasePredictionServiceRestTransport._BaseListLocations,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.ListLocations")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
            self,
            request: locations_pb2.ListLocationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> locations_pb2.ListLocationsResponse:
            r"""Call the list locations method over HTTP.

            Args:
                request (locations_pb2.ListLocationsRequest):
                    The request object for ListLocations method.
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                locations_pb2.ListLocationsResponse: Response from ListLocations method.
            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseListLocations._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_locations(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseListLocations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseListLocations._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.ListLocations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "ListLocations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._ListLocations._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            content = await response.read()
            resp = locations_pb2.ListLocationsResponse()
            resp = json_format.Parse(content, resp)
            resp = await self._interceptor.post_list_locations(resp)
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
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.ListLocations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "ListLocations",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def get_iam_policy(self):
        return self._GetIamPolicy(self._session, self._host, self._interceptor)  # type: ignore

    class _GetIamPolicy(
        _BasePredictionServiceRestTransport._BaseGetIamPolicy,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.GetIamPolicy")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
            self,
            request: iam_policy_pb2.GetIamPolicyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> policy_pb2.Policy:
            r"""Call the get iam policy method over HTTP.

            Args:
                request (iam_policy_pb2.GetIamPolicyRequest):
                    The request object for GetIamPolicy method.
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                policy_pb2.Policy: Response from GetIamPolicy method.
            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseGetIamPolicy._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_iam_policy(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseGetIamPolicy._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseGetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "GetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._GetIamPolicy._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            content = await response.read()
            resp = policy_pb2.Policy()
            resp = json_format.Parse(content, resp)
            resp = await self._interceptor.post_get_iam_policy(resp)
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
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "GetIamPolicy",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def set_iam_policy(self):
        return self._SetIamPolicy(self._session, self._host, self._interceptor)  # type: ignore

    class _SetIamPolicy(
        _BasePredictionServiceRestTransport._BaseSetIamPolicy,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.SetIamPolicy")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        async def __call__(
            self,
            request: iam_policy_pb2.SetIamPolicyRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> policy_pb2.Policy:
            r"""Call the set iam policy method over HTTP.

            Args:
                request (iam_policy_pb2.SetIamPolicyRequest):
                    The request object for SetIamPolicy method.
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                policy_pb2.Policy: Response from SetIamPolicy method.
            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseSetIamPolicy._get_http_options()
            )

            request, metadata = await self._interceptor.pre_set_iam_policy(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseSetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BasePredictionServiceRestTransport._BaseSetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseSetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "SetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._SetIamPolicy._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            content = await response.read()
            resp = policy_pb2.Policy()
            resp = json_format.Parse(content, resp)
            resp = await self._interceptor.post_set_iam_policy(resp)
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
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "SetIamPolicy",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def test_iam_permissions(self):
        return self._TestIamPermissions(self._session, self._host, self._interceptor)  # type: ignore

    class _TestIamPermissions(
        _BasePredictionServiceRestTransport._BaseTestIamPermissions,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.TestIamPermissions")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
            self,
            request: iam_policy_pb2.TestIamPermissionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> iam_policy_pb2.TestIamPermissionsResponse:
            r"""Call the test iam permissions method over HTTP.

            Args:
                request (iam_policy_pb2.TestIamPermissionsRequest):
                    The request object for TestIamPermissions method.
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                iam_policy_pb2.TestIamPermissionsResponse: Response from TestIamPermissions method.
            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseTestIamPermissions._get_http_options()
            )

            request, metadata = await self._interceptor.pre_test_iam_permissions(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseTestIamPermissions._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseTestIamPermissions._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "TestIamPermissions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._TestIamPermissions._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            content = await response.read()
            resp = iam_policy_pb2.TestIamPermissionsResponse()
            resp = json_format.Parse(content, resp)
            resp = await self._interceptor.post_test_iam_permissions(resp)
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
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "TestIamPermissions",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def cancel_operation(self):
        return self._CancelOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _CancelOperation(
        _BasePredictionServiceRestTransport._BaseCancelOperation,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.CancelOperation")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
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
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseCancelOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.CancelOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._CancelOperation._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            return await self._interceptor.post_cancel_operation(None)

    @property
    def delete_operation(self):
        return self._DeleteOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _DeleteOperation(
        _BasePredictionServiceRestTransport._BaseDeleteOperation,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.DeleteOperation")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
            self,
            request: operations_pb2.DeleteOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> None:
            r"""Call the delete operation method over HTTP.

            Args:
                request (operations_pb2.DeleteOperationRequest):
                    The request object for DeleteOperation method.
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseDeleteOperation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_delete_operation(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseDeleteOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseDeleteOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.DeleteOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "DeleteOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncPredictionServiceRestTransport._DeleteOperation._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            return await self._interceptor.post_delete_operation(None)

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BasePredictionServiceRestTransport._BaseGetOperation,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.GetOperation")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
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
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BasePredictionServiceRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_operation(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._GetOperation._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            content = await response.read()
            resp = operations_pb2.Operation()
            resp = json_format.Parse(content, resp)
            resp = await self._interceptor.post_get_operation(resp)
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
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
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
        _BasePredictionServiceRestTransport._BaseListOperations,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.ListOperations")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
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
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BasePredictionServiceRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_operations(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._ListOperations._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            content = await response.read()
            resp = operations_pb2.ListOperationsResponse()
            resp = json_format.Parse(content, resp)
            resp = await self._interceptor.post_list_operations(resp)
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
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "ListOperations",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def wait_operation(self):
        return self._WaitOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _WaitOperation(
        _BasePredictionServiceRestTransport._BaseWaitOperation,
        AsyncPredictionServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncPredictionServiceRestTransport.WaitOperation")

        @staticmethod
        async def _get_response(
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
            response = await getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        async def __call__(
            self,
            request: operations_pb2.WaitOperationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the wait operation method over HTTP.

            Args:
                request (operations_pb2.WaitOperationRequest):
                    The request object for WaitOperation method.
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                operations_pb2.Operation: Response from WaitOperation method.
            """

            http_options = (
                _BasePredictionServiceRestTransport._BaseWaitOperation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_wait_operation(
                request, metadata
            )
            transcoded_request = _BasePredictionServiceRestTransport._BaseWaitOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BasePredictionServiceRestTransport._BaseWaitOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1.PredictionServiceClient.WaitOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "WaitOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncPredictionServiceRestTransport._WaitOperation._get_response(
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
                content = await response.read()
                payload = json.loads(content.decode("utf-8"))
                request_url = "{host}{uri}".format(
                    host=self._host, uri=transcoded_request["uri"]
                )
                method = transcoded_request["method"]
                raise core_exceptions.format_http_response_error(response, method, request_url, payload)  # type: ignore

            content = await response.read()
            resp = operations_pb2.Operation()
            resp = json_format.Parse(content, resp)
            resp = await self._interceptor.post_wait_operation(resp)
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
                    "Received response for google.cloud.aiplatform_v1.PredictionServiceAsyncClient.WaitOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1.PredictionService",
                        "rpcName": "WaitOperation",
                        "httpResponse": http_response,
                        "metadata": http_response["headers"],
                    },
                )
            return resp

    @property
    def kind(self) -> str:
        return "rest_asyncio"

    async def close(self):
        await self._session.close()
