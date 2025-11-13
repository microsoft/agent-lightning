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
from google.api_core import operations_v1
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.api_core import retry_async as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming_async  # type: ignore
import google.protobuf

from google.protobuf import json_format
from google.api_core import operations_v1
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore

import json  # type: ignore
import dataclasses
from typing import Any, Dict, List, Callable, Tuple, Optional, Sequence, Union


from google.cloud.aiplatform_v1beta1.types import artifact
from google.cloud.aiplatform_v1beta1.types import artifact as gca_artifact
from google.cloud.aiplatform_v1beta1.types import context
from google.cloud.aiplatform_v1beta1.types import context as gca_context
from google.cloud.aiplatform_v1beta1.types import execution
from google.cloud.aiplatform_v1beta1.types import execution as gca_execution
from google.cloud.aiplatform_v1beta1.types import lineage_subgraph
from google.cloud.aiplatform_v1beta1.types import metadata_schema
from google.cloud.aiplatform_v1beta1.types import metadata_schema as gca_metadata_schema
from google.cloud.aiplatform_v1beta1.types import metadata_service
from google.cloud.aiplatform_v1beta1.types import metadata_store
from google.longrunning import operations_pb2  # type: ignore


from .rest_base import _BaseMetadataServiceRestTransport

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


class AsyncMetadataServiceRestInterceptor:
    """Asynchronous Interceptor for MetadataService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the AsyncMetadataServiceRestTransport.

    .. code-block:: python
        class MyCustomMetadataServiceInterceptor(MetadataServiceRestInterceptor):
            async def pre_add_context_artifacts_and_executions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_add_context_artifacts_and_executions(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_add_context_children(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_add_context_children(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_add_execution_events(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_add_execution_events(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_create_artifact(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_create_artifact(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_create_context(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_create_context(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_create_execution(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_create_execution(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_create_metadata_schema(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_create_metadata_schema(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_create_metadata_store(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_create_metadata_store(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_delete_artifact(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_delete_artifact(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_delete_context(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_delete_context(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_delete_execution(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_delete_execution(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_delete_metadata_store(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_delete_metadata_store(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_get_artifact(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_get_artifact(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_get_context(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_get_context(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_get_execution(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_get_execution(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_get_metadata_schema(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_get_metadata_schema(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_get_metadata_store(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_get_metadata_store(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_list_artifacts(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_list_artifacts(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_list_contexts(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_list_contexts(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_list_executions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_list_executions(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_list_metadata_schemas(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_list_metadata_schemas(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_list_metadata_stores(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_list_metadata_stores(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_purge_artifacts(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_purge_artifacts(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_purge_contexts(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_purge_contexts(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_purge_executions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_purge_executions(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_query_artifact_lineage_subgraph(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_query_artifact_lineage_subgraph(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_query_context_lineage_subgraph(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_query_context_lineage_subgraph(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_query_execution_inputs_and_outputs(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_query_execution_inputs_and_outputs(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_remove_context_children(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_remove_context_children(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_update_artifact(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_update_artifact(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_update_context(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_update_context(self, response):
                logging.log(f"Received response: {response}")
                return response

            async def pre_update_execution(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            async def post_update_execution(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = AsyncMetadataServiceRestTransport(interceptor=MyCustomMetadataServiceInterceptor())
        client = async MetadataServiceClient(transport=transport)


    """

    async def pre_add_context_artifacts_and_executions(
        self,
        request: metadata_service.AddContextArtifactsAndExecutionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.AddContextArtifactsAndExecutionsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for add_context_artifacts_and_executions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_add_context_artifacts_and_executions(
        self, response: metadata_service.AddContextArtifactsAndExecutionsResponse
    ) -> metadata_service.AddContextArtifactsAndExecutionsResponse:
        """Post-rpc interceptor for add_context_artifacts_and_executions

        DEPRECATED. Please use the `post_add_context_artifacts_and_executions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_add_context_artifacts_and_executions` interceptor runs
        before the `post_add_context_artifacts_and_executions_with_metadata` interceptor.
        """
        return response

    async def post_add_context_artifacts_and_executions_with_metadata(
        self,
        response: metadata_service.AddContextArtifactsAndExecutionsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.AddContextArtifactsAndExecutionsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for add_context_artifacts_and_executions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_add_context_artifacts_and_executions_with_metadata`
        interceptor in new development instead of the `post_add_context_artifacts_and_executions` interceptor.
        When both interceptors are used, this `post_add_context_artifacts_and_executions_with_metadata` interceptor runs after the
        `post_add_context_artifacts_and_executions` interceptor. The (possibly modified) response returned by
        `post_add_context_artifacts_and_executions` will be passed to
        `post_add_context_artifacts_and_executions_with_metadata`.
        """
        return response, metadata

    async def pre_add_context_children(
        self,
        request: metadata_service.AddContextChildrenRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.AddContextChildrenRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for add_context_children

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_add_context_children(
        self, response: metadata_service.AddContextChildrenResponse
    ) -> metadata_service.AddContextChildrenResponse:
        """Post-rpc interceptor for add_context_children

        DEPRECATED. Please use the `post_add_context_children_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_add_context_children` interceptor runs
        before the `post_add_context_children_with_metadata` interceptor.
        """
        return response

    async def post_add_context_children_with_metadata(
        self,
        response: metadata_service.AddContextChildrenResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.AddContextChildrenResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for add_context_children

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_add_context_children_with_metadata`
        interceptor in new development instead of the `post_add_context_children` interceptor.
        When both interceptors are used, this `post_add_context_children_with_metadata` interceptor runs after the
        `post_add_context_children` interceptor. The (possibly modified) response returned by
        `post_add_context_children` will be passed to
        `post_add_context_children_with_metadata`.
        """
        return response, metadata

    async def pre_add_execution_events(
        self,
        request: metadata_service.AddExecutionEventsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.AddExecutionEventsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for add_execution_events

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_add_execution_events(
        self, response: metadata_service.AddExecutionEventsResponse
    ) -> metadata_service.AddExecutionEventsResponse:
        """Post-rpc interceptor for add_execution_events

        DEPRECATED. Please use the `post_add_execution_events_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_add_execution_events` interceptor runs
        before the `post_add_execution_events_with_metadata` interceptor.
        """
        return response

    async def post_add_execution_events_with_metadata(
        self,
        response: metadata_service.AddExecutionEventsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.AddExecutionEventsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for add_execution_events

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_add_execution_events_with_metadata`
        interceptor in new development instead of the `post_add_execution_events` interceptor.
        When both interceptors are used, this `post_add_execution_events_with_metadata` interceptor runs after the
        `post_add_execution_events` interceptor. The (possibly modified) response returned by
        `post_add_execution_events` will be passed to
        `post_add_execution_events_with_metadata`.
        """
        return response, metadata

    async def pre_create_artifact(
        self,
        request: metadata_service.CreateArtifactRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.CreateArtifactRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for create_artifact

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_create_artifact(
        self, response: gca_artifact.Artifact
    ) -> gca_artifact.Artifact:
        """Post-rpc interceptor for create_artifact

        DEPRECATED. Please use the `post_create_artifact_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_create_artifact` interceptor runs
        before the `post_create_artifact_with_metadata` interceptor.
        """
        return response

    async def post_create_artifact_with_metadata(
        self,
        response: gca_artifact.Artifact,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gca_artifact.Artifact, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_artifact

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_create_artifact_with_metadata`
        interceptor in new development instead of the `post_create_artifact` interceptor.
        When both interceptors are used, this `post_create_artifact_with_metadata` interceptor runs after the
        `post_create_artifact` interceptor. The (possibly modified) response returned by
        `post_create_artifact` will be passed to
        `post_create_artifact_with_metadata`.
        """
        return response, metadata

    async def pre_create_context(
        self,
        request: metadata_service.CreateContextRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.CreateContextRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for create_context

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_create_context(
        self, response: gca_context.Context
    ) -> gca_context.Context:
        """Post-rpc interceptor for create_context

        DEPRECATED. Please use the `post_create_context_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_create_context` interceptor runs
        before the `post_create_context_with_metadata` interceptor.
        """
        return response

    async def post_create_context_with_metadata(
        self,
        response: gca_context.Context,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gca_context.Context, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_context

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_create_context_with_metadata`
        interceptor in new development instead of the `post_create_context` interceptor.
        When both interceptors are used, this `post_create_context_with_metadata` interceptor runs after the
        `post_create_context` interceptor. The (possibly modified) response returned by
        `post_create_context` will be passed to
        `post_create_context_with_metadata`.
        """
        return response, metadata

    async def pre_create_execution(
        self,
        request: metadata_service.CreateExecutionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.CreateExecutionRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for create_execution

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_create_execution(
        self, response: gca_execution.Execution
    ) -> gca_execution.Execution:
        """Post-rpc interceptor for create_execution

        DEPRECATED. Please use the `post_create_execution_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_create_execution` interceptor runs
        before the `post_create_execution_with_metadata` interceptor.
        """
        return response

    async def post_create_execution_with_metadata(
        self,
        response: gca_execution.Execution,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gca_execution.Execution, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_execution

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_create_execution_with_metadata`
        interceptor in new development instead of the `post_create_execution` interceptor.
        When both interceptors are used, this `post_create_execution_with_metadata` interceptor runs after the
        `post_create_execution` interceptor. The (possibly modified) response returned by
        `post_create_execution` will be passed to
        `post_create_execution_with_metadata`.
        """
        return response, metadata

    async def pre_create_metadata_schema(
        self,
        request: metadata_service.CreateMetadataSchemaRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.CreateMetadataSchemaRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_metadata_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_create_metadata_schema(
        self, response: gca_metadata_schema.MetadataSchema
    ) -> gca_metadata_schema.MetadataSchema:
        """Post-rpc interceptor for create_metadata_schema

        DEPRECATED. Please use the `post_create_metadata_schema_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_create_metadata_schema` interceptor runs
        before the `post_create_metadata_schema_with_metadata` interceptor.
        """
        return response

    async def post_create_metadata_schema_with_metadata(
        self,
        response: gca_metadata_schema.MetadataSchema,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gca_metadata_schema.MetadataSchema, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for create_metadata_schema

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_create_metadata_schema_with_metadata`
        interceptor in new development instead of the `post_create_metadata_schema` interceptor.
        When both interceptors are used, this `post_create_metadata_schema_with_metadata` interceptor runs after the
        `post_create_metadata_schema` interceptor. The (possibly modified) response returned by
        `post_create_metadata_schema` will be passed to
        `post_create_metadata_schema_with_metadata`.
        """
        return response, metadata

    async def pre_create_metadata_store(
        self,
        request: metadata_service.CreateMetadataStoreRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.CreateMetadataStoreRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_metadata_store

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_create_metadata_store(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_metadata_store

        DEPRECATED. Please use the `post_create_metadata_store_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_create_metadata_store` interceptor runs
        before the `post_create_metadata_store_with_metadata` interceptor.
        """
        return response

    async def post_create_metadata_store_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_metadata_store

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_create_metadata_store_with_metadata`
        interceptor in new development instead of the `post_create_metadata_store` interceptor.
        When both interceptors are used, this `post_create_metadata_store_with_metadata` interceptor runs after the
        `post_create_metadata_store` interceptor. The (possibly modified) response returned by
        `post_create_metadata_store` will be passed to
        `post_create_metadata_store_with_metadata`.
        """
        return response, metadata

    async def pre_delete_artifact(
        self,
        request: metadata_service.DeleteArtifactRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.DeleteArtifactRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_artifact

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_delete_artifact(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_artifact

        DEPRECATED. Please use the `post_delete_artifact_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_delete_artifact` interceptor runs
        before the `post_delete_artifact_with_metadata` interceptor.
        """
        return response

    async def post_delete_artifact_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_artifact

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_delete_artifact_with_metadata`
        interceptor in new development instead of the `post_delete_artifact` interceptor.
        When both interceptors are used, this `post_delete_artifact_with_metadata` interceptor runs after the
        `post_delete_artifact` interceptor. The (possibly modified) response returned by
        `post_delete_artifact` will be passed to
        `post_delete_artifact_with_metadata`.
        """
        return response, metadata

    async def pre_delete_context(
        self,
        request: metadata_service.DeleteContextRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.DeleteContextRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_context

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_delete_context(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_context

        DEPRECATED. Please use the `post_delete_context_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_delete_context` interceptor runs
        before the `post_delete_context_with_metadata` interceptor.
        """
        return response

    async def post_delete_context_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_context

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_delete_context_with_metadata`
        interceptor in new development instead of the `post_delete_context` interceptor.
        When both interceptors are used, this `post_delete_context_with_metadata` interceptor runs after the
        `post_delete_context` interceptor. The (possibly modified) response returned by
        `post_delete_context` will be passed to
        `post_delete_context_with_metadata`.
        """
        return response, metadata

    async def pre_delete_execution(
        self,
        request: metadata_service.DeleteExecutionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.DeleteExecutionRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_execution

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_delete_execution(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_execution

        DEPRECATED. Please use the `post_delete_execution_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_delete_execution` interceptor runs
        before the `post_delete_execution_with_metadata` interceptor.
        """
        return response

    async def post_delete_execution_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_execution

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_delete_execution_with_metadata`
        interceptor in new development instead of the `post_delete_execution` interceptor.
        When both interceptors are used, this `post_delete_execution_with_metadata` interceptor runs after the
        `post_delete_execution` interceptor. The (possibly modified) response returned by
        `post_delete_execution` will be passed to
        `post_delete_execution_with_metadata`.
        """
        return response, metadata

    async def pre_delete_metadata_store(
        self,
        request: metadata_service.DeleteMetadataStoreRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.DeleteMetadataStoreRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_metadata_store

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_delete_metadata_store(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_metadata_store

        DEPRECATED. Please use the `post_delete_metadata_store_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_delete_metadata_store` interceptor runs
        before the `post_delete_metadata_store_with_metadata` interceptor.
        """
        return response

    async def post_delete_metadata_store_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_metadata_store

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_delete_metadata_store_with_metadata`
        interceptor in new development instead of the `post_delete_metadata_store` interceptor.
        When both interceptors are used, this `post_delete_metadata_store_with_metadata` interceptor runs after the
        `post_delete_metadata_store` interceptor. The (possibly modified) response returned by
        `post_delete_metadata_store` will be passed to
        `post_delete_metadata_store_with_metadata`.
        """
        return response, metadata

    async def pre_get_artifact(
        self,
        request: metadata_service.GetArtifactRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.GetArtifactRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_artifact

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_get_artifact(self, response: artifact.Artifact) -> artifact.Artifact:
        """Post-rpc interceptor for get_artifact

        DEPRECATED. Please use the `post_get_artifact_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_get_artifact` interceptor runs
        before the `post_get_artifact_with_metadata` interceptor.
        """
        return response

    async def post_get_artifact_with_metadata(
        self,
        response: artifact.Artifact,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[artifact.Artifact, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_artifact

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_get_artifact_with_metadata`
        interceptor in new development instead of the `post_get_artifact` interceptor.
        When both interceptors are used, this `post_get_artifact_with_metadata` interceptor runs after the
        `post_get_artifact` interceptor. The (possibly modified) response returned by
        `post_get_artifact` will be passed to
        `post_get_artifact_with_metadata`.
        """
        return response, metadata

    async def pre_get_context(
        self,
        request: metadata_service.GetContextRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.GetContextRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_context

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_get_context(self, response: context.Context) -> context.Context:
        """Post-rpc interceptor for get_context

        DEPRECATED. Please use the `post_get_context_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_get_context` interceptor runs
        before the `post_get_context_with_metadata` interceptor.
        """
        return response

    async def post_get_context_with_metadata(
        self,
        response: context.Context,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[context.Context, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_context

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_get_context_with_metadata`
        interceptor in new development instead of the `post_get_context` interceptor.
        When both interceptors are used, this `post_get_context_with_metadata` interceptor runs after the
        `post_get_context` interceptor. The (possibly modified) response returned by
        `post_get_context` will be passed to
        `post_get_context_with_metadata`.
        """
        return response, metadata

    async def pre_get_execution(
        self,
        request: metadata_service.GetExecutionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.GetExecutionRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_execution

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_get_execution(
        self, response: execution.Execution
    ) -> execution.Execution:
        """Post-rpc interceptor for get_execution

        DEPRECATED. Please use the `post_get_execution_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_get_execution` interceptor runs
        before the `post_get_execution_with_metadata` interceptor.
        """
        return response

    async def post_get_execution_with_metadata(
        self,
        response: execution.Execution,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[execution.Execution, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_execution

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_get_execution_with_metadata`
        interceptor in new development instead of the `post_get_execution` interceptor.
        When both interceptors are used, this `post_get_execution_with_metadata` interceptor runs after the
        `post_get_execution` interceptor. The (possibly modified) response returned by
        `post_get_execution` will be passed to
        `post_get_execution_with_metadata`.
        """
        return response, metadata

    async def pre_get_metadata_schema(
        self,
        request: metadata_service.GetMetadataSchemaRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.GetMetadataSchemaRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_metadata_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_get_metadata_schema(
        self, response: metadata_schema.MetadataSchema
    ) -> metadata_schema.MetadataSchema:
        """Post-rpc interceptor for get_metadata_schema

        DEPRECATED. Please use the `post_get_metadata_schema_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_get_metadata_schema` interceptor runs
        before the `post_get_metadata_schema_with_metadata` interceptor.
        """
        return response

    async def post_get_metadata_schema_with_metadata(
        self,
        response: metadata_schema.MetadataSchema,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[metadata_schema.MetadataSchema, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_metadata_schema

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_get_metadata_schema_with_metadata`
        interceptor in new development instead of the `post_get_metadata_schema` interceptor.
        When both interceptors are used, this `post_get_metadata_schema_with_metadata` interceptor runs after the
        `post_get_metadata_schema` interceptor. The (possibly modified) response returned by
        `post_get_metadata_schema` will be passed to
        `post_get_metadata_schema_with_metadata`.
        """
        return response, metadata

    async def pre_get_metadata_store(
        self,
        request: metadata_service.GetMetadataStoreRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.GetMetadataStoreRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_metadata_store

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_get_metadata_store(
        self, response: metadata_store.MetadataStore
    ) -> metadata_store.MetadataStore:
        """Post-rpc interceptor for get_metadata_store

        DEPRECATED. Please use the `post_get_metadata_store_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_get_metadata_store` interceptor runs
        before the `post_get_metadata_store_with_metadata` interceptor.
        """
        return response

    async def post_get_metadata_store_with_metadata(
        self,
        response: metadata_store.MetadataStore,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[metadata_store.MetadataStore, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_metadata_store

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_get_metadata_store_with_metadata`
        interceptor in new development instead of the `post_get_metadata_store` interceptor.
        When both interceptors are used, this `post_get_metadata_store_with_metadata` interceptor runs after the
        `post_get_metadata_store` interceptor. The (possibly modified) response returned by
        `post_get_metadata_store` will be passed to
        `post_get_metadata_store_with_metadata`.
        """
        return response, metadata

    async def pre_list_artifacts(
        self,
        request: metadata_service.ListArtifactsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListArtifactsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_artifacts

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_list_artifacts(
        self, response: metadata_service.ListArtifactsResponse
    ) -> metadata_service.ListArtifactsResponse:
        """Post-rpc interceptor for list_artifacts

        DEPRECATED. Please use the `post_list_artifacts_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_list_artifacts` interceptor runs
        before the `post_list_artifacts_with_metadata` interceptor.
        """
        return response

    async def post_list_artifacts_with_metadata(
        self,
        response: metadata_service.ListArtifactsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListArtifactsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_artifacts

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_list_artifacts_with_metadata`
        interceptor in new development instead of the `post_list_artifacts` interceptor.
        When both interceptors are used, this `post_list_artifacts_with_metadata` interceptor runs after the
        `post_list_artifacts` interceptor. The (possibly modified) response returned by
        `post_list_artifacts` will be passed to
        `post_list_artifacts_with_metadata`.
        """
        return response, metadata

    async def pre_list_contexts(
        self,
        request: metadata_service.ListContextsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListContextsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_contexts

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_list_contexts(
        self, response: metadata_service.ListContextsResponse
    ) -> metadata_service.ListContextsResponse:
        """Post-rpc interceptor for list_contexts

        DEPRECATED. Please use the `post_list_contexts_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_list_contexts` interceptor runs
        before the `post_list_contexts_with_metadata` interceptor.
        """
        return response

    async def post_list_contexts_with_metadata(
        self,
        response: metadata_service.ListContextsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListContextsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_contexts

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_list_contexts_with_metadata`
        interceptor in new development instead of the `post_list_contexts` interceptor.
        When both interceptors are used, this `post_list_contexts_with_metadata` interceptor runs after the
        `post_list_contexts` interceptor. The (possibly modified) response returned by
        `post_list_contexts` will be passed to
        `post_list_contexts_with_metadata`.
        """
        return response, metadata

    async def pre_list_executions(
        self,
        request: metadata_service.ListExecutionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListExecutionsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_executions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_list_executions(
        self, response: metadata_service.ListExecutionsResponse
    ) -> metadata_service.ListExecutionsResponse:
        """Post-rpc interceptor for list_executions

        DEPRECATED. Please use the `post_list_executions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_list_executions` interceptor runs
        before the `post_list_executions_with_metadata` interceptor.
        """
        return response

    async def post_list_executions_with_metadata(
        self,
        response: metadata_service.ListExecutionsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListExecutionsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_executions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_list_executions_with_metadata`
        interceptor in new development instead of the `post_list_executions` interceptor.
        When both interceptors are used, this `post_list_executions_with_metadata` interceptor runs after the
        `post_list_executions` interceptor. The (possibly modified) response returned by
        `post_list_executions` will be passed to
        `post_list_executions_with_metadata`.
        """
        return response, metadata

    async def pre_list_metadata_schemas(
        self,
        request: metadata_service.ListMetadataSchemasRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListMetadataSchemasRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_metadata_schemas

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_list_metadata_schemas(
        self, response: metadata_service.ListMetadataSchemasResponse
    ) -> metadata_service.ListMetadataSchemasResponse:
        """Post-rpc interceptor for list_metadata_schemas

        DEPRECATED. Please use the `post_list_metadata_schemas_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_list_metadata_schemas` interceptor runs
        before the `post_list_metadata_schemas_with_metadata` interceptor.
        """
        return response

    async def post_list_metadata_schemas_with_metadata(
        self,
        response: metadata_service.ListMetadataSchemasResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListMetadataSchemasResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_metadata_schemas

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_list_metadata_schemas_with_metadata`
        interceptor in new development instead of the `post_list_metadata_schemas` interceptor.
        When both interceptors are used, this `post_list_metadata_schemas_with_metadata` interceptor runs after the
        `post_list_metadata_schemas` interceptor. The (possibly modified) response returned by
        `post_list_metadata_schemas` will be passed to
        `post_list_metadata_schemas_with_metadata`.
        """
        return response, metadata

    async def pre_list_metadata_stores(
        self,
        request: metadata_service.ListMetadataStoresRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListMetadataStoresRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_metadata_stores

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_list_metadata_stores(
        self, response: metadata_service.ListMetadataStoresResponse
    ) -> metadata_service.ListMetadataStoresResponse:
        """Post-rpc interceptor for list_metadata_stores

        DEPRECATED. Please use the `post_list_metadata_stores_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_list_metadata_stores` interceptor runs
        before the `post_list_metadata_stores_with_metadata` interceptor.
        """
        return response

    async def post_list_metadata_stores_with_metadata(
        self,
        response: metadata_service.ListMetadataStoresResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.ListMetadataStoresResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_metadata_stores

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_list_metadata_stores_with_metadata`
        interceptor in new development instead of the `post_list_metadata_stores` interceptor.
        When both interceptors are used, this `post_list_metadata_stores_with_metadata` interceptor runs after the
        `post_list_metadata_stores` interceptor. The (possibly modified) response returned by
        `post_list_metadata_stores` will be passed to
        `post_list_metadata_stores_with_metadata`.
        """
        return response, metadata

    async def pre_purge_artifacts(
        self,
        request: metadata_service.PurgeArtifactsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.PurgeArtifactsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for purge_artifacts

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_purge_artifacts(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for purge_artifacts

        DEPRECATED. Please use the `post_purge_artifacts_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_purge_artifacts` interceptor runs
        before the `post_purge_artifacts_with_metadata` interceptor.
        """
        return response

    async def post_purge_artifacts_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for purge_artifacts

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_purge_artifacts_with_metadata`
        interceptor in new development instead of the `post_purge_artifacts` interceptor.
        When both interceptors are used, this `post_purge_artifacts_with_metadata` interceptor runs after the
        `post_purge_artifacts` interceptor. The (possibly modified) response returned by
        `post_purge_artifacts` will be passed to
        `post_purge_artifacts_with_metadata`.
        """
        return response, metadata

    async def pre_purge_contexts(
        self,
        request: metadata_service.PurgeContextsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.PurgeContextsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for purge_contexts

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_purge_contexts(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for purge_contexts

        DEPRECATED. Please use the `post_purge_contexts_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_purge_contexts` interceptor runs
        before the `post_purge_contexts_with_metadata` interceptor.
        """
        return response

    async def post_purge_contexts_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for purge_contexts

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_purge_contexts_with_metadata`
        interceptor in new development instead of the `post_purge_contexts` interceptor.
        When both interceptors are used, this `post_purge_contexts_with_metadata` interceptor runs after the
        `post_purge_contexts` interceptor. The (possibly modified) response returned by
        `post_purge_contexts` will be passed to
        `post_purge_contexts_with_metadata`.
        """
        return response, metadata

    async def pre_purge_executions(
        self,
        request: metadata_service.PurgeExecutionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.PurgeExecutionsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for purge_executions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_purge_executions(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for purge_executions

        DEPRECATED. Please use the `post_purge_executions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_purge_executions` interceptor runs
        before the `post_purge_executions_with_metadata` interceptor.
        """
        return response

    async def post_purge_executions_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for purge_executions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_purge_executions_with_metadata`
        interceptor in new development instead of the `post_purge_executions` interceptor.
        When both interceptors are used, this `post_purge_executions_with_metadata` interceptor runs after the
        `post_purge_executions` interceptor. The (possibly modified) response returned by
        `post_purge_executions` will be passed to
        `post_purge_executions_with_metadata`.
        """
        return response, metadata

    async def pre_query_artifact_lineage_subgraph(
        self,
        request: metadata_service.QueryArtifactLineageSubgraphRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.QueryArtifactLineageSubgraphRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for query_artifact_lineage_subgraph

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_query_artifact_lineage_subgraph(
        self, response: lineage_subgraph.LineageSubgraph
    ) -> lineage_subgraph.LineageSubgraph:
        """Post-rpc interceptor for query_artifact_lineage_subgraph

        DEPRECATED. Please use the `post_query_artifact_lineage_subgraph_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_query_artifact_lineage_subgraph` interceptor runs
        before the `post_query_artifact_lineage_subgraph_with_metadata` interceptor.
        """
        return response

    async def post_query_artifact_lineage_subgraph_with_metadata(
        self,
        response: lineage_subgraph.LineageSubgraph,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        lineage_subgraph.LineageSubgraph, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for query_artifact_lineage_subgraph

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_query_artifact_lineage_subgraph_with_metadata`
        interceptor in new development instead of the `post_query_artifact_lineage_subgraph` interceptor.
        When both interceptors are used, this `post_query_artifact_lineage_subgraph_with_metadata` interceptor runs after the
        `post_query_artifact_lineage_subgraph` interceptor. The (possibly modified) response returned by
        `post_query_artifact_lineage_subgraph` will be passed to
        `post_query_artifact_lineage_subgraph_with_metadata`.
        """
        return response, metadata

    async def pre_query_context_lineage_subgraph(
        self,
        request: metadata_service.QueryContextLineageSubgraphRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.QueryContextLineageSubgraphRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for query_context_lineage_subgraph

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_query_context_lineage_subgraph(
        self, response: lineage_subgraph.LineageSubgraph
    ) -> lineage_subgraph.LineageSubgraph:
        """Post-rpc interceptor for query_context_lineage_subgraph

        DEPRECATED. Please use the `post_query_context_lineage_subgraph_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_query_context_lineage_subgraph` interceptor runs
        before the `post_query_context_lineage_subgraph_with_metadata` interceptor.
        """
        return response

    async def post_query_context_lineage_subgraph_with_metadata(
        self,
        response: lineage_subgraph.LineageSubgraph,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        lineage_subgraph.LineageSubgraph, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for query_context_lineage_subgraph

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_query_context_lineage_subgraph_with_metadata`
        interceptor in new development instead of the `post_query_context_lineage_subgraph` interceptor.
        When both interceptors are used, this `post_query_context_lineage_subgraph_with_metadata` interceptor runs after the
        `post_query_context_lineage_subgraph` interceptor. The (possibly modified) response returned by
        `post_query_context_lineage_subgraph` will be passed to
        `post_query_context_lineage_subgraph_with_metadata`.
        """
        return response, metadata

    async def pre_query_execution_inputs_and_outputs(
        self,
        request: metadata_service.QueryExecutionInputsAndOutputsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.QueryExecutionInputsAndOutputsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for query_execution_inputs_and_outputs

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_query_execution_inputs_and_outputs(
        self, response: lineage_subgraph.LineageSubgraph
    ) -> lineage_subgraph.LineageSubgraph:
        """Post-rpc interceptor for query_execution_inputs_and_outputs

        DEPRECATED. Please use the `post_query_execution_inputs_and_outputs_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_query_execution_inputs_and_outputs` interceptor runs
        before the `post_query_execution_inputs_and_outputs_with_metadata` interceptor.
        """
        return response

    async def post_query_execution_inputs_and_outputs_with_metadata(
        self,
        response: lineage_subgraph.LineageSubgraph,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        lineage_subgraph.LineageSubgraph, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for query_execution_inputs_and_outputs

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_query_execution_inputs_and_outputs_with_metadata`
        interceptor in new development instead of the `post_query_execution_inputs_and_outputs` interceptor.
        When both interceptors are used, this `post_query_execution_inputs_and_outputs_with_metadata` interceptor runs after the
        `post_query_execution_inputs_and_outputs` interceptor. The (possibly modified) response returned by
        `post_query_execution_inputs_and_outputs` will be passed to
        `post_query_execution_inputs_and_outputs_with_metadata`.
        """
        return response, metadata

    async def pre_remove_context_children(
        self,
        request: metadata_service.RemoveContextChildrenRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.RemoveContextChildrenRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for remove_context_children

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_remove_context_children(
        self, response: metadata_service.RemoveContextChildrenResponse
    ) -> metadata_service.RemoveContextChildrenResponse:
        """Post-rpc interceptor for remove_context_children

        DEPRECATED. Please use the `post_remove_context_children_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_remove_context_children` interceptor runs
        before the `post_remove_context_children_with_metadata` interceptor.
        """
        return response

    async def post_remove_context_children_with_metadata(
        self,
        response: metadata_service.RemoveContextChildrenResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.RemoveContextChildrenResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for remove_context_children

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_remove_context_children_with_metadata`
        interceptor in new development instead of the `post_remove_context_children` interceptor.
        When both interceptors are used, this `post_remove_context_children_with_metadata` interceptor runs after the
        `post_remove_context_children` interceptor. The (possibly modified) response returned by
        `post_remove_context_children` will be passed to
        `post_remove_context_children_with_metadata`.
        """
        return response, metadata

    async def pre_update_artifact(
        self,
        request: metadata_service.UpdateArtifactRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.UpdateArtifactRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for update_artifact

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_update_artifact(
        self, response: gca_artifact.Artifact
    ) -> gca_artifact.Artifact:
        """Post-rpc interceptor for update_artifact

        DEPRECATED. Please use the `post_update_artifact_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_update_artifact` interceptor runs
        before the `post_update_artifact_with_metadata` interceptor.
        """
        return response

    async def post_update_artifact_with_metadata(
        self,
        response: gca_artifact.Artifact,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gca_artifact.Artifact, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_artifact

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_update_artifact_with_metadata`
        interceptor in new development instead of the `post_update_artifact` interceptor.
        When both interceptors are used, this `post_update_artifact_with_metadata` interceptor runs after the
        `post_update_artifact` interceptor. The (possibly modified) response returned by
        `post_update_artifact` will be passed to
        `post_update_artifact_with_metadata`.
        """
        return response, metadata

    async def pre_update_context(
        self,
        request: metadata_service.UpdateContextRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.UpdateContextRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for update_context

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_update_context(
        self, response: gca_context.Context
    ) -> gca_context.Context:
        """Post-rpc interceptor for update_context

        DEPRECATED. Please use the `post_update_context_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_update_context` interceptor runs
        before the `post_update_context_with_metadata` interceptor.
        """
        return response

    async def post_update_context_with_metadata(
        self,
        response: gca_context.Context,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gca_context.Context, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_context

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_update_context_with_metadata`
        interceptor in new development instead of the `post_update_context` interceptor.
        When both interceptors are used, this `post_update_context_with_metadata` interceptor runs after the
        `post_update_context` interceptor. The (possibly modified) response returned by
        `post_update_context` will be passed to
        `post_update_context_with_metadata`.
        """
        return response, metadata

    async def pre_update_execution(
        self,
        request: metadata_service.UpdateExecutionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        metadata_service.UpdateExecutionRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for update_execution

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_update_execution(
        self, response: gca_execution.Execution
    ) -> gca_execution.Execution:
        """Post-rpc interceptor for update_execution

        DEPRECATED. Please use the `post_update_execution_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code. This `post_update_execution` interceptor runs
        before the `post_update_execution_with_metadata` interceptor.
        """
        return response

    async def post_update_execution_with_metadata(
        self,
        response: gca_execution.Execution,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gca_execution.Execution, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_execution

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the MetadataService server but before it is returned to user code.

        We recommend only using this `post_update_execution_with_metadata`
        interceptor in new development instead of the `post_update_execution` interceptor.
        When both interceptors are used, this `post_update_execution_with_metadata` interceptor runs after the
        `post_update_execution` interceptor. The (possibly modified) response returned by
        `post_update_execution` will be passed to
        `post_update_execution_with_metadata`.
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_get_location(
        self, response: locations_pb2.Location
    ) -> locations_pb2.Location:
        """Post-rpc interceptor for get_location

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_list_locations(
        self, response: locations_pb2.ListLocationsResponse
    ) -> locations_pb2.ListLocationsResponse:
        """Post-rpc interceptor for list_locations

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_get_iam_policy(
        self, response: policy_pb2.Policy
    ) -> policy_pb2.Policy:
        """Post-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_set_iam_policy(
        self, response: policy_pb2.Policy
    ) -> policy_pb2.Policy:
        """Post-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_test_iam_permissions(
        self, response: iam_policy_pb2.TestIamPermissionsResponse
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        """Post-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_delete_operation(self, response: None) -> None:
        """Post-rpc interceptor for delete_operation

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
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
        before they are sent to the MetadataService server.
        """
        return request, metadata

    async def post_wait_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for wait_operation

        Override in a subclass to manipulate the response
        after it is returned by the MetadataService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class AsyncMetadataServiceRestStub:
    _session: AsyncAuthorizedSession
    _host: str
    _interceptor: AsyncMetadataServiceRestInterceptor


class AsyncMetadataServiceRestTransport(_BaseMetadataServiceRestTransport):
    """Asynchronous REST backend transport for MetadataService.

    Service for reading and writing metadata entries.

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
        interceptor: Optional[AsyncMetadataServiceRestInterceptor] = None,
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
        self._interceptor = interceptor or AsyncMetadataServiceRestInterceptor()
        self._wrap_with_kind = True
        self._prep_wrapped_messages(client_info)
        self._operations_client: Optional[operations_v1.AsyncOperationsRestClient] = (
            None
        )

    def _prep_wrapped_messages(self, client_info):
        """Precompute the wrapped methods, overriding the base class method to use async wrappers."""
        self._wrapped_methods = {
            self.create_metadata_store: self._wrap_method(
                self.create_metadata_store,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_metadata_store: self._wrap_method(
                self.get_metadata_store,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_metadata_stores: self._wrap_method(
                self.list_metadata_stores,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.delete_metadata_store: self._wrap_method(
                self.delete_metadata_store,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.create_artifact: self._wrap_method(
                self.create_artifact,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_artifact: self._wrap_method(
                self.get_artifact,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_artifacts: self._wrap_method(
                self.list_artifacts,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.update_artifact: self._wrap_method(
                self.update_artifact,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.delete_artifact: self._wrap_method(
                self.delete_artifact,
                default_timeout=None,
                client_info=client_info,
            ),
            self.purge_artifacts: self._wrap_method(
                self.purge_artifacts,
                default_timeout=None,
                client_info=client_info,
            ),
            self.create_context: self._wrap_method(
                self.create_context,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_context: self._wrap_method(
                self.get_context,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_contexts: self._wrap_method(
                self.list_contexts,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.update_context: self._wrap_method(
                self.update_context,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.delete_context: self._wrap_method(
                self.delete_context,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.purge_contexts: self._wrap_method(
                self.purge_contexts,
                default_timeout=None,
                client_info=client_info,
            ),
            self.add_context_artifacts_and_executions: self._wrap_method(
                self.add_context_artifacts_and_executions,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.add_context_children: self._wrap_method(
                self.add_context_children,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.remove_context_children: self._wrap_method(
                self.remove_context_children,
                default_timeout=None,
                client_info=client_info,
            ),
            self.query_context_lineage_subgraph: self._wrap_method(
                self.query_context_lineage_subgraph,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.create_execution: self._wrap_method(
                self.create_execution,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_execution: self._wrap_method(
                self.get_execution,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_executions: self._wrap_method(
                self.list_executions,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.update_execution: self._wrap_method(
                self.update_execution,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.delete_execution: self._wrap_method(
                self.delete_execution,
                default_timeout=None,
                client_info=client_info,
            ),
            self.purge_executions: self._wrap_method(
                self.purge_executions,
                default_timeout=None,
                client_info=client_info,
            ),
            self.add_execution_events: self._wrap_method(
                self.add_execution_events,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.query_execution_inputs_and_outputs: self._wrap_method(
                self.query_execution_inputs_and_outputs,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.create_metadata_schema: self._wrap_method(
                self.create_metadata_schema,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_metadata_schema: self._wrap_method(
                self.get_metadata_schema,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_metadata_schemas: self._wrap_method(
                self.list_metadata_schemas,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.query_artifact_lineage_subgraph: self._wrap_method(
                self.query_artifact_lineage_subgraph,
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

    class _AddContextArtifactsAndExecutions(
        _BaseMetadataServiceRestTransport._BaseAddContextArtifactsAndExecutions,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash(
                "AsyncMetadataServiceRestTransport.AddContextArtifactsAndExecutions"
            )

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
            request: metadata_service.AddContextArtifactsAndExecutionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.AddContextArtifactsAndExecutionsResponse:
            r"""Call the add context artifacts and
            executions method over HTTP.

                Args:
                    request (~.metadata_service.AddContextArtifactsAndExecutionsRequest):
                        The request object. Request message for
                    [MetadataService.AddContextArtifactsAndExecutions][google.cloud.aiplatform.v1beta1.MetadataService.AddContextArtifactsAndExecutions].
                    retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.metadata_service.AddContextArtifactsAndExecutionsResponse:
                        Response message for
                    [MetadataService.AddContextArtifactsAndExecutions][google.cloud.aiplatform.v1beta1.MetadataService.AddContextArtifactsAndExecutions].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseAddContextArtifactsAndExecutions._get_http_options()
            )

            request, metadata = (
                await self._interceptor.pre_add_context_artifacts_and_executions(
                    request, metadata
                )
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseAddContextArtifactsAndExecutions._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseAddContextArtifactsAndExecutions._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseAddContextArtifactsAndExecutions._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.AddContextArtifactsAndExecutions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "AddContextArtifactsAndExecutions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._AddContextArtifactsAndExecutions._get_response(
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
            resp = metadata_service.AddContextArtifactsAndExecutionsResponse()
            pb_resp = metadata_service.AddContextArtifactsAndExecutionsResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_add_context_artifacts_and_executions(
                resp
            )
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                await self._interceptor.post_add_context_artifacts_and_executions_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = metadata_service.AddContextArtifactsAndExecutionsResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.add_context_artifacts_and_executions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "AddContextArtifactsAndExecutions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _AddContextChildren(
        _BaseMetadataServiceRestTransport._BaseAddContextChildren,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.AddContextChildren")

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
            request: metadata_service.AddContextChildrenRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.AddContextChildrenResponse:
            r"""Call the add context children method over HTTP.

            Args:
                request (~.metadata_service.AddContextChildrenRequest):
                    The request object. Request message for
                [MetadataService.AddContextChildren][google.cloud.aiplatform.v1beta1.MetadataService.AddContextChildren].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_service.AddContextChildrenResponse:
                    Response message for
                [MetadataService.AddContextChildren][google.cloud.aiplatform.v1beta1.MetadataService.AddContextChildren].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseAddContextChildren._get_http_options()
            )

            request, metadata = await self._interceptor.pre_add_context_children(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseAddContextChildren._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseAddContextChildren._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseAddContextChildren._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.AddContextChildren",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "AddContextChildren",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._AddContextChildren._get_response(
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
            resp = metadata_service.AddContextChildrenResponse()
            pb_resp = metadata_service.AddContextChildrenResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_add_context_children(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_add_context_children_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        metadata_service.AddContextChildrenResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.add_context_children",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "AddContextChildren",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _AddExecutionEvents(
        _BaseMetadataServiceRestTransport._BaseAddExecutionEvents,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.AddExecutionEvents")

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
            request: metadata_service.AddExecutionEventsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.AddExecutionEventsResponse:
            r"""Call the add execution events method over HTTP.

            Args:
                request (~.metadata_service.AddExecutionEventsRequest):
                    The request object. Request message for
                [MetadataService.AddExecutionEvents][google.cloud.aiplatform.v1beta1.MetadataService.AddExecutionEvents].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_service.AddExecutionEventsResponse:
                    Response message for
                [MetadataService.AddExecutionEvents][google.cloud.aiplatform.v1beta1.MetadataService.AddExecutionEvents].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseAddExecutionEvents._get_http_options()
            )

            request, metadata = await self._interceptor.pre_add_execution_events(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseAddExecutionEvents._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseAddExecutionEvents._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseAddExecutionEvents._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.AddExecutionEvents",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "AddExecutionEvents",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._AddExecutionEvents._get_response(
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
            resp = metadata_service.AddExecutionEventsResponse()
            pb_resp = metadata_service.AddExecutionEventsResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_add_execution_events(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_add_execution_events_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        metadata_service.AddExecutionEventsResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.add_execution_events",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "AddExecutionEvents",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _CreateArtifact(
        _BaseMetadataServiceRestTransport._BaseCreateArtifact,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.CreateArtifact")

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
            request: metadata_service.CreateArtifactRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_artifact.Artifact:
            r"""Call the create artifact method over HTTP.

            Args:
                request (~.metadata_service.CreateArtifactRequest):
                    The request object. Request message for
                [MetadataService.CreateArtifact][google.cloud.aiplatform.v1beta1.MetadataService.CreateArtifact].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_artifact.Artifact:
                    Instance of a general artifact.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseCreateArtifact._get_http_options()
            )

            request, metadata = await self._interceptor.pre_create_artifact(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseCreateArtifact._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseCreateArtifact._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseCreateArtifact._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.CreateArtifact",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateArtifact",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._CreateArtifact._get_response(
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
            resp = gca_artifact.Artifact()
            pb_resp = gca_artifact.Artifact.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_create_artifact(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_create_artifact_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_artifact.Artifact.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.create_artifact",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateArtifact",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _CreateContext(
        _BaseMetadataServiceRestTransport._BaseCreateContext,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.CreateContext")

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
            request: metadata_service.CreateContextRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_context.Context:
            r"""Call the create context method over HTTP.

            Args:
                request (~.metadata_service.CreateContextRequest):
                    The request object. Request message for
                [MetadataService.CreateContext][google.cloud.aiplatform.v1beta1.MetadataService.CreateContext].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_context.Context:
                    Instance of a general context.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseCreateContext._get_http_options()
            )

            request, metadata = await self._interceptor.pre_create_context(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseCreateContext._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseCreateContext._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseCreateContext._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.CreateContext",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateContext",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._CreateContext._get_response(
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
            resp = gca_context.Context()
            pb_resp = gca_context.Context.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_create_context(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_create_context_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_context.Context.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.create_context",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateContext",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _CreateExecution(
        _BaseMetadataServiceRestTransport._BaseCreateExecution,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.CreateExecution")

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
            request: metadata_service.CreateExecutionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_execution.Execution:
            r"""Call the create execution method over HTTP.

            Args:
                request (~.metadata_service.CreateExecutionRequest):
                    The request object. Request message for
                [MetadataService.CreateExecution][google.cloud.aiplatform.v1beta1.MetadataService.CreateExecution].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_execution.Execution:
                    Instance of a general execution.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseCreateExecution._get_http_options()
            )

            request, metadata = await self._interceptor.pre_create_execution(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseCreateExecution._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseCreateExecution._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseCreateExecution._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.CreateExecution",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateExecution",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._CreateExecution._get_response(
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
            resp = gca_execution.Execution()
            pb_resp = gca_execution.Execution.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_create_execution(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_create_execution_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_execution.Execution.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.create_execution",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateExecution",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _CreateMetadataSchema(
        _BaseMetadataServiceRestTransport._BaseCreateMetadataSchema,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.CreateMetadataSchema")

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
            request: metadata_service.CreateMetadataSchemaRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_metadata_schema.MetadataSchema:
            r"""Call the create metadata schema method over HTTP.

            Args:
                request (~.metadata_service.CreateMetadataSchemaRequest):
                    The request object. Request message for
                [MetadataService.CreateMetadataSchema][google.cloud.aiplatform.v1beta1.MetadataService.CreateMetadataSchema].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_metadata_schema.MetadataSchema:
                    Instance of a general MetadataSchema.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseCreateMetadataSchema._get_http_options()
            )

            request, metadata = await self._interceptor.pre_create_metadata_schema(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseCreateMetadataSchema._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseCreateMetadataSchema._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseCreateMetadataSchema._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.CreateMetadataSchema",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateMetadataSchema",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._CreateMetadataSchema._get_response(
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
            resp = gca_metadata_schema.MetadataSchema()
            pb_resp = gca_metadata_schema.MetadataSchema.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_create_metadata_schema(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_create_metadata_schema_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_metadata_schema.MetadataSchema.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.create_metadata_schema",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateMetadataSchema",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _CreateMetadataStore(
        _BaseMetadataServiceRestTransport._BaseCreateMetadataStore,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.CreateMetadataStore")

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
            request: metadata_service.CreateMetadataStoreRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create metadata store method over HTTP.

            Args:
                request (~.metadata_service.CreateMetadataStoreRequest):
                    The request object. Request message for
                [MetadataService.CreateMetadataStore][google.cloud.aiplatform.v1beta1.MetadataService.CreateMetadataStore].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BaseMetadataServiceRestTransport._BaseCreateMetadataStore._get_http_options()
            )

            request, metadata = await self._interceptor.pre_create_metadata_store(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseCreateMetadataStore._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseCreateMetadataStore._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseCreateMetadataStore._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.CreateMetadataStore",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateMetadataStore",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._CreateMetadataStore._get_response(
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
            resp = operations_pb2.Operation()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_create_metadata_store(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_create_metadata_store_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.create_metadata_store",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CreateMetadataStore",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _DeleteArtifact(
        _BaseMetadataServiceRestTransport._BaseDeleteArtifact,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.DeleteArtifact")

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
            request: metadata_service.DeleteArtifactRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete artifact method over HTTP.

            Args:
                request (~.metadata_service.DeleteArtifactRequest):
                    The request object. Request message for
                [MetadataService.DeleteArtifact][google.cloud.aiplatform.v1beta1.MetadataService.DeleteArtifact].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BaseMetadataServiceRestTransport._BaseDeleteArtifact._get_http_options()
            )

            request, metadata = await self._interceptor.pre_delete_artifact(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseDeleteArtifact._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseDeleteArtifact._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.DeleteArtifact",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteArtifact",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._DeleteArtifact._get_response(
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

            # Return the response
            resp = operations_pb2.Operation()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_delete_artifact(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_delete_artifact_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.delete_artifact",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteArtifact",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _DeleteContext(
        _BaseMetadataServiceRestTransport._BaseDeleteContext,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.DeleteContext")

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
            request: metadata_service.DeleteContextRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete context method over HTTP.

            Args:
                request (~.metadata_service.DeleteContextRequest):
                    The request object. Request message for
                [MetadataService.DeleteContext][google.cloud.aiplatform.v1beta1.MetadataService.DeleteContext].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BaseMetadataServiceRestTransport._BaseDeleteContext._get_http_options()
            )

            request, metadata = await self._interceptor.pre_delete_context(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseDeleteContext._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseDeleteContext._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.DeleteContext",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteContext",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._DeleteContext._get_response(
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

            # Return the response
            resp = operations_pb2.Operation()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_delete_context(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_delete_context_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.delete_context",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteContext",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _DeleteExecution(
        _BaseMetadataServiceRestTransport._BaseDeleteExecution,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.DeleteExecution")

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
            request: metadata_service.DeleteExecutionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete execution method over HTTP.

            Args:
                request (~.metadata_service.DeleteExecutionRequest):
                    The request object. Request message for
                [MetadataService.DeleteExecution][google.cloud.aiplatform.v1beta1.MetadataService.DeleteExecution].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BaseMetadataServiceRestTransport._BaseDeleteExecution._get_http_options()
            )

            request, metadata = await self._interceptor.pre_delete_execution(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseDeleteExecution._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseDeleteExecution._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.DeleteExecution",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteExecution",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._DeleteExecution._get_response(
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

            # Return the response
            resp = operations_pb2.Operation()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_delete_execution(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_delete_execution_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.delete_execution",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteExecution",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _DeleteMetadataStore(
        _BaseMetadataServiceRestTransport._BaseDeleteMetadataStore,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.DeleteMetadataStore")

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
            request: metadata_service.DeleteMetadataStoreRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete metadata store method over HTTP.

            Args:
                request (~.metadata_service.DeleteMetadataStoreRequest):
                    The request object. Request message for
                [MetadataService.DeleteMetadataStore][google.cloud.aiplatform.v1beta1.MetadataService.DeleteMetadataStore].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BaseMetadataServiceRestTransport._BaseDeleteMetadataStore._get_http_options()
            )

            request, metadata = await self._interceptor.pre_delete_metadata_store(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseDeleteMetadataStore._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseDeleteMetadataStore._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.DeleteMetadataStore",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteMetadataStore",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._DeleteMetadataStore._get_response(
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

            # Return the response
            resp = operations_pb2.Operation()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_delete_metadata_store(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_delete_metadata_store_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.delete_metadata_store",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteMetadataStore",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _GetArtifact(
        _BaseMetadataServiceRestTransport._BaseGetArtifact, AsyncMetadataServiceRestStub
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.GetArtifact")

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
            request: metadata_service.GetArtifactRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> artifact.Artifact:
            r"""Call the get artifact method over HTTP.

            Args:
                request (~.metadata_service.GetArtifactRequest):
                    The request object. Request message for
                [MetadataService.GetArtifact][google.cloud.aiplatform.v1beta1.MetadataService.GetArtifact].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.artifact.Artifact:
                    Instance of a general artifact.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseGetArtifact._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_artifact(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseGetArtifact._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseGetArtifact._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.GetArtifact",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetArtifact",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._GetArtifact._get_response(
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

            # Return the response
            resp = artifact.Artifact()
            pb_resp = artifact.Artifact.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_get_artifact(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_get_artifact_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = artifact.Artifact.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.get_artifact",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetArtifact",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _GetContext(
        _BaseMetadataServiceRestTransport._BaseGetContext, AsyncMetadataServiceRestStub
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.GetContext")

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
            request: metadata_service.GetContextRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> context.Context:
            r"""Call the get context method over HTTP.

            Args:
                request (~.metadata_service.GetContextRequest):
                    The request object. Request message for
                [MetadataService.GetContext][google.cloud.aiplatform.v1beta1.MetadataService.GetContext].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.context.Context:
                    Instance of a general context.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseGetContext._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_context(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseGetContext._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseGetContext._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.GetContext",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetContext",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._GetContext._get_response(
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

            # Return the response
            resp = context.Context()
            pb_resp = context.Context.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_get_context(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_get_context_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = context.Context.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.get_context",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetContext",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _GetExecution(
        _BaseMetadataServiceRestTransport._BaseGetExecution,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.GetExecution")

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
            request: metadata_service.GetExecutionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> execution.Execution:
            r"""Call the get execution method over HTTP.

            Args:
                request (~.metadata_service.GetExecutionRequest):
                    The request object. Request message for
                [MetadataService.GetExecution][google.cloud.aiplatform.v1beta1.MetadataService.GetExecution].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.execution.Execution:
                    Instance of a general execution.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseGetExecution._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_execution(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseGetExecution._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseGetExecution._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.GetExecution",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetExecution",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._GetExecution._get_response(
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

            # Return the response
            resp = execution.Execution()
            pb_resp = execution.Execution.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_get_execution(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_get_execution_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = execution.Execution.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.get_execution",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetExecution",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _GetMetadataSchema(
        _BaseMetadataServiceRestTransport._BaseGetMetadataSchema,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.GetMetadataSchema")

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
            request: metadata_service.GetMetadataSchemaRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_schema.MetadataSchema:
            r"""Call the get metadata schema method over HTTP.

            Args:
                request (~.metadata_service.GetMetadataSchemaRequest):
                    The request object. Request message for
                [MetadataService.GetMetadataSchema][google.cloud.aiplatform.v1beta1.MetadataService.GetMetadataSchema].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_schema.MetadataSchema:
                    Instance of a general MetadataSchema.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseGetMetadataSchema._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_metadata_schema(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseGetMetadataSchema._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseGetMetadataSchema._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.GetMetadataSchema",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetMetadataSchema",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._GetMetadataSchema._get_response(
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

            # Return the response
            resp = metadata_schema.MetadataSchema()
            pb_resp = metadata_schema.MetadataSchema.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_get_metadata_schema(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_get_metadata_schema_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = metadata_schema.MetadataSchema.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.get_metadata_schema",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetMetadataSchema",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _GetMetadataStore(
        _BaseMetadataServiceRestTransport._BaseGetMetadataStore,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.GetMetadataStore")

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
            request: metadata_service.GetMetadataStoreRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_store.MetadataStore:
            r"""Call the get metadata store method over HTTP.

            Args:
                request (~.metadata_service.GetMetadataStoreRequest):
                    The request object. Request message for
                [MetadataService.GetMetadataStore][google.cloud.aiplatform.v1beta1.MetadataService.GetMetadataStore].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_store.MetadataStore:
                    Instance of a metadata store.
                Contains a set of metadata that can be
                queried.

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseGetMetadataStore._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_metadata_store(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseGetMetadataStore._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseGetMetadataStore._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.GetMetadataStore",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetMetadataStore",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._GetMetadataStore._get_response(
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

            # Return the response
            resp = metadata_store.MetadataStore()
            pb_resp = metadata_store.MetadataStore.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_get_metadata_store(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_get_metadata_store_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = metadata_store.MetadataStore.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.get_metadata_store",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetMetadataStore",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _ListArtifacts(
        _BaseMetadataServiceRestTransport._BaseListArtifacts,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.ListArtifacts")

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
            request: metadata_service.ListArtifactsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.ListArtifactsResponse:
            r"""Call the list artifacts method over HTTP.

            Args:
                request (~.metadata_service.ListArtifactsRequest):
                    The request object. Request message for
                [MetadataService.ListArtifacts][google.cloud.aiplatform.v1beta1.MetadataService.ListArtifacts].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_service.ListArtifactsResponse:
                    Response message for
                [MetadataService.ListArtifacts][google.cloud.aiplatform.v1beta1.MetadataService.ListArtifacts].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseListArtifacts._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_artifacts(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseListArtifacts._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseListArtifacts._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.ListArtifacts",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListArtifacts",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._ListArtifacts._get_response(
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

            # Return the response
            resp = metadata_service.ListArtifactsResponse()
            pb_resp = metadata_service.ListArtifactsResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_list_artifacts(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_list_artifacts_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = metadata_service.ListArtifactsResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.list_artifacts",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListArtifacts",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _ListContexts(
        _BaseMetadataServiceRestTransport._BaseListContexts,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.ListContexts")

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
            request: metadata_service.ListContextsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.ListContextsResponse:
            r"""Call the list contexts method over HTTP.

            Args:
                request (~.metadata_service.ListContextsRequest):
                    The request object. Request message for
                [MetadataService.ListContexts][google.cloud.aiplatform.v1beta1.MetadataService.ListContexts]
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_service.ListContextsResponse:
                    Response message for
                [MetadataService.ListContexts][google.cloud.aiplatform.v1beta1.MetadataService.ListContexts].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseListContexts._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_contexts(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseListContexts._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseListContexts._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.ListContexts",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListContexts",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._ListContexts._get_response(
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

            # Return the response
            resp = metadata_service.ListContextsResponse()
            pb_resp = metadata_service.ListContextsResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_list_contexts(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_list_contexts_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = metadata_service.ListContextsResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.list_contexts",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListContexts",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _ListExecutions(
        _BaseMetadataServiceRestTransport._BaseListExecutions,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.ListExecutions")

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
            request: metadata_service.ListExecutionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.ListExecutionsResponse:
            r"""Call the list executions method over HTTP.

            Args:
                request (~.metadata_service.ListExecutionsRequest):
                    The request object. Request message for
                [MetadataService.ListExecutions][google.cloud.aiplatform.v1beta1.MetadataService.ListExecutions].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_service.ListExecutionsResponse:
                    Response message for
                [MetadataService.ListExecutions][google.cloud.aiplatform.v1beta1.MetadataService.ListExecutions].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseListExecutions._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_executions(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseListExecutions._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseListExecutions._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.ListExecutions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListExecutions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._ListExecutions._get_response(
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

            # Return the response
            resp = metadata_service.ListExecutionsResponse()
            pb_resp = metadata_service.ListExecutionsResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_list_executions(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_list_executions_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = metadata_service.ListExecutionsResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.list_executions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListExecutions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _ListMetadataSchemas(
        _BaseMetadataServiceRestTransport._BaseListMetadataSchemas,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.ListMetadataSchemas")

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
            request: metadata_service.ListMetadataSchemasRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.ListMetadataSchemasResponse:
            r"""Call the list metadata schemas method over HTTP.

            Args:
                request (~.metadata_service.ListMetadataSchemasRequest):
                    The request object. Request message for
                [MetadataService.ListMetadataSchemas][google.cloud.aiplatform.v1beta1.MetadataService.ListMetadataSchemas].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_service.ListMetadataSchemasResponse:
                    Response message for
                [MetadataService.ListMetadataSchemas][google.cloud.aiplatform.v1beta1.MetadataService.ListMetadataSchemas].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseListMetadataSchemas._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_metadata_schemas(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseListMetadataSchemas._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseListMetadataSchemas._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.ListMetadataSchemas",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListMetadataSchemas",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._ListMetadataSchemas._get_response(
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

            # Return the response
            resp = metadata_service.ListMetadataSchemasResponse()
            pb_resp = metadata_service.ListMetadataSchemasResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_list_metadata_schemas(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_list_metadata_schemas_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        metadata_service.ListMetadataSchemasResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.list_metadata_schemas",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListMetadataSchemas",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _ListMetadataStores(
        _BaseMetadataServiceRestTransport._BaseListMetadataStores,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.ListMetadataStores")

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
            request: metadata_service.ListMetadataStoresRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.ListMetadataStoresResponse:
            r"""Call the list metadata stores method over HTTP.

            Args:
                request (~.metadata_service.ListMetadataStoresRequest):
                    The request object. Request message for
                [MetadataService.ListMetadataStores][google.cloud.aiplatform.v1beta1.MetadataService.ListMetadataStores].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_service.ListMetadataStoresResponse:
                    Response message for
                [MetadataService.ListMetadataStores][google.cloud.aiplatform.v1beta1.MetadataService.ListMetadataStores].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseListMetadataStores._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_metadata_stores(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseListMetadataStores._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseListMetadataStores._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.ListMetadataStores",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListMetadataStores",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._ListMetadataStores._get_response(
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

            # Return the response
            resp = metadata_service.ListMetadataStoresResponse()
            pb_resp = metadata_service.ListMetadataStoresResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_list_metadata_stores(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_list_metadata_stores_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        metadata_service.ListMetadataStoresResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.list_metadata_stores",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListMetadataStores",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _PurgeArtifacts(
        _BaseMetadataServiceRestTransport._BasePurgeArtifacts,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.PurgeArtifacts")

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
            request: metadata_service.PurgeArtifactsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the purge artifacts method over HTTP.

            Args:
                request (~.metadata_service.PurgeArtifactsRequest):
                    The request object. Request message for
                [MetadataService.PurgeArtifacts][google.cloud.aiplatform.v1beta1.MetadataService.PurgeArtifacts].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BaseMetadataServiceRestTransport._BasePurgeArtifacts._get_http_options()
            )

            request, metadata = await self._interceptor.pre_purge_artifacts(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BasePurgeArtifacts._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BasePurgeArtifacts._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BasePurgeArtifacts._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.PurgeArtifacts",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "PurgeArtifacts",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._PurgeArtifacts._get_response(
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
            resp = operations_pb2.Operation()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_purge_artifacts(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_purge_artifacts_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.purge_artifacts",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "PurgeArtifacts",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _PurgeContexts(
        _BaseMetadataServiceRestTransport._BasePurgeContexts,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.PurgeContexts")

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
            request: metadata_service.PurgeContextsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the purge contexts method over HTTP.

            Args:
                request (~.metadata_service.PurgeContextsRequest):
                    The request object. Request message for
                [MetadataService.PurgeContexts][google.cloud.aiplatform.v1beta1.MetadataService.PurgeContexts].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BaseMetadataServiceRestTransport._BasePurgeContexts._get_http_options()
            )

            request, metadata = await self._interceptor.pre_purge_contexts(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BasePurgeContexts._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BasePurgeContexts._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BasePurgeContexts._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.PurgeContexts",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "PurgeContexts",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._PurgeContexts._get_response(
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
            resp = operations_pb2.Operation()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_purge_contexts(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_purge_contexts_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.purge_contexts",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "PurgeContexts",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _PurgeExecutions(
        _BaseMetadataServiceRestTransport._BasePurgeExecutions,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.PurgeExecutions")

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
            request: metadata_service.PurgeExecutionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the purge executions method over HTTP.

            Args:
                request (~.metadata_service.PurgeExecutionsRequest):
                    The request object. Request message for
                [MetadataService.PurgeExecutions][google.cloud.aiplatform.v1beta1.MetadataService.PurgeExecutions].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
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
                _BaseMetadataServiceRestTransport._BasePurgeExecutions._get_http_options()
            )

            request, metadata = await self._interceptor.pre_purge_executions(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BasePurgeExecutions._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BasePurgeExecutions._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BasePurgeExecutions._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.PurgeExecutions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "PurgeExecutions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._PurgeExecutions._get_response(
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
            resp = operations_pb2.Operation()
            pb_resp = resp
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_purge_executions(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_purge_executions_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.purge_executions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "PurgeExecutions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _QueryArtifactLineageSubgraph(
        _BaseMetadataServiceRestTransport._BaseQueryArtifactLineageSubgraph,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash(
                "AsyncMetadataServiceRestTransport.QueryArtifactLineageSubgraph"
            )

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
            request: metadata_service.QueryArtifactLineageSubgraphRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> lineage_subgraph.LineageSubgraph:
            r"""Call the query artifact lineage
            subgraph method over HTTP.

                Args:
                    request (~.metadata_service.QueryArtifactLineageSubgraphRequest):
                        The request object. Request message for
                    [MetadataService.QueryArtifactLineageSubgraph][google.cloud.aiplatform.v1beta1.MetadataService.QueryArtifactLineageSubgraph].
                    retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.lineage_subgraph.LineageSubgraph:
                        A subgraph of the overall lineage
                    graph. Event edges connect Artifact and
                    Execution nodes.

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseQueryArtifactLineageSubgraph._get_http_options()
            )

            request, metadata = (
                await self._interceptor.pre_query_artifact_lineage_subgraph(
                    request, metadata
                )
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseQueryArtifactLineageSubgraph._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseQueryArtifactLineageSubgraph._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.QueryArtifactLineageSubgraph",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "QueryArtifactLineageSubgraph",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._QueryArtifactLineageSubgraph._get_response(
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

            # Return the response
            resp = lineage_subgraph.LineageSubgraph()
            pb_resp = lineage_subgraph.LineageSubgraph.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_query_artifact_lineage_subgraph(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                await self._interceptor.post_query_artifact_lineage_subgraph_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = lineage_subgraph.LineageSubgraph.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.query_artifact_lineage_subgraph",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "QueryArtifactLineageSubgraph",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _QueryContextLineageSubgraph(
        _BaseMetadataServiceRestTransport._BaseQueryContextLineageSubgraph,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.QueryContextLineageSubgraph")

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
            request: metadata_service.QueryContextLineageSubgraphRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> lineage_subgraph.LineageSubgraph:
            r"""Call the query context lineage
            subgraph method over HTTP.

                Args:
                    request (~.metadata_service.QueryContextLineageSubgraphRequest):
                        The request object. Request message for
                    [MetadataService.QueryContextLineageSubgraph][google.cloud.aiplatform.v1beta1.MetadataService.QueryContextLineageSubgraph].
                    retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.lineage_subgraph.LineageSubgraph:
                        A subgraph of the overall lineage
                    graph. Event edges connect Artifact and
                    Execution nodes.

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseQueryContextLineageSubgraph._get_http_options()
            )

            request, metadata = (
                await self._interceptor.pre_query_context_lineage_subgraph(
                    request, metadata
                )
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseQueryContextLineageSubgraph._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseQueryContextLineageSubgraph._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.QueryContextLineageSubgraph",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "QueryContextLineageSubgraph",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._QueryContextLineageSubgraph._get_response(
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

            # Return the response
            resp = lineage_subgraph.LineageSubgraph()
            pb_resp = lineage_subgraph.LineageSubgraph.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_query_context_lineage_subgraph(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                await self._interceptor.post_query_context_lineage_subgraph_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = lineage_subgraph.LineageSubgraph.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.query_context_lineage_subgraph",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "QueryContextLineageSubgraph",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _QueryExecutionInputsAndOutputs(
        _BaseMetadataServiceRestTransport._BaseQueryExecutionInputsAndOutputs,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash(
                "AsyncMetadataServiceRestTransport.QueryExecutionInputsAndOutputs"
            )

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
            request: metadata_service.QueryExecutionInputsAndOutputsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> lineage_subgraph.LineageSubgraph:
            r"""Call the query execution inputs
            and outputs method over HTTP.

                Args:
                    request (~.metadata_service.QueryExecutionInputsAndOutputsRequest):
                        The request object. Request message for
                    [MetadataService.QueryExecutionInputsAndOutputs][google.cloud.aiplatform.v1beta1.MetadataService.QueryExecutionInputsAndOutputs].
                    retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.lineage_subgraph.LineageSubgraph:
                        A subgraph of the overall lineage
                    graph. Event edges connect Artifact and
                    Execution nodes.

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseQueryExecutionInputsAndOutputs._get_http_options()
            )

            request, metadata = (
                await self._interceptor.pre_query_execution_inputs_and_outputs(
                    request, metadata
                )
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseQueryExecutionInputsAndOutputs._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseQueryExecutionInputsAndOutputs._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.QueryExecutionInputsAndOutputs",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "QueryExecutionInputsAndOutputs",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._QueryExecutionInputsAndOutputs._get_response(
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

            # Return the response
            resp = lineage_subgraph.LineageSubgraph()
            pb_resp = lineage_subgraph.LineageSubgraph.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_query_execution_inputs_and_outputs(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                await self._interceptor.post_query_execution_inputs_and_outputs_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = lineage_subgraph.LineageSubgraph.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.query_execution_inputs_and_outputs",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "QueryExecutionInputsAndOutputs",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _RemoveContextChildren(
        _BaseMetadataServiceRestTransport._BaseRemoveContextChildren,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.RemoveContextChildren")

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
            request: metadata_service.RemoveContextChildrenRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> metadata_service.RemoveContextChildrenResponse:
            r"""Call the remove context children method over HTTP.

            Args:
                request (~.metadata_service.RemoveContextChildrenRequest):
                    The request object. Request message for
                [MetadataService.DeleteContextChildrenRequest][].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.metadata_service.RemoveContextChildrenResponse:
                    Response message for
                [MetadataService.RemoveContextChildren][google.cloud.aiplatform.v1beta1.MetadataService.RemoveContextChildren].

            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseRemoveContextChildren._get_http_options()
            )

            request, metadata = await self._interceptor.pre_remove_context_children(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseRemoveContextChildren._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseRemoveContextChildren._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseRemoveContextChildren._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.RemoveContextChildren",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "RemoveContextChildren",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._RemoveContextChildren._get_response(
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
            resp = metadata_service.RemoveContextChildrenResponse()
            pb_resp = metadata_service.RemoveContextChildrenResponse.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_remove_context_children(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                await self._interceptor.post_remove_context_children_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        metadata_service.RemoveContextChildrenResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.remove_context_children",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "RemoveContextChildren",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _UpdateArtifact(
        _BaseMetadataServiceRestTransport._BaseUpdateArtifact,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.UpdateArtifact")

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
            request: metadata_service.UpdateArtifactRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_artifact.Artifact:
            r"""Call the update artifact method over HTTP.

            Args:
                request (~.metadata_service.UpdateArtifactRequest):
                    The request object. Request message for
                [MetadataService.UpdateArtifact][google.cloud.aiplatform.v1beta1.MetadataService.UpdateArtifact].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_artifact.Artifact:
                    Instance of a general artifact.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseUpdateArtifact._get_http_options()
            )

            request, metadata = await self._interceptor.pre_update_artifact(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseUpdateArtifact._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseUpdateArtifact._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseUpdateArtifact._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.UpdateArtifact",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "UpdateArtifact",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._UpdateArtifact._get_response(
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
            resp = gca_artifact.Artifact()
            pb_resp = gca_artifact.Artifact.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_update_artifact(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_update_artifact_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_artifact.Artifact.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.update_artifact",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "UpdateArtifact",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _UpdateContext(
        _BaseMetadataServiceRestTransport._BaseUpdateContext,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.UpdateContext")

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
            request: metadata_service.UpdateContextRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_context.Context:
            r"""Call the update context method over HTTP.

            Args:
                request (~.metadata_service.UpdateContextRequest):
                    The request object. Request message for
                [MetadataService.UpdateContext][google.cloud.aiplatform.v1beta1.MetadataService.UpdateContext].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_context.Context:
                    Instance of a general context.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseUpdateContext._get_http_options()
            )

            request, metadata = await self._interceptor.pre_update_context(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseUpdateContext._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseUpdateContext._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseUpdateContext._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.UpdateContext",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "UpdateContext",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._UpdateContext._get_response(
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
            resp = gca_context.Context()
            pb_resp = gca_context.Context.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_update_context(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_update_context_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_context.Context.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.update_context",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "UpdateContext",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    class _UpdateExecution(
        _BaseMetadataServiceRestTransport._BaseUpdateExecution,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.UpdateExecution")

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
            request: metadata_service.UpdateExecutionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_execution.Execution:
            r"""Call the update execution method over HTTP.

            Args:
                request (~.metadata_service.UpdateExecutionRequest):
                    The request object. Request message for
                [MetadataService.UpdateExecution][google.cloud.aiplatform.v1beta1.MetadataService.UpdateExecution].
                retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_execution.Execution:
                    Instance of a general execution.
            """

            http_options = (
                _BaseMetadataServiceRestTransport._BaseUpdateExecution._get_http_options()
            )

            request, metadata = await self._interceptor.pre_update_execution(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseUpdateExecution._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseUpdateExecution._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseUpdateExecution._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.UpdateExecution",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "UpdateExecution",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._UpdateExecution._get_response(
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
            resp = gca_execution.Execution()
            pb_resp = gca_execution.Execution.pb(resp)
            content = await response.read()
            json_format.Parse(content, pb_resp, ignore_unknown_fields=True)
            resp = await self._interceptor.post_update_execution(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = await self._interceptor.post_update_execution_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_execution.Execution.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": "OK",  # need to obtain this properly
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.update_execution",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "UpdateExecution",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )

            return resp

    @property
    def operations_client(self) -> AsyncOperationsRestClient:
        """Create the async client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        # Only create a new client if we do not already have one.
        if self._operations_client is None:
            http_options: Dict[str, List[Dict[str, str]]] = {
                "google.longrunning.Operations.CancelOperation": [
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/agents/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/apps/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/savedQueries/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/annotationSpecs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/deploymentResourcePools/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/edgeDevices/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/endpoints/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/extensionControllers/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/extensions/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/customJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/dataLabelingJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/hyperparameterTuningJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tuningJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/indexes/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/indexEndpoints/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/artifacts/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/contexts/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/executions/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/modelMonitors/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/migratableResources/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/evaluations/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/notebookExecutionJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimes/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimeTemplates/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/persistentResources/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/trials/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/trainingPipelines/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/pipelineJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/schedules/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/ragEngineConfig/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/specialistPools/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/agents/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/apps/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/savedQueries/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/annotationSpecs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/deploymentResourcePools/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/edgeDevices/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/endpoints/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/exampleStores/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensionControllers/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensions/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/customJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/dataLabelingJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/hyperparameterTuningJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexes/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexEndpoints/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/artifacts/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/contexts/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/executions/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelMonitors/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/migratableResources/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/evaluations/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookExecutionJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimes/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimeTemplates/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/persistentResources/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragEngineConfig/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/ragFiles/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/trials/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/trainingPipelines/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/pipelineJobs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/memories/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/sessions/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/schedules/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/specialistPools/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*/operations/*}:cancel",
                    },
                ],
                "google.longrunning.Operations.DeleteOperation": [
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/agents/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/apps/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/savedQueries/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/annotationSpecs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/deploymentResourcePools/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/edgeDevices/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/endpoints/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/extensionControllers/*}/operations",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/extensions/*}/operations",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/customJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/dataLabelingJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/hyperparameterTuningJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/indexes/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/indexEndpoints/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/artifacts/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/contexts/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/executions/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/modelMonitors/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/migratableResources/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/evaluations/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/notebookExecutionJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimes/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimeTemplates/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/persistentResources/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/trials/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/trainingPipelines/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/pipelineJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/schedules/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/specialistPools/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/ragEngineConfig/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/featureOnlineStores/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/features/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/featureMonitors/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/ui/{name=projects/*/locations/*/featureOnlineStores/*/featureViews/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/agents/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/apps/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/savedQueries/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/annotationSpecs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/deploymentResourcePools/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/edgeDevices/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/endpoints/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/customJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/dataLabelingJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/evaluationTasks/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/exampleStores/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensionControllers/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensions/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/hyperparameterTuningJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexes/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexEndpoints/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/artifacts/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/contexts/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/executions/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelMonitors/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/migratableResources/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/evaluations/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookExecutionJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimes/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimeTemplates/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/persistentResources/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragEngineConfig/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/ragFiles/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/memories/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/sessions/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/solvers/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/trials/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/trainingPipelines/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/pipelineJobs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/schedules/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/specialistPools/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureOnlineStores/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/features/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/featureMonitors/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureOnlineStores/*/featureViews/*/operations/*}",
                    },
                ],
                "google.longrunning.Operations.GetOperation": [
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/agents/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/apps/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/savedQueries/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/annotationSpecs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/deploymentResourcePools/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/edgeDeploymentJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/edgeDevices/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/endpoints/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/extensionControllers/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/extensions/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/customJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/dataLabelingJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/hyperparameterTuningJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tuningJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/indexes/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/indexEndpoints/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/artifacts/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/contexts/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/executions/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/modelMonitors/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/migratableResources/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/evaluations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/notebookExecutionJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimes/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimeTemplates/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/persistentResources/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/trials/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/trainingPipelines/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/pipelineJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/ragEngineConfig/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/schedules/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/specialistPools/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureOnlineStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureOnlineStores/*/featureViews/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/features/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/featureMonitors/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/agents/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/apps/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/savedQueries/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/annotationSpecs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/deploymentResourcePools/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/edgeDevices/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/endpoints/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/evaluationTasks/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/exampleStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensionControllers/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensions/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/customJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/dataLabelingJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/hyperparameterTuningJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexes/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexEndpoints/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/artifacts/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/contexts/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/executions/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelMonitors/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/migratableResources/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/evaluations/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookExecutionJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimes/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimeTemplates/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/persistentResources/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragEngineConfig/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/ragFiles/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/memories/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/sessions/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/solvers/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/trials/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/trainingPipelines/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/pipelineJobs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/schedules/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/specialistPools/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureOnlineStores/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureOnlineStores/*/featureViews/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/features/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/featureMonitors/*/operations/*}",
                    },
                ],
                "google.longrunning.Operations.ListOperations": [
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/agents/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/apps/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/savedQueries/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/annotationSpecs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/deploymentResourcePools/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/edgeDevices/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/endpoints/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/extensionControllers/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/extensions/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/customJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/dataLabelingJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/hyperparameterTuningJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tuningJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/indexes/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/indexEndpoints/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/artifacts/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/contexts/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/executions/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/modelMonitors/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/migratableResources/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/models/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/evaluations/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/notebookExecutionJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimes/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimeTemplates/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/trials/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/trainingPipelines/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/persistentResources/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/pipelineJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/ragEngineConfig}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/schedules/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/specialistPools/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureOnlineStores/*/operations/*}:wait",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureOnlineStores/*/featureViews/*/operations/*}:wait",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/operations/*}:wait",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/features/*/operations/*}:wait",
                    },
                    {
                        "method": "get",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/featureMonitors/*/operations/*}:wait",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/agents/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/apps/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/savedQueries/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/annotationSpecs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/deploymentResourcePools/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/edgeDevices/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/endpoints/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/evaluationTasks/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/exampleStores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensionControllers/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensions/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/customJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/dataLabelingJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/hyperparameterTuningJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexes/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexEndpoints/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/artifacts/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/contexts/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/executions/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelMonitors/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/migratableResources/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/evaluations/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookExecutionJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimes/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimeTemplates/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/persistentResources/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragEngineConfig}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/ragFiles/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/memories/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/sessions/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/solvers/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/trials/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/trainingPipelines/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/pipelineJobs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/schedules/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/specialistPools/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureOnlineStores/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureOnlineStores/*/featureViews/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/features/*}/operations",
                    },
                    {
                        "method": "get",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/featureMonitors/*}/operations",
                    },
                ],
                "google.longrunning.Operations.WaitOperation": [
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/agents/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/apps/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/savedQueries/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/annotationSpecs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/deploymentResourcePools/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/edgeDevices/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/endpoints/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/extensionControllers/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/extensions/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/customJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/dataLabelingJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/hyperparameterTuningJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tuningJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/indexes/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/indexEndpoints/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/artifacts/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/contexts/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/metadataStores/*/executions/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/modelMonitors/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/migratableResources/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/models/*/evaluations/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/notebookExecutionJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimes/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/notebookRuntimeTemplates/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/studies/*/trials/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/trainingPipelines/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/persistentResources/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/pipelineJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/schedules/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/specialistPools/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/ragEngineConfig/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featureOnlineStores/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featureOnlineStores/*/featureViews/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/features/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/ui/{name=projects/*/locations/*/featureGroups/*/featureMonitors/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/agents/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/apps/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/savedQueries/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/annotationSpecs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/datasets/*/dataItems/*/annotations/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/deploymentResourcePools/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/edgeDevices/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/endpoints/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/evaluationTasks/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/exampleStores/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensionControllers/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/extensions/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featurestores/*/entityTypes/*/features/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/customJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/dataLabelingJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/hyperparameterTuningJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexes/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/indexEndpoints/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/artifacts/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/contexts/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/metadataStores/*/executions/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelDeploymentMonitoringJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/modelMonitors/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/migratableResources/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/models/*/evaluations/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookExecutionJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimes/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/notebookRuntimeTemplates/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/persistentResources/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragEngineConfig/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/ragCorpora/*/ragFiles/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/memories/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/reasoningEngines/*/sessions/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/studies/*/trials/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/trainingPipelines/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/pipelineJobs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/schedules/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/specialistPools/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/tensorboards/*/experiments/*/runs/*/timeSeries/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureOnlineStores/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureOnlineStores/*/featureViews/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/features/*/operations/*}:wait",
                    },
                    {
                        "method": "post",
                        "uri": "/v1beta1/{name=projects/*/locations/*/featureGroups/*/featureMonitors/*/operations/*}:wait",
                    },
                ],
            }

            rest_transport = operations_v1.AsyncOperationsRestTransport(  # type: ignore
                host=self._host,
                # use the credentials which are saved
                credentials=self._credentials,  # type: ignore
                http_options=http_options,
                path_prefix="v1beta1",
            )

            self._operations_client = AsyncOperationsRestClient(
                transport=rest_transport
            )

        # Return the client from cache.
        return self._operations_client

    @property
    def add_context_artifacts_and_executions(
        self,
    ) -> Callable[
        [metadata_service.AddContextArtifactsAndExecutionsRequest],
        metadata_service.AddContextArtifactsAndExecutionsResponse,
    ]:
        return self._AddContextArtifactsAndExecutions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def add_context_children(
        self,
    ) -> Callable[
        [metadata_service.AddContextChildrenRequest],
        metadata_service.AddContextChildrenResponse,
    ]:
        return self._AddContextChildren(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def add_execution_events(
        self,
    ) -> Callable[
        [metadata_service.AddExecutionEventsRequest],
        metadata_service.AddExecutionEventsResponse,
    ]:
        return self._AddExecutionEvents(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_artifact(
        self,
    ) -> Callable[[metadata_service.CreateArtifactRequest], gca_artifact.Artifact]:
        return self._CreateArtifact(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_context(
        self,
    ) -> Callable[[metadata_service.CreateContextRequest], gca_context.Context]:
        return self._CreateContext(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_execution(
        self,
    ) -> Callable[[metadata_service.CreateExecutionRequest], gca_execution.Execution]:
        return self._CreateExecution(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_metadata_schema(
        self,
    ) -> Callable[
        [metadata_service.CreateMetadataSchemaRequest],
        gca_metadata_schema.MetadataSchema,
    ]:
        return self._CreateMetadataSchema(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_metadata_store(
        self,
    ) -> Callable[
        [metadata_service.CreateMetadataStoreRequest], operations_pb2.Operation
    ]:
        return self._CreateMetadataStore(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_artifact(
        self,
    ) -> Callable[[metadata_service.DeleteArtifactRequest], operations_pb2.Operation]:
        return self._DeleteArtifact(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_context(
        self,
    ) -> Callable[[metadata_service.DeleteContextRequest], operations_pb2.Operation]:
        return self._DeleteContext(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_execution(
        self,
    ) -> Callable[[metadata_service.DeleteExecutionRequest], operations_pb2.Operation]:
        return self._DeleteExecution(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_metadata_store(
        self,
    ) -> Callable[
        [metadata_service.DeleteMetadataStoreRequest], operations_pb2.Operation
    ]:
        return self._DeleteMetadataStore(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_artifact(
        self,
    ) -> Callable[[metadata_service.GetArtifactRequest], artifact.Artifact]:
        return self._GetArtifact(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_context(
        self,
    ) -> Callable[[metadata_service.GetContextRequest], context.Context]:
        return self._GetContext(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_execution(
        self,
    ) -> Callable[[metadata_service.GetExecutionRequest], execution.Execution]:
        return self._GetExecution(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_metadata_schema(
        self,
    ) -> Callable[
        [metadata_service.GetMetadataSchemaRequest], metadata_schema.MetadataSchema
    ]:
        return self._GetMetadataSchema(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_metadata_store(
        self,
    ) -> Callable[
        [metadata_service.GetMetadataStoreRequest], metadata_store.MetadataStore
    ]:
        return self._GetMetadataStore(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_artifacts(
        self,
    ) -> Callable[
        [metadata_service.ListArtifactsRequest], metadata_service.ListArtifactsResponse
    ]:
        return self._ListArtifacts(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_contexts(
        self,
    ) -> Callable[
        [metadata_service.ListContextsRequest], metadata_service.ListContextsResponse
    ]:
        return self._ListContexts(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_executions(
        self,
    ) -> Callable[
        [metadata_service.ListExecutionsRequest],
        metadata_service.ListExecutionsResponse,
    ]:
        return self._ListExecutions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_metadata_schemas(
        self,
    ) -> Callable[
        [metadata_service.ListMetadataSchemasRequest],
        metadata_service.ListMetadataSchemasResponse,
    ]:
        return self._ListMetadataSchemas(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_metadata_stores(
        self,
    ) -> Callable[
        [metadata_service.ListMetadataStoresRequest],
        metadata_service.ListMetadataStoresResponse,
    ]:
        return self._ListMetadataStores(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def purge_artifacts(
        self,
    ) -> Callable[[metadata_service.PurgeArtifactsRequest], operations_pb2.Operation]:
        return self._PurgeArtifacts(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def purge_contexts(
        self,
    ) -> Callable[[metadata_service.PurgeContextsRequest], operations_pb2.Operation]:
        return self._PurgeContexts(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def purge_executions(
        self,
    ) -> Callable[[metadata_service.PurgeExecutionsRequest], operations_pb2.Operation]:
        return self._PurgeExecutions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def query_artifact_lineage_subgraph(
        self,
    ) -> Callable[
        [metadata_service.QueryArtifactLineageSubgraphRequest],
        lineage_subgraph.LineageSubgraph,
    ]:
        return self._QueryArtifactLineageSubgraph(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def query_context_lineage_subgraph(
        self,
    ) -> Callable[
        [metadata_service.QueryContextLineageSubgraphRequest],
        lineage_subgraph.LineageSubgraph,
    ]:
        return self._QueryContextLineageSubgraph(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def query_execution_inputs_and_outputs(
        self,
    ) -> Callable[
        [metadata_service.QueryExecutionInputsAndOutputsRequest],
        lineage_subgraph.LineageSubgraph,
    ]:
        return self._QueryExecutionInputsAndOutputs(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def remove_context_children(
        self,
    ) -> Callable[
        [metadata_service.RemoveContextChildrenRequest],
        metadata_service.RemoveContextChildrenResponse,
    ]:
        return self._RemoveContextChildren(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_artifact(
        self,
    ) -> Callable[[metadata_service.UpdateArtifactRequest], gca_artifact.Artifact]:
        return self._UpdateArtifact(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_context(
        self,
    ) -> Callable[[metadata_service.UpdateContextRequest], gca_context.Context]:
        return self._UpdateContext(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_execution(
        self,
    ) -> Callable[[metadata_service.UpdateExecutionRequest], gca_execution.Execution]:
        return self._UpdateExecution(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_location(self):
        return self._GetLocation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetLocation(
        _BaseMetadataServiceRestTransport._BaseGetLocation, AsyncMetadataServiceRestStub
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.GetLocation")

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
                _BaseMetadataServiceRestTransport._BaseGetLocation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_location(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseGetLocation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseGetLocation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.GetLocation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetLocation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._GetLocation._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.GetLocation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
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
        _BaseMetadataServiceRestTransport._BaseListLocations,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.ListLocations")

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
                _BaseMetadataServiceRestTransport._BaseListLocations._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_locations(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseListLocations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseListLocations._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.ListLocations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListLocations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._ListLocations._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.ListLocations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
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
        _BaseMetadataServiceRestTransport._BaseGetIamPolicy,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.GetIamPolicy")

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
                _BaseMetadataServiceRestTransport._BaseGetIamPolicy._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_iam_policy(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseGetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseGetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseGetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._GetIamPolicy._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
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
        _BaseMetadataServiceRestTransport._BaseSetIamPolicy,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.SetIamPolicy")

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
                _BaseMetadataServiceRestTransport._BaseSetIamPolicy._get_http_options()
            )

            request, metadata = await self._interceptor.pre_set_iam_policy(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseSetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseSetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseSetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "SetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._SetIamPolicy._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
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
        _BaseMetadataServiceRestTransport._BaseTestIamPermissions,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.TestIamPermissions")

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
                _BaseMetadataServiceRestTransport._BaseTestIamPermissions._get_http_options()
            )

            request, metadata = await self._interceptor.pre_test_iam_permissions(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseTestIamPermissions._get_transcoded_request(
                http_options, request
            )

            body = _BaseMetadataServiceRestTransport._BaseTestIamPermissions._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseTestIamPermissions._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "TestIamPermissions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = await AsyncMetadataServiceRestTransport._TestIamPermissions._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
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
        _BaseMetadataServiceRestTransport._BaseCancelOperation,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.CancelOperation")

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
                _BaseMetadataServiceRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseCancelOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.CancelOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._CancelOperation._get_response(
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

            return await self._interceptor.post_cancel_operation(None)

    @property
    def delete_operation(self):
        return self._DeleteOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _DeleteOperation(
        _BaseMetadataServiceRestTransport._BaseDeleteOperation,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.DeleteOperation")

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
                _BaseMetadataServiceRestTransport._BaseDeleteOperation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_delete_operation(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseDeleteOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseDeleteOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.DeleteOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "DeleteOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._DeleteOperation._get_response(
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

            return await self._interceptor.post_delete_operation(None)

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BaseMetadataServiceRestTransport._BaseGetOperation,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.GetOperation")

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
                _BaseMetadataServiceRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_get_operation(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._GetOperation._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
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
        _BaseMetadataServiceRestTransport._BaseListOperations,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.ListOperations")

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
                _BaseMetadataServiceRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = await self._interceptor.pre_list_operations(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._ListOperations._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
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
        _BaseMetadataServiceRestTransport._BaseWaitOperation,
        AsyncMetadataServiceRestStub,
    ):
        def __hash__(self):
            return hash("AsyncMetadataServiceRestTransport.WaitOperation")

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
                _BaseMetadataServiceRestTransport._BaseWaitOperation._get_http_options()
            )

            request, metadata = await self._interceptor.pre_wait_operation(
                request, metadata
            )
            transcoded_request = _BaseMetadataServiceRestTransport._BaseWaitOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseMetadataServiceRestTransport._BaseWaitOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.MetadataServiceClient.WaitOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
                        "rpcName": "WaitOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                await AsyncMetadataServiceRestTransport._WaitOperation._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.MetadataServiceAsyncClient.WaitOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.MetadataService",
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
