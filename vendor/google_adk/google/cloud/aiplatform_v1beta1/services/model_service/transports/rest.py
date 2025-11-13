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
from google.api_core import operations_v1
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore

from requests import __version__ as requests_version
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings


from google.cloud.aiplatform_v1beta1.types import model
from google.cloud.aiplatform_v1beta1.types import model as gca_model
from google.cloud.aiplatform_v1beta1.types import model_evaluation
from google.cloud.aiplatform_v1beta1.types import (
    model_evaluation as gca_model_evaluation,
)
from google.cloud.aiplatform_v1beta1.types import model_evaluation_slice
from google.cloud.aiplatform_v1beta1.types import model_service
from google.longrunning import operations_pb2  # type: ignore


from .rest_base import _BaseModelServiceRestTransport
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


class ModelServiceRestInterceptor:
    """Interceptor for ModelService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the ModelServiceRestTransport.

    .. code-block:: python
        class MyCustomModelServiceInterceptor(ModelServiceRestInterceptor):
            def pre_batch_import_evaluated_annotations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_batch_import_evaluated_annotations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_batch_import_model_evaluation_slices(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_batch_import_model_evaluation_slices(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_copy_model(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_copy_model(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_model(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_delete_model(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_model_version(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_delete_model_version(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_export_model(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_export_model(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_model(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_model(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_model_evaluation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_model_evaluation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_model_evaluation_slice(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_model_evaluation_slice(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_import_model_evaluation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_import_model_evaluation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_model_evaluations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_model_evaluations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_model_evaluation_slices(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_model_evaluation_slices(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_models(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_models(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_model_version_checkpoints(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_model_version_checkpoints(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_model_versions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_model_versions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_merge_version_aliases(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_merge_version_aliases(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_recommend_spec(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_recommend_spec(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_explanation_dataset(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_explanation_dataset(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_model(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_model(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_upload_model(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_upload_model(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = ModelServiceRestTransport(interceptor=MyCustomModelServiceInterceptor())
        client = ModelServiceClient(transport=transport)


    """

    def pre_batch_import_evaluated_annotations(
        self,
        request: model_service.BatchImportEvaluatedAnnotationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.BatchImportEvaluatedAnnotationsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for batch_import_evaluated_annotations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_batch_import_evaluated_annotations(
        self, response: model_service.BatchImportEvaluatedAnnotationsResponse
    ) -> model_service.BatchImportEvaluatedAnnotationsResponse:
        """Post-rpc interceptor for batch_import_evaluated_annotations

        DEPRECATED. Please use the `post_batch_import_evaluated_annotations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_batch_import_evaluated_annotations` interceptor runs
        before the `post_batch_import_evaluated_annotations_with_metadata` interceptor.
        """
        return response

    def post_batch_import_evaluated_annotations_with_metadata(
        self,
        response: model_service.BatchImportEvaluatedAnnotationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.BatchImportEvaluatedAnnotationsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for batch_import_evaluated_annotations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_batch_import_evaluated_annotations_with_metadata`
        interceptor in new development instead of the `post_batch_import_evaluated_annotations` interceptor.
        When both interceptors are used, this `post_batch_import_evaluated_annotations_with_metadata` interceptor runs after the
        `post_batch_import_evaluated_annotations` interceptor. The (possibly modified) response returned by
        `post_batch_import_evaluated_annotations` will be passed to
        `post_batch_import_evaluated_annotations_with_metadata`.
        """
        return response, metadata

    def pre_batch_import_model_evaluation_slices(
        self,
        request: model_service.BatchImportModelEvaluationSlicesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.BatchImportModelEvaluationSlicesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for batch_import_model_evaluation_slices

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_batch_import_model_evaluation_slices(
        self, response: model_service.BatchImportModelEvaluationSlicesResponse
    ) -> model_service.BatchImportModelEvaluationSlicesResponse:
        """Post-rpc interceptor for batch_import_model_evaluation_slices

        DEPRECATED. Please use the `post_batch_import_model_evaluation_slices_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_batch_import_model_evaluation_slices` interceptor runs
        before the `post_batch_import_model_evaluation_slices_with_metadata` interceptor.
        """
        return response

    def post_batch_import_model_evaluation_slices_with_metadata(
        self,
        response: model_service.BatchImportModelEvaluationSlicesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.BatchImportModelEvaluationSlicesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for batch_import_model_evaluation_slices

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_batch_import_model_evaluation_slices_with_metadata`
        interceptor in new development instead of the `post_batch_import_model_evaluation_slices` interceptor.
        When both interceptors are used, this `post_batch_import_model_evaluation_slices_with_metadata` interceptor runs after the
        `post_batch_import_model_evaluation_slices` interceptor. The (possibly modified) response returned by
        `post_batch_import_model_evaluation_slices` will be passed to
        `post_batch_import_model_evaluation_slices_with_metadata`.
        """
        return response, metadata

    def pre_copy_model(
        self,
        request: model_service.CopyModelRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[model_service.CopyModelRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for copy_model

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_copy_model(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for copy_model

        DEPRECATED. Please use the `post_copy_model_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_copy_model` interceptor runs
        before the `post_copy_model_with_metadata` interceptor.
        """
        return response

    def post_copy_model_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for copy_model

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_copy_model_with_metadata`
        interceptor in new development instead of the `post_copy_model` interceptor.
        When both interceptors are used, this `post_copy_model_with_metadata` interceptor runs after the
        `post_copy_model` interceptor. The (possibly modified) response returned by
        `post_copy_model` will be passed to
        `post_copy_model_with_metadata`.
        """
        return response, metadata

    def pre_delete_model(
        self,
        request: model_service.DeleteModelRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.DeleteModelRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_model

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_delete_model(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_model

        DEPRECATED. Please use the `post_delete_model_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_delete_model` interceptor runs
        before the `post_delete_model_with_metadata` interceptor.
        """
        return response

    def post_delete_model_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_model

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_delete_model_with_metadata`
        interceptor in new development instead of the `post_delete_model` interceptor.
        When both interceptors are used, this `post_delete_model_with_metadata` interceptor runs after the
        `post_delete_model` interceptor. The (possibly modified) response returned by
        `post_delete_model` will be passed to
        `post_delete_model_with_metadata`.
        """
        return response, metadata

    def pre_delete_model_version(
        self,
        request: model_service.DeleteModelVersionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.DeleteModelVersionRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_model_version

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_delete_model_version(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_model_version

        DEPRECATED. Please use the `post_delete_model_version_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_delete_model_version` interceptor runs
        before the `post_delete_model_version_with_metadata` interceptor.
        """
        return response

    def post_delete_model_version_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for delete_model_version

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_delete_model_version_with_metadata`
        interceptor in new development instead of the `post_delete_model_version` interceptor.
        When both interceptors are used, this `post_delete_model_version_with_metadata` interceptor runs after the
        `post_delete_model_version` interceptor. The (possibly modified) response returned by
        `post_delete_model_version` will be passed to
        `post_delete_model_version_with_metadata`.
        """
        return response, metadata

    def pre_export_model(
        self,
        request: model_service.ExportModelRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ExportModelRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for export_model

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_export_model(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for export_model

        DEPRECATED. Please use the `post_export_model_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_export_model` interceptor runs
        before the `post_export_model_with_metadata` interceptor.
        """
        return response

    def post_export_model_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for export_model

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_export_model_with_metadata`
        interceptor in new development instead of the `post_export_model` interceptor.
        When both interceptors are used, this `post_export_model_with_metadata` interceptor runs after the
        `post_export_model` interceptor. The (possibly modified) response returned by
        `post_export_model` will be passed to
        `post_export_model_with_metadata`.
        """
        return response, metadata

    def pre_get_model(
        self,
        request: model_service.GetModelRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[model_service.GetModelRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_model

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_get_model(self, response: model.Model) -> model.Model:
        """Post-rpc interceptor for get_model

        DEPRECATED. Please use the `post_get_model_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_get_model` interceptor runs
        before the `post_get_model_with_metadata` interceptor.
        """
        return response

    def post_get_model_with_metadata(
        self, response: model.Model, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[model.Model, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_model

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_get_model_with_metadata`
        interceptor in new development instead of the `post_get_model` interceptor.
        When both interceptors are used, this `post_get_model_with_metadata` interceptor runs after the
        `post_get_model` interceptor. The (possibly modified) response returned by
        `post_get_model` will be passed to
        `post_get_model_with_metadata`.
        """
        return response, metadata

    def pre_get_model_evaluation(
        self,
        request: model_service.GetModelEvaluationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.GetModelEvaluationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_model_evaluation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_get_model_evaluation(
        self, response: model_evaluation.ModelEvaluation
    ) -> model_evaluation.ModelEvaluation:
        """Post-rpc interceptor for get_model_evaluation

        DEPRECATED. Please use the `post_get_model_evaluation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_get_model_evaluation` interceptor runs
        before the `post_get_model_evaluation_with_metadata` interceptor.
        """
        return response

    def post_get_model_evaluation_with_metadata(
        self,
        response: model_evaluation.ModelEvaluation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_evaluation.ModelEvaluation, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for get_model_evaluation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_get_model_evaluation_with_metadata`
        interceptor in new development instead of the `post_get_model_evaluation` interceptor.
        When both interceptors are used, this `post_get_model_evaluation_with_metadata` interceptor runs after the
        `post_get_model_evaluation` interceptor. The (possibly modified) response returned by
        `post_get_model_evaluation` will be passed to
        `post_get_model_evaluation_with_metadata`.
        """
        return response, metadata

    def pre_get_model_evaluation_slice(
        self,
        request: model_service.GetModelEvaluationSliceRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.GetModelEvaluationSliceRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_model_evaluation_slice

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_get_model_evaluation_slice(
        self, response: model_evaluation_slice.ModelEvaluationSlice
    ) -> model_evaluation_slice.ModelEvaluationSlice:
        """Post-rpc interceptor for get_model_evaluation_slice

        DEPRECATED. Please use the `post_get_model_evaluation_slice_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_get_model_evaluation_slice` interceptor runs
        before the `post_get_model_evaluation_slice_with_metadata` interceptor.
        """
        return response

    def post_get_model_evaluation_slice_with_metadata(
        self,
        response: model_evaluation_slice.ModelEvaluationSlice,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_evaluation_slice.ModelEvaluationSlice,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for get_model_evaluation_slice

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_get_model_evaluation_slice_with_metadata`
        interceptor in new development instead of the `post_get_model_evaluation_slice` interceptor.
        When both interceptors are used, this `post_get_model_evaluation_slice_with_metadata` interceptor runs after the
        `post_get_model_evaluation_slice` interceptor. The (possibly modified) response returned by
        `post_get_model_evaluation_slice` will be passed to
        `post_get_model_evaluation_slice_with_metadata`.
        """
        return response, metadata

    def pre_import_model_evaluation(
        self,
        request: model_service.ImportModelEvaluationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ImportModelEvaluationRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for import_model_evaluation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_import_model_evaluation(
        self, response: gca_model_evaluation.ModelEvaluation
    ) -> gca_model_evaluation.ModelEvaluation:
        """Post-rpc interceptor for import_model_evaluation

        DEPRECATED. Please use the `post_import_model_evaluation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_import_model_evaluation` interceptor runs
        before the `post_import_model_evaluation_with_metadata` interceptor.
        """
        return response

    def post_import_model_evaluation_with_metadata(
        self,
        response: gca_model_evaluation.ModelEvaluation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        gca_model_evaluation.ModelEvaluation, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for import_model_evaluation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_import_model_evaluation_with_metadata`
        interceptor in new development instead of the `post_import_model_evaluation` interceptor.
        When both interceptors are used, this `post_import_model_evaluation_with_metadata` interceptor runs after the
        `post_import_model_evaluation` interceptor. The (possibly modified) response returned by
        `post_import_model_evaluation` will be passed to
        `post_import_model_evaluation_with_metadata`.
        """
        return response, metadata

    def pre_list_model_evaluations(
        self,
        request: model_service.ListModelEvaluationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelEvaluationsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_model_evaluations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_list_model_evaluations(
        self, response: model_service.ListModelEvaluationsResponse
    ) -> model_service.ListModelEvaluationsResponse:
        """Post-rpc interceptor for list_model_evaluations

        DEPRECATED. Please use the `post_list_model_evaluations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_list_model_evaluations` interceptor runs
        before the `post_list_model_evaluations_with_metadata` interceptor.
        """
        return response

    def post_list_model_evaluations_with_metadata(
        self,
        response: model_service.ListModelEvaluationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelEvaluationsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_model_evaluations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_list_model_evaluations_with_metadata`
        interceptor in new development instead of the `post_list_model_evaluations` interceptor.
        When both interceptors are used, this `post_list_model_evaluations_with_metadata` interceptor runs after the
        `post_list_model_evaluations` interceptor. The (possibly modified) response returned by
        `post_list_model_evaluations` will be passed to
        `post_list_model_evaluations_with_metadata`.
        """
        return response, metadata

    def pre_list_model_evaluation_slices(
        self,
        request: model_service.ListModelEvaluationSlicesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelEvaluationSlicesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_model_evaluation_slices

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_list_model_evaluation_slices(
        self, response: model_service.ListModelEvaluationSlicesResponse
    ) -> model_service.ListModelEvaluationSlicesResponse:
        """Post-rpc interceptor for list_model_evaluation_slices

        DEPRECATED. Please use the `post_list_model_evaluation_slices_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_list_model_evaluation_slices` interceptor runs
        before the `post_list_model_evaluation_slices_with_metadata` interceptor.
        """
        return response

    def post_list_model_evaluation_slices_with_metadata(
        self,
        response: model_service.ListModelEvaluationSlicesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelEvaluationSlicesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_model_evaluation_slices

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_list_model_evaluation_slices_with_metadata`
        interceptor in new development instead of the `post_list_model_evaluation_slices` interceptor.
        When both interceptors are used, this `post_list_model_evaluation_slices_with_metadata` interceptor runs after the
        `post_list_model_evaluation_slices` interceptor. The (possibly modified) response returned by
        `post_list_model_evaluation_slices` will be passed to
        `post_list_model_evaluation_slices_with_metadata`.
        """
        return response, metadata

    def pre_list_models(
        self,
        request: model_service.ListModelsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_models

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_list_models(
        self, response: model_service.ListModelsResponse
    ) -> model_service.ListModelsResponse:
        """Post-rpc interceptor for list_models

        DEPRECATED. Please use the `post_list_models_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_list_models` interceptor runs
        before the `post_list_models_with_metadata` interceptor.
        """
        return response

    def post_list_models_with_metadata(
        self,
        response: model_service.ListModelsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_models

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_list_models_with_metadata`
        interceptor in new development instead of the `post_list_models` interceptor.
        When both interceptors are used, this `post_list_models_with_metadata` interceptor runs after the
        `post_list_models` interceptor. The (possibly modified) response returned by
        `post_list_models` will be passed to
        `post_list_models_with_metadata`.
        """
        return response, metadata

    def pre_list_model_version_checkpoints(
        self,
        request: model_service.ListModelVersionCheckpointsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelVersionCheckpointsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_model_version_checkpoints

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_list_model_version_checkpoints(
        self, response: model_service.ListModelVersionCheckpointsResponse
    ) -> model_service.ListModelVersionCheckpointsResponse:
        """Post-rpc interceptor for list_model_version_checkpoints

        DEPRECATED. Please use the `post_list_model_version_checkpoints_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_list_model_version_checkpoints` interceptor runs
        before the `post_list_model_version_checkpoints_with_metadata` interceptor.
        """
        return response

    def post_list_model_version_checkpoints_with_metadata(
        self,
        response: model_service.ListModelVersionCheckpointsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelVersionCheckpointsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_model_version_checkpoints

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_list_model_version_checkpoints_with_metadata`
        interceptor in new development instead of the `post_list_model_version_checkpoints` interceptor.
        When both interceptors are used, this `post_list_model_version_checkpoints_with_metadata` interceptor runs after the
        `post_list_model_version_checkpoints` interceptor. The (possibly modified) response returned by
        `post_list_model_version_checkpoints` will be passed to
        `post_list_model_version_checkpoints_with_metadata`.
        """
        return response, metadata

    def pre_list_model_versions(
        self,
        request: model_service.ListModelVersionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelVersionsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_model_versions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_list_model_versions(
        self, response: model_service.ListModelVersionsResponse
    ) -> model_service.ListModelVersionsResponse:
        """Post-rpc interceptor for list_model_versions

        DEPRECATED. Please use the `post_list_model_versions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_list_model_versions` interceptor runs
        before the `post_list_model_versions_with_metadata` interceptor.
        """
        return response

    def post_list_model_versions_with_metadata(
        self,
        response: model_service.ListModelVersionsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.ListModelVersionsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_model_versions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_list_model_versions_with_metadata`
        interceptor in new development instead of the `post_list_model_versions` interceptor.
        When both interceptors are used, this `post_list_model_versions_with_metadata` interceptor runs after the
        `post_list_model_versions` interceptor. The (possibly modified) response returned by
        `post_list_model_versions` will be passed to
        `post_list_model_versions_with_metadata`.
        """
        return response, metadata

    def pre_merge_version_aliases(
        self,
        request: model_service.MergeVersionAliasesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.MergeVersionAliasesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for merge_version_aliases

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_merge_version_aliases(self, response: model.Model) -> model.Model:
        """Post-rpc interceptor for merge_version_aliases

        DEPRECATED. Please use the `post_merge_version_aliases_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_merge_version_aliases` interceptor runs
        before the `post_merge_version_aliases_with_metadata` interceptor.
        """
        return response

    def post_merge_version_aliases_with_metadata(
        self, response: model.Model, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[model.Model, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for merge_version_aliases

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_merge_version_aliases_with_metadata`
        interceptor in new development instead of the `post_merge_version_aliases` interceptor.
        When both interceptors are used, this `post_merge_version_aliases_with_metadata` interceptor runs after the
        `post_merge_version_aliases` interceptor. The (possibly modified) response returned by
        `post_merge_version_aliases` will be passed to
        `post_merge_version_aliases_with_metadata`.
        """
        return response, metadata

    def pre_recommend_spec(
        self,
        request: model_service.RecommendSpecRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.RecommendSpecRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for recommend_spec

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_recommend_spec(
        self, response: model_service.RecommendSpecResponse
    ) -> model_service.RecommendSpecResponse:
        """Post-rpc interceptor for recommend_spec

        DEPRECATED. Please use the `post_recommend_spec_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_recommend_spec` interceptor runs
        before the `post_recommend_spec_with_metadata` interceptor.
        """
        return response

    def post_recommend_spec_with_metadata(
        self,
        response: model_service.RecommendSpecResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.RecommendSpecResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for recommend_spec

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_recommend_spec_with_metadata`
        interceptor in new development instead of the `post_recommend_spec` interceptor.
        When both interceptors are used, this `post_recommend_spec_with_metadata` interceptor runs after the
        `post_recommend_spec` interceptor. The (possibly modified) response returned by
        `post_recommend_spec` will be passed to
        `post_recommend_spec_with_metadata`.
        """
        return response, metadata

    def pre_update_explanation_dataset(
        self,
        request: model_service.UpdateExplanationDatasetRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.UpdateExplanationDatasetRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_explanation_dataset

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_update_explanation_dataset(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_explanation_dataset

        DEPRECATED. Please use the `post_update_explanation_dataset_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_update_explanation_dataset` interceptor runs
        before the `post_update_explanation_dataset_with_metadata` interceptor.
        """
        return response

    def post_update_explanation_dataset_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_explanation_dataset

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_update_explanation_dataset_with_metadata`
        interceptor in new development instead of the `post_update_explanation_dataset` interceptor.
        When both interceptors are used, this `post_update_explanation_dataset_with_metadata` interceptor runs after the
        `post_update_explanation_dataset` interceptor. The (possibly modified) response returned by
        `post_update_explanation_dataset` will be passed to
        `post_update_explanation_dataset_with_metadata`.
        """
        return response, metadata

    def pre_update_model(
        self,
        request: model_service.UpdateModelRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.UpdateModelRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for update_model

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_update_model(self, response: gca_model.Model) -> gca_model.Model:
        """Post-rpc interceptor for update_model

        DEPRECATED. Please use the `post_update_model_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_update_model` interceptor runs
        before the `post_update_model_with_metadata` interceptor.
        """
        return response

    def post_update_model_with_metadata(
        self,
        response: gca_model.Model,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[gca_model.Model, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_model

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_update_model_with_metadata`
        interceptor in new development instead of the `post_update_model` interceptor.
        When both interceptors are used, this `post_update_model_with_metadata` interceptor runs after the
        `post_update_model` interceptor. The (possibly modified) response returned by
        `post_update_model` will be passed to
        `post_update_model_with_metadata`.
        """
        return response, metadata

    def pre_upload_model(
        self,
        request: model_service.UploadModelRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        model_service.UploadModelRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for upload_model

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_upload_model(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for upload_model

        DEPRECATED. Please use the `post_upload_model_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code. This `post_upload_model` interceptor runs
        before the `post_upload_model_with_metadata` interceptor.
        """
        return response

    def post_upload_model_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for upload_model

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the ModelService server but before it is returned to user code.

        We recommend only using this `post_upload_model_with_metadata`
        interceptor in new development instead of the `post_upload_model` interceptor.
        When both interceptors are used, this `post_upload_model_with_metadata` interceptor runs after the
        `post_upload_model` interceptor. The (possibly modified) response returned by
        `post_upload_model` will be passed to
        `post_upload_model_with_metadata`.
        """
        return response, metadata

    def pre_get_location(
        self,
        request: locations_pb2.GetLocationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        locations_pb2.GetLocationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_location

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_get_location(
        self, response: locations_pb2.Location
    ) -> locations_pb2.Location:
        """Post-rpc interceptor for get_location

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code.
        """
        return response

    def pre_list_locations(
        self,
        request: locations_pb2.ListLocationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        locations_pb2.ListLocationsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_locations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_list_locations(
        self, response: locations_pb2.ListLocationsResponse
    ) -> locations_pb2.ListLocationsResponse:
        """Post-rpc interceptor for list_locations

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code.
        """
        return response

    def pre_get_iam_policy(
        self,
        request: iam_policy_pb2.GetIamPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.GetIamPolicyRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_get_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code.
        """
        return response

    def pre_set_iam_policy(
        self,
        request: iam_policy_pb2.SetIamPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.SetIamPolicyRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_set_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code.
        """
        return response

    def pre_test_iam_permissions(
        self,
        request: iam_policy_pb2.TestIamPermissionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.TestIamPermissionsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_test_iam_permissions(
        self, response: iam_policy_pb2.TestIamPermissionsResponse
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        """Post-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code.
        """
        return response

    def pre_cancel_operation(
        self,
        request: operations_pb2.CancelOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.CancelOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code.
        """
        return response

    def pre_delete_operation(
        self,
        request: operations_pb2.DeleteOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.DeleteOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for delete_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_delete_operation(self, response: None) -> None:
        """Post-rpc interceptor for delete_operation

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
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
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
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
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code.
        """
        return response

    def pre_wait_operation(
        self,
        request: operations_pb2.WaitOperationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        operations_pb2.WaitOperationRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for wait_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ModelService server.
        """
        return request, metadata

    def post_wait_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for wait_operation

        Override in a subclass to manipulate the response
        after it is returned by the ModelService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class ModelServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: ModelServiceRestInterceptor


class ModelServiceRestTransport(_BaseModelServiceRestTransport):
    """REST backend synchronous transport for ModelService.

    A service for managing Vertex AI's machine learning Models.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "aiplatform.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[ModelServiceRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'aiplatform.googleapis.com').
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
        self._operations_client: Optional[operations_v1.AbstractOperationsClient] = None
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or ModelServiceRestInterceptor()
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

            rest_transport = operations_v1.OperationsRestTransport(
                host=self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                scopes=self._scopes,
                http_options=http_options,
                path_prefix="v1beta1",
            )

            self._operations_client = operations_v1.AbstractOperationsClient(
                transport=rest_transport
            )

        # Return the client from cache.
        return self._operations_client

    class _BatchImportEvaluatedAnnotations(
        _BaseModelServiceRestTransport._BaseBatchImportEvaluatedAnnotations,
        ModelServiceRestStub,
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.BatchImportEvaluatedAnnotations")

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
            request: model_service.BatchImportEvaluatedAnnotationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_service.BatchImportEvaluatedAnnotationsResponse:
            r"""Call the batch import evaluated
            annotations method over HTTP.

                Args:
                    request (~.model_service.BatchImportEvaluatedAnnotationsRequest):
                        The request object. Request message for
                    [ModelService.BatchImportEvaluatedAnnotations][google.cloud.aiplatform.v1beta1.ModelService.BatchImportEvaluatedAnnotations]
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.model_service.BatchImportEvaluatedAnnotationsResponse:
                        Response message for
                    [ModelService.BatchImportEvaluatedAnnotations][google.cloud.aiplatform.v1beta1.ModelService.BatchImportEvaluatedAnnotations]

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseBatchImportEvaluatedAnnotations._get_http_options()
            )

            request, metadata = (
                self._interceptor.pre_batch_import_evaluated_annotations(
                    request, metadata
                )
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseBatchImportEvaluatedAnnotations._get_transcoded_request(
                http_options, request
            )

            body = _BaseModelServiceRestTransport._BaseBatchImportEvaluatedAnnotations._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseBatchImportEvaluatedAnnotations._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.BatchImportEvaluatedAnnotations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "BatchImportEvaluatedAnnotations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._BatchImportEvaluatedAnnotations._get_response(
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
            resp = model_service.BatchImportEvaluatedAnnotationsResponse()
            pb_resp = model_service.BatchImportEvaluatedAnnotationsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_batch_import_evaluated_annotations(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                self._interceptor.post_batch_import_evaluated_annotations_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        model_service.BatchImportEvaluatedAnnotationsResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.batch_import_evaluated_annotations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "BatchImportEvaluatedAnnotations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _BatchImportModelEvaluationSlices(
        _BaseModelServiceRestTransport._BaseBatchImportModelEvaluationSlices,
        ModelServiceRestStub,
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.BatchImportModelEvaluationSlices")

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
            request: model_service.BatchImportModelEvaluationSlicesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_service.BatchImportModelEvaluationSlicesResponse:
            r"""Call the batch import model
            evaluation slices method over HTTP.

                Args:
                    request (~.model_service.BatchImportModelEvaluationSlicesRequest):
                        The request object. Request message for
                    [ModelService.BatchImportModelEvaluationSlices][google.cloud.aiplatform.v1beta1.ModelService.BatchImportModelEvaluationSlices]
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.model_service.BatchImportModelEvaluationSlicesResponse:
                        Response message for
                    [ModelService.BatchImportModelEvaluationSlices][google.cloud.aiplatform.v1beta1.ModelService.BatchImportModelEvaluationSlices]

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseBatchImportModelEvaluationSlices._get_http_options()
            )

            request, metadata = (
                self._interceptor.pre_batch_import_model_evaluation_slices(
                    request, metadata
                )
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseBatchImportModelEvaluationSlices._get_transcoded_request(
                http_options, request
            )

            body = _BaseModelServiceRestTransport._BaseBatchImportModelEvaluationSlices._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseBatchImportModelEvaluationSlices._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.BatchImportModelEvaluationSlices",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "BatchImportModelEvaluationSlices",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._BatchImportModelEvaluationSlices._get_response(
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
            resp = model_service.BatchImportModelEvaluationSlicesResponse()
            pb_resp = model_service.BatchImportModelEvaluationSlicesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_batch_import_model_evaluation_slices(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                self._interceptor.post_batch_import_model_evaluation_slices_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        model_service.BatchImportModelEvaluationSlicesResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.batch_import_model_evaluation_slices",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "BatchImportModelEvaluationSlices",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CopyModel(
        _BaseModelServiceRestTransport._BaseCopyModel, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.CopyModel")

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
            request: model_service.CopyModelRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the copy model method over HTTP.

            Args:
                request (~.model_service.CopyModelRequest):
                    The request object. Request message for
                [ModelService.CopyModel][google.cloud.aiplatform.v1beta1.ModelService.CopyModel].
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
                _BaseModelServiceRestTransport._BaseCopyModel._get_http_options()
            )

            request, metadata = self._interceptor.pre_copy_model(request, metadata)
            transcoded_request = (
                _BaseModelServiceRestTransport._BaseCopyModel._get_transcoded_request(
                    http_options, request
                )
            )

            body = _BaseModelServiceRestTransport._BaseCopyModel._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseCopyModel._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.CopyModel",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "CopyModel",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._CopyModel._get_response(
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

            resp = self._interceptor.post_copy_model(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_copy_model_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.copy_model",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "CopyModel",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteModel(
        _BaseModelServiceRestTransport._BaseDeleteModel, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.DeleteModel")

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
            request: model_service.DeleteModelRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete model method over HTTP.

            Args:
                request (~.model_service.DeleteModelRequest):
                    The request object. Request message for
                [ModelService.DeleteModel][google.cloud.aiplatform.v1beta1.ModelService.DeleteModel].
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
                _BaseModelServiceRestTransport._BaseDeleteModel._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_model(request, metadata)
            transcoded_request = (
                _BaseModelServiceRestTransport._BaseDeleteModel._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseDeleteModel._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.DeleteModel",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "DeleteModel",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._DeleteModel._get_response(
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

            resp = self._interceptor.post_delete_model(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_delete_model_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.delete_model",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "DeleteModel",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteModelVersion(
        _BaseModelServiceRestTransport._BaseDeleteModelVersion, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.DeleteModelVersion")

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
            request: model_service.DeleteModelVersionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the delete model version method over HTTP.

            Args:
                request (~.model_service.DeleteModelVersionRequest):
                    The request object. Request message for
                [ModelService.DeleteModelVersion][google.cloud.aiplatform.v1beta1.ModelService.DeleteModelVersion].
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
                _BaseModelServiceRestTransport._BaseDeleteModelVersion._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_model_version(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseDeleteModelVersion._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseDeleteModelVersion._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.DeleteModelVersion",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "DeleteModelVersion",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._DeleteModelVersion._get_response(
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

            resp = self._interceptor.post_delete_model_version(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_delete_model_version_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.delete_model_version",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "DeleteModelVersion",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ExportModel(
        _BaseModelServiceRestTransport._BaseExportModel, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ExportModel")

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
            request: model_service.ExportModelRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the export model method over HTTP.

            Args:
                request (~.model_service.ExportModelRequest):
                    The request object. Request message for
                [ModelService.ExportModel][google.cloud.aiplatform.v1beta1.ModelService.ExportModel].
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
                _BaseModelServiceRestTransport._BaseExportModel._get_http_options()
            )

            request, metadata = self._interceptor.pre_export_model(request, metadata)
            transcoded_request = (
                _BaseModelServiceRestTransport._BaseExportModel._get_transcoded_request(
                    http_options, request
                )
            )

            body = (
                _BaseModelServiceRestTransport._BaseExportModel._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseExportModel._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ExportModel",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ExportModel",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._ExportModel._get_response(
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

            resp = self._interceptor.post_export_model(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_export_model_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.export_model",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ExportModel",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetModel(_BaseModelServiceRestTransport._BaseGetModel, ModelServiceRestStub):
        def __hash__(self):
            return hash("ModelServiceRestTransport.GetModel")

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
            request: model_service.GetModelRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model.Model:
            r"""Call the get model method over HTTP.

            Args:
                request (~.model_service.GetModelRequest):
                    The request object. Request message for
                [ModelService.GetModel][google.cloud.aiplatform.v1beta1.ModelService.GetModel].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.model.Model:
                    A trained machine learning Model.
            """

            http_options = (
                _BaseModelServiceRestTransport._BaseGetModel._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_model(request, metadata)
            transcoded_request = (
                _BaseModelServiceRestTransport._BaseGetModel._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseGetModel._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.GetModel",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetModel",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._GetModel._get_response(
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
            resp = model.Model()
            pb_resp = model.Model.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_model(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_model_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = model.Model.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.get_model",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetModel",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetModelEvaluation(
        _BaseModelServiceRestTransport._BaseGetModelEvaluation, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.GetModelEvaluation")

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
            request: model_service.GetModelEvaluationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_evaluation.ModelEvaluation:
            r"""Call the get model evaluation method over HTTP.

            Args:
                request (~.model_service.GetModelEvaluationRequest):
                    The request object. Request message for
                [ModelService.GetModelEvaluation][google.cloud.aiplatform.v1beta1.ModelService.GetModelEvaluation].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.model_evaluation.ModelEvaluation:
                    A collection of metrics calculated by
                comparing Model's predictions on all of
                the test data against annotations from
                the test data.

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseGetModelEvaluation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_model_evaluation(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseGetModelEvaluation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseGetModelEvaluation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.GetModelEvaluation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetModelEvaluation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._GetModelEvaluation._get_response(
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
            resp = model_evaluation.ModelEvaluation()
            pb_resp = model_evaluation.ModelEvaluation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_model_evaluation(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_model_evaluation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = model_evaluation.ModelEvaluation.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.get_model_evaluation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetModelEvaluation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetModelEvaluationSlice(
        _BaseModelServiceRestTransport._BaseGetModelEvaluationSlice,
        ModelServiceRestStub,
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.GetModelEvaluationSlice")

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
            request: model_service.GetModelEvaluationSliceRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_evaluation_slice.ModelEvaluationSlice:
            r"""Call the get model evaluation
            slice method over HTTP.

                Args:
                    request (~.model_service.GetModelEvaluationSliceRequest):
                        The request object. Request message for
                    [ModelService.GetModelEvaluationSlice][google.cloud.aiplatform.v1beta1.ModelService.GetModelEvaluationSlice].
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.model_evaluation_slice.ModelEvaluationSlice:
                        A collection of metrics calculated by
                    comparing Model's predictions on a slice
                    of the test data against ground truth
                    annotations.

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseGetModelEvaluationSlice._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_model_evaluation_slice(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseGetModelEvaluationSlice._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseGetModelEvaluationSlice._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.GetModelEvaluationSlice",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetModelEvaluationSlice",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._GetModelEvaluationSlice._get_response(
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
            resp = model_evaluation_slice.ModelEvaluationSlice()
            pb_resp = model_evaluation_slice.ModelEvaluationSlice.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_model_evaluation_slice(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_model_evaluation_slice_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        model_evaluation_slice.ModelEvaluationSlice.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.get_model_evaluation_slice",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetModelEvaluationSlice",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ImportModelEvaluation(
        _BaseModelServiceRestTransport._BaseImportModelEvaluation, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ImportModelEvaluation")

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
            request: model_service.ImportModelEvaluationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_model_evaluation.ModelEvaluation:
            r"""Call the import model evaluation method over HTTP.

            Args:
                request (~.model_service.ImportModelEvaluationRequest):
                    The request object. Request message for
                [ModelService.ImportModelEvaluation][google.cloud.aiplatform.v1beta1.ModelService.ImportModelEvaluation]
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_model_evaluation.ModelEvaluation:
                    A collection of metrics calculated by
                comparing Model's predictions on all of
                the test data against annotations from
                the test data.

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseImportModelEvaluation._get_http_options()
            )

            request, metadata = self._interceptor.pre_import_model_evaluation(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseImportModelEvaluation._get_transcoded_request(
                http_options, request
            )

            body = _BaseModelServiceRestTransport._BaseImportModelEvaluation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseImportModelEvaluation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ImportModelEvaluation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ImportModelEvaluation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._ImportModelEvaluation._get_response(
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
            resp = gca_model_evaluation.ModelEvaluation()
            pb_resp = gca_model_evaluation.ModelEvaluation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_import_model_evaluation(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_import_model_evaluation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_model_evaluation.ModelEvaluation.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.import_model_evaluation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ImportModelEvaluation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListModelEvaluations(
        _BaseModelServiceRestTransport._BaseListModelEvaluations, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ListModelEvaluations")

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
            request: model_service.ListModelEvaluationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_service.ListModelEvaluationsResponse:
            r"""Call the list model evaluations method over HTTP.

            Args:
                request (~.model_service.ListModelEvaluationsRequest):
                    The request object. Request message for
                [ModelService.ListModelEvaluations][google.cloud.aiplatform.v1beta1.ModelService.ListModelEvaluations].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.model_service.ListModelEvaluationsResponse:
                    Response message for
                [ModelService.ListModelEvaluations][google.cloud.aiplatform.v1beta1.ModelService.ListModelEvaluations].

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseListModelEvaluations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_model_evaluations(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseListModelEvaluations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseListModelEvaluations._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ListModelEvaluations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModelEvaluations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._ListModelEvaluations._get_response(
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
            resp = model_service.ListModelEvaluationsResponse()
            pb_resp = model_service.ListModelEvaluationsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_model_evaluations(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_model_evaluations_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        model_service.ListModelEvaluationsResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.list_model_evaluations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModelEvaluations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListModelEvaluationSlices(
        _BaseModelServiceRestTransport._BaseListModelEvaluationSlices,
        ModelServiceRestStub,
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ListModelEvaluationSlices")

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
            request: model_service.ListModelEvaluationSlicesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_service.ListModelEvaluationSlicesResponse:
            r"""Call the list model evaluation
            slices method over HTTP.

                Args:
                    request (~.model_service.ListModelEvaluationSlicesRequest):
                        The request object. Request message for
                    [ModelService.ListModelEvaluationSlices][google.cloud.aiplatform.v1beta1.ModelService.ListModelEvaluationSlices].
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.model_service.ListModelEvaluationSlicesResponse:
                        Response message for
                    [ModelService.ListModelEvaluationSlices][google.cloud.aiplatform.v1beta1.ModelService.ListModelEvaluationSlices].

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseListModelEvaluationSlices._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_model_evaluation_slices(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseListModelEvaluationSlices._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseListModelEvaluationSlices._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ListModelEvaluationSlices",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModelEvaluationSlices",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ModelServiceRestTransport._ListModelEvaluationSlices._get_response(
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
            resp = model_service.ListModelEvaluationSlicesResponse()
            pb_resp = model_service.ListModelEvaluationSlicesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_model_evaluation_slices(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_model_evaluation_slices_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        model_service.ListModelEvaluationSlicesResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.list_model_evaluation_slices",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModelEvaluationSlices",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListModels(
        _BaseModelServiceRestTransport._BaseListModels, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ListModels")

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
            request: model_service.ListModelsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_service.ListModelsResponse:
            r"""Call the list models method over HTTP.

            Args:
                request (~.model_service.ListModelsRequest):
                    The request object. Request message for
                [ModelService.ListModels][google.cloud.aiplatform.v1beta1.ModelService.ListModels].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.model_service.ListModelsResponse:
                    Response message for
                [ModelService.ListModels][google.cloud.aiplatform.v1beta1.ModelService.ListModels]

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseListModels._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_models(request, metadata)
            transcoded_request = (
                _BaseModelServiceRestTransport._BaseListModels._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseListModels._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ListModels",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModels",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._ListModels._get_response(
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
            resp = model_service.ListModelsResponse()
            pb_resp = model_service.ListModelsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_models(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_models_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = model_service.ListModelsResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.list_models",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModels",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListModelVersionCheckpoints(
        _BaseModelServiceRestTransport._BaseListModelVersionCheckpoints,
        ModelServiceRestStub,
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ListModelVersionCheckpoints")

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
            request: model_service.ListModelVersionCheckpointsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_service.ListModelVersionCheckpointsResponse:
            r"""Call the list model version
            checkpoints method over HTTP.

                Args:
                    request (~.model_service.ListModelVersionCheckpointsRequest):
                        The request object. Request message for
                    [ModelService.ListModelVersionCheckpoints][google.cloud.aiplatform.v1beta1.ModelService.ListModelVersionCheckpoints].
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.model_service.ListModelVersionCheckpointsResponse:
                        Response message for
                    [ModelService.ListModelVersionCheckpoints][google.cloud.aiplatform.v1beta1.ModelService.ListModelVersionCheckpoints]

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseListModelVersionCheckpoints._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_model_version_checkpoints(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseListModelVersionCheckpoints._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseListModelVersionCheckpoints._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ListModelVersionCheckpoints",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModelVersionCheckpoints",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ModelServiceRestTransport._ListModelVersionCheckpoints._get_response(
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
            resp = model_service.ListModelVersionCheckpointsResponse()
            pb_resp = model_service.ListModelVersionCheckpointsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_model_version_checkpoints(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = (
                self._interceptor.post_list_model_version_checkpoints_with_metadata(
                    resp, response_metadata
                )
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        model_service.ListModelVersionCheckpointsResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.list_model_version_checkpoints",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModelVersionCheckpoints",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListModelVersions(
        _BaseModelServiceRestTransport._BaseListModelVersions, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ListModelVersions")

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
            request: model_service.ListModelVersionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_service.ListModelVersionsResponse:
            r"""Call the list model versions method over HTTP.

            Args:
                request (~.model_service.ListModelVersionsRequest):
                    The request object. Request message for
                [ModelService.ListModelVersions][google.cloud.aiplatform.v1beta1.ModelService.ListModelVersions].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.model_service.ListModelVersionsResponse:
                    Response message for
                [ModelService.ListModelVersions][google.cloud.aiplatform.v1beta1.ModelService.ListModelVersions]

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseListModelVersions._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_model_versions(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseListModelVersions._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseListModelVersions._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ListModelVersions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModelVersions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._ListModelVersions._get_response(
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
            resp = model_service.ListModelVersionsResponse()
            pb_resp = model_service.ListModelVersionsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_model_versions(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_model_versions_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = model_service.ListModelVersionsResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.list_model_versions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListModelVersions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _MergeVersionAliases(
        _BaseModelServiceRestTransport._BaseMergeVersionAliases, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.MergeVersionAliases")

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
            request: model_service.MergeVersionAliasesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model.Model:
            r"""Call the merge version aliases method over HTTP.

            Args:
                request (~.model_service.MergeVersionAliasesRequest):
                    The request object. Request message for
                [ModelService.MergeVersionAliases][google.cloud.aiplatform.v1beta1.ModelService.MergeVersionAliases].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.model.Model:
                    A trained machine learning Model.
            """

            http_options = (
                _BaseModelServiceRestTransport._BaseMergeVersionAliases._get_http_options()
            )

            request, metadata = self._interceptor.pre_merge_version_aliases(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseMergeVersionAliases._get_transcoded_request(
                http_options, request
            )

            body = _BaseModelServiceRestTransport._BaseMergeVersionAliases._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseMergeVersionAliases._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.MergeVersionAliases",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "MergeVersionAliases",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._MergeVersionAliases._get_response(
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
            resp = model.Model()
            pb_resp = model.Model.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_merge_version_aliases(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_merge_version_aliases_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = model.Model.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.merge_version_aliases",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "MergeVersionAliases",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _RecommendSpec(
        _BaseModelServiceRestTransport._BaseRecommendSpec, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.RecommendSpec")

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
            request: model_service.RecommendSpecRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> model_service.RecommendSpecResponse:
            r"""Call the recommend spec method over HTTP.

            Args:
                request (~.model_service.RecommendSpecRequest):
                    The request object. Request message for
                [ModelService.RecommendSpec][google.cloud.aiplatform.v1beta1.ModelService.RecommendSpec].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.model_service.RecommendSpecResponse:
                    Response message for
                [ModelService.RecommendSpec][google.cloud.aiplatform.v1beta1.ModelService.RecommendSpec].

            """

            http_options = (
                _BaseModelServiceRestTransport._BaseRecommendSpec._get_http_options()
            )

            request, metadata = self._interceptor.pre_recommend_spec(request, metadata)
            transcoded_request = _BaseModelServiceRestTransport._BaseRecommendSpec._get_transcoded_request(
                http_options, request
            )

            body = _BaseModelServiceRestTransport._BaseRecommendSpec._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseRecommendSpec._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.RecommendSpec",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "RecommendSpec",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._RecommendSpec._get_response(
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
            resp = model_service.RecommendSpecResponse()
            pb_resp = model_service.RecommendSpecResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_recommend_spec(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_recommend_spec_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = model_service.RecommendSpecResponse.to_json(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.recommend_spec",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "RecommendSpec",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateExplanationDataset(
        _BaseModelServiceRestTransport._BaseUpdateExplanationDataset,
        ModelServiceRestStub,
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.UpdateExplanationDataset")

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
            request: model_service.UpdateExplanationDatasetRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update explanation
            dataset method over HTTP.

                Args:
                    request (~.model_service.UpdateExplanationDatasetRequest):
                        The request object. Request message for
                    [ModelService.UpdateExplanationDataset][google.cloud.aiplatform.v1beta1.ModelService.UpdateExplanationDataset].
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
                _BaseModelServiceRestTransport._BaseUpdateExplanationDataset._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_explanation_dataset(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseUpdateExplanationDataset._get_transcoded_request(
                http_options, request
            )

            body = _BaseModelServiceRestTransport._BaseUpdateExplanationDataset._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseUpdateExplanationDataset._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.UpdateExplanationDataset",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "UpdateExplanationDataset",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                ModelServiceRestTransport._UpdateExplanationDataset._get_response(
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

            resp = self._interceptor.post_update_explanation_dataset(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_explanation_dataset_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.update_explanation_dataset",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "UpdateExplanationDataset",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateModel(
        _BaseModelServiceRestTransport._BaseUpdateModel, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.UpdateModel")

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
            request: model_service.UpdateModelRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> gca_model.Model:
            r"""Call the update model method over HTTP.

            Args:
                request (~.model_service.UpdateModelRequest):
                    The request object. Request message for
                [ModelService.UpdateModel][google.cloud.aiplatform.v1beta1.ModelService.UpdateModel].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.gca_model.Model:
                    A trained machine learning Model.
            """

            http_options = (
                _BaseModelServiceRestTransport._BaseUpdateModel._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_model(request, metadata)
            transcoded_request = (
                _BaseModelServiceRestTransport._BaseUpdateModel._get_transcoded_request(
                    http_options, request
                )
            )

            body = (
                _BaseModelServiceRestTransport._BaseUpdateModel._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseUpdateModel._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.UpdateModel",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "UpdateModel",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._UpdateModel._get_response(
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
            resp = gca_model.Model()
            pb_resp = gca_model.Model.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_update_model(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_model_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = gca_model.Model.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.update_model",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "UpdateModel",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UploadModel(
        _BaseModelServiceRestTransport._BaseUploadModel, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.UploadModel")

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
            request: model_service.UploadModelRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the upload model method over HTTP.

            Args:
                request (~.model_service.UploadModelRequest):
                    The request object. Request message for
                [ModelService.UploadModel][google.cloud.aiplatform.v1beta1.ModelService.UploadModel].
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
                _BaseModelServiceRestTransport._BaseUploadModel._get_http_options()
            )

            request, metadata = self._interceptor.pre_upload_model(request, metadata)
            transcoded_request = (
                _BaseModelServiceRestTransport._BaseUploadModel._get_transcoded_request(
                    http_options, request
                )
            )

            body = (
                _BaseModelServiceRestTransport._BaseUploadModel._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseUploadModel._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.UploadModel",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "UploadModel",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._UploadModel._get_response(
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

            resp = self._interceptor.post_upload_model(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_upload_model_with_metadata(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceClient.upload_model",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "UploadModel",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def batch_import_evaluated_annotations(
        self,
    ) -> Callable[
        [model_service.BatchImportEvaluatedAnnotationsRequest],
        model_service.BatchImportEvaluatedAnnotationsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._BatchImportEvaluatedAnnotations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def batch_import_model_evaluation_slices(
        self,
    ) -> Callable[
        [model_service.BatchImportModelEvaluationSlicesRequest],
        model_service.BatchImportModelEvaluationSlicesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._BatchImportModelEvaluationSlices(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def copy_model(
        self,
    ) -> Callable[[model_service.CopyModelRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CopyModel(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_model(
        self,
    ) -> Callable[[model_service.DeleteModelRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteModel(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_model_version(
        self,
    ) -> Callable[[model_service.DeleteModelVersionRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteModelVersion(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def export_model(
        self,
    ) -> Callable[[model_service.ExportModelRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ExportModel(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_model(self) -> Callable[[model_service.GetModelRequest], model.Model]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetModel(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_model_evaluation(
        self,
    ) -> Callable[
        [model_service.GetModelEvaluationRequest], model_evaluation.ModelEvaluation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetModelEvaluation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_model_evaluation_slice(
        self,
    ) -> Callable[
        [model_service.GetModelEvaluationSliceRequest],
        model_evaluation_slice.ModelEvaluationSlice,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetModelEvaluationSlice(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def import_model_evaluation(
        self,
    ) -> Callable[
        [model_service.ImportModelEvaluationRequest],
        gca_model_evaluation.ModelEvaluation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ImportModelEvaluation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_model_evaluations(
        self,
    ) -> Callable[
        [model_service.ListModelEvaluationsRequest],
        model_service.ListModelEvaluationsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListModelEvaluations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_model_evaluation_slices(
        self,
    ) -> Callable[
        [model_service.ListModelEvaluationSlicesRequest],
        model_service.ListModelEvaluationSlicesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListModelEvaluationSlices(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_models(
        self,
    ) -> Callable[[model_service.ListModelsRequest], model_service.ListModelsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListModels(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_model_version_checkpoints(
        self,
    ) -> Callable[
        [model_service.ListModelVersionCheckpointsRequest],
        model_service.ListModelVersionCheckpointsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListModelVersionCheckpoints(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_model_versions(
        self,
    ) -> Callable[
        [model_service.ListModelVersionsRequest],
        model_service.ListModelVersionsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListModelVersions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def merge_version_aliases(
        self,
    ) -> Callable[[model_service.MergeVersionAliasesRequest], model.Model]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._MergeVersionAliases(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def recommend_spec(
        self,
    ) -> Callable[
        [model_service.RecommendSpecRequest], model_service.RecommendSpecResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._RecommendSpec(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_explanation_dataset(
        self,
    ) -> Callable[
        [model_service.UpdateExplanationDatasetRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateExplanationDataset(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_model(
        self,
    ) -> Callable[[model_service.UpdateModelRequest], gca_model.Model]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateModel(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def upload_model(
        self,
    ) -> Callable[[model_service.UploadModelRequest], operations_pb2.Operation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UploadModel(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_location(self):
        return self._GetLocation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetLocation(
        _BaseModelServiceRestTransport._BaseGetLocation, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.GetLocation")

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
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
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
                _BaseModelServiceRestTransport._BaseGetLocation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_location(request, metadata)
            transcoded_request = (
                _BaseModelServiceRestTransport._BaseGetLocation._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseGetLocation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.GetLocation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetLocation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._GetLocation._get_response(
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
            resp = locations_pb2.Location()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_get_location(resp)
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceAsyncClient.GetLocation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
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
        _BaseModelServiceRestTransport._BaseListLocations, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ListLocations")

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
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
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
                _BaseModelServiceRestTransport._BaseListLocations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_locations(request, metadata)
            transcoded_request = _BaseModelServiceRestTransport._BaseListLocations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseListLocations._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ListLocations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListLocations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._ListLocations._get_response(
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
            resp = locations_pb2.ListLocationsResponse()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_list_locations(resp)
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceAsyncClient.ListLocations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
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
        _BaseModelServiceRestTransport._BaseGetIamPolicy, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.GetIamPolicy")

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
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
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
                _BaseModelServiceRestTransport._BaseGetIamPolicy._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_iam_policy(request, metadata)
            transcoded_request = _BaseModelServiceRestTransport._BaseGetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = (
                _BaseModelServiceRestTransport._BaseGetIamPolicy._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseGetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._GetIamPolicy._get_response(
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

            content = response.content.decode("utf-8")
            resp = policy_pb2.Policy()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_get_iam_policy(resp)
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceAsyncClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
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
        _BaseModelServiceRestTransport._BaseSetIamPolicy, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.SetIamPolicy")

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
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
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
                _BaseModelServiceRestTransport._BaseSetIamPolicy._get_http_options()
            )

            request, metadata = self._interceptor.pre_set_iam_policy(request, metadata)
            transcoded_request = _BaseModelServiceRestTransport._BaseSetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = (
                _BaseModelServiceRestTransport._BaseSetIamPolicy._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseSetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "SetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._SetIamPolicy._get_response(
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

            content = response.content.decode("utf-8")
            resp = policy_pb2.Policy()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_set_iam_policy(resp)
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceAsyncClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
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
        _BaseModelServiceRestTransport._BaseTestIamPermissions, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.TestIamPermissions")

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
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
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
                _BaseModelServiceRestTransport._BaseTestIamPermissions._get_http_options()
            )

            request, metadata = self._interceptor.pre_test_iam_permissions(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseTestIamPermissions._get_transcoded_request(
                http_options, request
            )

            body = _BaseModelServiceRestTransport._BaseTestIamPermissions._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseTestIamPermissions._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "TestIamPermissions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._TestIamPermissions._get_response(
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

            content = response.content.decode("utf-8")
            resp = iam_policy_pb2.TestIamPermissionsResponse()
            resp = json_format.Parse(content, resp)
            resp = self._interceptor.post_test_iam_permissions(resp)
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceAsyncClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
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
        _BaseModelServiceRestTransport._BaseCancelOperation, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.CancelOperation")

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
                _BaseModelServiceRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseCancelOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.CancelOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._CancelOperation._get_response(
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

            return self._interceptor.post_cancel_operation(None)

    @property
    def delete_operation(self):
        return self._DeleteOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _DeleteOperation(
        _BaseModelServiceRestTransport._BaseDeleteOperation, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.DeleteOperation")

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
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseModelServiceRestTransport._BaseDeleteOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_operation(
                request, metadata
            )
            transcoded_request = _BaseModelServiceRestTransport._BaseDeleteOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseDeleteOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.DeleteOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "DeleteOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._DeleteOperation._get_response(
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

            return self._interceptor.post_delete_operation(None)

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _GetOperation(
        _BaseModelServiceRestTransport._BaseGetOperation, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.GetOperation")

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
                _BaseModelServiceRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BaseModelServiceRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseModelServiceRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._GetOperation._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
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
        _BaseModelServiceRestTransport._BaseListOperations, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.ListOperations")

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
                _BaseModelServiceRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseModelServiceRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._ListOperations._get_response(
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
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
        _BaseModelServiceRestTransport._BaseWaitOperation, ModelServiceRestStub
    ):
        def __hash__(self):
            return hash("ModelServiceRestTransport.WaitOperation")

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
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
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
                _BaseModelServiceRestTransport._BaseWaitOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_wait_operation(request, metadata)
            transcoded_request = _BaseModelServiceRestTransport._BaseWaitOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseModelServiceRestTransport._BaseWaitOperation._get_query_params_json(
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
                    f"Sending request for google.cloud.aiplatform_v1beta1.ModelServiceClient.WaitOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "WaitOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = ModelServiceRestTransport._WaitOperation._get_response(
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
            resp = self._interceptor.post_wait_operation(resp)
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
                    "Received response for google.cloud.aiplatform_v1beta1.ModelServiceAsyncClient.WaitOperation",
                    extra={
                        "serviceName": "google.cloud.aiplatform.v1beta1.ModelService",
                        "rpcName": "WaitOperation",
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


__all__ = ("ModelServiceRestTransport",)
