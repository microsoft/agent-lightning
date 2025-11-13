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

from requests import __version__ as requests_version
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings


from google.cloud.spanner_admin_instance_v1.types import spanner_instance_admin
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore


from .rest_base import _BaseInstanceAdminRestTransport
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


class InstanceAdminRestInterceptor:
    """Interceptor for InstanceAdmin.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the InstanceAdminRestTransport.

    .. code-block:: python
        class MyCustomInstanceAdminInterceptor(InstanceAdminRestInterceptor):
            def pre_create_instance(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_instance(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_instance_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_instance_config(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_instance_partition(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_instance_partition(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_instance(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_instance_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_instance_partition(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_iam_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_iam_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_instance(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_instance(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_instance_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_instance_config(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_instance_partition(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_instance_partition(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_instance_config_operations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_instance_config_operations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_instance_configs(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_instance_configs(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_instance_partition_operations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_instance_partition_operations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_instance_partitions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_instance_partitions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_instances(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_instances(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_move_instance(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_move_instance(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_set_iam_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_set_iam_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_test_iam_permissions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_test_iam_permissions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_instance(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_instance(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_instance_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_instance_config(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_instance_partition(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_instance_partition(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = InstanceAdminRestTransport(interceptor=MyCustomInstanceAdminInterceptor())
        client = InstanceAdminClient(transport=transport)


    """

    def pre_create_instance(
        self,
        request: spanner_instance_admin.CreateInstanceRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.CreateInstanceRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_instance

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_create_instance(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_instance

        DEPRECATED. Please use the `post_create_instance_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_create_instance` interceptor runs
        before the `post_create_instance_with_metadata` interceptor.
        """
        return response

    def post_create_instance_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_instance

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_create_instance_with_metadata`
        interceptor in new development instead of the `post_create_instance` interceptor.
        When both interceptors are used, this `post_create_instance_with_metadata` interceptor runs after the
        `post_create_instance` interceptor. The (possibly modified) response returned by
        `post_create_instance` will be passed to
        `post_create_instance_with_metadata`.
        """
        return response, metadata

    def pre_create_instance_config(
        self,
        request: spanner_instance_admin.CreateInstanceConfigRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.CreateInstanceConfigRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_instance_config

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_create_instance_config(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_instance_config

        DEPRECATED. Please use the `post_create_instance_config_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_create_instance_config` interceptor runs
        before the `post_create_instance_config_with_metadata` interceptor.
        """
        return response

    def post_create_instance_config_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_instance_config

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_create_instance_config_with_metadata`
        interceptor in new development instead of the `post_create_instance_config` interceptor.
        When both interceptors are used, this `post_create_instance_config_with_metadata` interceptor runs after the
        `post_create_instance_config` interceptor. The (possibly modified) response returned by
        `post_create_instance_config` will be passed to
        `post_create_instance_config_with_metadata`.
        """
        return response, metadata

    def pre_create_instance_partition(
        self,
        request: spanner_instance_admin.CreateInstancePartitionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.CreateInstancePartitionRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for create_instance_partition

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_create_instance_partition(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_instance_partition

        DEPRECATED. Please use the `post_create_instance_partition_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_create_instance_partition` interceptor runs
        before the `post_create_instance_partition_with_metadata` interceptor.
        """
        return response

    def post_create_instance_partition_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_instance_partition

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_create_instance_partition_with_metadata`
        interceptor in new development instead of the `post_create_instance_partition` interceptor.
        When both interceptors are used, this `post_create_instance_partition_with_metadata` interceptor runs after the
        `post_create_instance_partition` interceptor. The (possibly modified) response returned by
        `post_create_instance_partition` will be passed to
        `post_create_instance_partition_with_metadata`.
        """
        return response, metadata

    def pre_delete_instance(
        self,
        request: spanner_instance_admin.DeleteInstanceRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.DeleteInstanceRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_instance

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def pre_delete_instance_config(
        self,
        request: spanner_instance_admin.DeleteInstanceConfigRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.DeleteInstanceConfigRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_instance_config

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def pre_delete_instance_partition(
        self,
        request: spanner_instance_admin.DeleteInstancePartitionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.DeleteInstancePartitionRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for delete_instance_partition

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def pre_get_iam_policy(
        self,
        request: iam_policy_pb2.GetIamPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.GetIamPolicyRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_get_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for get_iam_policy

        DEPRECATED. Please use the `post_get_iam_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_get_iam_policy` interceptor runs
        before the `post_get_iam_policy_with_metadata` interceptor.
        """
        return response

    def post_get_iam_policy_with_metadata(
        self,
        response: policy_pb2.Policy,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[policy_pb2.Policy, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_iam_policy

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_get_iam_policy_with_metadata`
        interceptor in new development instead of the `post_get_iam_policy` interceptor.
        When both interceptors are used, this `post_get_iam_policy_with_metadata` interceptor runs after the
        `post_get_iam_policy` interceptor. The (possibly modified) response returned by
        `post_get_iam_policy` will be passed to
        `post_get_iam_policy_with_metadata`.
        """
        return response, metadata

    def pre_get_instance(
        self,
        request: spanner_instance_admin.GetInstanceRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.GetInstanceRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_instance

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_get_instance(
        self, response: spanner_instance_admin.Instance
    ) -> spanner_instance_admin.Instance:
        """Post-rpc interceptor for get_instance

        DEPRECATED. Please use the `post_get_instance_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_get_instance` interceptor runs
        before the `post_get_instance_with_metadata` interceptor.
        """
        return response

    def post_get_instance_with_metadata(
        self,
        response: spanner_instance_admin.Instance,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.Instance, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for get_instance

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_get_instance_with_metadata`
        interceptor in new development instead of the `post_get_instance` interceptor.
        When both interceptors are used, this `post_get_instance_with_metadata` interceptor runs after the
        `post_get_instance` interceptor. The (possibly modified) response returned by
        `post_get_instance` will be passed to
        `post_get_instance_with_metadata`.
        """
        return response, metadata

    def pre_get_instance_config(
        self,
        request: spanner_instance_admin.GetInstanceConfigRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.GetInstanceConfigRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_instance_config

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_get_instance_config(
        self, response: spanner_instance_admin.InstanceConfig
    ) -> spanner_instance_admin.InstanceConfig:
        """Post-rpc interceptor for get_instance_config

        DEPRECATED. Please use the `post_get_instance_config_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_get_instance_config` interceptor runs
        before the `post_get_instance_config_with_metadata` interceptor.
        """
        return response

    def post_get_instance_config_with_metadata(
        self,
        response: spanner_instance_admin.InstanceConfig,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.InstanceConfig, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for get_instance_config

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_get_instance_config_with_metadata`
        interceptor in new development instead of the `post_get_instance_config` interceptor.
        When both interceptors are used, this `post_get_instance_config_with_metadata` interceptor runs after the
        `post_get_instance_config` interceptor. The (possibly modified) response returned by
        `post_get_instance_config` will be passed to
        `post_get_instance_config_with_metadata`.
        """
        return response, metadata

    def pre_get_instance_partition(
        self,
        request: spanner_instance_admin.GetInstancePartitionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.GetInstancePartitionRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for get_instance_partition

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_get_instance_partition(
        self, response: spanner_instance_admin.InstancePartition
    ) -> spanner_instance_admin.InstancePartition:
        """Post-rpc interceptor for get_instance_partition

        DEPRECATED. Please use the `post_get_instance_partition_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_get_instance_partition` interceptor runs
        before the `post_get_instance_partition_with_metadata` interceptor.
        """
        return response

    def post_get_instance_partition_with_metadata(
        self,
        response: spanner_instance_admin.InstancePartition,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.InstancePartition,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for get_instance_partition

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_get_instance_partition_with_metadata`
        interceptor in new development instead of the `post_get_instance_partition` interceptor.
        When both interceptors are used, this `post_get_instance_partition_with_metadata` interceptor runs after the
        `post_get_instance_partition` interceptor. The (possibly modified) response returned by
        `post_get_instance_partition` will be passed to
        `post_get_instance_partition_with_metadata`.
        """
        return response, metadata

    def pre_list_instance_config_operations(
        self,
        request: spanner_instance_admin.ListInstanceConfigOperationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstanceConfigOperationsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_instance_config_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_list_instance_config_operations(
        self, response: spanner_instance_admin.ListInstanceConfigOperationsResponse
    ) -> spanner_instance_admin.ListInstanceConfigOperationsResponse:
        """Post-rpc interceptor for list_instance_config_operations

        DEPRECATED. Please use the `post_list_instance_config_operations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_list_instance_config_operations` interceptor runs
        before the `post_list_instance_config_operations_with_metadata` interceptor.
        """
        return response

    def post_list_instance_config_operations_with_metadata(
        self,
        response: spanner_instance_admin.ListInstanceConfigOperationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstanceConfigOperationsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_instance_config_operations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_list_instance_config_operations_with_metadata`
        interceptor in new development instead of the `post_list_instance_config_operations` interceptor.
        When both interceptors are used, this `post_list_instance_config_operations_with_metadata` interceptor runs after the
        `post_list_instance_config_operations` interceptor. The (possibly modified) response returned by
        `post_list_instance_config_operations` will be passed to
        `post_list_instance_config_operations_with_metadata`.
        """
        return response, metadata

    def pre_list_instance_configs(
        self,
        request: spanner_instance_admin.ListInstanceConfigsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstanceConfigsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_instance_configs

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_list_instance_configs(
        self, response: spanner_instance_admin.ListInstanceConfigsResponse
    ) -> spanner_instance_admin.ListInstanceConfigsResponse:
        """Post-rpc interceptor for list_instance_configs

        DEPRECATED. Please use the `post_list_instance_configs_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_list_instance_configs` interceptor runs
        before the `post_list_instance_configs_with_metadata` interceptor.
        """
        return response

    def post_list_instance_configs_with_metadata(
        self,
        response: spanner_instance_admin.ListInstanceConfigsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstanceConfigsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_instance_configs

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_list_instance_configs_with_metadata`
        interceptor in new development instead of the `post_list_instance_configs` interceptor.
        When both interceptors are used, this `post_list_instance_configs_with_metadata` interceptor runs after the
        `post_list_instance_configs` interceptor. The (possibly modified) response returned by
        `post_list_instance_configs` will be passed to
        `post_list_instance_configs_with_metadata`.
        """
        return response, metadata

    def pre_list_instance_partition_operations(
        self,
        request: spanner_instance_admin.ListInstancePartitionOperationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstancePartitionOperationsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_instance_partition_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_list_instance_partition_operations(
        self, response: spanner_instance_admin.ListInstancePartitionOperationsResponse
    ) -> spanner_instance_admin.ListInstancePartitionOperationsResponse:
        """Post-rpc interceptor for list_instance_partition_operations

        DEPRECATED. Please use the `post_list_instance_partition_operations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_list_instance_partition_operations` interceptor runs
        before the `post_list_instance_partition_operations_with_metadata` interceptor.
        """
        return response

    def post_list_instance_partition_operations_with_metadata(
        self,
        response: spanner_instance_admin.ListInstancePartitionOperationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstancePartitionOperationsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_instance_partition_operations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_list_instance_partition_operations_with_metadata`
        interceptor in new development instead of the `post_list_instance_partition_operations` interceptor.
        When both interceptors are used, this `post_list_instance_partition_operations_with_metadata` interceptor runs after the
        `post_list_instance_partition_operations` interceptor. The (possibly modified) response returned by
        `post_list_instance_partition_operations` will be passed to
        `post_list_instance_partition_operations_with_metadata`.
        """
        return response, metadata

    def pre_list_instance_partitions(
        self,
        request: spanner_instance_admin.ListInstancePartitionsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstancePartitionsRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_instance_partitions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_list_instance_partitions(
        self, response: spanner_instance_admin.ListInstancePartitionsResponse
    ) -> spanner_instance_admin.ListInstancePartitionsResponse:
        """Post-rpc interceptor for list_instance_partitions

        DEPRECATED. Please use the `post_list_instance_partitions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_list_instance_partitions` interceptor runs
        before the `post_list_instance_partitions_with_metadata` interceptor.
        """
        return response

    def post_list_instance_partitions_with_metadata(
        self,
        response: spanner_instance_admin.ListInstancePartitionsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstancePartitionsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_instance_partitions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_list_instance_partitions_with_metadata`
        interceptor in new development instead of the `post_list_instance_partitions` interceptor.
        When both interceptors are used, this `post_list_instance_partitions_with_metadata` interceptor runs after the
        `post_list_instance_partitions` interceptor. The (possibly modified) response returned by
        `post_list_instance_partitions` will be passed to
        `post_list_instance_partitions_with_metadata`.
        """
        return response, metadata

    def pre_list_instances(
        self,
        request: spanner_instance_admin.ListInstancesRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstancesRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for list_instances

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_list_instances(
        self, response: spanner_instance_admin.ListInstancesResponse
    ) -> spanner_instance_admin.ListInstancesResponse:
        """Post-rpc interceptor for list_instances

        DEPRECATED. Please use the `post_list_instances_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_list_instances` interceptor runs
        before the `post_list_instances_with_metadata` interceptor.
        """
        return response

    def post_list_instances_with_metadata(
        self,
        response: spanner_instance_admin.ListInstancesResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.ListInstancesResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for list_instances

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_list_instances_with_metadata`
        interceptor in new development instead of the `post_list_instances` interceptor.
        When both interceptors are used, this `post_list_instances_with_metadata` interceptor runs after the
        `post_list_instances` interceptor. The (possibly modified) response returned by
        `post_list_instances` will be passed to
        `post_list_instances_with_metadata`.
        """
        return response, metadata

    def pre_move_instance(
        self,
        request: spanner_instance_admin.MoveInstanceRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.MoveInstanceRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for move_instance

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_move_instance(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for move_instance

        DEPRECATED. Please use the `post_move_instance_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_move_instance` interceptor runs
        before the `post_move_instance_with_metadata` interceptor.
        """
        return response

    def post_move_instance_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for move_instance

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_move_instance_with_metadata`
        interceptor in new development instead of the `post_move_instance` interceptor.
        When both interceptors are used, this `post_move_instance_with_metadata` interceptor runs after the
        `post_move_instance` interceptor. The (possibly modified) response returned by
        `post_move_instance` will be passed to
        `post_move_instance_with_metadata`.
        """
        return response, metadata

    def pre_set_iam_policy(
        self,
        request: iam_policy_pb2.SetIamPolicyRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.SetIamPolicyRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_set_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for set_iam_policy

        DEPRECATED. Please use the `post_set_iam_policy_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_set_iam_policy` interceptor runs
        before the `post_set_iam_policy_with_metadata` interceptor.
        """
        return response

    def post_set_iam_policy_with_metadata(
        self,
        response: policy_pb2.Policy,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[policy_pb2.Policy, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for set_iam_policy

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_set_iam_policy_with_metadata`
        interceptor in new development instead of the `post_set_iam_policy` interceptor.
        When both interceptors are used, this `post_set_iam_policy_with_metadata` interceptor runs after the
        `post_set_iam_policy` interceptor. The (possibly modified) response returned by
        `post_set_iam_policy` will be passed to
        `post_set_iam_policy_with_metadata`.
        """
        return response, metadata

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
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_test_iam_permissions(
        self, response: iam_policy_pb2.TestIamPermissionsResponse
    ) -> iam_policy_pb2.TestIamPermissionsResponse:
        """Post-rpc interceptor for test_iam_permissions

        DEPRECATED. Please use the `post_test_iam_permissions_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_test_iam_permissions` interceptor runs
        before the `post_test_iam_permissions_with_metadata` interceptor.
        """
        return response

    def post_test_iam_permissions_with_metadata(
        self,
        response: iam_policy_pb2.TestIamPermissionsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        iam_policy_pb2.TestIamPermissionsResponse,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Post-rpc interceptor for test_iam_permissions

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_test_iam_permissions_with_metadata`
        interceptor in new development instead of the `post_test_iam_permissions` interceptor.
        When both interceptors are used, this `post_test_iam_permissions_with_metadata` interceptor runs after the
        `post_test_iam_permissions` interceptor. The (possibly modified) response returned by
        `post_test_iam_permissions` will be passed to
        `post_test_iam_permissions_with_metadata`.
        """
        return response, metadata

    def pre_update_instance(
        self,
        request: spanner_instance_admin.UpdateInstanceRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.UpdateInstanceRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_instance

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_update_instance(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_instance

        DEPRECATED. Please use the `post_update_instance_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_update_instance` interceptor runs
        before the `post_update_instance_with_metadata` interceptor.
        """
        return response

    def post_update_instance_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_instance

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_update_instance_with_metadata`
        interceptor in new development instead of the `post_update_instance` interceptor.
        When both interceptors are used, this `post_update_instance_with_metadata` interceptor runs after the
        `post_update_instance` interceptor. The (possibly modified) response returned by
        `post_update_instance` will be passed to
        `post_update_instance_with_metadata`.
        """
        return response, metadata

    def pre_update_instance_config(
        self,
        request: spanner_instance_admin.UpdateInstanceConfigRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.UpdateInstanceConfigRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_instance_config

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_update_instance_config(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_instance_config

        DEPRECATED. Please use the `post_update_instance_config_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_update_instance_config` interceptor runs
        before the `post_update_instance_config_with_metadata` interceptor.
        """
        return response

    def post_update_instance_config_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_instance_config

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_update_instance_config_with_metadata`
        interceptor in new development instead of the `post_update_instance_config` interceptor.
        When both interceptors are used, this `post_update_instance_config_with_metadata` interceptor runs after the
        `post_update_instance_config` interceptor. The (possibly modified) response returned by
        `post_update_instance_config` will be passed to
        `post_update_instance_config_with_metadata`.
        """
        return response, metadata

    def pre_update_instance_partition(
        self,
        request: spanner_instance_admin.UpdateInstancePartitionRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        spanner_instance_admin.UpdateInstancePartitionRequest,
        Sequence[Tuple[str, Union[str, bytes]]],
    ]:
        """Pre-rpc interceptor for update_instance_partition

        Override in a subclass to manipulate the request or metadata
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_update_instance_partition(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_instance_partition

        DEPRECATED. Please use the `post_update_instance_partition_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code. This `post_update_instance_partition` interceptor runs
        before the `post_update_instance_partition_with_metadata` interceptor.
        """
        return response

    def post_update_instance_partition_with_metadata(
        self,
        response: operations_pb2.Operation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[operations_pb2.Operation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_instance_partition

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the InstanceAdmin server but before it is returned to user code.

        We recommend only using this `post_update_instance_partition_with_metadata`
        interceptor in new development instead of the `post_update_instance_partition` interceptor.
        When both interceptors are used, this `post_update_instance_partition_with_metadata` interceptor runs after the
        `post_update_instance_partition` interceptor. The (possibly modified) response returned by
        `post_update_instance_partition` will be passed to
        `post_update_instance_partition_with_metadata`.
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
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_cancel_operation(self, response: None) -> None:
        """Post-rpc interceptor for cancel_operation

        Override in a subclass to manipulate the response
        after it is returned by the InstanceAdmin server but before
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
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_delete_operation(self, response: None) -> None:
        """Post-rpc interceptor for delete_operation

        Override in a subclass to manipulate the response
        after it is returned by the InstanceAdmin server but before
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
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_get_operation(
        self, response: operations_pb2.Operation
    ) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the InstanceAdmin server but before
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
        before they are sent to the InstanceAdmin server.
        """
        return request, metadata

    def post_list_operations(
        self, response: operations_pb2.ListOperationsResponse
    ) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the InstanceAdmin server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class InstanceAdminRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: InstanceAdminRestInterceptor


class InstanceAdminRestTransport(_BaseInstanceAdminRestTransport):
    """REST backend synchronous transport for InstanceAdmin.

    Cloud Spanner Instance Admin API

    The Cloud Spanner Instance Admin API can be used to create,
    delete, modify and list instances. Instances are dedicated Cloud
    Spanner serving and storage resources to be used by Cloud
    Spanner databases.

    Each instance has a "configuration", which dictates where the
    serving resources for the Cloud Spanner instance are located
    (e.g., US-central, Europe). Configurations are created by Google
    based on resource availability.

    Cloud Spanner billing is based on the instances that exist and
    their sizes. After an instance exists, there are no additional
    per-database or per-operation charges for use of the instance
    (though there may be additional network bandwidth charges).
    Instances offer isolation: problems with databases in one
    instance will not affect other instances. However, within an
    instance databases can affect each other. For example, if one
    database in an instance receives a lot of requests and consumes
    most of the instance resources, fewer resources are available
    for other databases in that instance, and their performance may
    suffer.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = "spanner.googleapis.com",
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = "https",
        interceptor: Optional[InstanceAdminRestInterceptor] = None,
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'spanner.googleapis.com').
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
        self._interceptor = interceptor or InstanceAdminRestInterceptor()
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
                        "uri": "/v1/{name=projects/*/instances/*/databases/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1/{name=projects/*/instances/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1/{name=projects/*/instances/*/backups/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1/{name=projects/*/instances/*/instancePartitions/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/operations/*}:cancel",
                    },
                    {
                        "method": "post",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/ssdCaches/*/operations/*}:cancel",
                    },
                ],
                "google.longrunning.Operations.DeleteOperation": [
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instances/*/databases/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instances/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instances/*/backups/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instances/*/instancePartitions/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/operations/*}",
                    },
                    {
                        "method": "delete",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/ssdCaches/*/operations/*}",
                    },
                ],
                "google.longrunning.Operations.GetOperation": [
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/databases/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/backups/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/instancePartitions/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/operations/*}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/ssdCaches/*/operations/*}",
                    },
                ],
                "google.longrunning.Operations.ListOperations": [
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/databases/*/operations}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/operations}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/backups/*/operations}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instances/*/instancePartitions/*/operations}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/operations}",
                    },
                    {
                        "method": "get",
                        "uri": "/v1/{name=projects/*/instanceConfigs/*/ssdCaches/*/operations}",
                    },
                ],
            }

            rest_transport = operations_v1.OperationsRestTransport(
                host=self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                scopes=self._scopes,
                http_options=http_options,
                path_prefix="v1",
            )

            self._operations_client = operations_v1.AbstractOperationsClient(
                transport=rest_transport
            )

        # Return the client from cache.
        return self._operations_client

    class _CreateInstance(
        _BaseInstanceAdminRestTransport._BaseCreateInstance, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.CreateInstance")

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
            request: spanner_instance_admin.CreateInstanceRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create instance method over HTTP.

            Args:
                request (~.spanner_instance_admin.CreateInstanceRequest):
                    The request object. The request for
                [CreateInstance][google.spanner.admin.instance.v1.InstanceAdmin.CreateInstance].
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
                _BaseInstanceAdminRestTransport._BaseCreateInstance._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_instance(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseCreateInstance._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseCreateInstance._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseCreateInstance._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.CreateInstance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "CreateInstance",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._CreateInstance._get_response(
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

            resp = self._interceptor.post_create_instance(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_instance_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.create_instance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "CreateInstance",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateInstanceConfig(
        _BaseInstanceAdminRestTransport._BaseCreateInstanceConfig, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.CreateInstanceConfig")

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
            request: spanner_instance_admin.CreateInstanceConfigRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create instance config method over HTTP.

            Args:
                request (~.spanner_instance_admin.CreateInstanceConfigRequest):
                    The request object. The request for
                [CreateInstanceConfig][google.spanner.admin.instance.v1.InstanceAdmin.CreateInstanceConfig].
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
                _BaseInstanceAdminRestTransport._BaseCreateInstanceConfig._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_instance_config(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseCreateInstanceConfig._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseCreateInstanceConfig._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseCreateInstanceConfig._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.CreateInstanceConfig",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "CreateInstanceConfig",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._CreateInstanceConfig._get_response(
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

            resp = self._interceptor.post_create_instance_config(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_instance_config_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.create_instance_config",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "CreateInstanceConfig",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateInstancePartition(
        _BaseInstanceAdminRestTransport._BaseCreateInstancePartition,
        InstanceAdminRestStub,
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.CreateInstancePartition")

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
            request: spanner_instance_admin.CreateInstancePartitionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the create instance partition method over HTTP.

            Args:
                request (~.spanner_instance_admin.CreateInstancePartitionRequest):
                    The request object. The request for
                [CreateInstancePartition][google.spanner.admin.instance.v1.InstanceAdmin.CreateInstancePartition].
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
                _BaseInstanceAdminRestTransport._BaseCreateInstancePartition._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_instance_partition(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseCreateInstancePartition._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseCreateInstancePartition._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseCreateInstancePartition._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.CreateInstancePartition",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "CreateInstancePartition",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                InstanceAdminRestTransport._CreateInstancePartition._get_response(
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

            resp = self._interceptor.post_create_instance_partition(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_instance_partition_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.create_instance_partition",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "CreateInstancePartition",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteInstance(
        _BaseInstanceAdminRestTransport._BaseDeleteInstance, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.DeleteInstance")

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
            request: spanner_instance_admin.DeleteInstanceRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete instance method over HTTP.

            Args:
                request (~.spanner_instance_admin.DeleteInstanceRequest):
                    The request object. The request for
                [DeleteInstance][google.spanner.admin.instance.v1.InstanceAdmin.DeleteInstance].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseDeleteInstance._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_instance(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseDeleteInstance._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseDeleteInstance._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.DeleteInstance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "DeleteInstance",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._DeleteInstance._get_response(
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

    class _DeleteInstanceConfig(
        _BaseInstanceAdminRestTransport._BaseDeleteInstanceConfig, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.DeleteInstanceConfig")

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
            request: spanner_instance_admin.DeleteInstanceConfigRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete instance config method over HTTP.

            Args:
                request (~.spanner_instance_admin.DeleteInstanceConfigRequest):
                    The request object. The request for
                [DeleteInstanceConfig][google.spanner.admin.instance.v1.InstanceAdmin.DeleteInstanceConfig].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseDeleteInstanceConfig._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_instance_config(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseDeleteInstanceConfig._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseDeleteInstanceConfig._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.DeleteInstanceConfig",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "DeleteInstanceConfig",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._DeleteInstanceConfig._get_response(
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

    class _DeleteInstancePartition(
        _BaseInstanceAdminRestTransport._BaseDeleteInstancePartition,
        InstanceAdminRestStub,
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.DeleteInstancePartition")

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
            request: spanner_instance_admin.DeleteInstancePartitionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete instance partition method over HTTP.

            Args:
                request (~.spanner_instance_admin.DeleteInstancePartitionRequest):
                    The request object. The request for
                [DeleteInstancePartition][google.spanner.admin.instance.v1.InstanceAdmin.DeleteInstancePartition].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseDeleteInstancePartition._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_instance_partition(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseDeleteInstancePartition._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseDeleteInstancePartition._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.DeleteInstancePartition",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "DeleteInstancePartition",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                InstanceAdminRestTransport._DeleteInstancePartition._get_response(
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

    class _GetIamPolicy(
        _BaseInstanceAdminRestTransport._BaseGetIamPolicy, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.GetIamPolicy")

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
                request (~.iam_policy_pb2.GetIamPolicyRequest):
                    The request object. Request message for ``GetIamPolicy`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.policy_pb2.Policy:
                    An Identity and Access Management (IAM) policy, which
                specifies access controls for Google Cloud resources.

                A ``Policy`` is a collection of ``bindings``. A
                ``binding`` binds one or more ``members``, or
                principals, to a single ``role``. Principals can be user
                accounts, service accounts, Google groups, and domains
                (such as G Suite). A ``role`` is a named list of
                permissions; each ``role`` can be an IAM predefined role
                or a user-created custom role.

                For some types of Google Cloud resources, a ``binding``
                can also specify a ``condition``, which is a logical
                expression that allows access to a resource only if the
                expression evaluates to ``true``. A condition can add
                constraints based on attributes of the request, the
                resource, or both. To learn which resources support
                conditions in their IAM policies, see the `IAM
                documentation <https://cloud.google.com/iam/help/conditions/resource-policies>`__.

                **JSON example:**

                ::

                       {
                         "bindings": [
                           {
                             "role": "roles/resourcemanager.organizationAdmin",
                             "members": [
                               "user:mike@example.com",
                               "group:admins@example.com",
                               "domain:google.com",
                               "serviceAccount:my-project-id@appspot.gserviceaccount.com"
                             ]
                           },
                           {
                             "role": "roles/resourcemanager.organizationViewer",
                             "members": [
                               "user:eve@example.com"
                             ],
                             "condition": {
                               "title": "expirable access",
                               "description": "Does not grant access after Sep 2020",
                               "expression": "request.time <
                               timestamp('2020-10-01T00:00:00.000Z')",
                             }
                           }
                         ],
                         "etag": "BwWWja0YfJA=",
                         "version": 3
                       }

                **YAML example:**

                ::

                       bindings:
                       - members:
                         - user:mike@example.com
                         - group:admins@example.com
                         - domain:google.com
                         - serviceAccount:my-project-id@appspot.gserviceaccount.com
                         role: roles/resourcemanager.organizationAdmin
                       - members:
                         - user:eve@example.com
                         role: roles/resourcemanager.organizationViewer
                         condition:
                           title: expirable access
                           description: Does not grant access after Sep 2020
                           expression: request.time < timestamp('2020-10-01T00:00:00.000Z')
                       etag: BwWWja0YfJA=
                       version: 3

                For a description of IAM and its features, see the `IAM
                documentation <https://cloud.google.com/iam/docs/>`__.

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseGetIamPolicy._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_iam_policy(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseGetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseGetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseGetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.GetIamPolicy",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._GetIamPolicy._get_response(
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
            resp = policy_pb2.Policy()
            pb_resp = resp

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_iam_policy(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_iam_policy_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.get_iam_policy",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetIamPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetInstance(
        _BaseInstanceAdminRestTransport._BaseGetInstance, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.GetInstance")

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
            request: spanner_instance_admin.GetInstanceRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_instance_admin.Instance:
            r"""Call the get instance method over HTTP.

            Args:
                request (~.spanner_instance_admin.GetInstanceRequest):
                    The request object. The request for
                [GetInstance][google.spanner.admin.instance.v1.InstanceAdmin.GetInstance].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_instance_admin.Instance:
                    An isolated set of Cloud Spanner
                resources on which databases can be
                hosted.

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseGetInstance._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_instance(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseGetInstance._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseInstanceAdminRestTransport._BaseGetInstance._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.GetInstance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetInstance",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._GetInstance._get_response(
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
            resp = spanner_instance_admin.Instance()
            pb_resp = spanner_instance_admin.Instance.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_instance(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_instance_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = spanner_instance_admin.Instance.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.get_instance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetInstance",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetInstanceConfig(
        _BaseInstanceAdminRestTransport._BaseGetInstanceConfig, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.GetInstanceConfig")

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
            request: spanner_instance_admin.GetInstanceConfigRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_instance_admin.InstanceConfig:
            r"""Call the get instance config method over HTTP.

            Args:
                request (~.spanner_instance_admin.GetInstanceConfigRequest):
                    The request object. The request for
                [GetInstanceConfigRequest][google.spanner.admin.instance.v1.InstanceAdmin.GetInstanceConfig].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_instance_admin.InstanceConfig:
                    A possible configuration for a Cloud
                Spanner instance. Configurations define
                the geographic placement of nodes and
                their replication.

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseGetInstanceConfig._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_instance_config(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseGetInstanceConfig._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseGetInstanceConfig._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.GetInstanceConfig",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetInstanceConfig",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._GetInstanceConfig._get_response(
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
            resp = spanner_instance_admin.InstanceConfig()
            pb_resp = spanner_instance_admin.InstanceConfig.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_instance_config(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_instance_config_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = spanner_instance_admin.InstanceConfig.to_json(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.get_instance_config",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetInstanceConfig",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetInstancePartition(
        _BaseInstanceAdminRestTransport._BaseGetInstancePartition, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.GetInstancePartition")

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
            request: spanner_instance_admin.GetInstancePartitionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_instance_admin.InstancePartition:
            r"""Call the get instance partition method over HTTP.

            Args:
                request (~.spanner_instance_admin.GetInstancePartitionRequest):
                    The request object. The request for
                [GetInstancePartition][google.spanner.admin.instance.v1.InstanceAdmin.GetInstancePartition].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_instance_admin.InstancePartition:
                    An isolated set of Cloud Spanner
                resources that databases can define
                placements on.

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseGetInstancePartition._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_instance_partition(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseGetInstancePartition._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseGetInstancePartition._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.GetInstancePartition",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetInstancePartition",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._GetInstancePartition._get_response(
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
            resp = spanner_instance_admin.InstancePartition()
            pb_resp = spanner_instance_admin.InstancePartition.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_instance_partition(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_instance_partition_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = spanner_instance_admin.InstancePartition.to_json(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.get_instance_partition",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetInstancePartition",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListInstanceConfigOperations(
        _BaseInstanceAdminRestTransport._BaseListInstanceConfigOperations,
        InstanceAdminRestStub,
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.ListInstanceConfigOperations")

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
            request: spanner_instance_admin.ListInstanceConfigOperationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_instance_admin.ListInstanceConfigOperationsResponse:
            r"""Call the list instance config
            operations method over HTTP.

                Args:
                    request (~.spanner_instance_admin.ListInstanceConfigOperationsRequest):
                        The request object. The request for
                    [ListInstanceConfigOperations][google.spanner.admin.instance.v1.InstanceAdmin.ListInstanceConfigOperations].
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.spanner_instance_admin.ListInstanceConfigOperationsResponse:
                        The response for
                    [ListInstanceConfigOperations][google.spanner.admin.instance.v1.InstanceAdmin.ListInstanceConfigOperations].

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseListInstanceConfigOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_instance_config_operations(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseListInstanceConfigOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseListInstanceConfigOperations._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.ListInstanceConfigOperations",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstanceConfigOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                InstanceAdminRestTransport._ListInstanceConfigOperations._get_response(
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
            resp = spanner_instance_admin.ListInstanceConfigOperationsResponse()
            pb_resp = spanner_instance_admin.ListInstanceConfigOperationsResponse.pb(
                resp
            )

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_instance_config_operations(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_list_instance_config_operations_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = spanner_instance_admin.ListInstanceConfigOperationsResponse.to_json(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.list_instance_config_operations",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstanceConfigOperations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListInstanceConfigs(
        _BaseInstanceAdminRestTransport._BaseListInstanceConfigs, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.ListInstanceConfigs")

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
            request: spanner_instance_admin.ListInstanceConfigsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_instance_admin.ListInstanceConfigsResponse:
            r"""Call the list instance configs method over HTTP.

            Args:
                request (~.spanner_instance_admin.ListInstanceConfigsRequest):
                    The request object. The request for
                [ListInstanceConfigs][google.spanner.admin.instance.v1.InstanceAdmin.ListInstanceConfigs].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_instance_admin.ListInstanceConfigsResponse:
                    The response for
                [ListInstanceConfigs][google.spanner.admin.instance.v1.InstanceAdmin.ListInstanceConfigs].

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseListInstanceConfigs._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_instance_configs(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseListInstanceConfigs._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseListInstanceConfigs._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.ListInstanceConfigs",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstanceConfigs",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._ListInstanceConfigs._get_response(
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
            resp = spanner_instance_admin.ListInstanceConfigsResponse()
            pb_resp = spanner_instance_admin.ListInstanceConfigsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_instance_configs(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_instance_configs_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        spanner_instance_admin.ListInstanceConfigsResponse.to_json(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.list_instance_configs",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstanceConfigs",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListInstancePartitionOperations(
        _BaseInstanceAdminRestTransport._BaseListInstancePartitionOperations,
        InstanceAdminRestStub,
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.ListInstancePartitionOperations")

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
            request: spanner_instance_admin.ListInstancePartitionOperationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_instance_admin.ListInstancePartitionOperationsResponse:
            r"""Call the list instance partition
            operations method over HTTP.

                Args:
                    request (~.spanner_instance_admin.ListInstancePartitionOperationsRequest):
                        The request object. The request for
                    [ListInstancePartitionOperations][google.spanner.admin.instance.v1.InstanceAdmin.ListInstancePartitionOperations].
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.spanner_instance_admin.ListInstancePartitionOperationsResponse:
                        The response for
                    [ListInstancePartitionOperations][google.spanner.admin.instance.v1.InstanceAdmin.ListInstancePartitionOperations].

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseListInstancePartitionOperations._get_http_options()
            )

            (
                request,
                metadata,
            ) = self._interceptor.pre_list_instance_partition_operations(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseListInstancePartitionOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseListInstancePartitionOperations._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.ListInstancePartitionOperations",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstancePartitionOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._ListInstancePartitionOperations._get_response(
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
            resp = spanner_instance_admin.ListInstancePartitionOperationsResponse()
            pb_resp = spanner_instance_admin.ListInstancePartitionOperationsResponse.pb(
                resp
            )

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_instance_partition_operations(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            (
                resp,
                _,
            ) = self._interceptor.post_list_instance_partition_operations_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = spanner_instance_admin.ListInstancePartitionOperationsResponse.to_json(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.list_instance_partition_operations",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstancePartitionOperations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListInstancePartitions(
        _BaseInstanceAdminRestTransport._BaseListInstancePartitions,
        InstanceAdminRestStub,
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.ListInstancePartitions")

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
            request: spanner_instance_admin.ListInstancePartitionsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_instance_admin.ListInstancePartitionsResponse:
            r"""Call the list instance partitions method over HTTP.

            Args:
                request (~.spanner_instance_admin.ListInstancePartitionsRequest):
                    The request object. The request for
                [ListInstancePartitions][google.spanner.admin.instance.v1.InstanceAdmin.ListInstancePartitions].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_instance_admin.ListInstancePartitionsResponse:
                    The response for
                [ListInstancePartitions][google.spanner.admin.instance.v1.InstanceAdmin.ListInstancePartitions].

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseListInstancePartitions._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_instance_partitions(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseListInstancePartitions._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseListInstancePartitions._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.ListInstancePartitions",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstancePartitions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._ListInstancePartitions._get_response(
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
            resp = spanner_instance_admin.ListInstancePartitionsResponse()
            pb_resp = spanner_instance_admin.ListInstancePartitionsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_instance_partitions(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_instance_partitions_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        spanner_instance_admin.ListInstancePartitionsResponse.to_json(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.list_instance_partitions",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstancePartitions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListInstances(
        _BaseInstanceAdminRestTransport._BaseListInstances, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.ListInstances")

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
            request: spanner_instance_admin.ListInstancesRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> spanner_instance_admin.ListInstancesResponse:
            r"""Call the list instances method over HTTP.

            Args:
                request (~.spanner_instance_admin.ListInstancesRequest):
                    The request object. The request for
                [ListInstances][google.spanner.admin.instance.v1.InstanceAdmin.ListInstances].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.spanner_instance_admin.ListInstancesResponse:
                    The response for
                [ListInstances][google.spanner.admin.instance.v1.InstanceAdmin.ListInstances].

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseListInstances._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_instances(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseListInstances._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseListInstances._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.ListInstances",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstances",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._ListInstances._get_response(
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
            resp = spanner_instance_admin.ListInstancesResponse()
            pb_resp = spanner_instance_admin.ListInstancesResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_instances(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_instances_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                logging.DEBUG
            ):  # pragma: NO COVER
                try:
                    response_payload = (
                        spanner_instance_admin.ListInstancesResponse.to_json(response)
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.list_instances",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListInstances",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _MoveInstance(
        _BaseInstanceAdminRestTransport._BaseMoveInstance, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.MoveInstance")

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
            request: spanner_instance_admin.MoveInstanceRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the move instance method over HTTP.

            Args:
                request (~.spanner_instance_admin.MoveInstanceRequest):
                    The request object. The request for
                [MoveInstance][google.spanner.admin.instance.v1.InstanceAdmin.MoveInstance].
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
                _BaseInstanceAdminRestTransport._BaseMoveInstance._get_http_options()
            )

            request, metadata = self._interceptor.pre_move_instance(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseMoveInstance._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseMoveInstance._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseMoveInstance._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.MoveInstance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "MoveInstance",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._MoveInstance._get_response(
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

            resp = self._interceptor.post_move_instance(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_move_instance_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.move_instance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "MoveInstance",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _SetIamPolicy(
        _BaseInstanceAdminRestTransport._BaseSetIamPolicy, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.SetIamPolicy")

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
                request (~.iam_policy_pb2.SetIamPolicyRequest):
                    The request object. Request message for ``SetIamPolicy`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.policy_pb2.Policy:
                    An Identity and Access Management (IAM) policy, which
                specifies access controls for Google Cloud resources.

                A ``Policy`` is a collection of ``bindings``. A
                ``binding`` binds one or more ``members``, or
                principals, to a single ``role``. Principals can be user
                accounts, service accounts, Google groups, and domains
                (such as G Suite). A ``role`` is a named list of
                permissions; each ``role`` can be an IAM predefined role
                or a user-created custom role.

                For some types of Google Cloud resources, a ``binding``
                can also specify a ``condition``, which is a logical
                expression that allows access to a resource only if the
                expression evaluates to ``true``. A condition can add
                constraints based on attributes of the request, the
                resource, or both. To learn which resources support
                conditions in their IAM policies, see the `IAM
                documentation <https://cloud.google.com/iam/help/conditions/resource-policies>`__.

                **JSON example:**

                ::

                       {
                         "bindings": [
                           {
                             "role": "roles/resourcemanager.organizationAdmin",
                             "members": [
                               "user:mike@example.com",
                               "group:admins@example.com",
                               "domain:google.com",
                               "serviceAccount:my-project-id@appspot.gserviceaccount.com"
                             ]
                           },
                           {
                             "role": "roles/resourcemanager.organizationViewer",
                             "members": [
                               "user:eve@example.com"
                             ],
                             "condition": {
                               "title": "expirable access",
                               "description": "Does not grant access after Sep 2020",
                               "expression": "request.time <
                               timestamp('2020-10-01T00:00:00.000Z')",
                             }
                           }
                         ],
                         "etag": "BwWWja0YfJA=",
                         "version": 3
                       }

                **YAML example:**

                ::

                       bindings:
                       - members:
                         - user:mike@example.com
                         - group:admins@example.com
                         - domain:google.com
                         - serviceAccount:my-project-id@appspot.gserviceaccount.com
                         role: roles/resourcemanager.organizationAdmin
                       - members:
                         - user:eve@example.com
                         role: roles/resourcemanager.organizationViewer
                         condition:
                           title: expirable access
                           description: Does not grant access after Sep 2020
                           expression: request.time < timestamp('2020-10-01T00:00:00.000Z')
                       etag: BwWWja0YfJA=
                       version: 3

                For a description of IAM and its features, see the `IAM
                documentation <https://cloud.google.com/iam/docs/>`__.

            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseSetIamPolicy._get_http_options()
            )

            request, metadata = self._interceptor.pre_set_iam_policy(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseSetIamPolicy._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseSetIamPolicy._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseSetIamPolicy._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.SetIamPolicy",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "SetIamPolicy",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._SetIamPolicy._get_response(
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
            resp = policy_pb2.Policy()
            pb_resp = resp

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_set_iam_policy(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_set_iam_policy_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.set_iam_policy",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "SetIamPolicy",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _TestIamPermissions(
        _BaseInstanceAdminRestTransport._BaseTestIamPermissions, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.TestIamPermissions")

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
                request (~.iam_policy_pb2.TestIamPermissionsRequest):
                    The request object. Request message for ``TestIamPermissions`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.iam_policy_pb2.TestIamPermissionsResponse:
                    Response message for ``TestIamPermissions`` method.
            """

            http_options = (
                _BaseInstanceAdminRestTransport._BaseTestIamPermissions._get_http_options()
            )

            request, metadata = self._interceptor.pre_test_iam_permissions(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseTestIamPermissions._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseTestIamPermissions._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseTestIamPermissions._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.TestIamPermissions",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "TestIamPermissions",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._TestIamPermissions._get_response(
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
            resp = iam_policy_pb2.TestIamPermissionsResponse()
            pb_resp = resp

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_test_iam_permissions(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_test_iam_permissions_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.test_iam_permissions",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "TestIamPermissions",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateInstance(
        _BaseInstanceAdminRestTransport._BaseUpdateInstance, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.UpdateInstance")

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
            request: spanner_instance_admin.UpdateInstanceRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update instance method over HTTP.

            Args:
                request (~.spanner_instance_admin.UpdateInstanceRequest):
                    The request object. The request for
                [UpdateInstance][google.spanner.admin.instance.v1.InstanceAdmin.UpdateInstance].
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
                _BaseInstanceAdminRestTransport._BaseUpdateInstance._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_instance(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseUpdateInstance._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseUpdateInstance._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseUpdateInstance._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.UpdateInstance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "UpdateInstance",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._UpdateInstance._get_response(
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

            resp = self._interceptor.post_update_instance(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_instance_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.update_instance",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "UpdateInstance",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateInstanceConfig(
        _BaseInstanceAdminRestTransport._BaseUpdateInstanceConfig, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.UpdateInstanceConfig")

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
            request: spanner_instance_admin.UpdateInstanceConfigRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update instance config method over HTTP.

            Args:
                request (~.spanner_instance_admin.UpdateInstanceConfigRequest):
                    The request object. The request for
                [UpdateInstanceConfig][google.spanner.admin.instance.v1.InstanceAdmin.UpdateInstanceConfig].
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
                _BaseInstanceAdminRestTransport._BaseUpdateInstanceConfig._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_instance_config(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseUpdateInstanceConfig._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseUpdateInstanceConfig._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseUpdateInstanceConfig._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.UpdateInstanceConfig",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "UpdateInstanceConfig",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._UpdateInstanceConfig._get_response(
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

            resp = self._interceptor.post_update_instance_config(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_instance_config_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.update_instance_config",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "UpdateInstanceConfig",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateInstancePartition(
        _BaseInstanceAdminRestTransport._BaseUpdateInstancePartition,
        InstanceAdminRestStub,
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.UpdateInstancePartition")

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
            request: spanner_instance_admin.UpdateInstancePartitionRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> operations_pb2.Operation:
            r"""Call the update instance partition method over HTTP.

            Args:
                request (~.spanner_instance_admin.UpdateInstancePartitionRequest):
                    The request object. The request for
                [UpdateInstancePartition][google.spanner.admin.instance.v1.InstanceAdmin.UpdateInstancePartition].
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
                _BaseInstanceAdminRestTransport._BaseUpdateInstancePartition._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_instance_partition(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseUpdateInstancePartition._get_transcoded_request(
                http_options, request
            )

            body = _BaseInstanceAdminRestTransport._BaseUpdateInstancePartition._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseUpdateInstancePartition._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.UpdateInstancePartition",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "UpdateInstancePartition",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                InstanceAdminRestTransport._UpdateInstancePartition._get_response(
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

            resp = self._interceptor.post_update_instance_partition(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_instance_partition_with_metadata(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminClient.update_instance_partition",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "UpdateInstancePartition",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def create_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.CreateInstanceRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateInstance(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_instance_config(
        self,
    ) -> Callable[
        [spanner_instance_admin.CreateInstanceConfigRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateInstanceConfig(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_instance_partition(
        self,
    ) -> Callable[
        [spanner_instance_admin.CreateInstancePartitionRequest],
        operations_pb2.Operation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateInstancePartition(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_instance(
        self,
    ) -> Callable[[spanner_instance_admin.DeleteInstanceRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteInstance(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_instance_config(
        self,
    ) -> Callable[
        [spanner_instance_admin.DeleteInstanceConfigRequest], empty_pb2.Empty
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteInstanceConfig(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_instance_partition(
        self,
    ) -> Callable[
        [spanner_instance_admin.DeleteInstancePartitionRequest], empty_pb2.Empty
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteInstancePartition(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetIamPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.GetInstanceRequest], spanner_instance_admin.Instance
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetInstance(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_instance_config(
        self,
    ) -> Callable[
        [spanner_instance_admin.GetInstanceConfigRequest],
        spanner_instance_admin.InstanceConfig,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetInstanceConfig(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_instance_partition(
        self,
    ) -> Callable[
        [spanner_instance_admin.GetInstancePartitionRequest],
        spanner_instance_admin.InstancePartition,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetInstancePartition(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_instance_config_operations(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstanceConfigOperationsRequest],
        spanner_instance_admin.ListInstanceConfigOperationsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListInstanceConfigOperations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_instance_configs(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstanceConfigsRequest],
        spanner_instance_admin.ListInstanceConfigsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListInstanceConfigs(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_instance_partition_operations(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstancePartitionOperationsRequest],
        spanner_instance_admin.ListInstancePartitionOperationsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListInstancePartitionOperations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_instance_partitions(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstancePartitionsRequest],
        spanner_instance_admin.ListInstancePartitionsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListInstancePartitions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_instances(
        self,
    ) -> Callable[
        [spanner_instance_admin.ListInstancesRequest],
        spanner_instance_admin.ListInstancesResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListInstances(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def move_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.MoveInstanceRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._MoveInstance(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def set_iam_policy(
        self,
    ) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._SetIamPolicy(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        iam_policy_pb2.TestIamPermissionsResponse,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._TestIamPermissions(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_instance(
        self,
    ) -> Callable[
        [spanner_instance_admin.UpdateInstanceRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateInstance(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_instance_config(
        self,
    ) -> Callable[
        [spanner_instance_admin.UpdateInstanceConfigRequest], operations_pb2.Operation
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateInstanceConfig(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_instance_partition(
        self,
    ) -> Callable[
        [spanner_instance_admin.UpdateInstancePartitionRequest],
        operations_pb2.Operation,
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateInstancePartition(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def cancel_operation(self):
        return self._CancelOperation(self._session, self._host, self._interceptor)  # type: ignore

    class _CancelOperation(
        _BaseInstanceAdminRestTransport._BaseCancelOperation, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.CancelOperation")

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
                _BaseInstanceAdminRestTransport._BaseCancelOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_operation(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseCancelOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseCancelOperation._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.CancelOperation",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "CancelOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._CancelOperation._get_response(
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
        _BaseInstanceAdminRestTransport._BaseDeleteOperation, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.DeleteOperation")

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
                _BaseInstanceAdminRestTransport._BaseDeleteOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_operation(
                request, metadata
            )
            transcoded_request = _BaseInstanceAdminRestTransport._BaseDeleteOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseDeleteOperation._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.DeleteOperation",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "DeleteOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._DeleteOperation._get_response(
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
        _BaseInstanceAdminRestTransport._BaseGetOperation, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.GetOperation")

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
                _BaseInstanceAdminRestTransport._BaseGetOperation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_operation(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseGetOperation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseGetOperation._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.GetOperation",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "GetOperation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._GetOperation._get_response(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminAsyncClient.GetOperation",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
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
        _BaseInstanceAdminRestTransport._BaseListOperations, InstanceAdminRestStub
    ):
        def __hash__(self):
            return hash("InstanceAdminRestTransport.ListOperations")

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
                _BaseInstanceAdminRestTransport._BaseListOperations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_operations(request, metadata)
            transcoded_request = _BaseInstanceAdminRestTransport._BaseListOperations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseInstanceAdminRestTransport._BaseListOperations._get_query_params_json(
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
                    f"Sending request for google.spanner.admin.instance_v1.InstanceAdminClient.ListOperations",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
                        "rpcName": "ListOperations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = InstanceAdminRestTransport._ListOperations._get_response(
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
                    "Received response for google.spanner.admin.instance_v1.InstanceAdminAsyncClient.ListOperations",
                    extra={
                        "serviceName": "google.spanner.admin.instance.v1.InstanceAdmin",
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


__all__ = ("InstanceAdminRestTransport",)
