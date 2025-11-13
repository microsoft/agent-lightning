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
from google.cloud.spanner_admin_instance_v1 import gapic_version as package_version

__version__ = package_version.__version__


from .services.instance_admin import InstanceAdminClient
from .services.instance_admin import InstanceAdminAsyncClient

from .types.common import OperationProgress
from .types.common import ReplicaSelection
from .types.common import FulfillmentPeriod
from .types.spanner_instance_admin import AutoscalingConfig
from .types.spanner_instance_admin import CreateInstanceConfigMetadata
from .types.spanner_instance_admin import CreateInstanceConfigRequest
from .types.spanner_instance_admin import CreateInstanceMetadata
from .types.spanner_instance_admin import CreateInstancePartitionMetadata
from .types.spanner_instance_admin import CreateInstancePartitionRequest
from .types.spanner_instance_admin import CreateInstanceRequest
from .types.spanner_instance_admin import DeleteInstanceConfigRequest
from .types.spanner_instance_admin import DeleteInstancePartitionRequest
from .types.spanner_instance_admin import DeleteInstanceRequest
from .types.spanner_instance_admin import FreeInstanceMetadata
from .types.spanner_instance_admin import GetInstanceConfigRequest
from .types.spanner_instance_admin import GetInstancePartitionRequest
from .types.spanner_instance_admin import GetInstanceRequest
from .types.spanner_instance_admin import Instance
from .types.spanner_instance_admin import InstanceConfig
from .types.spanner_instance_admin import InstancePartition
from .types.spanner_instance_admin import ListInstanceConfigOperationsRequest
from .types.spanner_instance_admin import ListInstanceConfigOperationsResponse
from .types.spanner_instance_admin import ListInstanceConfigsRequest
from .types.spanner_instance_admin import ListInstanceConfigsResponse
from .types.spanner_instance_admin import ListInstancePartitionOperationsRequest
from .types.spanner_instance_admin import ListInstancePartitionOperationsResponse
from .types.spanner_instance_admin import ListInstancePartitionsRequest
from .types.spanner_instance_admin import ListInstancePartitionsResponse
from .types.spanner_instance_admin import ListInstancesRequest
from .types.spanner_instance_admin import ListInstancesResponse
from .types.spanner_instance_admin import MoveInstanceMetadata
from .types.spanner_instance_admin import MoveInstanceRequest
from .types.spanner_instance_admin import MoveInstanceResponse
from .types.spanner_instance_admin import ReplicaComputeCapacity
from .types.spanner_instance_admin import ReplicaInfo
from .types.spanner_instance_admin import UpdateInstanceConfigMetadata
from .types.spanner_instance_admin import UpdateInstanceConfigRequest
from .types.spanner_instance_admin import UpdateInstanceMetadata
from .types.spanner_instance_admin import UpdateInstancePartitionMetadata
from .types.spanner_instance_admin import UpdateInstancePartitionRequest
from .types.spanner_instance_admin import UpdateInstanceRequest

__all__ = (
    "InstanceAdminAsyncClient",
    "AutoscalingConfig",
    "CreateInstanceConfigMetadata",
    "CreateInstanceConfigRequest",
    "CreateInstanceMetadata",
    "CreateInstancePartitionMetadata",
    "CreateInstancePartitionRequest",
    "CreateInstanceRequest",
    "DeleteInstanceConfigRequest",
    "DeleteInstancePartitionRequest",
    "DeleteInstanceRequest",
    "FreeInstanceMetadata",
    "FulfillmentPeriod",
    "GetInstanceConfigRequest",
    "GetInstancePartitionRequest",
    "GetInstanceRequest",
    "Instance",
    "InstanceAdminClient",
    "InstanceConfig",
    "InstancePartition",
    "ListInstanceConfigOperationsRequest",
    "ListInstanceConfigOperationsResponse",
    "ListInstanceConfigsRequest",
    "ListInstanceConfigsResponse",
    "ListInstancePartitionOperationsRequest",
    "ListInstancePartitionOperationsResponse",
    "ListInstancePartitionsRequest",
    "ListInstancePartitionsResponse",
    "ListInstancesRequest",
    "ListInstancesResponse",
    "MoveInstanceMetadata",
    "MoveInstanceRequest",
    "MoveInstanceResponse",
    "OperationProgress",
    "ReplicaComputeCapacity",
    "ReplicaInfo",
    "ReplicaSelection",
    "UpdateInstanceConfigMetadata",
    "UpdateInstanceConfigRequest",
    "UpdateInstanceMetadata",
    "UpdateInstancePartitionMetadata",
    "UpdateInstancePartitionRequest",
    "UpdateInstanceRequest",
)
