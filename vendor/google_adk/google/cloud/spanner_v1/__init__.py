# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
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
from __future__ import absolute_import

from google.cloud.spanner_v1 import gapic_version as package_version

__version__: str = package_version.__version__

from .services.spanner import SpannerClient
from .services.spanner import SpannerAsyncClient
from .types.commit_response import CommitResponse
from .types.keys import KeyRange as KeyRangePB
from .types.keys import KeySet as KeySetPB
from .types.mutation import Mutation
from .types.query_plan import PlanNode
from .types.query_plan import QueryPlan
from .types.result_set import PartialResultSet
from .types import RequestOptions
from .types.result_set import ResultSet
from .types.result_set import ResultSetMetadata
from .types.result_set import ResultSetStats
from .types.spanner import BatchCreateSessionsRequest
from .types.spanner import BatchCreateSessionsResponse
from .types.spanner import BatchWriteRequest
from .types.spanner import BatchWriteResponse
from .types.spanner import BeginTransactionRequest
from .types.spanner import CommitRequest
from .types.spanner import CreateSessionRequest
from .types.spanner import DeleteSessionRequest
from .types.spanner import DirectedReadOptions
from .types.spanner import ExecuteBatchDmlRequest
from .types.spanner import ExecuteBatchDmlResponse
from .types.spanner import ExecuteSqlRequest
from .types.spanner import GetSessionRequest
from .types.spanner import ListSessionsRequest
from .types.spanner import ListSessionsResponse
from .types.spanner import Partition
from .types.spanner import PartitionOptions
from .types.spanner import PartitionQueryRequest
from .types.spanner import PartitionReadRequest
from .types.spanner import PartitionResponse
from .types.spanner import ReadRequest
from .types.spanner import RollbackRequest
from .types.spanner import Session
from .types.transaction import Transaction
from .types.transaction import TransactionOptions
from .types.transaction import TransactionSelector
from .types.type import StructType
from .types.type import Type
from .types.type import TypeAnnotationCode
from .types.type import TypeCode
from .data_types import JsonObject, Interval
from .transaction import BatchTransactionId, DefaultTransactionOptions

from google.cloud.spanner_v1 import param_types
from google.cloud.spanner_v1.client import Client
from google.cloud.spanner_v1.keyset import KeyRange
from google.cloud.spanner_v1.keyset import KeySet
from google.cloud.spanner_v1.pool import AbstractSessionPool
from google.cloud.spanner_v1.pool import BurstyPool
from google.cloud.spanner_v1.pool import FixedSizePool
from google.cloud.spanner_v1.pool import PingingPool
from google.cloud.spanner_v1.pool import TransactionPingingPool


COMMIT_TIMESTAMP = "spanner.commit_timestamp()"
"""Placeholder be used to store commit timestamp of a transaction in a column.
This value can only be used for timestamp columns that have set the option
``(allow_commit_timestamp=true)`` in the schema.
"""


__all__ = (
    # google.cloud.spanner_v1
    "__version__",
    "param_types",
    # google.cloud.spanner_v1.client
    "Client",
    # google.cloud.spanner_v1.keyset
    "KeyRange",
    "KeySet",
    # google.cloud.spanner_v1.pool
    "AbstractSessionPool",
    "BurstyPool",
    "FixedSizePool",
    "PingingPool",
    "TransactionPingingPool",
    # local
    "COMMIT_TIMESTAMP",
    # google.cloud.spanner_v1.types
    "BatchCreateSessionsRequest",
    "BatchCreateSessionsResponse",
    "BatchWriteRequest",
    "BatchWriteResponse",
    "BeginTransactionRequest",
    "CommitRequest",
    "CommitResponse",
    "CreateSessionRequest",
    "DeleteSessionRequest",
    "DirectedReadOptions",
    "ExecuteBatchDmlRequest",
    "ExecuteBatchDmlResponse",
    "ExecuteSqlRequest",
    "GetSessionRequest",
    "KeyRangePB",
    "KeySetPB",
    "ListSessionsRequest",
    "ListSessionsResponse",
    "Mutation",
    "PartialResultSet",
    "Partition",
    "PartitionOptions",
    "PartitionQueryRequest",
    "PartitionReadRequest",
    "PartitionResponse",
    "PlanNode",
    "QueryPlan",
    "ReadRequest",
    "RequestOptions",
    "ResultSet",
    "ResultSetMetadata",
    "ResultSetStats",
    "RollbackRequest",
    "Session",
    "StructType",
    "Transaction",
    "TransactionOptions",
    "TransactionSelector",
    "Type",
    "TypeAnnotationCode",
    "TypeCode",
    # Custom spanner related data types
    "JsonObject",
    "Interval",
    # google.cloud.spanner_v1.services
    "SpannerClient",
    "SpannerAsyncClient",
    "BatchTransactionId",
    "DefaultTransactionOptions",
)
