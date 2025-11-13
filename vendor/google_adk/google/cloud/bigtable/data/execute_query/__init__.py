# Copyright 2024 Google LLC
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

from google.cloud.bigtable.data.execute_query._async.execute_query_iterator import (
    ExecuteQueryIteratorAsync,
)
from google.cloud.bigtable.data.execute_query._sync_autogen.execute_query_iterator import (
    ExecuteQueryIterator,
)
from google.cloud.bigtable.data.execute_query.metadata import (
    Metadata,
    SqlType,
)
from google.cloud.bigtable.data.execute_query.values import (
    ExecuteQueryValueType,
    QueryResultRow,
    Struct,
)
from google.cloud.bigtable.data._cross_sync import CrossSync

CrossSync.add_mapping("ExecuteQueryIterator", ExecuteQueryIteratorAsync)
CrossSync._Sync_Impl.add_mapping("ExecuteQueryIterator", ExecuteQueryIterator)

__all__ = [
    "ExecuteQueryValueType",
    "SqlType",
    "QueryResultRow",
    "Struct",
    "Metadata",
    "ExecuteQueryIteratorAsync",
    "ExecuteQueryIterator",
]
