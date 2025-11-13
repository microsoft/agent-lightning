# Copyright 2024 Google LLC All rights reserved.
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

from google.longrunning import operations_pb2 as operations_pb2
from google.protobuf import empty_pb2
import google.cloud.spanner_v1.testing.spanner_database_admin_pb2_grpc as database_admin_grpc


# An in-memory mock DatabaseAdmin server that can be used for testing.
class DatabaseAdminServicer(database_admin_grpc.DatabaseAdminServicer):
    def __init__(self):
        self._requests = []

    @property
    def requests(self):
        return self._requests

    def clear_requests(self):
        self._requests = []

    def UpdateDatabaseDdl(self, request, context):
        self._requests.append(request)
        operation = operations_pb2.Operation()
        operation.done = True
        operation.name = "projects/test-project/operations/test-operation"
        operation.response.Pack(empty_pb2.Empty())
        return operation
