# Copyright 2023 Google LLC All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import gzip
import pickle
import base64

from google.cloud.spanner_v1 import BatchTransactionId


def decode_from_string(encoded_partition_id):
    gzip_bytes = base64.b64decode(bytes(encoded_partition_id, "utf-8"))
    partition_id_bytes = gzip.decompress(gzip_bytes)
    return pickle.loads(partition_id_bytes)


def encode_to_string(batch_transaction_id, partition_result):
    partition_id = PartitionId(batch_transaction_id, partition_result)
    partition_id_bytes = pickle.dumps(partition_id)
    gzip_bytes = gzip.compress(partition_id_bytes)
    return str(base64.b64encode(gzip_bytes), "utf-8")


@dataclass
class PartitionId:
    batch_transaction_id: BatchTransactionId
    partition_result: Any
