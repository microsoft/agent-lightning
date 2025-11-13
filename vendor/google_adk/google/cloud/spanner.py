# Copyright 2016, Google LLC All rights reserved.
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

from __future__ import absolute_import

from google.cloud.spanner_v1 import __version__
from google.cloud.spanner_v1 import param_types
from google.cloud.spanner_v1 import Client
from google.cloud.spanner_v1 import KeyRange
from google.cloud.spanner_v1 import KeySet
from google.cloud.spanner_v1 import AbstractSessionPool
from google.cloud.spanner_v1 import BurstyPool
from google.cloud.spanner_v1 import FixedSizePool
from google.cloud.spanner_v1 import PingingPool
from google.cloud.spanner_v1 import TransactionPingingPool
from google.cloud.spanner_v1 import COMMIT_TIMESTAMP


__all__ = (
    # google.cloud.spanner
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
)
