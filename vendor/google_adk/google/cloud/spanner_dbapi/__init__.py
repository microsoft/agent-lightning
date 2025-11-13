# Copyright 2020 Google LLC All rights reserved.
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

"""Connection-based DB API for Cloud Spanner."""

from google.cloud.spanner_dbapi.connection import Connection
from google.cloud.spanner_dbapi.connection import connect

from google.cloud.spanner_dbapi.cursor import Cursor

from google.cloud.spanner_dbapi.exceptions import DatabaseError
from google.cloud.spanner_dbapi.exceptions import DataError
from google.cloud.spanner_dbapi.exceptions import Error
from google.cloud.spanner_dbapi.exceptions import IntegrityError
from google.cloud.spanner_dbapi.exceptions import InterfaceError
from google.cloud.spanner_dbapi.exceptions import InternalError
from google.cloud.spanner_dbapi.exceptions import NotSupportedError
from google.cloud.spanner_dbapi.exceptions import OperationalError
from google.cloud.spanner_dbapi.exceptions import ProgrammingError
from google.cloud.spanner_dbapi.exceptions import Warning

from google.cloud.spanner_dbapi.parse_utils import get_param_types

from google.cloud.spanner_dbapi.types import BINARY
from google.cloud.spanner_dbapi.types import DATETIME
from google.cloud.spanner_dbapi.types import NUMBER
from google.cloud.spanner_dbapi.types import ROWID
from google.cloud.spanner_dbapi.types import STRING
from google.cloud.spanner_dbapi.types import Binary
from google.cloud.spanner_dbapi.types import Date
from google.cloud.spanner_dbapi.types import DateFromTicks
from google.cloud.spanner_dbapi.types import Time
from google.cloud.spanner_dbapi.types import TimeFromTicks
from google.cloud.spanner_dbapi.types import Timestamp
from google.cloud.spanner_dbapi.types import TimestampStr
from google.cloud.spanner_dbapi.types import TimestampFromTicks

from google.cloud.spanner_dbapi.version import DEFAULT_USER_AGENT

apilevel = "2.0"  # supports DP-API 2.0 level.
paramstyle = "format"  # ANSI C printf format codes, e.g. ...WHERE name=%s.

# Threads may share the module, but not connections. This is a paranoid threadsafety
# level, but it is necessary for starters to use when debugging failures.
# Eventually once transactions are working properly, we'll update the
# threadsafety level.
threadsafety = 1


__all__ = [
    "Connection",
    "connect",
    "Cursor",
    "DatabaseError",
    "DataError",
    "Error",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    "Warning",
    "DEFAULT_USER_AGENT",
    "apilevel",
    "paramstyle",
    "threadsafety",
    "get_param_types",
    "Binary",
    "Date",
    "DateFromTicks",
    "Time",
    "TimeFromTicks",
    "Timestamp",
    "TimestampFromTicks",
    "BINARY",
    "STRING",
    "NUMBER",
    "DATETIME",
    "ROWID",
    "TimestampStr",
]
