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

from google.cloud.spanner_v1 import param_types


SQL_LIST_TABLES = """
SELECT table_name
FROM information_schema.tables
WHERE table_catalog = ''
AND table_schema = @table_schema
AND table_type = 'BASE TABLE'
"""

SQL_LIST_TABLES_AND_VIEWS = """
SELECT table_name
FROM information_schema.tables
WHERE table_catalog = '' AND table_schema = @table_schema
"""

SQL_GET_TABLE_COLUMN_SCHEMA = """
SELECT COLUMN_NAME, IS_NULLABLE, SPANNER_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = @schema_name AND TABLE_NAME = @table_name
"""

# This table maps spanner_types to Spanner's data type sizes as per
#   https://cloud.google.com/spanner/docs/data-types#allowable-types
# It is used to map `display_size` to a known type for Cursor.description
# after a row fetch.
# Since ResultMetadata
#   https://cloud.google.com/spanner/docs/reference/rest/v1/ResultSetMetadata
# does not send back the actual size, we have to lookup the respective size.
# Some fields' sizes are dependent upon the dynamic data hence aren't sent back
# by Cloud Spanner.
CODE_TO_DISPLAY_SIZE = {
    param_types.BOOL.code: 1,
    param_types.DATE.code: 4,
    param_types.FLOAT64.code: 8,
    param_types.FLOAT32.code: 4,
    param_types.INT64.code: 8,
    param_types.TIMESTAMP.code: 12,
}


class ColumnInfo:
    """Row column description object."""

    def __init__(
        self,
        name,
        type_code,
        display_size=None,
        internal_size=None,
        precision=None,
        scale=None,
        null_ok=False,
    ):
        self.name = name
        self.type_code = type_code
        self.display_size = display_size
        self.internal_size = internal_size
        self.precision = precision
        self.scale = scale
        self.null_ok = null_ok

        self.fields = (
            self.name,
            self.type_code,
            self.display_size,
            self.internal_size,
            self.precision,
            self.scale,
            self.null_ok,
        )

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, index):
        return self.fields[index]

    def __str__(self):
        str_repr = ", ".join(
            filter(
                lambda part: part is not None,
                [
                    "name='%s'" % self.name,
                    "type_code=%d" % self.type_code,
                    "display_size=%d" % self.display_size
                    if self.display_size
                    else None,
                    "internal_size=%d" % self.internal_size
                    if self.internal_size
                    else None,
                    "precision='%s'" % self.precision if self.precision else None,
                    "scale='%s'" % self.scale if self.scale else None,
                    "null_ok='%s'" % self.null_ok if self.null_ok else None,
                ],
            )
        )
        return "ColumnInfo(%s)" % str_repr
