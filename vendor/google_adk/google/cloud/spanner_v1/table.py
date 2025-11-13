# Copyright 2021 Google LLC
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

"""User friendly container for Cloud Spanner Table."""

from google.cloud.exceptions import NotFound

from google.cloud.spanner_admin_database_v1 import DatabaseDialect
from google.cloud.spanner_v1.types import (
    Type,
    TypeCode,
)


_EXISTS_TEMPLATE = """
SELECT EXISTS(
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    {}
)
"""
_GET_SCHEMA_TEMPLATE = "SELECT * FROM {} LIMIT 0"


class Table(object):
    """Representation of a Cloud Spanner Table.

    :type table_id: str
    :param table_id: The ID of the table.

    :type database: :class:`~google.cloud.spanner_v1.database.Database`
    :param database: The database that owns the table.
    """

    def __init__(self, table_id, database, schema_name=None):
        if schema_name is None:
            self._schema_name = database.default_schema_name
        else:
            self._schema_name = schema_name
        self._table_id = table_id
        self._database = database

        # Calculated properties.
        self._schema = None

    @property
    def schema_name(self):
        """The schema name of the table used in SQL.

        :rtype: str
        :returns: The table schema name.
        """
        return self._schema_name

    @property
    def table_id(self):
        """The ID of the table used in SQL.

        :rtype: str
        :returns: The table ID.
        """
        return self._table_id

    @property
    def qualified_table_name(self):
        """The qualified name of the table used in SQL.

        :rtype: str
        :returns: The qualified table name.
        """
        if self.schema_name == self._database.default_schema_name:
            return self._quote_identifier(self.table_id)
        return "{}.{}".format(
            self._quote_identifier(self.schema_name),
            self._quote_identifier(self.table_id),
        )

    def _quote_identifier(self, identifier):
        """Quotes the given identifier using the rules of the dialect of the database of this table.

        :rtype: str
        :returns: The quoted identifier.
        """
        if self._database.database_dialect == DatabaseDialect.POSTGRESQL:
            return '"{}"'.format(identifier)
        return "`{}`".format(identifier)

    def exists(self):
        """Test whether this table exists.

        :rtype: bool
        :returns: True if the table exists, else false.
        """
        with self._database.snapshot() as snapshot:
            return self._exists(snapshot)

    def _exists(self, snapshot):
        """Query to check that the table exists.

        :type snapshot: :class:`~google.cloud.spanner_v1.snapshot.Snapshot`
        :param snapshot: snapshot to use for database queries

        :rtype: bool
        :returns: True if the table exists, else false.
        """
        if self._database.database_dialect == DatabaseDialect.POSTGRESQL:
            results = snapshot.execute_sql(
                sql=_EXISTS_TEMPLATE.format(
                    "WHERE TABLE_SCHEMA=$1 AND TABLE_NAME = $2"
                ),
                params={"p1": self.schema_name, "p2": self.table_id},
                param_types={
                    "p1": Type(code=TypeCode.STRING),
                    "p2": Type(code=TypeCode.STRING),
                },
            )
        else:
            results = snapshot.execute_sql(
                sql=_EXISTS_TEMPLATE.format(
                    "WHERE TABLE_SCHEMA = @schema_name AND TABLE_NAME = @table_id"
                ),
                params={"schema_name": self.schema_name, "table_id": self.table_id},
                param_types={
                    "schema_name": Type(code=TypeCode.STRING),
                    "table_id": Type(code=TypeCode.STRING),
                },
            )
        return next(iter(results))[0]

    @property
    def schema(self):
        """The schema of this table.

        :rtype: list of :class:`~google.cloud.spanner_v1.types.StructType.Field`
        :returns: The table schema.
        """
        if self._schema is None:
            with self._database.snapshot() as snapshot:
                self._schema = self._get_schema(snapshot)
        return self._schema

    def _get_schema(self, snapshot):
        """Get the schema of this table.

        :type snapshot: :class:`~google.cloud.spanner_v1.snapshot.Snapshot`
        :param snapshot: snapshot to use for database queries

        :rtype: list of :class:`~google.cloud.spanner_v1.types.StructType.Field`
        :returns: The table schema.
        """
        query = _GET_SCHEMA_TEMPLATE.format(self.qualified_table_name)
        results = snapshot.execute_sql(query)
        # Start iterating to force the schema to download.
        try:
            next(iter(results))
        except StopIteration:
            pass
        return list(results.fields)

    def reload(self):
        """Reload this table.

        Refresh any configured schema into :attr:`schema`.

        :raises NotFound: if the table does not exist
        """
        with self._database.snapshot() as snapshot:
            if not self._exists(snapshot):
                raise NotFound("table '{}' does not exist".format(self.table_id))
            self._schema = self._get_schema(snapshot)
