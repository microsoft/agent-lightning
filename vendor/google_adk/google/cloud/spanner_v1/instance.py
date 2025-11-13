# Copyright 2016 Google LLC All rights reserved.
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

"""User friendly container for Cloud Spanner Instance."""

import google.api_core.operation
from google.api_core.exceptions import InvalidArgument
import re
import typing

from google.protobuf.empty_pb2 import Empty
from google.protobuf.field_mask_pb2 import FieldMask
from google.cloud.exceptions import NotFound

from google.cloud.spanner_admin_instance_v1 import Instance as InstancePB
from google.cloud.spanner_admin_database_v1.types import backup
from google.cloud.spanner_admin_database_v1.types import spanner_database_admin
from google.cloud.spanner_admin_database_v1 import DatabaseDialect
from google.cloud.spanner_admin_database_v1 import ListBackupsRequest
from google.cloud.spanner_admin_database_v1 import ListBackupOperationsRequest
from google.cloud.spanner_admin_database_v1 import ListDatabasesRequest
from google.cloud.spanner_admin_database_v1 import ListDatabaseOperationsRequest
from google.cloud.spanner_v1._helpers import _metadata_with_prefix
from google.cloud.spanner_v1.backup import Backup
from google.cloud.spanner_v1.database import Database
from google.cloud.spanner_v1.testing.database_test import TestDatabase

_INSTANCE_NAME_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/" r"instances/(?P<instance_id>[a-z][-a-z0-9]*)$"
)

DEFAULT_NODE_COUNT = 1
PROCESSING_UNITS_PER_NODE = 1000

_OPERATION_METADATA_MESSAGES: typing.Tuple = (
    backup.Backup,
    backup.CreateBackupMetadata,
    backup.CopyBackupMetadata,
    spanner_database_admin.CreateDatabaseMetadata,
    spanner_database_admin.Database,
    spanner_database_admin.OptimizeRestoredDatabaseMetadata,
    spanner_database_admin.RestoreDatabaseMetadata,
    spanner_database_admin.UpdateDatabaseDdlMetadata,
)

_OPERATION_METADATA_TYPES = {
    "type.googleapis.com/{}".format(message._meta.full_name): message
    for message in _OPERATION_METADATA_MESSAGES
}

_OPERATION_RESPONSE_TYPES = {
    backup.CreateBackupMetadata: backup.Backup,
    backup.CopyBackupMetadata: backup.Backup,
    spanner_database_admin.CreateDatabaseMetadata: spanner_database_admin.Database,
    spanner_database_admin.OptimizeRestoredDatabaseMetadata: spanner_database_admin.Database,
    spanner_database_admin.RestoreDatabaseMetadata: spanner_database_admin.Database,
    spanner_database_admin.UpdateDatabaseDdlMetadata: Empty,
}


def _type_string_to_type_pb(type_string):
    return _OPERATION_METADATA_TYPES.get(type_string, Empty)


class Instance(object):
    """Representation of a Cloud Spanner Instance.

    We can use a :class:`Instance` to:

    * :meth:`reload` itself
    * :meth:`create` itself
    * :meth:`update` itself
    * :meth:`delete` itself

    :type instance_id: str
    :param instance_id: The ID of the instance.

    :type client: :class:`~google.cloud.spanner_v1.client.Client`
    :param client: The client that owns the instance. Provides
                   authorization and a project ID.

    :type configuration_name: str
    :param configuration_name: Name of the instance configuration defining
                        how the instance will be created.
                        Required for instances which do not yet exist.

    :type node_count: int
    :param node_count: (Optional) Number of nodes allocated to the instance.

    :type processing_units: int
    :param processing_units: (Optional) The number of processing units
                            allocated to this instance.

    :type display_name: str
    :param display_name: (Optional) The display name for the instance in the
                         Cloud Console UI. (Must be between 4 and 30
                         characters.) If this value is not set in the
                         constructor, will fall back to the instance ID.

    :type labels: dict (str -> str) or None
    :param labels: (Optional) User-assigned labels for this instance.
    """

    def __init__(
        self,
        instance_id,
        client,
        configuration_name=None,
        node_count=None,
        display_name=None,
        emulator_host=None,
        labels=None,
        processing_units=None,
    ):
        self.instance_id = instance_id
        self._client = client
        self.configuration_name = configuration_name
        if node_count is not None and processing_units is not None:
            if processing_units != node_count * PROCESSING_UNITS_PER_NODE:
                raise InvalidArgument(
                    "Only one of node count and processing units can be set."
                )
        if node_count is None and processing_units is None:
            self._node_count = DEFAULT_NODE_COUNT
            self._processing_units = DEFAULT_NODE_COUNT * PROCESSING_UNITS_PER_NODE
        elif node_count is not None:
            self._node_count = node_count
            self._processing_units = node_count * PROCESSING_UNITS_PER_NODE
        else:
            self._processing_units = processing_units
            self._node_count = processing_units // PROCESSING_UNITS_PER_NODE
        self.display_name = display_name or instance_id
        self.emulator_host = emulator_host
        if labels is None:
            labels = {}
        self.labels = labels

    def _update_from_pb(self, instance_pb):
        """Refresh self from the server-provided protobuf.

        Helper for :meth:`from_pb` and :meth:`reload`.
        """
        if not instance_pb.display_name:  # Simple field (string)
            raise ValueError("Instance protobuf does not contain display_name")
        self.display_name = instance_pb.display_name
        self.configuration_name = instance_pb.config
        self._node_count = instance_pb.node_count
        self._processing_units = instance_pb.processing_units
        self.labels = instance_pb.labels

    @classmethod
    def from_pb(cls, instance_pb, client):
        """Creates an instance from a protobuf.

        :type instance_pb:
            :class:`~google.spanner.v2.spanner_instance_admin_pb2.Instance`
        :param instance_pb: A instance protobuf object.

        :type client: :class:`~google.cloud.spanner_v1.client.Client`
        :param client: The client that owns the instance.

        :rtype: :class:`Instance`
        :returns: The instance parsed from the protobuf response.
        :raises ValueError:
            if the instance name does not match
            ``projects/{project}/instances/{instance_id}`` or if the parsed
            project ID does not match the project ID on the client.
        """
        match = _INSTANCE_NAME_RE.match(instance_pb.name)
        if match is None:
            raise ValueError(
                "Instance protobuf name was not in the " "expected format.",
                instance_pb.name,
            )
        if match.group("project") != client.project:
            raise ValueError(
                "Project ID on instance does not match the " "project ID on the client"
            )
        instance_id = match.group("instance_id")
        configuration_name = instance_pb.config

        result = cls(instance_id, client, configuration_name)
        result._update_from_pb(instance_pb)
        return result

    @property
    def name(self):
        """Instance name used in requests.

        .. note::

           This property will not change if ``instance_id`` does not,
           but the return value is not cached.

        The instance name is of the form

            ``"projects/{project}/instances/{instance_id}"``

        :rtype: str
        :returns: The instance name.
        """
        return self._client.project_name + "/instances/" + self.instance_id

    @property
    def processing_units(self):
        """Processing units used in requests.

        :rtype: int
        :returns: The number of processing units allocated to this instance.
        """
        return self._processing_units

    @processing_units.setter
    def processing_units(self, value):
        """Sets the processing units for requests. Affects node_count.

        :param value: The number of processing units allocated to this instance.
        """
        self._processing_units = value
        self._node_count = value // PROCESSING_UNITS_PER_NODE

    @property
    def node_count(self):
        """Node count used in requests.

        :rtype: int
        :returns:
            The number of nodes in the instance's cluster;
            used to set up the instance's cluster.
        """
        return self._node_count

    @node_count.setter
    def node_count(self, value):
        """Sets the node count for requests. Affects processing_units.

        :param value: The number of nodes in the instance's cluster.
        """
        self._node_count = value
        self._processing_units = value * PROCESSING_UNITS_PER_NODE

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        # NOTE: This does not compare the configuration values, such as
        #       the display_name. Instead, it only compares
        #       identifying values instance ID and client. This is
        #       intentional, since the same instance can be in different states
        #       if not synchronized. Instances with similar instance
        #       settings but different clients can't be used in the same way.
        return other.instance_id == self.instance_id and other._client == self._client

    def __ne__(self, other):
        return not self == other

    def copy(self):
        """Make a copy of this instance.

        Copies the local data stored as simple types and copies the client
        attached to this instance.

        :rtype: :class:`~google.cloud.spanner_v1.instance.Instance`
        :returns: A copy of the current instance.
        """
        new_client = self._client.copy()
        return self.__class__(
            self.instance_id,
            new_client,
            self.configuration_name,
            node_count=self._node_count,
            processing_units=self._processing_units,
            display_name=self.display_name,
        )

    def create(self):
        """Create this instance.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.instance.v1#google.spanner.admin.instance.v1.InstanceAdmin.CreateInstance

        .. note::

           Uses the ``project`` and ``instance_id`` on the current
           :class:`Instance` in addition to the ``display_name``.
           To change them before creating, reset the values via

           .. code:: python

              instance.display_name = 'New display name'
              instance.instance_id = 'i-changed-my-mind'

           before calling :meth:`create`.

        :rtype: :class:`~google.api_core.operation.Operation`
        :returns: an operation instance
        :raises Conflict: if the instance already exists
        """
        api = self._client.instance_admin_api
        instance_pb = InstancePB(
            name=self.name,
            config=self.configuration_name,
            display_name=self.display_name,
            processing_units=self._processing_units,
            labels=self.labels,
        )
        metadata = _metadata_with_prefix(self.name)

        future = api.create_instance(
            parent=self._client.project_name,
            instance_id=self.instance_id,
            instance=instance_pb,
            metadata=metadata,
        )

        return future

    def exists(self):
        """Test whether this instance exists.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.instance.v1#google.spanner.admin.instance.v1.InstanceAdmin.GetInstanceConfig

        :rtype: bool
        :returns: True if the instance exists, else false
        """
        api = self._client.instance_admin_api
        metadata = _metadata_with_prefix(self.name)

        try:
            api.get_instance(name=self.name, metadata=metadata)
        except NotFound:
            return False

        return True

    def reload(self):
        """Reload the metadata for this instance.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.instance.v1#google.spanner.admin.instance.v1.InstanceAdmin.GetInstanceConfig

        :raises NotFound: if the instance does not exist
        """
        api = self._client.instance_admin_api
        metadata = _metadata_with_prefix(self.name)

        instance_pb = api.get_instance(name=self.name, metadata=metadata)

        self._update_from_pb(instance_pb)

    def update(self):
        """Update this instance.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.instance.v1#google.spanner.admin.instance.v1.InstanceAdmin.UpdateInstance

        .. note::

            Updates the ``display_name``, ``node_count``, ``processing_units``
            and ``labels``. To change those values before updating, set them via

            .. code:: python

                instance.display_name = 'New display name'
                instance.node_count = 5

           before calling :meth:`update`.

        :rtype: :class:`google.api_core.operation.Operation`
        :returns: an operation instance
        :raises NotFound: if the instance does not exist
        """
        api = self._client.instance_admin_api
        instance_pb = InstancePB(
            name=self.name,
            config=self.configuration_name,
            display_name=self.display_name,
            node_count=self._node_count,
            processing_units=self._processing_units,
            labels=self.labels,
        )

        # Always update only processing_units, not nodes
        field_mask = FieldMask(
            paths=["config", "display_name", "processing_units", "labels"]
        )
        metadata = _metadata_with_prefix(self.name)

        future = api.update_instance(
            instance=instance_pb, field_mask=field_mask, metadata=metadata
        )

        return future

    def delete(self):
        """Mark an instance and all of its databases for permanent deletion.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.instance.v1#google.spanner.admin.instance.v1.InstanceAdmin.DeleteInstance

        Immediately upon completion of the request:

        * Billing will cease for all of the instance's reserved resources.

        Soon afterward:

        * The instance and all databases within the instance will be deleted.
          All data in the databases will be permanently deleted.
        """
        api = self._client.instance_admin_api
        metadata = _metadata_with_prefix(self.name)

        api.delete_instance(name=self.name, metadata=metadata)

    def database(
        self,
        database_id,
        ddl_statements=(),
        pool=None,
        logger=None,
        encryption_config=None,
        database_dialect=DatabaseDialect.DATABASE_DIALECT_UNSPECIFIED,
        database_role=None,
        enable_drop_protection=False,
        # should be only set for tests if tests want to use interceptors
        enable_interceptors_in_tests=False,
        proto_descriptors=None,
    ):
        """Factory to create a database within this instance.

        :type database_id: str
        :param database_id: The ID of the database.

        :type ddl_statements: list of string
        :param ddl_statements: (Optional) DDL statements, excluding the
                               'CREATE DATABASE' statement.

        :type pool: concrete subclass of
                    :class:`~google.cloud.spanner_v1.pool.AbstractSessionPool`.
        :param pool: (Optional) session pool to be used by database.

        :type logger: :class:`logging.Logger`
        :param logger: (Optional) a custom logger that is used if `log_commit_stats`
                       is `True` to log commit statistics. If not passed, a logger
                       will be created when needed that will log the commit statistics
                       to stdout.

        :type encryption_config:
            :class:`~google.cloud.spanner_admin_database_v1.types.EncryptionConfig`
            or :class:`~google.cloud.spanner_admin_database_v1.types.RestoreDatabaseEncryptionConfig`
            or :class:`dict`
        :param encryption_config:
            (Optional) Encryption configuration for the database.
            If a dict is provided, it must be of the same form as either of the protobuf
            messages :class:`~google.cloud.spanner_admin_database_v1.types.EncryptionConfig`
            or :class:`~google.cloud.spanner_admin_database_v1.types.RestoreDatabaseEncryptionConfig`

        :type database_dialect:
            :class:`~google.cloud.spanner_admin_database_v1.types.DatabaseDialect`
        :param database_dialect:
            (Optional) database dialect for the database

        :type enable_drop_protection: boolean
        :param enable_drop_protection: (Optional) Represents whether the database
            has drop protection enabled or not.

        :type enable_interceptors_in_tests: boolean
        :param enable_interceptors_in_tests: (Optional) should only be set to True
            for tests if the tests want to use interceptors.

        :type proto_descriptors: bytes
        :param proto_descriptors: (Optional) Proto descriptors used by CREATE/ALTER PROTO BUNDLE
                                  statements in 'ddl_statements' above.

        :rtype: :class:`~google.cloud.spanner_v1.database.Database`
        :returns: a database owned by this instance.
        """

        if not enable_interceptors_in_tests:
            return Database(
                database_id,
                self,
                ddl_statements=ddl_statements,
                pool=pool,
                logger=logger,
                encryption_config=encryption_config,
                database_dialect=database_dialect,
                database_role=database_role,
                enable_drop_protection=enable_drop_protection,
                proto_descriptors=proto_descriptors,
            )
        else:
            return TestDatabase(
                database_id,
                self,
                ddl_statements=ddl_statements,
                pool=pool,
                logger=logger,
                encryption_config=encryption_config,
                database_dialect=database_dialect,
                database_role=database_role,
                enable_drop_protection=enable_drop_protection,
            )

    def list_databases(self, page_size=None):
        """List databases for the instance.

        See
        https://cloud.google.com/spanner/reference/rpc/google.spanner.admin.database.v1#google.spanner.admin.database.v1.DatabaseAdmin.ListDatabases

        :type page_size: int
        :param page_size:
            Optional. The maximum number of databases in each page of results
            from this request. Non-positive values are ignored. Defaults
            to a sensible value set by the API.

        :rtype: :class:`~google.api._ore.page_iterator.Iterator`
        :returns:
            Iterator of :class:`~google.cloud.spanner_admin_database_v1.types.Database`
            resources within the current instance.
        """
        metadata = _metadata_with_prefix(self.name)
        request = ListDatabasesRequest(parent=self.name, page_size=page_size)
        page_iter = self._client.database_admin_api.list_databases(
            request=request, metadata=metadata
        )
        return page_iter

    def backup(
        self,
        backup_id,
        database="",
        expire_time=None,
        version_time=None,
        encryption_config=None,
    ):
        """Factory to create a backup within this instance.

        :type backup_id: str
        :param backup_id: The ID of the backup.

        :type database: :class:`~google.cloud.spanner_v1.database.Database`
        :param database:
            Optional. The database that will be used when creating the backup.
            Required if the create method needs to be called.

        :type expire_time: :class:`datetime.datetime`
        :param expire_time:
            Optional. The expire time that will be used when creating the backup.
            Required if the create method needs to be called.

        :type version_time: :class:`datetime.datetime`
        :param version_time:
            Optional. The version time that will be used to create the externally
            consistent copy of the database. If not present, it is the same as
            the `create_time` of the backup.

        :type encryption_config:
            :class:`~google.cloud.spanner_admin_database_v1.types.CreateBackupEncryptionConfig`
            or :class:`dict`
        :param encryption_config:
            (Optional) Encryption configuration for the backup.
            If a dict is provided, it must be of the same form as the protobuf
            message :class:`~google.cloud.spanner_admin_database_v1.types.CreateBackupEncryptionConfig`

        :rtype: :class:`~google.cloud.spanner_v1.backup.Backup`
        :returns: a backup owned by this instance.
        """
        try:
            return Backup(
                backup_id,
                self,
                database=database.name,
                expire_time=expire_time,
                version_time=version_time,
                encryption_config=encryption_config,
            )
        except AttributeError:
            return Backup(
                backup_id,
                self,
                database=database,
                expire_time=expire_time,
                version_time=version_time,
                encryption_config=encryption_config,
            )

    def copy_backup(
        self,
        backup_id,
        source_backup,
        expire_time=None,
        encryption_config=None,
    ):
        """Factory to create a copy backup within this instance.

        :type backup_id: str
        :param backup_id: The ID of the backup copy.
        :type source_backup: str
        :param source_backup_id: The full path of the source backup to be copied.
        :type expire_time: :class:`datetime.datetime`
        :param expire_time:
            Optional. The expire time that will be used when creating the copy backup.
            Required if the create method needs to be called.
        :type encryption_config:
            :class:`~google.cloud.spanner_admin_database_v1.types.CopyBackupEncryptionConfig`
            or :class:`dict`
        :param encryption_config:
            (Optional) Encryption configuration for the backup.
            If a dict is provided, it must be of the same form as the protobuf
            message :class:`~google.cloud.spanner_admin_database_v1.types.CopyBackupEncryptionConfig`
        :rtype: :class:`~google.cloud.spanner_v1.backup.Backup`
        :returns: a copy backup owned by this instance.
        """
        return Backup(
            backup_id,
            self,
            source_backup=source_backup,
            expire_time=expire_time,
            encryption_config=encryption_config,
        )

    def list_backups(self, filter_="", page_size=None):
        """List backups for the instance.

        :type filter_: str
        :param filter_:
            Optional. A string specifying a filter for which backups to list.

        :type page_size: int
        :param page_size:
            Optional. The maximum number of databases in each page of results
            from this request. Non-positive values are ignored. Defaults to a
            sensible value set by the API.

        :rtype: :class:`~google.api_core.page_iterator.Iterator`
        :returns:
            Iterator of :class:`~google.cloud.spanner_admin_database_v1.types.Backup`
            resources within the current instance.
        """
        metadata = _metadata_with_prefix(self.name)
        request = ListBackupsRequest(
            parent=self.name,
            filter=filter_,
            page_size=page_size,
        )
        page_iter = self._client.database_admin_api.list_backups(
            request=request, metadata=metadata
        )
        return page_iter

    def list_backup_operations(self, filter_="", page_size=None):
        """List backup operations for the instance.

        :type filter_: str
        :param filter_:
            Optional. A string specifying a filter for which backup operations
            to list.

        :type page_size: int
        :param page_size:
            Optional. The maximum number of operations in each page of results
            from this request. Non-positive values are ignored. Defaults to a
            sensible value set by the API.

        :rtype: :class:`~google.api_core.page_iterator.Iterator`
        :returns:
            Iterator of :class:`~google.api_core.operation.Operation`
            resources within the current instance.
        """
        metadata = _metadata_with_prefix(self.name)
        request = ListBackupOperationsRequest(
            parent=self.name,
            filter=filter_,
            page_size=page_size,
        )
        page_iter = self._client.database_admin_api.list_backup_operations(
            request=request, metadata=metadata
        )
        return map(self._item_to_operation, page_iter)

    def list_database_operations(self, filter_="", page_size=None):
        """List database operations for the instance.

        :type filter_: str
        :param filter_:
            Optional. A string specifying a filter for which database operations
            to list.

        :type page_size: int
        :param page_size:
            Optional. The maximum number of operations in each page of results
            from this request. Non-positive values are ignored. Defaults to a
            sensible value set by the API.

        :rtype: :class:`~google.api_core.page_iterator.Iterator`
        :returns:
            Iterator of :class:`~google.api_core.operation.Operation`
            resources within the current instance.
        """
        metadata = _metadata_with_prefix(self.name)
        request = ListDatabaseOperationsRequest(
            parent=self.name,
            filter=filter_,
            page_size=page_size,
        )
        page_iter = self._client.database_admin_api.list_database_operations(
            request=request, metadata=metadata
        )
        return map(self._item_to_operation, page_iter)

    def _item_to_operation(self, operation_pb):
        """Convert an operation protobuf to the native object.
        :type operation_pb: :class:`~google.longrunning.operations.Operation`
        :param operation_pb: An operation returned from the API.
        :rtype: :class:`~google.api_core.operation.Operation`
        :returns: The next operation in the page.
        """
        operations_client = self._client.database_admin_api.transport.operations_client
        metadata_type = _type_string_to_type_pb(operation_pb.metadata.type_url)
        response_type = _OPERATION_RESPONSE_TYPES[metadata_type]
        return google.api_core.operation.from_gapic(
            operation_pb, operations_client, response_type, metadata_type=metadata_type
        )
