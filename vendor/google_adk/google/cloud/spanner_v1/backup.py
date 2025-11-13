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

"""User friendly container for Cloud Spanner Backup."""

import re

from google.cloud.exceptions import NotFound

from google.cloud.spanner_admin_database_v1 import Backup as BackupPB
from google.cloud.spanner_admin_database_v1 import CreateBackupEncryptionConfig
from google.cloud.spanner_admin_database_v1 import CreateBackupRequest
from google.cloud.spanner_admin_database_v1 import CopyBackupEncryptionConfig
from google.cloud.spanner_admin_database_v1 import CopyBackupRequest
from google.cloud.spanner_v1._helpers import _metadata_with_prefix

_BACKUP_NAME_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/"
    r"instances/(?P<instance_id>[a-z][-a-z0-9]*)/"
    r"backups/(?P<backup_id>[a-z][a-z0-9_\-]*[a-z0-9])$"
)


class Backup(object):
    """Representation of a Cloud Spanner Backup.

    We can use a :class`Backup` to:

    * :meth:`create` the backup
    * :meth:`update` the backup
    * :meth:`delete` the backup

    :type backup_id: str
    :param backup_id: The ID of the backup.

    :type instance: :class:`~google.cloud.spanner_v1.instance.Instance`
    :param instance: The instance that owns the backup.

    :type database: str
    :param database: (Optional) The URI of the database that the backup is
                     for. Required if the create method needs to be called.

    :type expire_time: :class:`datetime.datetime`
    :param expire_time: (Optional) The expire time that will be used to
                        create the backup. Required if the create method
                        needs to be called.

    :type version_time: :class:`datetime.datetime`
    :param version_time: (Optional) The version time that was specified for
                        the externally consistent copy of the database. If
                        not present, it is the same as the `create_time` of
                        the backup.

    :type encryption_config:
        :class:`~google.cloud.spanner_admin_database_v1.types.CreateBackupEncryptionConfig`
        or :class:`dict`
    :param encryption_config:
        (Optional) Encryption configuration for the backup.
        If a dict is provided, it must be of the same form as the protobuf
        message :class:`~google.cloud.spanner_admin_database_v1.types.CreateBackupEncryptionConfig`
    """

    def __init__(
        self,
        backup_id,
        instance,
        database="",
        expire_time=None,
        version_time=None,
        encryption_config=None,
        source_backup=None,
    ):
        self.backup_id = backup_id
        self._instance = instance
        self._database = database
        self._source_backup = source_backup
        self._expire_time = expire_time
        self._create_time = None
        self._version_time = version_time
        self._size_bytes = None
        self._state = None
        self._referencing_databases = None
        self._encryption_info = None
        self._max_expire_time = None
        self._referencing_backups = None
        self._database_dialect = None
        if type(encryption_config) is dict:
            if source_backup:
                self._encryption_config = CopyBackupEncryptionConfig(
                    **encryption_config
                )
            else:
                self._encryption_config = CreateBackupEncryptionConfig(
                    **encryption_config
                )
        else:
            self._encryption_config = encryption_config

    @property
    def name(self):
        """Backup name used in requests.

        The backup name is of the form

            ``"projects/../instances/../backups/{backup_id}"``

        :rtype: str
        :returns: The backup name.
        """
        return self._instance.name + "/backups/" + self.backup_id

    @property
    def database(self):
        """Database name used in requests.

        The database name is of the form

            ``"projects/../instances/../backups/{backup_id}"``

        :rtype: str
        :returns: The database name.
        """
        return self._database

    @property
    def expire_time(self):
        """Expire time used in creation requests.

        :rtype: :class:`datetime.datetime`
        :returns: a datetime object representing the expire time of
            this backup
        """
        return self._expire_time

    @property
    def create_time(self):
        """Create time of this backup.

        :rtype: :class:`datetime.datetime`
        :returns: a datetime object representing the create time of
            this backup
        """
        return self._create_time

    @property
    def version_time(self):
        """Version time of this backup.

        :rtype: :class:`datetime.datetime`
        :returns: a datetime object representing the version time of
            this backup
        """
        return self._version_time

    @property
    def size_bytes(self):
        """Size of this backup in bytes.

        :rtype: int
        :returns: the number size of this backup measured in bytes
        """
        return self._size_bytes

    @property
    def state(self):
        """State of this backup.

        :rtype: :class:`~google.cloud.spanner_admin_database_v1.types.Backup.State`
        :returns: an enum describing the state of the backup
        """
        return self._state

    @property
    def referencing_databases(self):
        """List of databases referencing this backup.

        :rtype: list of strings
        :returns: a list of database path strings which specify the databases still
            referencing this backup
        """
        return self._referencing_databases

    @property
    def encryption_info(self):
        """Encryption info for this backup.
        :rtype: :class:`~google.cloud.spanner_admin_database_v1.types.EncryptionInfo`
        :returns: a class representing the encryption info
        """
        return self._encryption_info

    @property
    def max_expire_time(self):
        """The max allowed expiration time of the backup.
        :rtype: :class:`datetime.datetime`
        :returns: a datetime object representing the max expire time of
            this backup
        """
        return self._max_expire_time

    @property
    def referencing_backups(self):
        """The names of the destination backups being created by copying this source backup.
        :rtype: list of strings
        :returns: a list of backup path strings which specify the backups that are
            referencing this copy backup
        """
        return self._referencing_backups

    def database_dialect(self):
        """Database Dialect for this backup.
        :rtype: :class:`~google.cloud.spanner_admin_database_v1.types.DatabaseDialect`
        :returns: a class representing the dialect of this backup's database
        """
        return self._database_dialect

    @classmethod
    def from_pb(cls, backup_pb, instance):
        """Create an instance of this class from a protobuf message.

        :type backup_pb: :class:`~google.cloud.spanner_admin_database_v1.types.Backup`
        :param backup_pb: A backup protobuf object.

        :type instance: :class:`~google.cloud.spanner_v1.instance.Instance`
        :param instance: The instance that owns the backup.

        :rtype: :class:`Backup`
        :returns: The backup parsed from the protobuf response.
        :raises ValueError:
            if the backup name does not match the expected format or if
            the parsed project ID does not match the project ID on the
            instance's client, or if the parsed instance ID does not match
            the instance's ID.
        """
        match = _BACKUP_NAME_RE.match(backup_pb.name)
        if match is None:
            raise ValueError(
                "Backup protobuf name was not in the expected format.", backup_pb.name
            )
        if match.group("project") != instance._client.project:
            raise ValueError(
                "Project ID on backup does not match the project ID"
                "on the instance's client"
            )
        instance_id = match.group("instance_id")
        if instance_id != instance.instance_id:
            raise ValueError(
                "Instance ID on database does not match the instance ID"
                "on the instance"
            )
        backup_id = match.group("backup_id")
        return cls(backup_id, instance)

    def create(self):
        """Create this backup or backup copy within its instance.

        :rtype: :class:`~google.api_core.operation.Operation`
        :returns: a future used to poll the status of the create request
        :raises Conflict: if the backup already exists
        :raises NotFound: if the instance owning the backup does not exist
        :raises BadRequest: if the database or expire_time values are invalid
                            or expire_time is not set
        """
        if not self._expire_time:
            raise ValueError("expire_time not set")

        if not self._database and not self._source_backup:
            raise ValueError("database and source backup both not set")

        if (
            (
                self._encryption_config
                and self._encryption_config.kms_key_name
                and self._encryption_config.encryption_type
                != CreateBackupEncryptionConfig.EncryptionType.CUSTOMER_MANAGED_ENCRYPTION
            )
            and self._encryption_config
            and self._encryption_config.kms_key_name
            and self._encryption_config.encryption_type
            != CopyBackupEncryptionConfig.EncryptionType.CUSTOMER_MANAGED_ENCRYPTION
        ):
            raise ValueError("kms_key_name only used with CUSTOMER_MANAGED_ENCRYPTION")

        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)

        if self._source_backup:
            request = CopyBackupRequest(
                parent=self._instance.name,
                backup_id=self.backup_id,
                source_backup=self._source_backup,
                expire_time=self._expire_time,
                encryption_config=self._encryption_config,
            )

            future = api.copy_backup(
                request=request,
                metadata=metadata,
            )
            return future

        backup = BackupPB(
            database=self._database,
            expire_time=self.expire_time,
            version_time=self.version_time,
        )

        request = CreateBackupRequest(
            parent=self._instance.name,
            backup_id=self.backup_id,
            backup=backup,
            encryption_config=self._encryption_config,
        )

        future = api.create_backup(
            request=request,
            metadata=metadata,
        )
        return future

    def exists(self):
        """Test whether this backup exists.

        :rtype: bool
        :returns: True if the backup exists, else False.
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)

        try:
            api.get_backup(name=self.name, metadata=metadata)
        except NotFound:
            return False
        return True

    def reload(self):
        """Reload this backup.

        Refresh the stored backup properties.

        :raises NotFound: if the backup does not exist
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)
        pb = api.get_backup(name=self.name, metadata=metadata)
        self._database = pb.database
        self._expire_time = pb.expire_time
        self._create_time = pb.create_time
        self._version_time = pb.version_time
        self._size_bytes = pb.size_bytes
        self._state = BackupPB.State(pb.state)
        self._referencing_databases = pb.referencing_databases
        self._encryption_info = pb.encryption_info
        self._max_expire_time = pb.max_expire_time
        self._referencing_backups = pb.referencing_backups

    def update_expire_time(self, new_expire_time):
        """Update the expire time of this backup.

        :type new_expire_time: :class:`datetime.datetime`
        :param new_expire_time: the new expire time timestamp
        """
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)
        backup_update = BackupPB(
            name=self.name,
            expire_time=new_expire_time,
        )
        update_mask = {"paths": ["expire_time"]}
        api.update_backup(
            backup=backup_update, update_mask=update_mask, metadata=metadata
        )
        self._expire_time = new_expire_time

    def is_ready(self):
        """Test whether this backup is ready for use.

        :rtype: bool
        :returns: True if the backup state is READY, else False.
        """
        return self.state == BackupPB.State.READY

    def delete(self):
        """Delete this backup."""
        api = self._instance._client.database_admin_api
        metadata = _metadata_with_prefix(self.name)
        api.delete_backup(name=self.name, metadata=metadata)
