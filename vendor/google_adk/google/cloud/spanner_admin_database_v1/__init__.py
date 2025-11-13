# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
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
from google.cloud.spanner_admin_database_v1 import gapic_version as package_version

__version__ = package_version.__version__


from .services.database_admin import DatabaseAdminClient
from .services.database_admin import DatabaseAdminAsyncClient

from .types.backup import Backup
from .types.backup import BackupInfo
from .types.backup import BackupInstancePartition
from .types.backup import CopyBackupEncryptionConfig
from .types.backup import CopyBackupMetadata
from .types.backup import CopyBackupRequest
from .types.backup import CreateBackupEncryptionConfig
from .types.backup import CreateBackupMetadata
from .types.backup import CreateBackupRequest
from .types.backup import DeleteBackupRequest
from .types.backup import FullBackupSpec
from .types.backup import GetBackupRequest
from .types.backup import IncrementalBackupSpec
from .types.backup import ListBackupOperationsRequest
from .types.backup import ListBackupOperationsResponse
from .types.backup import ListBackupsRequest
from .types.backup import ListBackupsResponse
from .types.backup import UpdateBackupRequest
from .types.backup_schedule import BackupSchedule
from .types.backup_schedule import BackupScheduleSpec
from .types.backup_schedule import CreateBackupScheduleRequest
from .types.backup_schedule import CrontabSpec
from .types.backup_schedule import DeleteBackupScheduleRequest
from .types.backup_schedule import GetBackupScheduleRequest
from .types.backup_schedule import ListBackupSchedulesRequest
from .types.backup_schedule import ListBackupSchedulesResponse
from .types.backup_schedule import UpdateBackupScheduleRequest
from .types.common import EncryptionConfig
from .types.common import EncryptionInfo
from .types.common import OperationProgress
from .types.common import DatabaseDialect
from .types.spanner_database_admin import AddSplitPointsRequest
from .types.spanner_database_admin import AddSplitPointsResponse
from .types.spanner_database_admin import CreateDatabaseMetadata
from .types.spanner_database_admin import CreateDatabaseRequest
from .types.spanner_database_admin import Database
from .types.spanner_database_admin import DatabaseRole
from .types.spanner_database_admin import DdlStatementActionInfo
from .types.spanner_database_admin import DropDatabaseRequest
from .types.spanner_database_admin import GetDatabaseDdlRequest
from .types.spanner_database_admin import GetDatabaseDdlResponse
from .types.spanner_database_admin import GetDatabaseRequest
from .types.spanner_database_admin import InternalUpdateGraphOperationRequest
from .types.spanner_database_admin import InternalUpdateGraphOperationResponse
from .types.spanner_database_admin import ListDatabaseOperationsRequest
from .types.spanner_database_admin import ListDatabaseOperationsResponse
from .types.spanner_database_admin import ListDatabaseRolesRequest
from .types.spanner_database_admin import ListDatabaseRolesResponse
from .types.spanner_database_admin import ListDatabasesRequest
from .types.spanner_database_admin import ListDatabasesResponse
from .types.spanner_database_admin import OptimizeRestoredDatabaseMetadata
from .types.spanner_database_admin import RestoreDatabaseEncryptionConfig
from .types.spanner_database_admin import RestoreDatabaseMetadata
from .types.spanner_database_admin import RestoreDatabaseRequest
from .types.spanner_database_admin import RestoreInfo
from .types.spanner_database_admin import SplitPoints
from .types.spanner_database_admin import UpdateDatabaseDdlMetadata
from .types.spanner_database_admin import UpdateDatabaseDdlRequest
from .types.spanner_database_admin import UpdateDatabaseMetadata
from .types.spanner_database_admin import UpdateDatabaseRequest
from .types.spanner_database_admin import RestoreSourceType

__all__ = (
    "DatabaseAdminAsyncClient",
    "AddSplitPointsRequest",
    "AddSplitPointsResponse",
    "Backup",
    "BackupInfo",
    "BackupInstancePartition",
    "BackupSchedule",
    "BackupScheduleSpec",
    "CopyBackupEncryptionConfig",
    "CopyBackupMetadata",
    "CopyBackupRequest",
    "CreateBackupEncryptionConfig",
    "CreateBackupMetadata",
    "CreateBackupRequest",
    "CreateBackupScheduleRequest",
    "CreateDatabaseMetadata",
    "CreateDatabaseRequest",
    "CrontabSpec",
    "Database",
    "DatabaseAdminClient",
    "DatabaseDialect",
    "DatabaseRole",
    "DdlStatementActionInfo",
    "DeleteBackupRequest",
    "DeleteBackupScheduleRequest",
    "DropDatabaseRequest",
    "EncryptionConfig",
    "EncryptionInfo",
    "FullBackupSpec",
    "GetBackupRequest",
    "GetBackupScheduleRequest",
    "GetDatabaseDdlRequest",
    "GetDatabaseDdlResponse",
    "GetDatabaseRequest",
    "IncrementalBackupSpec",
    "InternalUpdateGraphOperationRequest",
    "InternalUpdateGraphOperationResponse",
    "ListBackupOperationsRequest",
    "ListBackupOperationsResponse",
    "ListBackupSchedulesRequest",
    "ListBackupSchedulesResponse",
    "ListBackupsRequest",
    "ListBackupsResponse",
    "ListDatabaseOperationsRequest",
    "ListDatabaseOperationsResponse",
    "ListDatabaseRolesRequest",
    "ListDatabaseRolesResponse",
    "ListDatabasesRequest",
    "ListDatabasesResponse",
    "OperationProgress",
    "OptimizeRestoredDatabaseMetadata",
    "RestoreDatabaseEncryptionConfig",
    "RestoreDatabaseMetadata",
    "RestoreDatabaseRequest",
    "RestoreInfo",
    "RestoreSourceType",
    "SplitPoints",
    "UpdateBackupRequest",
    "UpdateBackupScheduleRequest",
    "UpdateDatabaseDdlMetadata",
    "UpdateDatabaseDdlRequest",
    "UpdateDatabaseMetadata",
    "UpdateDatabaseRequest",
)
