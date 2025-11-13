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

from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.iam.v1 import policy_pb2 as _policy_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import message as _message
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.rpc import status_pb2 as _status_pb2

DESCRIPTOR: _descriptor.FileDescriptor

class BigQueryAuditMetadata(_message.Message):
    __slots__ = (
        "job_insertion",
        "job_change",
        "job_deletion",
        "dataset_creation",
        "dataset_change",
        "dataset_deletion",
        "table_creation",
        "table_change",
        "table_deletion",
        "table_data_read",
        "table_data_change",
        "model_deletion",
        "model_creation",
        "model_metadata_change",
        "model_data_change",
        "model_data_read",
        "routine_creation",
        "routine_change",
        "routine_deletion",
        "row_access_policy_creation",
        "row_access_policy_change",
        "row_access_policy_deletion",
        "unlink_dataset",
        "first_party_app_metadata",
    )

    class CreateDisposition(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        CREATE_DISPOSITION_UNSPECIFIED: _ClassVar[
            BigQueryAuditMetadata.CreateDisposition
        ]
        CREATE_NEVER: _ClassVar[BigQueryAuditMetadata.CreateDisposition]
        CREATE_IF_NEEDED: _ClassVar[BigQueryAuditMetadata.CreateDisposition]
    CREATE_DISPOSITION_UNSPECIFIED: BigQueryAuditMetadata.CreateDisposition
    CREATE_NEVER: BigQueryAuditMetadata.CreateDisposition
    CREATE_IF_NEEDED: BigQueryAuditMetadata.CreateDisposition

    class WriteDisposition(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        WRITE_DISPOSITION_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.WriteDisposition]
        WRITE_EMPTY: _ClassVar[BigQueryAuditMetadata.WriteDisposition]
        WRITE_TRUNCATE: _ClassVar[BigQueryAuditMetadata.WriteDisposition]
        WRITE_APPEND: _ClassVar[BigQueryAuditMetadata.WriteDisposition]
    WRITE_DISPOSITION_UNSPECIFIED: BigQueryAuditMetadata.WriteDisposition
    WRITE_EMPTY: BigQueryAuditMetadata.WriteDisposition
    WRITE_TRUNCATE: BigQueryAuditMetadata.WriteDisposition
    WRITE_APPEND: BigQueryAuditMetadata.WriteDisposition

    class OperationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        OPERATION_TYPE_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.OperationType]
        COPY: _ClassVar[BigQueryAuditMetadata.OperationType]
        SNAPSHOT: _ClassVar[BigQueryAuditMetadata.OperationType]
        RESTORE: _ClassVar[BigQueryAuditMetadata.OperationType]
    OPERATION_TYPE_UNSPECIFIED: BigQueryAuditMetadata.OperationType
    COPY: BigQueryAuditMetadata.OperationType
    SNAPSHOT: BigQueryAuditMetadata.OperationType
    RESTORE: BigQueryAuditMetadata.OperationType

    class JobState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOB_STATE_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.JobState]
        PENDING: _ClassVar[BigQueryAuditMetadata.JobState]
        RUNNING: _ClassVar[BigQueryAuditMetadata.JobState]
        DONE: _ClassVar[BigQueryAuditMetadata.JobState]
    JOB_STATE_UNSPECIFIED: BigQueryAuditMetadata.JobState
    PENDING: BigQueryAuditMetadata.JobState
    RUNNING: BigQueryAuditMetadata.JobState
    DONE: BigQueryAuditMetadata.JobState

    class QueryStatementType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        QUERY_STATEMENT_TYPE_UNSPECIFIED: _ClassVar[
            BigQueryAuditMetadata.QueryStatementType
        ]
        SELECT: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        ASSERT: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        INSERT: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        UPDATE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DELETE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        MERGE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_TABLE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_TABLE_AS_SELECT: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_VIEW: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_MODEL: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_MATERIALIZED_VIEW: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_FUNCTION: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_TABLE_FUNCTION: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_PROCEDURE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_ROW_ACCESS_POLICY: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_SCHEMA: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_SNAPSHOT_TABLE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_TABLE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_EXTERNAL_TABLE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_VIEW: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_MODEL: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_MATERIALIZED_VIEW: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_FUNCTION: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_PROCEDURE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_SCHEMA: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_ROW_ACCESS_POLICY: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        DROP_SNAPSHOT_TABLE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        ALTER_TABLE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        ALTER_VIEW: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        ALTER_MATERIALIZED_VIEW: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        ALTER_SCHEMA: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        SCRIPT: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        TRUNCATE_TABLE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CREATE_EXTERNAL_TABLE: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        EXPORT_DATA: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
        CALL: _ClassVar[BigQueryAuditMetadata.QueryStatementType]
    QUERY_STATEMENT_TYPE_UNSPECIFIED: BigQueryAuditMetadata.QueryStatementType
    SELECT: BigQueryAuditMetadata.QueryStatementType
    ASSERT: BigQueryAuditMetadata.QueryStatementType
    INSERT: BigQueryAuditMetadata.QueryStatementType
    UPDATE: BigQueryAuditMetadata.QueryStatementType
    DELETE: BigQueryAuditMetadata.QueryStatementType
    MERGE: BigQueryAuditMetadata.QueryStatementType
    CREATE_TABLE: BigQueryAuditMetadata.QueryStatementType
    CREATE_TABLE_AS_SELECT: BigQueryAuditMetadata.QueryStatementType
    CREATE_VIEW: BigQueryAuditMetadata.QueryStatementType
    CREATE_MODEL: BigQueryAuditMetadata.QueryStatementType
    CREATE_MATERIALIZED_VIEW: BigQueryAuditMetadata.QueryStatementType
    CREATE_FUNCTION: BigQueryAuditMetadata.QueryStatementType
    CREATE_TABLE_FUNCTION: BigQueryAuditMetadata.QueryStatementType
    CREATE_PROCEDURE: BigQueryAuditMetadata.QueryStatementType
    CREATE_ROW_ACCESS_POLICY: BigQueryAuditMetadata.QueryStatementType
    CREATE_SCHEMA: BigQueryAuditMetadata.QueryStatementType
    CREATE_SNAPSHOT_TABLE: BigQueryAuditMetadata.QueryStatementType
    DROP_TABLE: BigQueryAuditMetadata.QueryStatementType
    DROP_EXTERNAL_TABLE: BigQueryAuditMetadata.QueryStatementType
    DROP_VIEW: BigQueryAuditMetadata.QueryStatementType
    DROP_MODEL: BigQueryAuditMetadata.QueryStatementType
    DROP_MATERIALIZED_VIEW: BigQueryAuditMetadata.QueryStatementType
    DROP_FUNCTION: BigQueryAuditMetadata.QueryStatementType
    DROP_PROCEDURE: BigQueryAuditMetadata.QueryStatementType
    DROP_SCHEMA: BigQueryAuditMetadata.QueryStatementType
    DROP_ROW_ACCESS_POLICY: BigQueryAuditMetadata.QueryStatementType
    DROP_SNAPSHOT_TABLE: BigQueryAuditMetadata.QueryStatementType
    ALTER_TABLE: BigQueryAuditMetadata.QueryStatementType
    ALTER_VIEW: BigQueryAuditMetadata.QueryStatementType
    ALTER_MATERIALIZED_VIEW: BigQueryAuditMetadata.QueryStatementType
    ALTER_SCHEMA: BigQueryAuditMetadata.QueryStatementType
    SCRIPT: BigQueryAuditMetadata.QueryStatementType
    TRUNCATE_TABLE: BigQueryAuditMetadata.QueryStatementType
    CREATE_EXTERNAL_TABLE: BigQueryAuditMetadata.QueryStatementType
    EXPORT_DATA: BigQueryAuditMetadata.QueryStatementType
    CALL: BigQueryAuditMetadata.QueryStatementType

    class JobInsertion(_message.Message):
        __slots__ = ("job", "reason")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.JobInsertion.Reason]
            JOB_INSERT_REQUEST: _ClassVar[BigQueryAuditMetadata.JobInsertion.Reason]
            QUERY_REQUEST: _ClassVar[BigQueryAuditMetadata.JobInsertion.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.JobInsertion.Reason
        JOB_INSERT_REQUEST: BigQueryAuditMetadata.JobInsertion.Reason
        QUERY_REQUEST: BigQueryAuditMetadata.JobInsertion.Reason
        JOB_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        job: BigQueryAuditMetadata.Job
        reason: BigQueryAuditMetadata.JobInsertion.Reason
        def __init__(
            self,
            job: _Optional[_Union[BigQueryAuditMetadata.Job, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.JobInsertion.Reason, str]
            ] = ...,
        ) -> None: ...

    class JobChange(_message.Message):
        __slots__ = ("before", "after", "job")
        BEFORE_FIELD_NUMBER: _ClassVar[int]
        AFTER_FIELD_NUMBER: _ClassVar[int]
        JOB_FIELD_NUMBER: _ClassVar[int]
        before: BigQueryAuditMetadata.JobState
        after: BigQueryAuditMetadata.JobState
        job: BigQueryAuditMetadata.Job
        def __init__(
            self,
            before: _Optional[_Union[BigQueryAuditMetadata.JobState, str]] = ...,
            after: _Optional[_Union[BigQueryAuditMetadata.JobState, str]] = ...,
            job: _Optional[_Union[BigQueryAuditMetadata.Job, _Mapping]] = ...,
        ) -> None: ...

    class JobDeletion(_message.Message):
        __slots__ = ("job_name", "reason")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.JobDeletion.Reason]
            JOB_DELETE_REQUEST: _ClassVar[BigQueryAuditMetadata.JobDeletion.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.JobDeletion.Reason
        JOB_DELETE_REQUEST: BigQueryAuditMetadata.JobDeletion.Reason
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        job_name: str
        reason: BigQueryAuditMetadata.JobDeletion.Reason
        def __init__(
            self,
            job_name: _Optional[str] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.JobDeletion.Reason, str]
            ] = ...,
        ) -> None: ...

    class DatasetCreation(_message.Message):
        __slots__ = ("dataset", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.DatasetCreation.Reason]
            CREATE: _ClassVar[BigQueryAuditMetadata.DatasetCreation.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.DatasetCreation.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.DatasetCreation.Reason
        CREATE: BigQueryAuditMetadata.DatasetCreation.Reason
        QUERY: BigQueryAuditMetadata.DatasetCreation.Reason
        DATASET_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        dataset: BigQueryAuditMetadata.Dataset
        reason: BigQueryAuditMetadata.DatasetCreation.Reason
        job_name: str
        def __init__(
            self,
            dataset: _Optional[_Union[BigQueryAuditMetadata.Dataset, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.DatasetCreation.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class DatasetChange(_message.Message):
        __slots__ = ("dataset", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.DatasetChange.Reason]
            UPDATE: _ClassVar[BigQueryAuditMetadata.DatasetChange.Reason]
            SET_IAM_POLICY: _ClassVar[BigQueryAuditMetadata.DatasetChange.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.DatasetChange.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.DatasetChange.Reason
        UPDATE: BigQueryAuditMetadata.DatasetChange.Reason
        SET_IAM_POLICY: BigQueryAuditMetadata.DatasetChange.Reason
        QUERY: BigQueryAuditMetadata.DatasetChange.Reason
        DATASET_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        dataset: BigQueryAuditMetadata.Dataset
        reason: BigQueryAuditMetadata.DatasetChange.Reason
        job_name: str
        def __init__(
            self,
            dataset: _Optional[_Union[BigQueryAuditMetadata.Dataset, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.DatasetChange.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class DatasetDeletion(_message.Message):
        __slots__ = ("reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.DatasetDeletion.Reason]
            DELETE: _ClassVar[BigQueryAuditMetadata.DatasetDeletion.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.DatasetDeletion.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.DatasetDeletion.Reason
        DELETE: BigQueryAuditMetadata.DatasetDeletion.Reason
        QUERY: BigQueryAuditMetadata.DatasetDeletion.Reason
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        reason: BigQueryAuditMetadata.DatasetDeletion.Reason
        job_name: str
        def __init__(
            self,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.DatasetDeletion.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class TableCreation(_message.Message):
        __slots__ = ("table", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.TableCreation.Reason]
            JOB: _ClassVar[BigQueryAuditMetadata.TableCreation.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.TableCreation.Reason]
            TABLE_INSERT_REQUEST: _ClassVar[BigQueryAuditMetadata.TableCreation.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.TableCreation.Reason
        JOB: BigQueryAuditMetadata.TableCreation.Reason
        QUERY: BigQueryAuditMetadata.TableCreation.Reason
        TABLE_INSERT_REQUEST: BigQueryAuditMetadata.TableCreation.Reason
        TABLE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        table: BigQueryAuditMetadata.Table
        reason: BigQueryAuditMetadata.TableCreation.Reason
        job_name: str
        def __init__(
            self,
            table: _Optional[_Union[BigQueryAuditMetadata.Table, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.TableCreation.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class ModelCreation(_message.Message):
        __slots__ = ("model", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.ModelCreation.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.ModelCreation.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.ModelCreation.Reason
        QUERY: BigQueryAuditMetadata.ModelCreation.Reason
        MODEL_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        model: BigQueryAuditMetadata.Model
        reason: BigQueryAuditMetadata.ModelCreation.Reason
        job_name: str
        def __init__(
            self,
            model: _Optional[_Union[BigQueryAuditMetadata.Model, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.ModelCreation.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class RoutineCreation(_message.Message):
        __slots__ = ("routine", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.RoutineCreation.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.RoutineCreation.Reason]
            ROUTINE_INSERT_REQUEST: _ClassVar[
                BigQueryAuditMetadata.RoutineCreation.Reason
            ]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.RoutineCreation.Reason
        QUERY: BigQueryAuditMetadata.RoutineCreation.Reason
        ROUTINE_INSERT_REQUEST: BigQueryAuditMetadata.RoutineCreation.Reason
        ROUTINE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        routine: BigQueryAuditMetadata.Routine
        reason: BigQueryAuditMetadata.RoutineCreation.Reason
        job_name: str
        def __init__(
            self,
            routine: _Optional[_Union[BigQueryAuditMetadata.Routine, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.RoutineCreation.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class TableDataRead(_message.Message):
        __slots__ = (
            "fields",
            "fields_truncated",
            "policy_tags",
            "policy_tags_truncated",
            "reason",
            "job_name",
            "session_name",
        )

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.TableDataRead.Reason]
            JOB: _ClassVar[BigQueryAuditMetadata.TableDataRead.Reason]
            TABLEDATA_LIST_REQUEST: _ClassVar[
                BigQueryAuditMetadata.TableDataRead.Reason
            ]
            GET_QUERY_RESULTS_REQUEST: _ClassVar[
                BigQueryAuditMetadata.TableDataRead.Reason
            ]
            QUERY_REQUEST: _ClassVar[BigQueryAuditMetadata.TableDataRead.Reason]
            CREATE_READ_SESSION: _ClassVar[BigQueryAuditMetadata.TableDataRead.Reason]
            MATERIALIZED_VIEW_REFRESH: _ClassVar[
                BigQueryAuditMetadata.TableDataRead.Reason
            ]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.TableDataRead.Reason
        JOB: BigQueryAuditMetadata.TableDataRead.Reason
        TABLEDATA_LIST_REQUEST: BigQueryAuditMetadata.TableDataRead.Reason
        GET_QUERY_RESULTS_REQUEST: BigQueryAuditMetadata.TableDataRead.Reason
        QUERY_REQUEST: BigQueryAuditMetadata.TableDataRead.Reason
        CREATE_READ_SESSION: BigQueryAuditMetadata.TableDataRead.Reason
        MATERIALIZED_VIEW_REFRESH: BigQueryAuditMetadata.TableDataRead.Reason
        FIELDS_FIELD_NUMBER: _ClassVar[int]
        FIELDS_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
        POLICY_TAGS_FIELD_NUMBER: _ClassVar[int]
        POLICY_TAGS_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        SESSION_NAME_FIELD_NUMBER: _ClassVar[int]
        fields: _containers.RepeatedScalarFieldContainer[str]
        fields_truncated: bool
        policy_tags: _containers.RepeatedScalarFieldContainer[str]
        policy_tags_truncated: bool
        reason: BigQueryAuditMetadata.TableDataRead.Reason
        job_name: str
        session_name: str
        def __init__(
            self,
            fields: _Optional[_Iterable[str]] = ...,
            fields_truncated: bool = ...,
            policy_tags: _Optional[_Iterable[str]] = ...,
            policy_tags_truncated: bool = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.TableDataRead.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
            session_name: _Optional[str] = ...,
        ) -> None: ...

    class TableChange(_message.Message):
        __slots__ = ("table", "truncated", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.TableChange.Reason]
            TABLE_UPDATE_REQUEST: _ClassVar[BigQueryAuditMetadata.TableChange.Reason]
            JOB: _ClassVar[BigQueryAuditMetadata.TableChange.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.TableChange.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.TableChange.Reason
        TABLE_UPDATE_REQUEST: BigQueryAuditMetadata.TableChange.Reason
        JOB: BigQueryAuditMetadata.TableChange.Reason
        QUERY: BigQueryAuditMetadata.TableChange.Reason
        TABLE_FIELD_NUMBER: _ClassVar[int]
        TRUNCATED_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        table: BigQueryAuditMetadata.Table
        truncated: bool
        reason: BigQueryAuditMetadata.TableChange.Reason
        job_name: str
        def __init__(
            self,
            table: _Optional[_Union[BigQueryAuditMetadata.Table, _Mapping]] = ...,
            truncated: bool = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.TableChange.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class ModelMetadataChange(_message.Message):
        __slots__ = ("model", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[
                BigQueryAuditMetadata.ModelMetadataChange.Reason
            ]
            MODEL_PATCH_REQUEST: _ClassVar[
                BigQueryAuditMetadata.ModelMetadataChange.Reason
            ]
            QUERY: _ClassVar[BigQueryAuditMetadata.ModelMetadataChange.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.ModelMetadataChange.Reason
        MODEL_PATCH_REQUEST: BigQueryAuditMetadata.ModelMetadataChange.Reason
        QUERY: BigQueryAuditMetadata.ModelMetadataChange.Reason
        MODEL_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        model: BigQueryAuditMetadata.Model
        reason: BigQueryAuditMetadata.ModelMetadataChange.Reason
        job_name: str
        def __init__(
            self,
            model: _Optional[_Union[BigQueryAuditMetadata.Model, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.ModelMetadataChange.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class RoutineChange(_message.Message):
        __slots__ = ("routine", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.RoutineChange.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.RoutineChange.Reason]
            ROUTINE_UPDATE_REQUEST: _ClassVar[
                BigQueryAuditMetadata.RoutineChange.Reason
            ]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.RoutineChange.Reason
        QUERY: BigQueryAuditMetadata.RoutineChange.Reason
        ROUTINE_UPDATE_REQUEST: BigQueryAuditMetadata.RoutineChange.Reason
        ROUTINE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        routine: BigQueryAuditMetadata.Routine
        reason: BigQueryAuditMetadata.RoutineChange.Reason
        job_name: str
        def __init__(
            self,
            routine: _Optional[_Union[BigQueryAuditMetadata.Routine, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.RoutineChange.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class TableDataChange(_message.Message):
        __slots__ = (
            "deleted_rows_count",
            "inserted_rows_count",
            "truncated",
            "reason",
            "job_name",
            "stream_name",
        )

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.TableDataChange.Reason]
            JOB: _ClassVar[BigQueryAuditMetadata.TableDataChange.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.TableDataChange.Reason]
            MATERIALIZED_VIEW_REFRESH: _ClassVar[
                BigQueryAuditMetadata.TableDataChange.Reason
            ]
            WRITE_API: _ClassVar[BigQueryAuditMetadata.TableDataChange.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.TableDataChange.Reason
        JOB: BigQueryAuditMetadata.TableDataChange.Reason
        QUERY: BigQueryAuditMetadata.TableDataChange.Reason
        MATERIALIZED_VIEW_REFRESH: BigQueryAuditMetadata.TableDataChange.Reason
        WRITE_API: BigQueryAuditMetadata.TableDataChange.Reason
        DELETED_ROWS_COUNT_FIELD_NUMBER: _ClassVar[int]
        INSERTED_ROWS_COUNT_FIELD_NUMBER: _ClassVar[int]
        TRUNCATED_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        STREAM_NAME_FIELD_NUMBER: _ClassVar[int]
        deleted_rows_count: int
        inserted_rows_count: int
        truncated: bool
        reason: BigQueryAuditMetadata.TableDataChange.Reason
        job_name: str
        stream_name: str
        def __init__(
            self,
            deleted_rows_count: _Optional[int] = ...,
            inserted_rows_count: _Optional[int] = ...,
            truncated: bool = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.TableDataChange.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
            stream_name: _Optional[str] = ...,
        ) -> None: ...

    class ModelDataChange(_message.Message):
        __slots__ = ("reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.ModelDataChange.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.ModelDataChange.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.ModelDataChange.Reason
        QUERY: BigQueryAuditMetadata.ModelDataChange.Reason
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        reason: BigQueryAuditMetadata.ModelDataChange.Reason
        job_name: str
        def __init__(
            self,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.ModelDataChange.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class ModelDataRead(_message.Message):
        __slots__ = ("reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.ModelDataRead.Reason]
            JOB: _ClassVar[BigQueryAuditMetadata.ModelDataRead.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.ModelDataRead.Reason
        JOB: BigQueryAuditMetadata.ModelDataRead.Reason
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        reason: BigQueryAuditMetadata.ModelDataRead.Reason
        job_name: str
        def __init__(
            self,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.ModelDataRead.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class TableDeletion(_message.Message):
        __slots__ = ("reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.TableDeletion.Reason]
            TABLE_DELETE_REQUEST: _ClassVar[BigQueryAuditMetadata.TableDeletion.Reason]
            EXPIRED: _ClassVar[BigQueryAuditMetadata.TableDeletion.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.TableDeletion.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.TableDeletion.Reason
        TABLE_DELETE_REQUEST: BigQueryAuditMetadata.TableDeletion.Reason
        EXPIRED: BigQueryAuditMetadata.TableDeletion.Reason
        QUERY: BigQueryAuditMetadata.TableDeletion.Reason
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        reason: BigQueryAuditMetadata.TableDeletion.Reason
        job_name: str
        def __init__(
            self,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.TableDeletion.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class ModelDeletion(_message.Message):
        __slots__ = ("reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.ModelDeletion.Reason]
            MODEL_DELETE_REQUEST: _ClassVar[BigQueryAuditMetadata.ModelDeletion.Reason]
            EXPIRED: _ClassVar[BigQueryAuditMetadata.ModelDeletion.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.ModelDeletion.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.ModelDeletion.Reason
        MODEL_DELETE_REQUEST: BigQueryAuditMetadata.ModelDeletion.Reason
        EXPIRED: BigQueryAuditMetadata.ModelDeletion.Reason
        QUERY: BigQueryAuditMetadata.ModelDeletion.Reason
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        reason: BigQueryAuditMetadata.ModelDeletion.Reason
        job_name: str
        def __init__(
            self,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.ModelDeletion.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class RoutineDeletion(_message.Message):
        __slots__ = ("routine", "reason", "job_name")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.RoutineDeletion.Reason]
            QUERY: _ClassVar[BigQueryAuditMetadata.RoutineDeletion.Reason]
            ROUTINE_DELETE_REQUEST: _ClassVar[
                BigQueryAuditMetadata.RoutineDeletion.Reason
            ]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.RoutineDeletion.Reason
        QUERY: BigQueryAuditMetadata.RoutineDeletion.Reason
        ROUTINE_DELETE_REQUEST: BigQueryAuditMetadata.RoutineDeletion.Reason
        ROUTINE_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        routine: BigQueryAuditMetadata.Routine
        reason: BigQueryAuditMetadata.RoutineDeletion.Reason
        job_name: str
        def __init__(
            self,
            routine: _Optional[_Union[BigQueryAuditMetadata.Routine, _Mapping]] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.RoutineDeletion.Reason, str]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class RowAccessPolicyCreation(_message.Message):
        __slots__ = ("row_access_policy", "job_name")
        ROW_ACCESS_POLICY_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        row_access_policy: BigQueryAuditMetadata.RowAccessPolicy
        job_name: str
        def __init__(
            self,
            row_access_policy: _Optional[
                _Union[BigQueryAuditMetadata.RowAccessPolicy, _Mapping]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class RowAccessPolicyChange(_message.Message):
        __slots__ = ("row_access_policy", "job_name")
        ROW_ACCESS_POLICY_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        row_access_policy: BigQueryAuditMetadata.RowAccessPolicy
        job_name: str
        def __init__(
            self,
            row_access_policy: _Optional[
                _Union[BigQueryAuditMetadata.RowAccessPolicy, _Mapping]
            ] = ...,
            job_name: _Optional[str] = ...,
        ) -> None: ...

    class RowAccessPolicyDeletion(_message.Message):
        __slots__ = (
            "row_access_policies",
            "job_name",
            "all_row_access_policies_dropped",
        )
        ROW_ACCESS_POLICIES_FIELD_NUMBER: _ClassVar[int]
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        ALL_ROW_ACCESS_POLICIES_DROPPED_FIELD_NUMBER: _ClassVar[int]
        row_access_policies: _containers.RepeatedCompositeFieldContainer[
            BigQueryAuditMetadata.RowAccessPolicy
        ]
        job_name: str
        all_row_access_policies_dropped: bool
        def __init__(
            self,
            row_access_policies: _Optional[
                _Iterable[_Union[BigQueryAuditMetadata.RowAccessPolicy, _Mapping]]
            ] = ...,
            job_name: _Optional[str] = ...,
            all_row_access_policies_dropped: bool = ...,
        ) -> None: ...

    class UnlinkDataset(_message.Message):
        __slots__ = ("linked_dataset", "source_dataset", "reason")

        class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            REASON_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.UnlinkDataset.Reason]
            UNLINK_API: _ClassVar[BigQueryAuditMetadata.UnlinkDataset.Reason]
        REASON_UNSPECIFIED: BigQueryAuditMetadata.UnlinkDataset.Reason
        UNLINK_API: BigQueryAuditMetadata.UnlinkDataset.Reason
        LINKED_DATASET_FIELD_NUMBER: _ClassVar[int]
        SOURCE_DATASET_FIELD_NUMBER: _ClassVar[int]
        REASON_FIELD_NUMBER: _ClassVar[int]
        linked_dataset: str
        source_dataset: str
        reason: BigQueryAuditMetadata.UnlinkDataset.Reason
        def __init__(
            self,
            linked_dataset: _Optional[str] = ...,
            source_dataset: _Optional[str] = ...,
            reason: _Optional[
                _Union[BigQueryAuditMetadata.UnlinkDataset.Reason, str]
            ] = ...,
        ) -> None: ...

    class Job(_message.Message):
        __slots__ = ("job_name", "job_config", "job_status", "job_stats")
        JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        JOB_CONFIG_FIELD_NUMBER: _ClassVar[int]
        JOB_STATUS_FIELD_NUMBER: _ClassVar[int]
        JOB_STATS_FIELD_NUMBER: _ClassVar[int]
        job_name: str
        job_config: BigQueryAuditMetadata.JobConfig
        job_status: BigQueryAuditMetadata.JobStatus
        job_stats: BigQueryAuditMetadata.JobStats
        def __init__(
            self,
            job_name: _Optional[str] = ...,
            job_config: _Optional[
                _Union[BigQueryAuditMetadata.JobConfig, _Mapping]
            ] = ...,
            job_status: _Optional[
                _Union[BigQueryAuditMetadata.JobStatus, _Mapping]
            ] = ...,
            job_stats: _Optional[
                _Union[BigQueryAuditMetadata.JobStats, _Mapping]
            ] = ...,
        ) -> None: ...

    class JobConfig(_message.Message):
        __slots__ = (
            "type",
            "query_config",
            "load_config",
            "extract_config",
            "table_copy_config",
            "labels",
        )

        class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            TYPE_UNSPECIFIED: _ClassVar[BigQueryAuditMetadata.JobConfig.Type]
            QUERY: _ClassVar[BigQueryAuditMetadata.JobConfig.Type]
            COPY: _ClassVar[BigQueryAuditMetadata.JobConfig.Type]
            EXPORT: _ClassVar[BigQueryAuditMetadata.JobConfig.Type]
            IMPORT: _ClassVar[BigQueryAuditMetadata.JobConfig.Type]
        TYPE_UNSPECIFIED: BigQueryAuditMetadata.JobConfig.Type
        QUERY: BigQueryAuditMetadata.JobConfig.Type
        COPY: BigQueryAuditMetadata.JobConfig.Type
        EXPORT: BigQueryAuditMetadata.JobConfig.Type
        IMPORT: BigQueryAuditMetadata.JobConfig.Type

        class Query(_message.Message):
            __slots__ = (
                "query",
                "query_truncated",
                "destination_table",
                "create_disposition",
                "write_disposition",
                "default_dataset",
                "table_definitions",
                "priority",
                "destination_table_encryption",
                "statement_type",
            )

            class Priority(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
                __slots__ = ()
                PRIORITY_UNSPECIFIED: _ClassVar[
                    BigQueryAuditMetadata.JobConfig.Query.Priority
                ]
                QUERY_INTERACTIVE: _ClassVar[
                    BigQueryAuditMetadata.JobConfig.Query.Priority
                ]
                QUERY_BATCH: _ClassVar[BigQueryAuditMetadata.JobConfig.Query.Priority]
            PRIORITY_UNSPECIFIED: BigQueryAuditMetadata.JobConfig.Query.Priority
            QUERY_INTERACTIVE: BigQueryAuditMetadata.JobConfig.Query.Priority
            QUERY_BATCH: BigQueryAuditMetadata.JobConfig.Query.Priority
            QUERY_FIELD_NUMBER: _ClassVar[int]
            QUERY_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
            DESTINATION_TABLE_FIELD_NUMBER: _ClassVar[int]
            CREATE_DISPOSITION_FIELD_NUMBER: _ClassVar[int]
            WRITE_DISPOSITION_FIELD_NUMBER: _ClassVar[int]
            DEFAULT_DATASET_FIELD_NUMBER: _ClassVar[int]
            TABLE_DEFINITIONS_FIELD_NUMBER: _ClassVar[int]
            PRIORITY_FIELD_NUMBER: _ClassVar[int]
            DESTINATION_TABLE_ENCRYPTION_FIELD_NUMBER: _ClassVar[int]
            STATEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
            query: str
            query_truncated: bool
            destination_table: str
            create_disposition: BigQueryAuditMetadata.CreateDisposition
            write_disposition: BigQueryAuditMetadata.WriteDisposition
            default_dataset: str
            table_definitions: _containers.RepeatedCompositeFieldContainer[
                BigQueryAuditMetadata.TableDefinition
            ]
            priority: BigQueryAuditMetadata.JobConfig.Query.Priority
            destination_table_encryption: BigQueryAuditMetadata.EncryptionInfo
            statement_type: BigQueryAuditMetadata.QueryStatementType
            def __init__(
                self,
                query: _Optional[str] = ...,
                query_truncated: bool = ...,
                destination_table: _Optional[str] = ...,
                create_disposition: _Optional[
                    _Union[BigQueryAuditMetadata.CreateDisposition, str]
                ] = ...,
                write_disposition: _Optional[
                    _Union[BigQueryAuditMetadata.WriteDisposition, str]
                ] = ...,
                default_dataset: _Optional[str] = ...,
                table_definitions: _Optional[
                    _Iterable[_Union[BigQueryAuditMetadata.TableDefinition, _Mapping]]
                ] = ...,
                priority: _Optional[
                    _Union[BigQueryAuditMetadata.JobConfig.Query.Priority, str]
                ] = ...,
                destination_table_encryption: _Optional[
                    _Union[BigQueryAuditMetadata.EncryptionInfo, _Mapping]
                ] = ...,
                statement_type: _Optional[
                    _Union[BigQueryAuditMetadata.QueryStatementType, str]
                ] = ...,
            ) -> None: ...

        class Load(_message.Message):
            __slots__ = (
                "source_uris",
                "source_uris_truncated",
                "schema_json",
                "schema_json_truncated",
                "destination_table",
                "create_disposition",
                "write_disposition",
                "destination_table_encryption",
            )
            SOURCE_URIS_FIELD_NUMBER: _ClassVar[int]
            SOURCE_URIS_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
            SCHEMA_JSON_FIELD_NUMBER: _ClassVar[int]
            SCHEMA_JSON_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
            DESTINATION_TABLE_FIELD_NUMBER: _ClassVar[int]
            CREATE_DISPOSITION_FIELD_NUMBER: _ClassVar[int]
            WRITE_DISPOSITION_FIELD_NUMBER: _ClassVar[int]
            DESTINATION_TABLE_ENCRYPTION_FIELD_NUMBER: _ClassVar[int]
            source_uris: _containers.RepeatedScalarFieldContainer[str]
            source_uris_truncated: bool
            schema_json: str
            schema_json_truncated: bool
            destination_table: str
            create_disposition: BigQueryAuditMetadata.CreateDisposition
            write_disposition: BigQueryAuditMetadata.WriteDisposition
            destination_table_encryption: BigQueryAuditMetadata.EncryptionInfo
            def __init__(
                self,
                source_uris: _Optional[_Iterable[str]] = ...,
                source_uris_truncated: bool = ...,
                schema_json: _Optional[str] = ...,
                schema_json_truncated: bool = ...,
                destination_table: _Optional[str] = ...,
                create_disposition: _Optional[
                    _Union[BigQueryAuditMetadata.CreateDisposition, str]
                ] = ...,
                write_disposition: _Optional[
                    _Union[BigQueryAuditMetadata.WriteDisposition, str]
                ] = ...,
                destination_table_encryption: _Optional[
                    _Union[BigQueryAuditMetadata.EncryptionInfo, _Mapping]
                ] = ...,
            ) -> None: ...

        class Extract(_message.Message):
            __slots__ = (
                "destination_uris",
                "destination_uris_truncated",
                "source_table",
                "source_model",
            )
            DESTINATION_URIS_FIELD_NUMBER: _ClassVar[int]
            DESTINATION_URIS_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
            SOURCE_TABLE_FIELD_NUMBER: _ClassVar[int]
            SOURCE_MODEL_FIELD_NUMBER: _ClassVar[int]
            destination_uris: _containers.RepeatedScalarFieldContainer[str]
            destination_uris_truncated: bool
            source_table: str
            source_model: str
            def __init__(
                self,
                destination_uris: _Optional[_Iterable[str]] = ...,
                destination_uris_truncated: bool = ...,
                source_table: _Optional[str] = ...,
                source_model: _Optional[str] = ...,
            ) -> None: ...

        class TableCopy(_message.Message):
            __slots__ = (
                "source_tables",
                "source_tables_truncated",
                "destination_table",
                "create_disposition",
                "write_disposition",
                "destination_table_encryption",
                "operation_type",
                "destination_expiration_time",
            )
            SOURCE_TABLES_FIELD_NUMBER: _ClassVar[int]
            SOURCE_TABLES_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
            DESTINATION_TABLE_FIELD_NUMBER: _ClassVar[int]
            CREATE_DISPOSITION_FIELD_NUMBER: _ClassVar[int]
            WRITE_DISPOSITION_FIELD_NUMBER: _ClassVar[int]
            DESTINATION_TABLE_ENCRYPTION_FIELD_NUMBER: _ClassVar[int]
            OPERATION_TYPE_FIELD_NUMBER: _ClassVar[int]
            DESTINATION_EXPIRATION_TIME_FIELD_NUMBER: _ClassVar[int]
            source_tables: _containers.RepeatedScalarFieldContainer[str]
            source_tables_truncated: bool
            destination_table: str
            create_disposition: BigQueryAuditMetadata.CreateDisposition
            write_disposition: BigQueryAuditMetadata.WriteDisposition
            destination_table_encryption: BigQueryAuditMetadata.EncryptionInfo
            operation_type: BigQueryAuditMetadata.OperationType
            destination_expiration_time: _timestamp_pb2.Timestamp
            def __init__(
                self,
                source_tables: _Optional[_Iterable[str]] = ...,
                source_tables_truncated: bool = ...,
                destination_table: _Optional[str] = ...,
                create_disposition: _Optional[
                    _Union[BigQueryAuditMetadata.CreateDisposition, str]
                ] = ...,
                write_disposition: _Optional[
                    _Union[BigQueryAuditMetadata.WriteDisposition, str]
                ] = ...,
                destination_table_encryption: _Optional[
                    _Union[BigQueryAuditMetadata.EncryptionInfo, _Mapping]
                ] = ...,
                operation_type: _Optional[
                    _Union[BigQueryAuditMetadata.OperationType, str]
                ] = ...,
                destination_expiration_time: _Optional[
                    _Union[_timestamp_pb2.Timestamp, _Mapping]
                ] = ...,
            ) -> None: ...

        class LabelsEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(
                self, key: _Optional[str] = ..., value: _Optional[str] = ...
            ) -> None: ...
        TYPE_FIELD_NUMBER: _ClassVar[int]
        QUERY_CONFIG_FIELD_NUMBER: _ClassVar[int]
        LOAD_CONFIG_FIELD_NUMBER: _ClassVar[int]
        EXTRACT_CONFIG_FIELD_NUMBER: _ClassVar[int]
        TABLE_COPY_CONFIG_FIELD_NUMBER: _ClassVar[int]
        LABELS_FIELD_NUMBER: _ClassVar[int]
        type: BigQueryAuditMetadata.JobConfig.Type
        query_config: BigQueryAuditMetadata.JobConfig.Query
        load_config: BigQueryAuditMetadata.JobConfig.Load
        extract_config: BigQueryAuditMetadata.JobConfig.Extract
        table_copy_config: BigQueryAuditMetadata.JobConfig.TableCopy
        labels: _containers.ScalarMap[str, str]
        def __init__(
            self,
            type: _Optional[_Union[BigQueryAuditMetadata.JobConfig.Type, str]] = ...,
            query_config: _Optional[
                _Union[BigQueryAuditMetadata.JobConfig.Query, _Mapping]
            ] = ...,
            load_config: _Optional[
                _Union[BigQueryAuditMetadata.JobConfig.Load, _Mapping]
            ] = ...,
            extract_config: _Optional[
                _Union[BigQueryAuditMetadata.JobConfig.Extract, _Mapping]
            ] = ...,
            table_copy_config: _Optional[
                _Union[BigQueryAuditMetadata.JobConfig.TableCopy, _Mapping]
            ] = ...,
            labels: _Optional[_Mapping[str, str]] = ...,
        ) -> None: ...

    class TableDefinition(_message.Message):
        __slots__ = ("name", "source_uris")
        NAME_FIELD_NUMBER: _ClassVar[int]
        SOURCE_URIS_FIELD_NUMBER: _ClassVar[int]
        name: str
        source_uris: _containers.RepeatedScalarFieldContainer[str]
        def __init__(
            self,
            name: _Optional[str] = ...,
            source_uris: _Optional[_Iterable[str]] = ...,
        ) -> None: ...

    class JobStatus(_message.Message):
        __slots__ = ("job_state", "error_result", "errors")
        JOB_STATE_FIELD_NUMBER: _ClassVar[int]
        ERROR_RESULT_FIELD_NUMBER: _ClassVar[int]
        ERRORS_FIELD_NUMBER: _ClassVar[int]
        job_state: BigQueryAuditMetadata.JobState
        error_result: _status_pb2.Status
        errors: _containers.RepeatedCompositeFieldContainer[_status_pb2.Status]
        def __init__(
            self,
            job_state: _Optional[_Union[BigQueryAuditMetadata.JobState, str]] = ...,
            error_result: _Optional[_Union[_status_pb2.Status, _Mapping]] = ...,
            errors: _Optional[_Iterable[_Union[_status_pb2.Status, _Mapping]]] = ...,
        ) -> None: ...

    class JobStats(_message.Message):
        __slots__ = (
            "create_time",
            "start_time",
            "end_time",
            "query_stats",
            "load_stats",
            "extract_stats",
            "total_slot_ms",
            "reservation_usage",
            "reservation",
            "parent_job_name",
        )

        class Query(_message.Message):
            __slots__ = (
                "total_processed_bytes",
                "total_billed_bytes",
                "billing_tier",
                "referenced_tables",
                "referenced_views",
                "referenced_routines",
                "output_row_count",
                "cache_hit",
            )
            TOTAL_PROCESSED_BYTES_FIELD_NUMBER: _ClassVar[int]
            TOTAL_BILLED_BYTES_FIELD_NUMBER: _ClassVar[int]
            BILLING_TIER_FIELD_NUMBER: _ClassVar[int]
            REFERENCED_TABLES_FIELD_NUMBER: _ClassVar[int]
            REFERENCED_VIEWS_FIELD_NUMBER: _ClassVar[int]
            REFERENCED_ROUTINES_FIELD_NUMBER: _ClassVar[int]
            OUTPUT_ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
            CACHE_HIT_FIELD_NUMBER: _ClassVar[int]
            total_processed_bytes: int
            total_billed_bytes: int
            billing_tier: int
            referenced_tables: _containers.RepeatedScalarFieldContainer[str]
            referenced_views: _containers.RepeatedScalarFieldContainer[str]
            referenced_routines: _containers.RepeatedScalarFieldContainer[str]
            output_row_count: int
            cache_hit: bool
            def __init__(
                self,
                total_processed_bytes: _Optional[int] = ...,
                total_billed_bytes: _Optional[int] = ...,
                billing_tier: _Optional[int] = ...,
                referenced_tables: _Optional[_Iterable[str]] = ...,
                referenced_views: _Optional[_Iterable[str]] = ...,
                referenced_routines: _Optional[_Iterable[str]] = ...,
                output_row_count: _Optional[int] = ...,
                cache_hit: bool = ...,
            ) -> None: ...

        class Load(_message.Message):
            __slots__ = ("total_output_bytes",)
            TOTAL_OUTPUT_BYTES_FIELD_NUMBER: _ClassVar[int]
            total_output_bytes: int
            def __init__(self, total_output_bytes: _Optional[int] = ...) -> None: ...

        class Extract(_message.Message):
            __slots__ = ("total_input_bytes",)
            TOTAL_INPUT_BYTES_FIELD_NUMBER: _ClassVar[int]
            total_input_bytes: int
            def __init__(self, total_input_bytes: _Optional[int] = ...) -> None: ...

        class ReservationResourceUsage(_message.Message):
            __slots__ = ("name", "slot_ms")
            NAME_FIELD_NUMBER: _ClassVar[int]
            SLOT_MS_FIELD_NUMBER: _ClassVar[int]
            name: str
            slot_ms: int
            def __init__(
                self, name: _Optional[str] = ..., slot_ms: _Optional[int] = ...
            ) -> None: ...
        CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
        START_TIME_FIELD_NUMBER: _ClassVar[int]
        END_TIME_FIELD_NUMBER: _ClassVar[int]
        QUERY_STATS_FIELD_NUMBER: _ClassVar[int]
        LOAD_STATS_FIELD_NUMBER: _ClassVar[int]
        EXTRACT_STATS_FIELD_NUMBER: _ClassVar[int]
        TOTAL_SLOT_MS_FIELD_NUMBER: _ClassVar[int]
        RESERVATION_USAGE_FIELD_NUMBER: _ClassVar[int]
        RESERVATION_FIELD_NUMBER: _ClassVar[int]
        PARENT_JOB_NAME_FIELD_NUMBER: _ClassVar[int]
        create_time: _timestamp_pb2.Timestamp
        start_time: _timestamp_pb2.Timestamp
        end_time: _timestamp_pb2.Timestamp
        query_stats: BigQueryAuditMetadata.JobStats.Query
        load_stats: BigQueryAuditMetadata.JobStats.Load
        extract_stats: BigQueryAuditMetadata.JobStats.Extract
        total_slot_ms: int
        reservation_usage: _containers.RepeatedCompositeFieldContainer[
            BigQueryAuditMetadata.JobStats.ReservationResourceUsage
        ]
        reservation: str
        parent_job_name: str
        def __init__(
            self,
            create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            query_stats: _Optional[
                _Union[BigQueryAuditMetadata.JobStats.Query, _Mapping]
            ] = ...,
            load_stats: _Optional[
                _Union[BigQueryAuditMetadata.JobStats.Load, _Mapping]
            ] = ...,
            extract_stats: _Optional[
                _Union[BigQueryAuditMetadata.JobStats.Extract, _Mapping]
            ] = ...,
            total_slot_ms: _Optional[int] = ...,
            reservation_usage: _Optional[
                _Iterable[
                    _Union[
                        BigQueryAuditMetadata.JobStats.ReservationResourceUsage,
                        _Mapping,
                    ]
                ]
            ] = ...,
            reservation: _Optional[str] = ...,
            parent_job_name: _Optional[str] = ...,
        ) -> None: ...

    class Table(_message.Message):
        __slots__ = (
            "table_name",
            "table_info",
            "schema_json",
            "schema_json_truncated",
            "view",
            "expire_time",
            "create_time",
            "update_time",
            "truncate_time",
            "encryption",
        )
        TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
        TABLE_INFO_FIELD_NUMBER: _ClassVar[int]
        SCHEMA_JSON_FIELD_NUMBER: _ClassVar[int]
        SCHEMA_JSON_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
        VIEW_FIELD_NUMBER: _ClassVar[int]
        EXPIRE_TIME_FIELD_NUMBER: _ClassVar[int]
        CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
        UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
        TRUNCATE_TIME_FIELD_NUMBER: _ClassVar[int]
        ENCRYPTION_FIELD_NUMBER: _ClassVar[int]
        table_name: str
        table_info: BigQueryAuditMetadata.EntityInfo
        schema_json: str
        schema_json_truncated: bool
        view: BigQueryAuditMetadata.TableViewDefinition
        expire_time: _timestamp_pb2.Timestamp
        create_time: _timestamp_pb2.Timestamp
        update_time: _timestamp_pb2.Timestamp
        truncate_time: _timestamp_pb2.Timestamp
        encryption: BigQueryAuditMetadata.EncryptionInfo
        def __init__(
            self,
            table_name: _Optional[str] = ...,
            table_info: _Optional[
                _Union[BigQueryAuditMetadata.EntityInfo, _Mapping]
            ] = ...,
            schema_json: _Optional[str] = ...,
            schema_json_truncated: bool = ...,
            view: _Optional[
                _Union[BigQueryAuditMetadata.TableViewDefinition, _Mapping]
            ] = ...,
            expire_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            truncate_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            encryption: _Optional[
                _Union[BigQueryAuditMetadata.EncryptionInfo, _Mapping]
            ] = ...,
        ) -> None: ...

    class Model(_message.Message):
        __slots__ = (
            "model_name",
            "model_info",
            "expire_time",
            "create_time",
            "update_time",
            "encryption",
        )
        MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
        MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
        EXPIRE_TIME_FIELD_NUMBER: _ClassVar[int]
        CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
        UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
        ENCRYPTION_FIELD_NUMBER: _ClassVar[int]
        model_name: str
        model_info: BigQueryAuditMetadata.EntityInfo
        expire_time: _timestamp_pb2.Timestamp
        create_time: _timestamp_pb2.Timestamp
        update_time: _timestamp_pb2.Timestamp
        encryption: BigQueryAuditMetadata.EncryptionInfo
        def __init__(
            self,
            model_name: _Optional[str] = ...,
            model_info: _Optional[
                _Union[BigQueryAuditMetadata.EntityInfo, _Mapping]
            ] = ...,
            expire_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            encryption: _Optional[
                _Union[BigQueryAuditMetadata.EncryptionInfo, _Mapping]
            ] = ...,
        ) -> None: ...

    class Routine(_message.Message):
        __slots__ = ("routine_name", "create_time", "update_time")
        ROUTINE_NAME_FIELD_NUMBER: _ClassVar[int]
        CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
        UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
        routine_name: str
        create_time: _timestamp_pb2.Timestamp
        update_time: _timestamp_pb2.Timestamp
        def __init__(
            self,
            routine_name: _Optional[str] = ...,
            create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        ) -> None: ...

    class EntityInfo(_message.Message):
        __slots__ = ("friendly_name", "description", "labels")

        class LabelsEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(
                self, key: _Optional[str] = ..., value: _Optional[str] = ...
            ) -> None: ...
        FRIENDLY_NAME_FIELD_NUMBER: _ClassVar[int]
        DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
        LABELS_FIELD_NUMBER: _ClassVar[int]
        friendly_name: str
        description: str
        labels: _containers.ScalarMap[str, str]
        def __init__(
            self,
            friendly_name: _Optional[str] = ...,
            description: _Optional[str] = ...,
            labels: _Optional[_Mapping[str, str]] = ...,
        ) -> None: ...

    class TableViewDefinition(_message.Message):
        __slots__ = ("query", "query_truncated")
        QUERY_FIELD_NUMBER: _ClassVar[int]
        QUERY_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
        query: str
        query_truncated: bool
        def __init__(
            self, query: _Optional[str] = ..., query_truncated: bool = ...
        ) -> None: ...

    class Dataset(_message.Message):
        __slots__ = (
            "dataset_name",
            "dataset_info",
            "create_time",
            "update_time",
            "acl",
            "default_table_expire_duration",
            "default_encryption",
            "default_collation",
        )
        DATASET_NAME_FIELD_NUMBER: _ClassVar[int]
        DATASET_INFO_FIELD_NUMBER: _ClassVar[int]
        CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
        UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
        ACL_FIELD_NUMBER: _ClassVar[int]
        DEFAULT_TABLE_EXPIRE_DURATION_FIELD_NUMBER: _ClassVar[int]
        DEFAULT_ENCRYPTION_FIELD_NUMBER: _ClassVar[int]
        DEFAULT_COLLATION_FIELD_NUMBER: _ClassVar[int]
        dataset_name: str
        dataset_info: BigQueryAuditMetadata.EntityInfo
        create_time: _timestamp_pb2.Timestamp
        update_time: _timestamp_pb2.Timestamp
        acl: BigQueryAuditMetadata.BigQueryAcl
        default_table_expire_duration: _duration_pb2.Duration
        default_encryption: BigQueryAuditMetadata.EncryptionInfo
        default_collation: str
        def __init__(
            self,
            dataset_name: _Optional[str] = ...,
            dataset_info: _Optional[
                _Union[BigQueryAuditMetadata.EntityInfo, _Mapping]
            ] = ...,
            create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
            acl: _Optional[_Union[BigQueryAuditMetadata.BigQueryAcl, _Mapping]] = ...,
            default_table_expire_duration: _Optional[
                _Union[_duration_pb2.Duration, _Mapping]
            ] = ...,
            default_encryption: _Optional[
                _Union[BigQueryAuditMetadata.EncryptionInfo, _Mapping]
            ] = ...,
            default_collation: _Optional[str] = ...,
        ) -> None: ...

    class BigQueryAcl(_message.Message):
        __slots__ = ("policy", "authorized_views")
        POLICY_FIELD_NUMBER: _ClassVar[int]
        AUTHORIZED_VIEWS_FIELD_NUMBER: _ClassVar[int]
        policy: _policy_pb2.Policy
        authorized_views: _containers.RepeatedScalarFieldContainer[str]
        def __init__(
            self,
            policy: _Optional[_Union[_policy_pb2.Policy, _Mapping]] = ...,
            authorized_views: _Optional[_Iterable[str]] = ...,
        ) -> None: ...

    class EncryptionInfo(_message.Message):
        __slots__ = ("kms_key_name",)
        KMS_KEY_NAME_FIELD_NUMBER: _ClassVar[int]
        kms_key_name: str
        def __init__(self, kms_key_name: _Optional[str] = ...) -> None: ...

    class RowAccessPolicy(_message.Message):
        __slots__ = ("row_access_policy_name",)
        ROW_ACCESS_POLICY_NAME_FIELD_NUMBER: _ClassVar[int]
        row_access_policy_name: str
        def __init__(self, row_access_policy_name: _Optional[str] = ...) -> None: ...

    class FirstPartyAppMetadata(_message.Message):
        __slots__ = ("sheets_metadata",)
        SHEETS_METADATA_FIELD_NUMBER: _ClassVar[int]
        sheets_metadata: BigQueryAuditMetadata.SheetsMetadata
        def __init__(
            self,
            sheets_metadata: _Optional[
                _Union[BigQueryAuditMetadata.SheetsMetadata, _Mapping]
            ] = ...,
        ) -> None: ...

    class SheetsMetadata(_message.Message):
        __slots__ = ("doc_id",)
        DOC_ID_FIELD_NUMBER: _ClassVar[int]
        doc_id: str
        def __init__(self, doc_id: _Optional[str] = ...) -> None: ...
    JOB_INSERTION_FIELD_NUMBER: _ClassVar[int]
    JOB_CHANGE_FIELD_NUMBER: _ClassVar[int]
    JOB_DELETION_FIELD_NUMBER: _ClassVar[int]
    DATASET_CREATION_FIELD_NUMBER: _ClassVar[int]
    DATASET_CHANGE_FIELD_NUMBER: _ClassVar[int]
    DATASET_DELETION_FIELD_NUMBER: _ClassVar[int]
    TABLE_CREATION_FIELD_NUMBER: _ClassVar[int]
    TABLE_CHANGE_FIELD_NUMBER: _ClassVar[int]
    TABLE_DELETION_FIELD_NUMBER: _ClassVar[int]
    TABLE_DATA_READ_FIELD_NUMBER: _ClassVar[int]
    TABLE_DATA_CHANGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_DELETION_FIELD_NUMBER: _ClassVar[int]
    MODEL_CREATION_FIELD_NUMBER: _ClassVar[int]
    MODEL_METADATA_CHANGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_CHANGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_READ_FIELD_NUMBER: _ClassVar[int]
    ROUTINE_CREATION_FIELD_NUMBER: _ClassVar[int]
    ROUTINE_CHANGE_FIELD_NUMBER: _ClassVar[int]
    ROUTINE_DELETION_FIELD_NUMBER: _ClassVar[int]
    ROW_ACCESS_POLICY_CREATION_FIELD_NUMBER: _ClassVar[int]
    ROW_ACCESS_POLICY_CHANGE_FIELD_NUMBER: _ClassVar[int]
    ROW_ACCESS_POLICY_DELETION_FIELD_NUMBER: _ClassVar[int]
    UNLINK_DATASET_FIELD_NUMBER: _ClassVar[int]
    FIRST_PARTY_APP_METADATA_FIELD_NUMBER: _ClassVar[int]
    job_insertion: BigQueryAuditMetadata.JobInsertion
    job_change: BigQueryAuditMetadata.JobChange
    job_deletion: BigQueryAuditMetadata.JobDeletion
    dataset_creation: BigQueryAuditMetadata.DatasetCreation
    dataset_change: BigQueryAuditMetadata.DatasetChange
    dataset_deletion: BigQueryAuditMetadata.DatasetDeletion
    table_creation: BigQueryAuditMetadata.TableCreation
    table_change: BigQueryAuditMetadata.TableChange
    table_deletion: BigQueryAuditMetadata.TableDeletion
    table_data_read: BigQueryAuditMetadata.TableDataRead
    table_data_change: BigQueryAuditMetadata.TableDataChange
    model_deletion: BigQueryAuditMetadata.ModelDeletion
    model_creation: BigQueryAuditMetadata.ModelCreation
    model_metadata_change: BigQueryAuditMetadata.ModelMetadataChange
    model_data_change: BigQueryAuditMetadata.ModelDataChange
    model_data_read: BigQueryAuditMetadata.ModelDataRead
    routine_creation: BigQueryAuditMetadata.RoutineCreation
    routine_change: BigQueryAuditMetadata.RoutineChange
    routine_deletion: BigQueryAuditMetadata.RoutineDeletion
    row_access_policy_creation: BigQueryAuditMetadata.RowAccessPolicyCreation
    row_access_policy_change: BigQueryAuditMetadata.RowAccessPolicyChange
    row_access_policy_deletion: BigQueryAuditMetadata.RowAccessPolicyDeletion
    unlink_dataset: BigQueryAuditMetadata.UnlinkDataset
    first_party_app_metadata: BigQueryAuditMetadata.FirstPartyAppMetadata
    def __init__(
        self,
        job_insertion: _Optional[
            _Union[BigQueryAuditMetadata.JobInsertion, _Mapping]
        ] = ...,
        job_change: _Optional[_Union[BigQueryAuditMetadata.JobChange, _Mapping]] = ...,
        job_deletion: _Optional[
            _Union[BigQueryAuditMetadata.JobDeletion, _Mapping]
        ] = ...,
        dataset_creation: _Optional[
            _Union[BigQueryAuditMetadata.DatasetCreation, _Mapping]
        ] = ...,
        dataset_change: _Optional[
            _Union[BigQueryAuditMetadata.DatasetChange, _Mapping]
        ] = ...,
        dataset_deletion: _Optional[
            _Union[BigQueryAuditMetadata.DatasetDeletion, _Mapping]
        ] = ...,
        table_creation: _Optional[
            _Union[BigQueryAuditMetadata.TableCreation, _Mapping]
        ] = ...,
        table_change: _Optional[
            _Union[BigQueryAuditMetadata.TableChange, _Mapping]
        ] = ...,
        table_deletion: _Optional[
            _Union[BigQueryAuditMetadata.TableDeletion, _Mapping]
        ] = ...,
        table_data_read: _Optional[
            _Union[BigQueryAuditMetadata.TableDataRead, _Mapping]
        ] = ...,
        table_data_change: _Optional[
            _Union[BigQueryAuditMetadata.TableDataChange, _Mapping]
        ] = ...,
        model_deletion: _Optional[
            _Union[BigQueryAuditMetadata.ModelDeletion, _Mapping]
        ] = ...,
        model_creation: _Optional[
            _Union[BigQueryAuditMetadata.ModelCreation, _Mapping]
        ] = ...,
        model_metadata_change: _Optional[
            _Union[BigQueryAuditMetadata.ModelMetadataChange, _Mapping]
        ] = ...,
        model_data_change: _Optional[
            _Union[BigQueryAuditMetadata.ModelDataChange, _Mapping]
        ] = ...,
        model_data_read: _Optional[
            _Union[BigQueryAuditMetadata.ModelDataRead, _Mapping]
        ] = ...,
        routine_creation: _Optional[
            _Union[BigQueryAuditMetadata.RoutineCreation, _Mapping]
        ] = ...,
        routine_change: _Optional[
            _Union[BigQueryAuditMetadata.RoutineChange, _Mapping]
        ] = ...,
        routine_deletion: _Optional[
            _Union[BigQueryAuditMetadata.RoutineDeletion, _Mapping]
        ] = ...,
        row_access_policy_creation: _Optional[
            _Union[BigQueryAuditMetadata.RowAccessPolicyCreation, _Mapping]
        ] = ...,
        row_access_policy_change: _Optional[
            _Union[BigQueryAuditMetadata.RowAccessPolicyChange, _Mapping]
        ] = ...,
        row_access_policy_deletion: _Optional[
            _Union[BigQueryAuditMetadata.RowAccessPolicyDeletion, _Mapping]
        ] = ...,
        unlink_dataset: _Optional[
            _Union[BigQueryAuditMetadata.UnlinkDataset, _Mapping]
        ] = ...,
        first_party_app_metadata: _Optional[
            _Union[BigQueryAuditMetadata.FirstPartyAppMetadata, _Mapping]
        ] = ...,
    ) -> None: ...
