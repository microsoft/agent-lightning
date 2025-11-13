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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.spanner.admin.database.v1",
    manifest={
        "DatabaseDialect",
        "OperationProgress",
        "EncryptionConfig",
        "EncryptionInfo",
    },
)


class DatabaseDialect(proto.Enum):
    r"""Indicates the dialect type of a database.

    Values:
        DATABASE_DIALECT_UNSPECIFIED (0):
            Default value. This value will create a database with the
            GOOGLE_STANDARD_SQL dialect.
        GOOGLE_STANDARD_SQL (1):
            GoogleSQL supported SQL.
        POSTGRESQL (2):
            PostgreSQL supported SQL.
    """
    DATABASE_DIALECT_UNSPECIFIED = 0
    GOOGLE_STANDARD_SQL = 1
    POSTGRESQL = 2


class OperationProgress(proto.Message):
    r"""Encapsulates progress related information for a Cloud Spanner
    long running operation.

    Attributes:
        progress_percent (int):
            Percent completion of the operation.
            Values are between 0 and 100 inclusive.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Time the request was received.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            If set, the time at which this operation
            failed or was completed successfully.
    """

    progress_percent: int = proto.Field(
        proto.INT32,
        number=1,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class EncryptionConfig(proto.Message):
    r"""Encryption configuration for a Cloud Spanner database.

    Attributes:
        kms_key_name (str):
            The Cloud KMS key to be used for encrypting and decrypting
            the database. Values are of the form
            ``projects/<project>/locations/<location>/keyRings/<key_ring>/cryptoKeys/<kms_key_name>``.
        kms_key_names (MutableSequence[str]):
            Specifies the KMS configuration for the one or more keys
            used to encrypt the database. Values are of the form
            ``projects/<project>/locations/<location>/keyRings/<key_ring>/cryptoKeys/<kms_key_name>``.

            The keys referenced by kms_key_names must fully cover all
            regions of the database instance configuration. Some
            examples:

            - For single region database instance configs, specify a
              single regional location KMS key.
            - For multi-regional database instance configs of type
              GOOGLE_MANAGED, either specify a multi-regional location
              KMS key or multiple regional location KMS keys that cover
              all regions in the instance config.
            - For a database instance config of type USER_MANAGED,
              please specify only regional location KMS keys to cover
              each region in the instance config. Multi-regional
              location KMS keys are not supported for USER_MANAGED
              instance configs.
    """

    kms_key_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    kms_key_names: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


class EncryptionInfo(proto.Message):
    r"""Encryption information for a Cloud Spanner database or
    backup.

    Attributes:
        encryption_type (google.cloud.spanner_admin_database_v1.types.EncryptionInfo.Type):
            Output only. The type of encryption.
        encryption_status (google.rpc.status_pb2.Status):
            Output only. If present, the status of a
            recent encrypt/decrypt call on underlying data
            for this database or backup. Regardless of
            status, data is always encrypted at rest.
        kms_key_version (str):
            Output only. A Cloud KMS key version that is
            being used to protect the database or backup.
    """

    class Type(proto.Enum):
        r"""Possible encryption types.

        Values:
            TYPE_UNSPECIFIED (0):
                Encryption type was not specified, though
                data at rest remains encrypted.
            GOOGLE_DEFAULT_ENCRYPTION (1):
                The data is encrypted at rest with a key that
                is fully managed by Google. No key version or
                status will be populated. This is the default
                state.
            CUSTOMER_MANAGED_ENCRYPTION (2):
                The data is encrypted at rest with a key that is managed by
                the customer. The active version of the key.
                ``kms_key_version`` will be populated, and
                ``encryption_status`` may be populated.
        """
        TYPE_UNSPECIFIED = 0
        GOOGLE_DEFAULT_ENCRYPTION = 1
        CUSTOMER_MANAGED_ENCRYPTION = 2

    encryption_type: Type = proto.Field(
        proto.ENUM,
        number=3,
        enum=Type,
    )
    encryption_status: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=4,
        message=status_pb2.Status,
    )
    kms_key_version: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
