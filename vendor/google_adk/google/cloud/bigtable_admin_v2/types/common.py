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


__protobuf__ = proto.module(
    package="google.bigtable.admin.v2",
    manifest={
        "StorageType",
        "OperationProgress",
    },
)


class StorageType(proto.Enum):
    r"""Storage media types for persisting Bigtable data.

    Values:
        STORAGE_TYPE_UNSPECIFIED (0):
            The user did not specify a storage type.
        SSD (1):
            Flash (SSD) storage should be used.
        HDD (2):
            Magnetic drive (HDD) storage should be used.
    """
    STORAGE_TYPE_UNSPECIFIED = 0
    SSD = 1
    HDD = 2


class OperationProgress(proto.Message):
    r"""Encapsulates progress related information for a Cloud
    Bigtable long running operation.

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


__all__ = tuple(sorted(__protobuf__.manifest))
