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
    package="google.spanner.admin.instance.v1",
    manifest={
        "FulfillmentPeriod",
        "OperationProgress",
        "ReplicaSelection",
    },
)


class FulfillmentPeriod(proto.Enum):
    r"""Indicates the expected fulfillment period of an operation.

    Values:
        FULFILLMENT_PERIOD_UNSPECIFIED (0):
            Not specified.
        FULFILLMENT_PERIOD_NORMAL (1):
            Normal fulfillment period. The operation is
            expected to complete within minutes.
        FULFILLMENT_PERIOD_EXTENDED (2):
            Extended fulfillment period. It can take up
            to an hour for the operation to complete.
    """
    FULFILLMENT_PERIOD_UNSPECIFIED = 0
    FULFILLMENT_PERIOD_NORMAL = 1
    FULFILLMENT_PERIOD_EXTENDED = 2


class OperationProgress(proto.Message):
    r"""Encapsulates progress related information for a Cloud Spanner
    long running instance operations.

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


class ReplicaSelection(proto.Message):
    r"""ReplicaSelection identifies replicas with common properties.

    Attributes:
        location (str):
            Required. Name of the location of the
            replicas (e.g., "us-central1").
    """

    location: str = proto.Field(
        proto.STRING,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
