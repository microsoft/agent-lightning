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

from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1beta.types import engine as gcd_engine

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "CreateEngineRequest",
        "CreateEngineMetadata",
        "DeleteEngineRequest",
        "DeleteEngineMetadata",
        "GetEngineRequest",
        "ListEnginesRequest",
        "ListEnginesResponse",
        "UpdateEngineRequest",
        "PauseEngineRequest",
        "ResumeEngineRequest",
        "TuneEngineRequest",
        "TuneEngineMetadata",
        "TuneEngineResponse",
    },
)


class CreateEngineRequest(proto.Message):
    r"""Request for
    [EngineService.CreateEngine][google.cloud.discoveryengine.v1beta.EngineService.CreateEngine]
    method.

    Attributes:
        parent (str):
            Required. The parent resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection}``.
        engine (google.cloud.discoveryengine_v1beta.types.Engine):
            Required. The
            [Engine][google.cloud.discoveryengine.v1beta.Engine] to
            create.
        engine_id (str):
            Required. The ID to use for the
            [Engine][google.cloud.discoveryengine.v1beta.Engine], which
            will become the final component of the
            [Engine][google.cloud.discoveryengine.v1beta.Engine]'s
            resource name.

            This field must conform to
            `RFC-1034 <https://tools.ietf.org/html/rfc1034>`__ standard
            with a length limit of 63 characters. Otherwise, an
            INVALID_ARGUMENT error is returned.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    engine: gcd_engine.Engine = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_engine.Engine,
    )
    engine_id: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CreateEngineMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [EngineService.CreateEngine][google.cloud.discoveryengine.v1beta.EngineService.CreateEngine]
    operation. This will be returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class DeleteEngineRequest(proto.Message):
    r"""Request message for
    [EngineService.DeleteEngine][google.cloud.discoveryengine.v1beta.EngineService.DeleteEngine]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [Engine][google.cloud.discoveryengine.v1beta.Engine], such
            as
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``.

            If the caller does not have permission to delete the
            [Engine][google.cloud.discoveryengine.v1beta.Engine],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the [Engine][google.cloud.discoveryengine.v1beta.Engine]
            to delete does not exist, a NOT_FOUND error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteEngineMetadata(proto.Message):
    r"""Metadata related to the progress of the
    [EngineService.DeleteEngine][google.cloud.discoveryengine.v1beta.EngineService.DeleteEngine]
    operation. This will be returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class GetEngineRequest(proto.Message):
    r"""Request message for
    [EngineService.GetEngine][google.cloud.discoveryengine.v1beta.EngineService.GetEngine]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [Engine][google.cloud.discoveryengine.v1beta.Engine], such
            as
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListEnginesRequest(proto.Message):
    r"""Request message for
    [EngineService.ListEngines][google.cloud.discoveryengine.v1beta.EngineService.ListEngines]
    method.

    Attributes:
        parent (str):
            Required. The parent resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection_id}``.
        page_size (int):
            Optional. Not supported.
        page_token (str):
            Optional. Not supported.
        filter (str):
            Optional. Filter by solution type. For example:
            solution_type=SOLUTION_TYPE_SEARCH
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListEnginesResponse(proto.Message):
    r"""Response message for
    [EngineService.ListEngines][google.cloud.discoveryengine.v1beta.EngineService.ListEngines]
    method.

    Attributes:
        engines (MutableSequence[google.cloud.discoveryengine_v1beta.types.Engine]):
            All the customer's
            [Engine][google.cloud.discoveryengine.v1beta.Engine]s.
        next_page_token (str):
            Not supported.
    """

    @property
    def raw_page(self):
        return self

    engines: MutableSequence[gcd_engine.Engine] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_engine.Engine,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateEngineRequest(proto.Message):
    r"""Request message for
    [EngineService.UpdateEngine][google.cloud.discoveryengine.v1beta.EngineService.UpdateEngine]
    method.

    Attributes:
        engine (google.cloud.discoveryengine_v1beta.types.Engine):
            Required. The
            [Engine][google.cloud.discoveryengine.v1beta.Engine] to
            update.

            If the caller does not have permission to update the
            [Engine][google.cloud.discoveryengine.v1beta.Engine],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the [Engine][google.cloud.discoveryengine.v1beta.Engine]
            to update does not exist, a NOT_FOUND error is returned.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            [Engine][google.cloud.discoveryengine.v1beta.Engine] to
            update.

            If an unsupported or unknown field is provided, an
            INVALID_ARGUMENT error is returned.
    """

    engine: gcd_engine.Engine = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_engine.Engine,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class PauseEngineRequest(proto.Message):
    r"""Request for pausing training of an engine.

    Attributes:
        name (str):
            Required. The name of the engine to pause. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ResumeEngineRequest(proto.Message):
    r"""Request for resuming training of an engine.

    Attributes:
        name (str):
            Required. The name of the engine to resume. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class TuneEngineRequest(proto.Message):
    r"""Request to manually start a tuning process now (instead of
    waiting for the periodically scheduled tuning to happen).

    Attributes:
        name (str):
            Required. The resource name of the engine to tune. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class TuneEngineMetadata(proto.Message):
    r"""Metadata associated with a tune operation.

    Attributes:
        engine (str):
            Required. The resource name of the engine that this tune
            applies to. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``
    """

    engine: str = proto.Field(
        proto.STRING,
        number=1,
    )


class TuneEngineResponse(proto.Message):
    r"""Response associated with a tune operation."""


__all__ = tuple(sorted(__protobuf__.manifest))
