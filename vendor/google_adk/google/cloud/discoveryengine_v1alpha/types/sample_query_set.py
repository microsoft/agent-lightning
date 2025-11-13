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

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "SampleQuerySet",
    },
)


class SampleQuerySet(proto.Message):
    r"""A SampleQuerySet is the parent resource of SampleQuery, and
    contains the configurations shared by all SampleQuery under it.

    Attributes:
        name (str):
            Identifier. The full resource name of the
            [SampleQuerySet][google.cloud.discoveryengine.v1alpha.SampleQuerySet],
            in the format of
            ``projects/{project}/locations/{location}/sampleQuerySets/{sample_query_set}``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        display_name (str):
            Required. The sample query set display name.

            This field must be a UTF-8 encoded string with a
            length limit of 128 characters.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp the
            [SampleQuerySet][google.cloud.discoveryengine.v1alpha.SampleQuerySet]
            was created at.
        description (str):
            The description of the
            [SampleQuerySet][google.cloud.discoveryengine.v1alpha.SampleQuerySet].
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    description: str = proto.Field(
        proto.STRING,
        number=4,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
