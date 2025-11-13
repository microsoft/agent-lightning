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

from google.cloud.discoveryengine_v1.types import user_event as gcd_user_event

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "WriteUserEventRequest",
        "CollectUserEventRequest",
    },
)


class WriteUserEventRequest(proto.Message):
    r"""Request message for WriteUserEvent method.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        parent (str):
            Required. The parent resource name. If the write user event
            action is applied in
            [DataStore][google.cloud.discoveryengine.v1.DataStore]
            level, the format is:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}``.
            If the write user event action is applied in
            [Location][google.cloud.location.Location] level, for
            example, the event with
            [Document][google.cloud.discoveryengine.v1.Document] across
            multiple
            [DataStore][google.cloud.discoveryengine.v1.DataStore], the
            format is: ``projects/{project}/locations/{location}``.
        user_event (google.cloud.discoveryengine_v1.types.UserEvent):
            Required. User event to write.

            This field is a member of `oneof`_ ``_user_event``.
        write_async (bool):
            If set to true, the user event is written
            asynchronously after validation, and the API
            responds without waiting for the write.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    user_event: gcd_user_event.UserEvent = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message=gcd_user_event.UserEvent,
    )
    write_async: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class CollectUserEventRequest(proto.Message):
    r"""Request message for CollectUserEvent method.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        parent (str):
            Required. The parent resource name. If the collect user
            event action is applied in
            [DataStore][google.cloud.discoveryengine.v1.DataStore]
            level, the format is:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}``.
            If the collect user event action is applied in
            [Location][google.cloud.location.Location] level, for
            example, the event with
            [Document][google.cloud.discoveryengine.v1.Document] across
            multiple
            [DataStore][google.cloud.discoveryengine.v1.DataStore], the
            format is: ``projects/{project}/locations/{location}``.
        user_event (str):
            Required. URL encoded UserEvent proto with a
            length limit of 2,000,000 characters.
        uri (str):
            The URL including cgi-parameters but
            excluding the hash fragment with a length limit
            of 5,000 characters. This is often more useful
            than the referer URL, because many browsers only
            send the domain for third-party requests.

            This field is a member of `oneof`_ ``_uri``.
        ets (int):
            The event timestamp in milliseconds. This
            prevents browser caching of otherwise identical
            get requests. The name is abbreviated to reduce
            the payload bytes.

            This field is a member of `oneof`_ ``_ets``.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    user_event: str = proto.Field(
        proto.STRING,
        number=2,
    )
    uri: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    ets: int = proto.Field(
        proto.INT64,
        number=4,
        optional=True,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
