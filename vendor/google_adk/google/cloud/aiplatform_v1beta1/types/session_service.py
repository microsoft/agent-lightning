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

from google.cloud.aiplatform_v1beta1.types import operation
from google.cloud.aiplatform_v1beta1.types import session as gca_session
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateSessionRequest",
        "CreateSessionOperationMetadata",
        "GetSessionRequest",
        "ListSessionsRequest",
        "ListSessionsResponse",
        "UpdateSessionRequest",
        "DeleteSessionRequest",
        "ListEventsRequest",
        "ListEventsResponse",
        "AppendEventRequest",
        "AppendEventResponse",
    },
)


class CreateSessionRequest(proto.Message):
    r"""Request message for
    [SessionService.CreateSession][google.cloud.aiplatform.v1beta1.SessionService.CreateSession].

    Attributes:
        parent (str):
            Required. The resource name of the location to create the
            session in. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
        session (google.cloud.aiplatform_v1beta1.types.Session):
            Required. The session to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    session: gca_session.Session = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_session.Session,
    )


class CreateSessionOperationMetadata(proto.Message):
    r"""Metadata associated with the
    [SessionService.CreateSession][google.cloud.aiplatform.v1beta1.SessionService.CreateSession]
    operation.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class GetSessionRequest(proto.Message):
    r"""Request message for
    [SessionService.GetSession][google.cloud.aiplatform.v1beta1.SessionService.GetSession].

    Attributes:
        name (str):
            Required. The resource name of the session. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/sessions/{session}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListSessionsRequest(proto.Message):
    r"""Request message for
    [SessionService.ListSessions][google.cloud.aiplatform.v1beta1.SessionService.ListSessions].

    Attributes:
        parent (str):
            Required. The resource name of the location to list sessions
            from. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
        page_size (int):
            Optional. The maximum number of sessions to
            return. The service may return fewer than this
            value. If unspecified, at most 100 sessions will
            be returned.
        page_token (str):
            Optional. The
            [next_page_token][google.cloud.aiplatform.v1beta1.ListSessionsResponse.next_page_token]
            value returned from a previous list
            [SessionService.ListSessions][google.cloud.aiplatform.v1beta1.SessionService.ListSessions]
            call.
        filter (str):
            Optional. The standard list filter. Supported fields: \*
            ``display_name``

            Example: ``display_name=abc``.
        order_by (str):
            Optional. A comma-separated list of fields to order by,
            sorted in ascending order. Use "desc" after a field name for
            descending. Supported fields:

            - ``create_time``
            - ``update_time``

            Example: ``create_time desc``.
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
    order_by: str = proto.Field(
        proto.STRING,
        number=5,
    )


class ListSessionsResponse(proto.Message):
    r"""Response message for
    [SessionService.ListSessions][google.cloud.aiplatform.v1beta1.SessionService.ListSessions].

    Attributes:
        sessions (MutableSequence[google.cloud.aiplatform_v1beta1.types.Session]):
            A list of sessions matching the request.
        next_page_token (str):
            A token, which can be sent as
            [ListSessionsRequest.page_token][google.cloud.aiplatform.v1beta1.ListSessionsRequest.page_token]
            to retrieve the next page. Absence of this field indicates
            there are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    sessions: MutableSequence[gca_session.Session] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_session.Session,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class UpdateSessionRequest(proto.Message):
    r"""Request message for
    [SessionService.UpdateSession][google.cloud.aiplatform.v1beta1.SessionService.UpdateSession].

    Attributes:
        session (google.cloud.aiplatform_v1beta1.types.Session):
            Required. The session to update. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/sessions/{session}``
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Field mask is used to control which
            fields get updated. If the mask is not present,
            all fields will be updated.
    """

    session: gca_session.Session = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_session.Session,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteSessionRequest(proto.Message):
    r"""Request message for
    [SessionService.DeleteSession][google.cloud.aiplatform.v1beta1.SessionService.DeleteSession].

    Attributes:
        name (str):
            Required. The resource name of the session. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/sessions/{session}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListEventsRequest(proto.Message):
    r"""Request message for
    [SessionService.ListEvents][google.cloud.aiplatform.v1beta1.SessionService.ListEvents].

    Attributes:
        parent (str):
            Required. The resource name of the session to list events
            from. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/sessions/{session}``
        page_size (int):
            Optional. The maximum number of events to
            return. The service may return fewer than this
            value. If unspecified, at most 100 events will
            be returned. These events are ordered by
            timestamp in ascending order.
        page_token (str):
            Optional. The
            [next_page_token][google.cloud.aiplatform.v1beta1.ListEventsResponse.next_page_token]
            value returned from a previous list
            [SessionService.ListEvents][google.cloud.aiplatform.v1beta1.SessionService.ListEvents]
            call.
        filter (str):
            Optional. The standard list filter. Supported fields: \*
            ``timestamp`` range (i.e.
            ``timestamp>="2025-01-31T11:30:00-04:00"`` where the
            timestamp is in RFC 3339 format)

            More detail in `AIP-160 <https://google.aip.dev/160>`__.
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


class ListEventsResponse(proto.Message):
    r"""Response message for
    [SessionService.ListEvents][google.cloud.aiplatform.v1beta1.SessionService.ListEvents].

    Attributes:
        session_events (MutableSequence[google.cloud.aiplatform_v1beta1.types.SessionEvent]):
            A list of events matching the request.
            Ordered by timestamp in ascending order.
        next_page_token (str):
            A token, which can be sent as
            [ListEventsRequest.page_token][google.cloud.aiplatform.v1beta1.ListEventsRequest.page_token]
            to retrieve the next page. Absence of this field indicates
            there are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    session_events: MutableSequence[gca_session.SessionEvent] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_session.SessionEvent,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class AppendEventRequest(proto.Message):
    r"""Request message for
    [SessionService.AppendEvent][google.cloud.aiplatform.v1beta1.SessionService.AppendEvent].

    Attributes:
        name (str):
            Required. The resource name of the session to append event
            to. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/sessions/{session}``
        event (google.cloud.aiplatform_v1beta1.types.SessionEvent):
            Required. The event to append to the session.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    event: gca_session.SessionEvent = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_session.SessionEvent,
    )


class AppendEventResponse(proto.Message):
    r"""Response message for
    [SessionService.AppendEvent][google.cloud.aiplatform.v1beta1.SessionService.AppendEvent].

    """


__all__ = tuple(sorted(__protobuf__.manifest))
