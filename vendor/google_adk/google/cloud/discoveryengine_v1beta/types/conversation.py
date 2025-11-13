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

from google.cloud.discoveryengine_v1beta.types import search_service

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "Conversation",
        "Reply",
        "ConversationContext",
        "TextInput",
        "ConversationMessage",
    },
)


class Conversation(proto.Message):
    r"""External conversation proto definition.

    Attributes:
        name (str):
            Immutable. Fully qualified name
            ``projects/{project}/locations/global/collections/{collection}/dataStore/*/conversations/*``
            or
            ``projects/{project}/locations/global/collections/{collection}/engines/*/conversations/*``.
        state (google.cloud.discoveryengine_v1beta.types.Conversation.State):
            The state of the Conversation.
        user_pseudo_id (str):
            A unique identifier for tracking users.
        messages (MutableSequence[google.cloud.discoveryengine_v1beta.types.ConversationMessage]):
            Conversation messages.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time the conversation
            started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The time the conversation
            finished.
    """

    class State(proto.Enum):
        r"""Enumeration of the state of the conversation.

        Values:
            STATE_UNSPECIFIED (0):
                Unknown.
            IN_PROGRESS (1):
                Conversation is currently open.
            COMPLETED (2):
                Conversation has been completed.
        """
        STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        COMPLETED = 2

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=2,
        enum=State,
    )
    user_pseudo_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    messages: MutableSequence["ConversationMessage"] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message="ConversationMessage",
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )


class Reply(proto.Message):
    r"""Defines a reply message to user.

    Attributes:
        reply (str):
            DEPRECATED: use ``summary`` instead. Text reply.
        references (MutableSequence[google.cloud.discoveryengine_v1beta.types.Reply.Reference]):
            References in the reply.
        summary (google.cloud.discoveryengine_v1beta.types.SearchResponse.Summary):
            Summary based on search results.
    """

    class Reference(proto.Message):
        r"""Defines reference in reply.

        Attributes:
            uri (str):
                URI link reference.
            anchor_text (str):
                Anchor text.
            start (int):
                Anchor text start index.
            end (int):
                Anchor text end index.
        """

        uri: str = proto.Field(
            proto.STRING,
            number=1,
        )
        anchor_text: str = proto.Field(
            proto.STRING,
            number=2,
        )
        start: int = proto.Field(
            proto.INT32,
            number=3,
        )
        end: int = proto.Field(
            proto.INT32,
            number=4,
        )

    reply: str = proto.Field(
        proto.STRING,
        number=1,
    )
    references: MutableSequence[Reference] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=Reference,
    )
    summary: search_service.SearchResponse.Summary = proto.Field(
        proto.MESSAGE,
        number=3,
        message=search_service.SearchResponse.Summary,
    )


class ConversationContext(proto.Message):
    r"""Defines context of the conversation

    Attributes:
        context_documents (MutableSequence[str]):
            The current list of documents the user is
            seeing. It contains the document resource
            references.
        active_document (str):
            The current active document the user opened.
            It contains the document resource reference.
    """

    context_documents: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    active_document: str = proto.Field(
        proto.STRING,
        number=2,
    )


class TextInput(proto.Message):
    r"""Defines text input.

    Attributes:
        input (str):
            Text input.
        context (google.cloud.discoveryengine_v1beta.types.ConversationContext):
            Conversation context of the input.
    """

    input: str = proto.Field(
        proto.STRING,
        number=1,
    )
    context: "ConversationContext" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ConversationContext",
    )


class ConversationMessage(proto.Message):
    r"""Defines a conversation message.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        user_input (google.cloud.discoveryengine_v1beta.types.TextInput):
            User text input.

            This field is a member of `oneof`_ ``message``.
        reply (google.cloud.discoveryengine_v1beta.types.Reply):
            Search reply.

            This field is a member of `oneof`_ ``message``.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Message creation timestamp.
    """

    user_input: "TextInput" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="message",
        message="TextInput",
    )
    reply: "Reply" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="message",
        message="Reply",
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
