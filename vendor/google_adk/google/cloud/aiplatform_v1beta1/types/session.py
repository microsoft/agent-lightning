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

from google.cloud.aiplatform_v1beta1.types import content as gca_content
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "Session",
        "SessionEvent",
        "EventMetadata",
        "EventActions",
    },
)


class Session(proto.Message):
    r"""A session contains a set of actions between users and Vertex
    agents.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            Optional. Timestamp of when this session is considered
            expired. This is *always* provided on output, regardless of
            what was sent on input.

            This field is a member of `oneof`_ ``expiration``.
        ttl (google.protobuf.duration_pb2.Duration):
            Optional. Input only. The TTL for this
            session.

            This field is a member of `oneof`_ ``expiration``.
        name (str):
            Identifier. The resource name of the session. Format:
            'projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/sessions/{session}'.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the session was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when the session was
            updated.
        display_name (str):
            Optional. The display name of the session.
        session_state (google.protobuf.struct_pb2.Struct):
            Optional. Session specific memory which
            stores key conversation points.
        user_id (str):
            Required. Immutable. String id provided by
            the user
    """

    expire_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="expiration",
        message=timestamp_pb2.Timestamp,
    )
    ttl: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="expiration",
        message=duration_pb2.Duration,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=5,
    )
    session_state: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=10,
        message=struct_pb2.Struct,
    )
    user_id: str = proto.Field(
        proto.STRING,
        number=12,
    )


class SessionEvent(proto.Message):
    r"""An event represents a message from either the user or agent.

    Attributes:
        name (str):
            Identifier. The resource name of the event.
            Format:``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/sessions/{session}/events/{event}``.
        author (str):
            Required. The name of the agent that sent the
            event, or user.
        content (google.cloud.aiplatform_v1beta1.types.Content):
            Optional. Content of the event provided by
            the author.
        invocation_id (str):
            Required. The invocation id of the event,
            multiple events can have the same invocation id.
        actions (google.cloud.aiplatform_v1beta1.types.EventActions):
            Optional. Actions executed by the agent.
        timestamp (google.protobuf.timestamp_pb2.Timestamp):
            Required. Timestamp when the event was
            created on client side.
        error_code (str):
            Optional. Error code if the response is an
            error. Code varies by model.
        error_message (str):
            Optional. Error message if the response is an
            error.
        event_metadata (google.cloud.aiplatform_v1beta1.types.EventMetadata):
            Optional. Metadata relating to this event.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    author: str = proto.Field(
        proto.STRING,
        number=3,
    )
    content: gca_content.Content = proto.Field(
        proto.MESSAGE,
        number=4,
        message=gca_content.Content,
    )
    invocation_id: str = proto.Field(
        proto.STRING,
        number=5,
    )
    actions: "EventActions" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="EventActions",
    )
    timestamp: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    error_code: str = proto.Field(
        proto.STRING,
        number=9,
    )
    error_message: str = proto.Field(
        proto.STRING,
        number=10,
    )
    event_metadata: "EventMetadata" = proto.Field(
        proto.MESSAGE,
        number=11,
        message="EventMetadata",
    )


class EventMetadata(proto.Message):
    r"""Metadata relating to a LLM response event.

    Attributes:
        grounding_metadata (google.cloud.aiplatform_v1beta1.types.GroundingMetadata):
            Optional. Metadata returned to client when
            grounding is enabled.
        partial (bool):
            Optional. Indicates whether the text content
            is part of a unfinished text stream. Only used
            for streaming mode and when the content is plain
            text.
        turn_complete (bool):
            Optional. Indicates whether the response from
            the model is complete. Only used for streaming
            mode.
        interrupted (bool):
            Optional. Flag indicating that LLM was
            interrupted when generating the content. Usually
            it's due to user interruption during a bidi
            streaming.
        long_running_tool_ids (MutableSequence[str]):
            Optional. Set of ids of the long running
            function calls. Agent client will know from this
            field about which function call is long running.
            Only valid for function call event.
        branch (str):
            Optional. The branch of the event. The format is like
            agent_1.agent_2.agent_3, where agent_1 is the parent of
            agent_2, and agent_2 is the parent of agent_3. Branch is
            used when multiple child agents shouldn't see their
            siblings' conversation history.
        custom_metadata (google.protobuf.struct_pb2.Struct):
            The custom metadata of the LlmResponse.
    """

    grounding_metadata: gca_content.GroundingMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_content.GroundingMetadata,
    )
    partial: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    turn_complete: bool = proto.Field(
        proto.BOOL,
        number=3,
    )
    interrupted: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    long_running_tool_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=5,
    )
    branch: str = proto.Field(
        proto.STRING,
        number=6,
    )
    custom_metadata: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=7,
        message=struct_pb2.Struct,
    )


class EventActions(proto.Message):
    r"""Actions are parts of events that are executed by the agent.

    Attributes:
        skip_summarization (bool):
            Optional. If true, it won't call model to summarize function
            response. Only used for function_response event.
        state_delta (google.protobuf.struct_pb2.Struct):
            Optional. Indicates that the event is
            updating the state with the given delta.
        artifact_delta (MutableMapping[str, int]):
            Optional. Indicates that the event is
            updating an artifact. key is the filename, value
            is the version.
        transfer_to_agent (bool):
            Deprecated. If set, the event transfers to
            the specified agent.
        escalate (bool):
            Optional. The agent is escalating to a higher
            level agent.
        requested_auth_configs (google.protobuf.struct_pb2.Struct):
            Optional. Will only be set by a tool response
            indicating tool request euc. Struct key is the
            function call id since one function call
            response (from model) could correspond to
            multiple function calls. Struct value is the
            required auth config, which can be another
            struct.
        transfer_agent (str):
            Optional. If set, the event transfers to the
            specified agent.
    """

    skip_summarization: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    state_delta: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Struct,
    )
    artifact_delta: MutableMapping[str, int] = proto.MapField(
        proto.STRING,
        proto.INT32,
        number=3,
    )
    transfer_to_agent: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    escalate: bool = proto.Field(
        proto.BOOL,
        number=6,
    )
    requested_auth_configs: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=7,
        message=struct_pb2.Struct,
    )
    transfer_agent: str = proto.Field(
        proto.STRING,
        number=8,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
