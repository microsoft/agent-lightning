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
from google.cloud.aiplatform_v1beta1.types import memory_bank
from google.cloud.aiplatform_v1beta1.types import operation
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateMemoryRequest",
        "CreateMemoryOperationMetadata",
        "GetMemoryRequest",
        "UpdateMemoryRequest",
        "UpdateMemoryOperationMetadata",
        "ListMemoriesRequest",
        "ListMemoriesResponse",
        "DeleteMemoryRequest",
        "DeleteMemoryOperationMetadata",
        "GenerateMemoriesRequest",
        "GenerateMemoriesResponse",
        "GenerateMemoriesOperationMetadata",
        "RetrieveMemoriesRequest",
        "RetrieveMemoriesResponse",
    },
)


class CreateMemoryRequest(proto.Message):
    r"""Request message for
    [MemoryBankService.CreateMemory][google.cloud.aiplatform.v1beta1.MemoryBankService.CreateMemory].

    Attributes:
        parent (str):
            Required. The resource name of the ReasoningEngine to create
            the Memory under. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
        memory (google.cloud.aiplatform_v1beta1.types.Memory):
            Required. The Memory to be created.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    memory: memory_bank.Memory = proto.Field(
        proto.MESSAGE,
        number=2,
        message=memory_bank.Memory,
    )


class CreateMemoryOperationMetadata(proto.Message):
    r"""Details of
    [MemoryBankService.CreateMemory][google.cloud.aiplatform.v1beta1.MemoryBankService.CreateMemory]
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


class GetMemoryRequest(proto.Message):
    r"""Request message for
    [MemoryBankService.GetMemory][google.cloud.aiplatform.v1beta1.MemoryBankService.GetMemory].

    Attributes:
        name (str):
            Required. The resource name of the Memory. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/memories/{memory}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class UpdateMemoryRequest(proto.Message):
    r"""Request message for
    [MemoryBankService.UpdateMemory][google.cloud.aiplatform.v1beta1.MemoryBankService.UpdateMemory].

    Attributes:
        memory (google.cloud.aiplatform_v1beta1.types.Memory):
            Required. The Memory which replaces the
            resource on the server.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Mask specifying which fields to update. Supported
            fields:

            ::

               * `display_name`
               * `description`
               * `fact`
    """

    memory: memory_bank.Memory = proto.Field(
        proto.MESSAGE,
        number=1,
        message=memory_bank.Memory,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class UpdateMemoryOperationMetadata(proto.Message):
    r"""Details of
    [MemoryBankService.UpdateMemory][google.cloud.aiplatform.v1beta1.MemoryBankService.UpdateMemory]
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


class ListMemoriesRequest(proto.Message):
    r"""Request message for
    [MemoryBankService.ListMemories][google.cloud.aiplatform.v1beta1.MemoryBankService.ListMemories].

    Attributes:
        parent (str):
            Required. The resource name of the ReasoningEngine to list
            the Memories under. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
        filter (str):
            Optional. The standard list filter. More detail in
            `AIP-160 <https://google.aip.dev/160>`__.

            Supported fields (equality match only):

            - ``scope`` (as a JSON string)
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListMemoriesResponse(proto.Message):
    r"""Response message for
    [MemoryBankService.ListMemories][google.cloud.aiplatform.v1beta1.MemoryBankService.ListMemories].

    Attributes:
        memories (MutableSequence[google.cloud.aiplatform_v1beta1.types.Memory]):
            List of Memories in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListMemoriesRequest.page_token][google.cloud.aiplatform.v1beta1.ListMemoriesRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    memories: MutableSequence[memory_bank.Memory] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=memory_bank.Memory,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteMemoryRequest(proto.Message):
    r"""Request message for
    [MemoryBankService.DeleteMemory][google.cloud.aiplatform.v1beta1.MemoryBankService.DeleteMemory].

    Attributes:
        name (str):
            Required. The resource name of the Memory to delete. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/memories/{memory}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class DeleteMemoryOperationMetadata(proto.Message):
    r"""Details of
    [MemoryBankService.DeleteMemory][google.cloud.aiplatform.v1beta1.MemoryBankService.DeleteMemory]
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


class GenerateMemoriesRequest(proto.Message):
    r"""Request message for
    [MemoryBankService.GenerateMemories][google.cloud.aiplatform.v1beta1.MemoryBankService.GenerateMemories].

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        vertex_session_source (google.cloud.aiplatform_v1beta1.types.GenerateMemoriesRequest.VertexSessionSource):
            Defines a Vertex Session as the source
            content from which to generate memories.

            This field is a member of `oneof`_ ``source``.
        direct_contents_source (google.cloud.aiplatform_v1beta1.types.GenerateMemoriesRequest.DirectContentsSource):
            Defines a direct source of content as the
            source content from which to generate memories.

            This field is a member of `oneof`_ ``source``.
        direct_memories_source (google.cloud.aiplatform_v1beta1.types.GenerateMemoriesRequest.DirectMemoriesSource):
            Defines a direct source of memories that should be uploaded
            to Memory Bank. This is similar to ``CreateMemory``, but it
            allows for consolidation between these new memories and
            existing memories for the same scope.

            This field is a member of `oneof`_ ``source``.
        parent (str):
            Required. The resource name of the ReasoningEngine to
            generate memories for. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
        disable_consolidation (bool):
            Optional. If true, generated memories will
            not be consolidated with existing memories; all
            generated memories will be added as new memories
            regardless of whether they are duplicates of or
            contradictory to existing memories. By default,
            memory consolidation is enabled.
        scope (MutableMapping[str, str]):
            Optional. The scope of the memories that should be
            generated. Memories will be consolidated across memories
            with the same scope. Must be provided unless the scope is
            defined in the source content. If ``scope`` is provided, it
            will override the scope defined in the source content. Scope
            values cannot contain the wildcard character '\*'.
    """

    class VertexSessionSource(proto.Message):
        r"""Defines an Agent Engine Session from which to generate the memories.
        If ``scope`` is not provided, the scope will be extracted from the
        Session (i.e. {"user_id": sesison.user_id}).

        Attributes:
            session (str):
                Required. The resource name of the Session to generate
                memories for. Format:
                ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}/sessions/{session}``
            start_time (google.protobuf.timestamp_pb2.Timestamp):
                Optional. Time range to define which session
                events should be used to generate memories.
                Start time (inclusive) of the time range. If not
                set, the start time is unbounded.
            end_time (google.protobuf.timestamp_pb2.Timestamp):
                Optional. End time (exclusive) of the time
                range. If not set, the end time is unbounded.
        """

        session: str = proto.Field(
            proto.STRING,
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

    class DirectContentsSource(proto.Message):
        r"""Defines a direct source of content from which to generate the
        memories.

        Attributes:
            events (MutableSequence[google.cloud.aiplatform_v1beta1.types.GenerateMemoriesRequest.DirectContentsSource.Event]):
                Required. The source content (i.e. chat
                history) to generate memories from.
        """

        class Event(proto.Message):
            r"""A single piece of conversation from which to generate
            memories.

            Attributes:
                content (google.cloud.aiplatform_v1beta1.types.Content):
                    Required. A single piece of content from
                    which to generate memories.
            """

            content: gca_content.Content = proto.Field(
                proto.MESSAGE,
                number=1,
                message=gca_content.Content,
            )

        events: MutableSequence[
            "GenerateMemoriesRequest.DirectContentsSource.Event"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="GenerateMemoriesRequest.DirectContentsSource.Event",
        )

    class DirectMemoriesSource(proto.Message):
        r"""Defines a direct source of memories that should be uploaded
        to Memory Bank with consolidation.

        Attributes:
            direct_memories (MutableSequence[google.cloud.aiplatform_v1beta1.types.GenerateMemoriesRequest.DirectMemoriesSource.DirectMemory]):
                Required. The direct memories to upload to
                Memory Bank. At most 5 direct memories are
                allowed per request.
        """

        class DirectMemory(proto.Message):
            r"""A direct memory to upload to Memory Bank.

            Attributes:
                fact (str):
                    Required. The fact to consolidate with
                    existing memories.
            """

            fact: str = proto.Field(
                proto.STRING,
                number=1,
            )

        direct_memories: MutableSequence[
            "GenerateMemoriesRequest.DirectMemoriesSource.DirectMemory"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="GenerateMemoriesRequest.DirectMemoriesSource.DirectMemory",
        )

    vertex_session_source: VertexSessionSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=VertexSessionSource,
    )
    direct_contents_source: DirectContentsSource = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="source",
        message=DirectContentsSource,
    )
    direct_memories_source: DirectMemoriesSource = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="source",
        message=DirectMemoriesSource,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    disable_consolidation: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    scope: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=8,
    )


class GenerateMemoriesResponse(proto.Message):
    r"""Response message for
    [MemoryBankService.GenerateMemories][google.cloud.aiplatform.v1beta1.MemoryBankService.GenerateMemories].

    Attributes:
        generated_memories (MutableSequence[google.cloud.aiplatform_v1beta1.types.GenerateMemoriesResponse.GeneratedMemory]):
            The generated memories.
    """

    class GeneratedMemory(proto.Message):
        r"""A memory generated by the operation.

        Attributes:
            memory (google.cloud.aiplatform_v1beta1.types.Memory):
                The generated Memory.
            action (google.cloud.aiplatform_v1beta1.types.GenerateMemoriesResponse.GeneratedMemory.Action):
                The action that was performed on the Memory.
        """

        class Action(proto.Enum):
            r"""Actions that can be performed on a Memory.

            Values:
                ACTION_UNSPECIFIED (0):
                    Action is unspecified.
                CREATED (1):
                    The memory was created.
                UPDATED (2):
                    The memory was updated. The ``fact`` field may not be
                    updated if the existing fact is still accurate.
                DELETED (3):
                    The memory was deleted.
            """

            ACTION_UNSPECIFIED = 0
            CREATED = 1
            UPDATED = 2
            DELETED = 3

        memory: memory_bank.Memory = proto.Field(
            proto.MESSAGE,
            number=1,
            message=memory_bank.Memory,
        )
        action: "GenerateMemoriesResponse.GeneratedMemory.Action" = proto.Field(
            proto.ENUM,
            number=2,
            enum="GenerateMemoriesResponse.GeneratedMemory.Action",
        )

    generated_memories: MutableSequence[GeneratedMemory] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=GeneratedMemory,
    )


class GenerateMemoriesOperationMetadata(proto.Message):
    r"""Details of
    [MemoryBankService.GenerateMemories][google.cloud.aiplatform.v1beta1.MemoryBankService.GenerateMemories]
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


class RetrieveMemoriesRequest(proto.Message):
    r"""Request message for
    [MemoryBankService.RetrieveMemories][google.cloud.aiplatform.v1beta1.MemoryBankService.RetrieveMemories].

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        similarity_search_params (google.cloud.aiplatform_v1beta1.types.RetrieveMemoriesRequest.SimilaritySearchParams):
            Parameters for semantic similarity search
            based retrieval.

            This field is a member of `oneof`_ ``retrieval_params``.
        simple_retrieval_params (google.cloud.aiplatform_v1beta1.types.RetrieveMemoriesRequest.SimpleRetrievalParams):
            Parameters for simple (non-similarity search)
            retrieval.

            This field is a member of `oneof`_ ``retrieval_params``.
        parent (str):
            Required. The resource name of the ReasoningEngine to
            retrieve memories from. Format:
            ``projects/{project}/locations/{location}/reasoningEngines/{reasoning_engine}``
        scope (MutableMapping[str, str]):
            Required. The scope of the memories to retrieve. A memory
            must have exactly the same scope (``Memory.scope``) as the
            scope provided here to be retrieved (same keys and values).
            Order does not matter, but it is case-sensitive.
    """

    class SimilaritySearchParams(proto.Message):
        r"""Parameters for semantic similarity search based retrieval.

        Attributes:
            search_query (str):
                Required. Query to use for similarity search retrieval. If
                provided, then the parent ReasoningEngine must have
                [ReasoningEngineContextSpec.MemoryBankConfig.SimilaritySearchConfig][google.cloud.aiplatform.v1beta1.ReasoningEngineContextSpec.MemoryBankConfig.SimilaritySearchConfig]
                set.
            top_k (int):
                Optional. The maximum number of memories to
                return. The service may return fewer than this
                value. If unspecified, at most 3 memories will
                be returned. The maximum value is 100; values
                above 100 will be coerced to 100.
        """

        search_query: str = proto.Field(
            proto.STRING,
            number=1,
        )
        top_k: int = proto.Field(
            proto.INT32,
            number=2,
        )

    class SimpleRetrievalParams(proto.Message):
        r"""Parameters for simple (non-similarity search) retrieval.

        Attributes:
            page_size (int):
                Optional. The maximum number of memories to
                return. The service may return fewer than this
                value. If unspecified, at most 3 memories will
                be returned. The maximum value is 100; values
                above 100 will be coerced to 100.
            page_token (str):
                Optional. A page token, received from a previous
                ``RetrieveMemories`` call. Provide this to retrieve the
                subsequent page.
        """

        page_size: int = proto.Field(
            proto.INT32,
            number=1,
        )
        page_token: str = proto.Field(
            proto.STRING,
            number=2,
        )

    similarity_search_params: SimilaritySearchParams = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="retrieval_params",
        message=SimilaritySearchParams,
    )
    simple_retrieval_params: SimpleRetrievalParams = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="retrieval_params",
        message=SimpleRetrievalParams,
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    scope: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=8,
    )


class RetrieveMemoriesResponse(proto.Message):
    r"""Response message for
    [MemoryBankService.RetrieveMemories][google.cloud.aiplatform.v1beta1.MemoryBankService.RetrieveMemories].

    Attributes:
        retrieved_memories (MutableSequence[google.cloud.aiplatform_v1beta1.types.RetrieveMemoriesResponse.RetrievedMemory]):
            The retrieved memories.
        next_page_token (str):
            A token that can be sent as ``page_token`` to retrieve the
            next page. If this field is omitted, there are no subsequent
            pages. This token is not set if similarity search was used
            for retrieval.
    """

    class RetrievedMemory(proto.Message):
        r"""A retrieved memory.

        Attributes:
            memory (google.cloud.aiplatform_v1beta1.types.Memory):
                The retrieved Memory.
            distance (float):
                The distance between the query and the
                retrieved Memory. Smaller values indicate more
                similar memories. This is only set if similarity
                search was used for retrieval.
        """

        memory: memory_bank.Memory = proto.Field(
            proto.MESSAGE,
            number=1,
            message=memory_bank.Memory,
        )
        distance: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )

    @property
    def raw_page(self):
        return self

    retrieved_memories: MutableSequence[RetrievedMemory] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=RetrievedMemory,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
