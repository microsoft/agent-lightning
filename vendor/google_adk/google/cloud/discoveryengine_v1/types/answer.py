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

from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import safety

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "Answer",
    },
)


class Answer(proto.Message):
    r"""Defines an answer.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Immutable. Fully qualified name
            ``projects/{project}/locations/global/collections/{collection}/engines/{engine}/sessions/*/answers/*``
        state (google.cloud.discoveryengine_v1.types.Answer.State):
            The state of the answer generation.
        answer_text (str):
            The textual answer.
        grounding_score (float):
            A score in the range of [0, 1] describing how grounded the
            answer is by the reference chunks.

            This field is a member of `oneof`_ ``_grounding_score``.
        citations (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.Citation]):
            Citations.
        grounding_supports (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.GroundingSupport]):
            Optional. Grounding supports.
        references (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.Reference]):
            References.
        related_questions (MutableSequence[str]):
            Suggested related questions.
        steps (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.Step]):
            Answer generation steps.
        query_understanding_info (google.cloud.discoveryengine_v1.types.Answer.QueryUnderstandingInfo):
            Query understanding information.
        answer_skipped_reasons (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.AnswerSkippedReason]):
            Additional answer-skipped reasons. This
            provides the reason for ignored cases. If
            nothing is skipped, this field is not set.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Answer creation timestamp.
        complete_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Answer completed timestamp.
        safety_ratings (MutableSequence[google.cloud.discoveryengine_v1.types.SafetyRating]):
            Optional. Safety ratings.
    """

    class State(proto.Enum):
        r"""Enumeration of the state of the answer generation.

        Values:
            STATE_UNSPECIFIED (0):
                Unknown.
            IN_PROGRESS (1):
                Answer generation is currently in progress.
            FAILED (2):
                Answer generation currently failed.
            SUCCEEDED (3):
                Answer generation has succeeded.
            STREAMING (4):
                Answer generation is currently in progress.
        """
        STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        FAILED = 2
        SUCCEEDED = 3
        STREAMING = 4

    class AnswerSkippedReason(proto.Enum):
        r"""An enum for answer skipped reasons.

        Values:
            ANSWER_SKIPPED_REASON_UNSPECIFIED (0):
                Default value. The answer skipped reason is
                not specified.
            ADVERSARIAL_QUERY_IGNORED (1):
                The adversarial query ignored case.
            NON_ANSWER_SEEKING_QUERY_IGNORED (2):
                The non-answer seeking query ignored case

                Google skips the answer if the query is chit
                chat.
            OUT_OF_DOMAIN_QUERY_IGNORED (3):
                The out-of-domain query ignored case.

                Google skips the answer if there are no
                high-relevance search results.
            POTENTIAL_POLICY_VIOLATION (4):
                The potential policy violation case.

                Google skips the answer if there is a potential
                policy violation detected. This includes content
                that may be violent or toxic.
            NO_RELEVANT_CONTENT (5):
                The no relevant content case.

                Google skips the answer if there is no relevant
                content in the retrieved search results.
            JAIL_BREAKING_QUERY_IGNORED (6):
                The jail-breaking query ignored case.

                For example, "Reply in the tone of a competing
                company's CEO". Google skips the answer if the
                query is classified as a jail-breaking query.
            CUSTOMER_POLICY_VIOLATION (7):
                The customer policy violation case.

                Google skips the summary if there is a customer
                policy violation detected. The policy is defined
                by the customer.
            NON_ANSWER_SEEKING_QUERY_IGNORED_V2 (8):
                The non-answer seeking query ignored case.

                Google skips the answer if the query doesn't
                have clear intent.
            LOW_GROUNDED_ANSWER (9):
                The low-grounded answer case.

                Google skips the answer if a well grounded
                answer was unable to be generated.
            USER_DEFINED_CLASSIFICATION_QUERY_IGNORED (10):
                The user defined query classification ignored
                case.
                Google skips the answer if the query is
                classified as a user defined query
                classification.
            UNHELPFUL_ANSWER (11):
                The unhelpful answer case.

                Google skips the answer if the answer is not
                helpful. This can be due to a variety of
                factors, including but not limited to: the query
                is not answerable, the answer is not relevant to
                the query, or the answer is not well-formatted.
        """
        ANSWER_SKIPPED_REASON_UNSPECIFIED = 0
        ADVERSARIAL_QUERY_IGNORED = 1
        NON_ANSWER_SEEKING_QUERY_IGNORED = 2
        OUT_OF_DOMAIN_QUERY_IGNORED = 3
        POTENTIAL_POLICY_VIOLATION = 4
        NO_RELEVANT_CONTENT = 5
        JAIL_BREAKING_QUERY_IGNORED = 6
        CUSTOMER_POLICY_VIOLATION = 7
        NON_ANSWER_SEEKING_QUERY_IGNORED_V2 = 8
        LOW_GROUNDED_ANSWER = 9
        USER_DEFINED_CLASSIFICATION_QUERY_IGNORED = 10
        UNHELPFUL_ANSWER = 11

    class Citation(proto.Message):
        r"""Citation info for a segment.

        Attributes:
            start_index (int):
                Index indicates the start of the segment,
                measured in bytes (UTF-8 unicode). If there are
                multi-byte characters,such as non-ASCII
                characters, the index measurement is longer than
                the string length.
            end_index (int):
                End of the attributed segment, exclusive.
                Measured in bytes (UTF-8 unicode). If there are
                multi-byte characters,such as non-ASCII
                characters, the index measurement is longer than
                the string length.
            sources (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.CitationSource]):
                Citation sources for the attributed segment.
        """

        start_index: int = proto.Field(
            proto.INT64,
            number=1,
        )
        end_index: int = proto.Field(
            proto.INT64,
            number=2,
        )
        sources: MutableSequence["Answer.CitationSource"] = proto.RepeatedField(
            proto.MESSAGE,
            number=3,
            message="Answer.CitationSource",
        )

    class CitationSource(proto.Message):
        r"""Citation source.

        Attributes:
            reference_id (str):
                ID of the citation source.
        """

        reference_id: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class GroundingSupport(proto.Message):
        r"""Grounding support for a claim in ``answer_text``.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            start_index (int):
                Required. Index indicates the start of the
                claim, measured in bytes (UTF-8 unicode).
            end_index (int):
                Required. End of the claim, exclusive.
            grounding_score (float):
                A score in the range of [0, 1] describing how grounded is a
                specific claim by the references. Higher value means that
                the claim is better supported by the reference chunks.

                This field is a member of `oneof`_ ``_grounding_score``.
            grounding_check_required (bool):
                Indicates that this claim required grounding check. When the
                system decided this claim didn't require
                attribution/grounding check, this field is set to false. In
                that case, no grounding check was done for the claim and
                therefore ``grounding_score``, ``sources`` is not returned.

                This field is a member of `oneof`_ ``_grounding_check_required``.
            sources (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.CitationSource]):
                Optional. Citation sources for the claim.
        """

        start_index: int = proto.Field(
            proto.INT64,
            number=1,
        )
        end_index: int = proto.Field(
            proto.INT64,
            number=2,
        )
        grounding_score: float = proto.Field(
            proto.DOUBLE,
            number=3,
            optional=True,
        )
        grounding_check_required: bool = proto.Field(
            proto.BOOL,
            number=4,
            optional=True,
        )
        sources: MutableSequence["Answer.CitationSource"] = proto.RepeatedField(
            proto.MESSAGE,
            number=5,
            message="Answer.CitationSource",
        )

    class Reference(proto.Message):
        r"""Reference.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            unstructured_document_info (google.cloud.discoveryengine_v1.types.Answer.Reference.UnstructuredDocumentInfo):
                Unstructured document information.

                This field is a member of `oneof`_ ``content``.
            chunk_info (google.cloud.discoveryengine_v1.types.Answer.Reference.ChunkInfo):
                Chunk information.

                This field is a member of `oneof`_ ``content``.
            structured_document_info (google.cloud.discoveryengine_v1.types.Answer.Reference.StructuredDocumentInfo):
                Structured document information.

                This field is a member of `oneof`_ ``content``.
        """

        class UnstructuredDocumentInfo(proto.Message):
            r"""Unstructured document information.

            Attributes:
                document (str):
                    Document resource name.
                uri (str):
                    URI for the document.
                title (str):
                    Title.
                chunk_contents (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.Reference.UnstructuredDocumentInfo.ChunkContent]):
                    List of cited chunk contents derived from
                    document content.
                struct_data (google.protobuf.struct_pb2.Struct):
                    The structured JSON metadata for the
                    document. It is populated from the struct data
                    from the Chunk in search result.
            """

            class ChunkContent(proto.Message):
                r"""Chunk content.

                .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                Attributes:
                    content (str):
                        Chunk textual content.
                    page_identifier (str):
                        Page identifier.
                    relevance_score (float):
                        The relevance of the chunk for a given query.
                        Values range from 0.0 (completely irrelevant) to
                        1.0 (completely relevant). This value is for
                        informational purpose only. It may change for
                        the same query and chunk at any time due to a
                        model retraining or change in implementation.

                        This field is a member of `oneof`_ ``_relevance_score``.
                """

                content: str = proto.Field(
                    proto.STRING,
                    number=1,
                )
                page_identifier: str = proto.Field(
                    proto.STRING,
                    number=2,
                )
                relevance_score: float = proto.Field(
                    proto.FLOAT,
                    number=3,
                    optional=True,
                )

            document: str = proto.Field(
                proto.STRING,
                number=1,
            )
            uri: str = proto.Field(
                proto.STRING,
                number=2,
            )
            title: str = proto.Field(
                proto.STRING,
                number=3,
            )
            chunk_contents: MutableSequence[
                "Answer.Reference.UnstructuredDocumentInfo.ChunkContent"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=4,
                message="Answer.Reference.UnstructuredDocumentInfo.ChunkContent",
            )
            struct_data: struct_pb2.Struct = proto.Field(
                proto.MESSAGE,
                number=5,
                message=struct_pb2.Struct,
            )

        class ChunkInfo(proto.Message):
            r"""Chunk information.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                chunk (str):
                    Chunk resource name.
                content (str):
                    Chunk textual content.
                relevance_score (float):
                    The relevance of the chunk for a given query.
                    Values range from 0.0 (completely irrelevant) to
                    1.0 (completely relevant). This value is for
                    informational purpose only. It may change for
                    the same query and chunk at any time due to a
                    model retraining or change in implementation.

                    This field is a member of `oneof`_ ``_relevance_score``.
                document_metadata (google.cloud.discoveryengine_v1.types.Answer.Reference.ChunkInfo.DocumentMetadata):
                    Document metadata.
            """

            class DocumentMetadata(proto.Message):
                r"""Document metadata.

                Attributes:
                    document (str):
                        Document resource name.
                    uri (str):
                        URI for the document.
                    title (str):
                        Title.
                    page_identifier (str):
                        Page identifier.
                    struct_data (google.protobuf.struct_pb2.Struct):
                        The structured JSON metadata for the
                        document. It is populated from the struct data
                        from the Chunk in search result.
                """

                document: str = proto.Field(
                    proto.STRING,
                    number=1,
                )
                uri: str = proto.Field(
                    proto.STRING,
                    number=2,
                )
                title: str = proto.Field(
                    proto.STRING,
                    number=3,
                )
                page_identifier: str = proto.Field(
                    proto.STRING,
                    number=4,
                )
                struct_data: struct_pb2.Struct = proto.Field(
                    proto.MESSAGE,
                    number=5,
                    message=struct_pb2.Struct,
                )

            chunk: str = proto.Field(
                proto.STRING,
                number=1,
            )
            content: str = proto.Field(
                proto.STRING,
                number=2,
            )
            relevance_score: float = proto.Field(
                proto.FLOAT,
                number=3,
                optional=True,
            )
            document_metadata: "Answer.Reference.ChunkInfo.DocumentMetadata" = (
                proto.Field(
                    proto.MESSAGE,
                    number=4,
                    message="Answer.Reference.ChunkInfo.DocumentMetadata",
                )
            )

        class StructuredDocumentInfo(proto.Message):
            r"""Structured search information.

            Attributes:
                document (str):
                    Document resource name.
                struct_data (google.protobuf.struct_pb2.Struct):
                    Structured search data.
                title (str):
                    Output only. The title of the document.
                uri (str):
                    Output only. The URI of the document.
            """

            document: str = proto.Field(
                proto.STRING,
                number=1,
            )
            struct_data: struct_pb2.Struct = proto.Field(
                proto.MESSAGE,
                number=2,
                message=struct_pb2.Struct,
            )
            title: str = proto.Field(
                proto.STRING,
                number=3,
            )
            uri: str = proto.Field(
                proto.STRING,
                number=4,
            )

        unstructured_document_info: "Answer.Reference.UnstructuredDocumentInfo" = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                oneof="content",
                message="Answer.Reference.UnstructuredDocumentInfo",
            )
        )
        chunk_info: "Answer.Reference.ChunkInfo" = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="content",
            message="Answer.Reference.ChunkInfo",
        )
        structured_document_info: "Answer.Reference.StructuredDocumentInfo" = (
            proto.Field(
                proto.MESSAGE,
                number=3,
                oneof="content",
                message="Answer.Reference.StructuredDocumentInfo",
            )
        )

    class Step(proto.Message):
        r"""Step information.

        Attributes:
            state (google.cloud.discoveryengine_v1.types.Answer.Step.State):
                The state of the step.
            description (str):
                The description of the step.
            thought (str):
                The thought of the step.
            actions (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.Step.Action]):
                Actions.
        """

        class State(proto.Enum):
            r"""Enumeration of the state of the step.

            Values:
                STATE_UNSPECIFIED (0):
                    Unknown.
                IN_PROGRESS (1):
                    Step is currently in progress.
                FAILED (2):
                    Step currently failed.
                SUCCEEDED (3):
                    Step has succeeded.
            """
            STATE_UNSPECIFIED = 0
            IN_PROGRESS = 1
            FAILED = 2
            SUCCEEDED = 3

        class Action(proto.Message):
            r"""Action.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                search_action (google.cloud.discoveryengine_v1.types.Answer.Step.Action.SearchAction):
                    Search action.

                    This field is a member of `oneof`_ ``action``.
                observation (google.cloud.discoveryengine_v1.types.Answer.Step.Action.Observation):
                    Observation.
            """

            class SearchAction(proto.Message):
                r"""Search action.

                Attributes:
                    query (str):
                        The query to search.
                """

                query: str = proto.Field(
                    proto.STRING,
                    number=1,
                )

            class Observation(proto.Message):
                r"""Observation.

                Attributes:
                    search_results (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.Step.Action.Observation.SearchResult]):
                        Search results observed by the search action,
                        it can be snippets info or chunk info, depending
                        on the citation type set by the user.
                """

                class SearchResult(proto.Message):
                    r"""

                    Attributes:
                        document (str):
                            Document resource name.
                        uri (str):
                            URI for the document.
                        title (str):
                            Title.
                        snippet_info (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.Step.Action.Observation.SearchResult.SnippetInfo]):
                            If citation_type is DOCUMENT_LEVEL_CITATION, populate
                            document level snippets.
                        chunk_info (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.Step.Action.Observation.SearchResult.ChunkInfo]):
                            If citation_type is CHUNK_LEVEL_CITATION and chunk mode is
                            on, populate chunk info.
                        struct_data (google.protobuf.struct_pb2.Struct):
                            Data representation.
                            The structured JSON data for the document.
                            It's populated from the struct data from the
                            Document, or the Chunk in search result.
                    """

                    class SnippetInfo(proto.Message):
                        r"""Snippet information.

                        Attributes:
                            snippet (str):
                                Snippet content.
                            snippet_status (str):
                                Status of the snippet defined by the search
                                team.
                        """

                        snippet: str = proto.Field(
                            proto.STRING,
                            number=1,
                        )
                        snippet_status: str = proto.Field(
                            proto.STRING,
                            number=2,
                        )

                    class ChunkInfo(proto.Message):
                        r"""Chunk information.

                        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                        Attributes:
                            chunk (str):
                                Chunk resource name.
                            content (str):
                                Chunk textual content.
                            relevance_score (float):
                                The relevance of the chunk for a given query.
                                Values range from 0.0 (completely irrelevant) to
                                1.0 (completely relevant). This value is for
                                informational purpose only. It may change for
                                the same query and chunk at any time due to a
                                model retraining or change in implementation.

                                This field is a member of `oneof`_ ``_relevance_score``.
                        """

                        chunk: str = proto.Field(
                            proto.STRING,
                            number=1,
                        )
                        content: str = proto.Field(
                            proto.STRING,
                            number=2,
                        )
                        relevance_score: float = proto.Field(
                            proto.FLOAT,
                            number=3,
                            optional=True,
                        )

                    document: str = proto.Field(
                        proto.STRING,
                        number=1,
                    )
                    uri: str = proto.Field(
                        proto.STRING,
                        number=2,
                    )
                    title: str = proto.Field(
                        proto.STRING,
                        number=3,
                    )
                    snippet_info: MutableSequence[
                        "Answer.Step.Action.Observation.SearchResult.SnippetInfo"
                    ] = proto.RepeatedField(
                        proto.MESSAGE,
                        number=4,
                        message="Answer.Step.Action.Observation.SearchResult.SnippetInfo",
                    )
                    chunk_info: MutableSequence[
                        "Answer.Step.Action.Observation.SearchResult.ChunkInfo"
                    ] = proto.RepeatedField(
                        proto.MESSAGE,
                        number=5,
                        message="Answer.Step.Action.Observation.SearchResult.ChunkInfo",
                    )
                    struct_data: struct_pb2.Struct = proto.Field(
                        proto.MESSAGE,
                        number=6,
                        message=struct_pb2.Struct,
                    )

                search_results: MutableSequence[
                    "Answer.Step.Action.Observation.SearchResult"
                ] = proto.RepeatedField(
                    proto.MESSAGE,
                    number=2,
                    message="Answer.Step.Action.Observation.SearchResult",
                )

            search_action: "Answer.Step.Action.SearchAction" = proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="action",
                message="Answer.Step.Action.SearchAction",
            )
            observation: "Answer.Step.Action.Observation" = proto.Field(
                proto.MESSAGE,
                number=3,
                message="Answer.Step.Action.Observation",
            )

        state: "Answer.Step.State" = proto.Field(
            proto.ENUM,
            number=1,
            enum="Answer.Step.State",
        )
        description: str = proto.Field(
            proto.STRING,
            number=2,
        )
        thought: str = proto.Field(
            proto.STRING,
            number=3,
        )
        actions: MutableSequence["Answer.Step.Action"] = proto.RepeatedField(
            proto.MESSAGE,
            number=4,
            message="Answer.Step.Action",
        )

    class QueryUnderstandingInfo(proto.Message):
        r"""Query understanding information.

        Attributes:
            query_classification_info (MutableSequence[google.cloud.discoveryengine_v1.types.Answer.QueryUnderstandingInfo.QueryClassificationInfo]):
                Query classification information.
        """

        class QueryClassificationInfo(proto.Message):
            r"""Query classification information.

            Attributes:
                type_ (google.cloud.discoveryengine_v1.types.Answer.QueryUnderstandingInfo.QueryClassificationInfo.Type):
                    Query classification type.
                positive (bool):
                    Classification output.
            """

            class Type(proto.Enum):
                r"""Query classification types.

                Values:
                    TYPE_UNSPECIFIED (0):
                        Unspecified query classification type.
                    ADVERSARIAL_QUERY (1):
                        Adversarial query classification type.
                    NON_ANSWER_SEEKING_QUERY (2):
                        Non-answer-seeking query classification type,
                        for chit chat.
                    JAIL_BREAKING_QUERY (3):
                        Jail-breaking query classification type.
                    NON_ANSWER_SEEKING_QUERY_V2 (4):
                        Non-answer-seeking query classification type,
                        for no clear intent.
                    USER_DEFINED_CLASSIFICATION_QUERY (5):
                        User defined query classification type.
                """
                TYPE_UNSPECIFIED = 0
                ADVERSARIAL_QUERY = 1
                NON_ANSWER_SEEKING_QUERY = 2
                JAIL_BREAKING_QUERY = 3
                NON_ANSWER_SEEKING_QUERY_V2 = 4
                USER_DEFINED_CLASSIFICATION_QUERY = 5

            type_: "Answer.QueryUnderstandingInfo.QueryClassificationInfo.Type" = (
                proto.Field(
                    proto.ENUM,
                    number=1,
                    enum="Answer.QueryUnderstandingInfo.QueryClassificationInfo.Type",
                )
            )
            positive: bool = proto.Field(
                proto.BOOL,
                number=2,
            )

        query_classification_info: MutableSequence[
            "Answer.QueryUnderstandingInfo.QueryClassificationInfo"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="Answer.QueryUnderstandingInfo.QueryClassificationInfo",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=2,
        enum=State,
    )
    answer_text: str = proto.Field(
        proto.STRING,
        number=3,
    )
    grounding_score: float = proto.Field(
        proto.DOUBLE,
        number=12,
        optional=True,
    )
    citations: MutableSequence[Citation] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Citation,
    )
    grounding_supports: MutableSequence[GroundingSupport] = proto.RepeatedField(
        proto.MESSAGE,
        number=13,
        message=GroundingSupport,
    )
    references: MutableSequence[Reference] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message=Reference,
    )
    related_questions: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=6,
    )
    steps: MutableSequence[Step] = proto.RepeatedField(
        proto.MESSAGE,
        number=7,
        message=Step,
    )
    query_understanding_info: QueryUnderstandingInfo = proto.Field(
        proto.MESSAGE,
        number=10,
        message=QueryUnderstandingInfo,
    )
    answer_skipped_reasons: MutableSequence[AnswerSkippedReason] = proto.RepeatedField(
        proto.ENUM,
        number=11,
        enum=AnswerSkippedReason,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    complete_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    safety_ratings: MutableSequence[safety.SafetyRating] = proto.RepeatedField(
        proto.MESSAGE,
        number=14,
        message=safety.SafetyRating,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
