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
import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import conversation as gcd_conversation
from google.cloud.discoveryengine_v1.types import answer as gcd_answer
from google.cloud.discoveryengine_v1.types import safety, search_service
from google.cloud.discoveryengine_v1.types import session as gcd_session

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "ConverseConversationRequest",
        "ConverseConversationResponse",
        "CreateConversationRequest",
        "UpdateConversationRequest",
        "DeleteConversationRequest",
        "GetConversationRequest",
        "ListConversationsRequest",
        "ListConversationsResponse",
        "AnswerQueryRequest",
        "AnswerQueryResponse",
        "GetAnswerRequest",
        "CreateSessionRequest",
        "UpdateSessionRequest",
        "DeleteSessionRequest",
        "GetSessionRequest",
        "ListSessionsRequest",
        "ListSessionsResponse",
    },
)


class ConverseConversationRequest(proto.Message):
    r"""Request message for
    [ConversationalSearchService.ConverseConversation][google.cloud.discoveryengine.v1.ConversationalSearchService.ConverseConversation]
    method.

    Attributes:
        name (str):
            Required. The resource name of the Conversation to get.
            Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/conversations/{conversation_id}``.
            Use
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/conversations/-``
            to activate auto session mode, which automatically creates a
            new conversation inside a ConverseConversation session.
        query (google.cloud.discoveryengine_v1.types.TextInput):
            Required. Current user input.
        serving_config (str):
            The resource name of the Serving Config to use. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/servingConfigs/{serving_config_id}``
            If this is not set, the default serving config will be used.
        conversation (google.cloud.discoveryengine_v1.types.Conversation):
            The conversation to be used by auto session
            only. The name field will be ignored as we
            automatically assign new name for the
            conversation in auto session.
        safe_search (bool):
            Whether to turn on safe search.
        user_labels (MutableMapping[str, str]):
            The user labels applied to a resource must meet the
            following requirements:

            - Each resource can have multiple labels, up to a maximum of
              64.
            - Each label must be a key-value pair.
            - Keys have a minimum length of 1 character and a maximum
              length of 63 characters and cannot be empty. Values can be
              empty and have a maximum length of 63 characters.
            - Keys and values can contain only lowercase letters,
              numeric characters, underscores, and dashes. All
              characters must use UTF-8 encoding, and international
              characters are allowed.
            - The key portion of a label must be unique. However, you
              can use the same key with multiple resources.
            - Keys must start with a lowercase letter or international
              character.

            See `Google Cloud
            Document <https://cloud.google.com/resource-manager/docs/creating-managing-labels#requirements>`__
            for more details.
        summary_spec (google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec.SummarySpec):
            A specification for configuring the summary
            returned in the response.
        filter (str):
            The filter syntax consists of an expression language for
            constructing a predicate from one or more fields of the
            documents being filtered. Filter expression is
            case-sensitive. This will be used to filter search results
            which may affect the summary response.

            If this field is unrecognizable, an ``INVALID_ARGUMENT`` is
            returned.

            Filtering in Vertex AI Search is done by mapping the LHS
            filter key to a key property defined in the Vertex AI Search
            backend -- this mapping is defined by the customer in their
            schema. For example a media customer might have a field
            'name' in their schema. In this case the filter would look
            like this: filter --> name:'ANY("king kong")'

            For more information about filtering including syntax and
            filter operators, see
            `Filter <https://cloud.google.com/generative-ai-app-builder/docs/filter-search-metadata>`__
        boost_spec (google.cloud.discoveryengine_v1.types.SearchRequest.BoostSpec):
            Boost specification to boost certain documents in search
            results which may affect the converse response. For more
            information on boosting, see
            `Boosting <https://cloud.google.com/retail/docs/boosting#boost>`__
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: gcd_conversation.TextInput = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_conversation.TextInput,
    )
    serving_config: str = proto.Field(
        proto.STRING,
        number=3,
    )
    conversation: gcd_conversation.Conversation = proto.Field(
        proto.MESSAGE,
        number=5,
        message=gcd_conversation.Conversation,
    )
    safe_search: bool = proto.Field(
        proto.BOOL,
        number=6,
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=7,
    )
    summary_spec: search_service.SearchRequest.ContentSearchSpec.SummarySpec = (
        proto.Field(
            proto.MESSAGE,
            number=8,
            message=search_service.SearchRequest.ContentSearchSpec.SummarySpec,
        )
    )
    filter: str = proto.Field(
        proto.STRING,
        number=9,
    )
    boost_spec: search_service.SearchRequest.BoostSpec = proto.Field(
        proto.MESSAGE,
        number=10,
        message=search_service.SearchRequest.BoostSpec,
    )


class ConverseConversationResponse(proto.Message):
    r"""Response message for
    [ConversationalSearchService.ConverseConversation][google.cloud.discoveryengine.v1.ConversationalSearchService.ConverseConversation]
    method.

    Attributes:
        reply (google.cloud.discoveryengine_v1.types.Reply):
            Answer to the current query.
        conversation (google.cloud.discoveryengine_v1.types.Conversation):
            Updated conversation including the answer.
        search_results (MutableSequence[google.cloud.discoveryengine_v1.types.SearchResponse.SearchResult]):
            Search Results.
    """

    reply: gcd_conversation.Reply = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_conversation.Reply,
    )
    conversation: gcd_conversation.Conversation = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_conversation.Conversation,
    )
    search_results: MutableSequence[
        search_service.SearchResponse.SearchResult
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=search_service.SearchResponse.SearchResult,
    )


class CreateConversationRequest(proto.Message):
    r"""Request for CreateConversation method.

    Attributes:
        parent (str):
            Required. Full resource name of parent data store. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}``
        conversation (google.cloud.discoveryengine_v1.types.Conversation):
            Required. The conversation to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    conversation: gcd_conversation.Conversation = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_conversation.Conversation,
    )


class UpdateConversationRequest(proto.Message):
    r"""Request for UpdateConversation method.

    Attributes:
        conversation (google.cloud.discoveryengine_v1.types.Conversation):
            Required. The Conversation to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            [Conversation][google.cloud.discoveryengine.v1.Conversation]
            to update. The following are NOT supported:

            - [Conversation.name][google.cloud.discoveryengine.v1.Conversation.name]

            If not set or empty, all supported fields are updated.
    """

    conversation: gcd_conversation.Conversation = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_conversation.Conversation,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteConversationRequest(proto.Message):
    r"""Request for DeleteConversation method.

    Attributes:
        name (str):
            Required. The resource name of the Conversation to delete.
            Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/conversations/{conversation_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetConversationRequest(proto.Message):
    r"""Request for GetConversation method.

    Attributes:
        name (str):
            Required. The resource name of the Conversation to get.
            Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/conversations/{conversation_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListConversationsRequest(proto.Message):
    r"""Request for ListConversations method.

    Attributes:
        parent (str):
            Required. The data store resource name. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}``
        page_size (int):
            Maximum number of results to return. If
            unspecified, defaults to 50. Max allowed value
            is 1000.
        page_token (str):
            A page token, received from a previous ``ListConversations``
            call. Provide this to retrieve the subsequent page.
        filter (str):
            A filter to apply on the list results. The supported
            features are: user_pseudo_id, state.

            Example: "user_pseudo_id = some_id".
        order_by (str):
            A comma-separated list of fields to order by, sorted in
            ascending order. Use "desc" after a field name for
            descending. Supported fields:

            - ``update_time``
            - ``create_time``
            - ``conversation_name``

            Example: "update_time desc" "create_time".
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


class ListConversationsResponse(proto.Message):
    r"""Response for ListConversations method.

    Attributes:
        conversations (MutableSequence[google.cloud.discoveryengine_v1.types.Conversation]):
            All the Conversations for a given data store.
        next_page_token (str):
            Pagination token, if not returned indicates
            the last page.
    """

    @property
    def raw_page(self):
        return self

    conversations: MutableSequence[gcd_conversation.Conversation] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_conversation.Conversation,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class AnswerQueryRequest(proto.Message):
    r"""Request message for
    [ConversationalSearchService.AnswerQuery][google.cloud.discoveryengine.v1.ConversationalSearchService.AnswerQuery]
    method.

    Attributes:
        serving_config (str):
            Required. The resource name of the Search serving config,
            such as
            ``projects/*/locations/global/collections/default_collection/engines/*/servingConfigs/default_serving_config``,
            or
            ``projects/*/locations/global/collections/default_collection/dataStores/*/servingConfigs/default_serving_config``.
            This field is used to identify the serving configuration
            name, set of models used to make the search.
        query (google.cloud.discoveryengine_v1.types.Query):
            Required. Current user query.
        session (str):
            The session resource name. Not required.

            When session field is not set, the API is in sessionless
            mode.

            We support auto session mode: users can use the wildcard
            symbol ``-`` as session ID. A new ID will be automatically
            generated and assigned.
        safety_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SafetySpec):
            Model specification.
        related_questions_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.RelatedQuestionsSpec):
            Related questions specification.
        grounding_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.GroundingSpec):
            Optional. Grounding specification.
        answer_generation_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.AnswerGenerationSpec):
            Answer generation specification.
        search_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec):
            Search specification.
        query_understanding_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.QueryUnderstandingSpec):
            Query understanding specification.
        asynchronous_mode (bool):
            Deprecated: This field is deprecated. Streaming Answer API
            will be supported.

            Asynchronous mode control.

            If enabled, the response will be returned with
            answer/session resource name without final answer. The API
            users need to do the polling to get the latest status of
            answer/session by calling
            [ConversationalSearchService.GetAnswer][google.cloud.discoveryengine.v1.ConversationalSearchService.GetAnswer]
            or
            [ConversationalSearchService.GetSession][google.cloud.discoveryengine.v1.ConversationalSearchService.GetSession]
            method.
        user_pseudo_id (str):
            A unique identifier for tracking visitors. For example, this
            could be implemented with an HTTP cookie, which should be
            able to uniquely identify a visitor on a single device. This
            unique identifier should not change if the visitor logs in
            or out of the website.

            This field should NOT have a fixed value such as
            ``unknown_visitor``.

            The field must be a UTF-8 encoded string with a length limit
            of 128 characters. Otherwise, an ``INVALID_ARGUMENT`` error
            is returned.
        user_labels (MutableMapping[str, str]):
            The user labels applied to a resource must meet the
            following requirements:

            - Each resource can have multiple labels, up to a maximum of
              64.
            - Each label must be a key-value pair.
            - Keys have a minimum length of 1 character and a maximum
              length of 63 characters and cannot be empty. Values can be
              empty and have a maximum length of 63 characters.
            - Keys and values can contain only lowercase letters,
              numeric characters, underscores, and dashes. All
              characters must use UTF-8 encoding, and international
              characters are allowed.
            - The key portion of a label must be unique. However, you
              can use the same key with multiple resources.
            - Keys must start with a lowercase letter or international
              character.

            See `Google Cloud
            Document <https://cloud.google.com/resource-manager/docs/creating-managing-labels#requirements>`__
            for more details.
        end_user_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.EndUserSpec):
            Optional. End user specification.
    """

    class SafetySpec(proto.Message):
        r"""Safety specification. There are two use cases:

        1. when only safety_spec.enable is set, the BLOCK_LOW_AND_ABOVE
           threshold will be applied for all categories.
        2. when safety_spec.enable is set and some safety_settings are set,
           only specified safety_settings are applied.

        Attributes:
            enable (bool):
                Enable the safety filtering on the answer
                response. It is false by default.
            safety_settings (MutableSequence[google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SafetySpec.SafetySetting]):
                Optional. Safety settings. This settings are effective only
                when the safety_spec.enable is true.
        """

        class SafetySetting(proto.Message):
            r"""Safety settings.

            Attributes:
                category (google.cloud.discoveryengine_v1.types.HarmCategory):
                    Required. Harm category.
                threshold (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SafetySpec.SafetySetting.HarmBlockThreshold):
                    Required. The harm block threshold.
            """

            class HarmBlockThreshold(proto.Enum):
                r"""Probability based thresholds levels for blocking.

                Values:
                    HARM_BLOCK_THRESHOLD_UNSPECIFIED (0):
                        Unspecified harm block threshold.
                    BLOCK_LOW_AND_ABOVE (1):
                        Block low threshold and above (i.e. block
                        more).
                    BLOCK_MEDIUM_AND_ABOVE (2):
                        Block medium threshold and above.
                    BLOCK_ONLY_HIGH (3):
                        Block only high threshold (i.e. block less).
                    BLOCK_NONE (4):
                        Block none.
                    OFF (5):
                        Turn off the safety filter.
                """
                HARM_BLOCK_THRESHOLD_UNSPECIFIED = 0
                BLOCK_LOW_AND_ABOVE = 1
                BLOCK_MEDIUM_AND_ABOVE = 2
                BLOCK_ONLY_HIGH = 3
                BLOCK_NONE = 4
                OFF = 5

            category: safety.HarmCategory = proto.Field(
                proto.ENUM,
                number=1,
                enum=safety.HarmCategory,
            )
            threshold: "AnswerQueryRequest.SafetySpec.SafetySetting.HarmBlockThreshold" = proto.Field(
                proto.ENUM,
                number=2,
                enum="AnswerQueryRequest.SafetySpec.SafetySetting.HarmBlockThreshold",
            )

        enable: bool = proto.Field(
            proto.BOOL,
            number=1,
        )
        safety_settings: MutableSequence[
            "AnswerQueryRequest.SafetySpec.SafetySetting"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message="AnswerQueryRequest.SafetySpec.SafetySetting",
        )

    class RelatedQuestionsSpec(proto.Message):
        r"""Related questions specification.

        Attributes:
            enable (bool):
                Enable related questions feature if true.
        """

        enable: bool = proto.Field(
            proto.BOOL,
            number=1,
        )

    class GroundingSpec(proto.Message):
        r"""Grounding specification.

        Attributes:
            include_grounding_supports (bool):
                Optional. Specifies whether to include grounding_supports in
                the answer. The default value is ``false``.

                When this field is set to ``true``, returned answer will
                have ``grounding_score`` and will contain GroundingSupports
                for each claim.
            filtering_level (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.GroundingSpec.FilteringLevel):
                Optional. Specifies whether to enable the
                filtering based on grounding score and at what
                level.
        """

        class FilteringLevel(proto.Enum):
            r"""Level to filter based on answer grounding.

            Values:
                FILTERING_LEVEL_UNSPECIFIED (0):
                    Default is no filter
                FILTERING_LEVEL_LOW (1):
                    Filter answers based on a low threshold.
                FILTERING_LEVEL_HIGH (2):
                    Filter answers based on a high threshold.
            """
            FILTERING_LEVEL_UNSPECIFIED = 0
            FILTERING_LEVEL_LOW = 1
            FILTERING_LEVEL_HIGH = 2

        include_grounding_supports: bool = proto.Field(
            proto.BOOL,
            number=2,
        )
        filtering_level: "AnswerQueryRequest.GroundingSpec.FilteringLevel" = (
            proto.Field(
                proto.ENUM,
                number=3,
                enum="AnswerQueryRequest.GroundingSpec.FilteringLevel",
            )
        )

    class AnswerGenerationSpec(proto.Message):
        r"""Answer generation specification.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            model_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec):
                Answer generation model specification.
            prompt_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec):
                Answer generation prompt specification.
            include_citations (bool):
                Specifies whether to include citation metadata in the
                answer. The default value is ``false``.
            answer_language_code (str):
                Language code for Answer. Use language tags defined by
                `BCP47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__.
                Note: This is an experimental feature.
            ignore_adversarial_query (bool):
                Specifies whether to filter out adversarial queries. The
                default value is ``false``.

                Google employs search-query classification to detect
                adversarial queries. No answer is returned if the search
                query is classified as an adversarial query. For example, a
                user might ask a question regarding negative comments about
                the company or submit a query designed to generate unsafe,
                policy-violating output. If this field is set to ``true``,
                we skip generating answers for adversarial queries and
                return fallback messages instead.
            ignore_non_answer_seeking_query (bool):
                Specifies whether to filter out queries that are not
                answer-seeking. The default value is ``false``.

                Google employs search-query classification to detect
                answer-seeking queries. No answer is returned if the search
                query is classified as a non-answer seeking query. If this
                field is set to ``true``, we skip generating answers for
                non-answer seeking queries and return fallback messages
                instead.
            ignore_low_relevant_content (bool):
                Specifies whether to filter out queries that have low
                relevance.

                If this field is set to ``false``, all search results are
                used regardless of relevance to generate answers. If set to
                ``true`` or unset, the behavior will be determined
                automatically by the service.

                This field is a member of `oneof`_ ``_ignore_low_relevant_content``.
            ignore_jail_breaking_query (bool):
                Optional. Specifies whether to filter out jail-breaking
                queries. The default value is ``false``.

                Google employs search-query classification to detect
                jail-breaking queries. No summary is returned if the search
                query is classified as a jail-breaking query. A user might
                add instructions to the query to change the tone, style,
                language, content of the answer, or ask the model to act as
                a different entity, e.g. "Reply in the tone of a competing
                company's CEO". If this field is set to ``true``, we skip
                generating summaries for jail-breaking queries and return
                fallback messages instead.
        """

        class ModelSpec(proto.Message):
            r"""Answer Generation Model specification.

            Attributes:
                model_version (str):
                    Model version. If not set, it will use the
                    default stable model. Allowed values are:
                    stable, preview.
            """

            model_version: str = proto.Field(
                proto.STRING,
                number=1,
            )

        class PromptSpec(proto.Message):
            r"""Answer generation prompt specification.

            Attributes:
                preamble (str):
                    Customized preamble.
            """

            preamble: str = proto.Field(
                proto.STRING,
                number=1,
            )

        model_spec: "AnswerQueryRequest.AnswerGenerationSpec.ModelSpec" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="AnswerQueryRequest.AnswerGenerationSpec.ModelSpec",
        )
        prompt_spec: "AnswerQueryRequest.AnswerGenerationSpec.PromptSpec" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="AnswerQueryRequest.AnswerGenerationSpec.PromptSpec",
        )
        include_citations: bool = proto.Field(
            proto.BOOL,
            number=3,
        )
        answer_language_code: str = proto.Field(
            proto.STRING,
            number=4,
        )
        ignore_adversarial_query: bool = proto.Field(
            proto.BOOL,
            number=5,
        )
        ignore_non_answer_seeking_query: bool = proto.Field(
            proto.BOOL,
            number=6,
        )
        ignore_low_relevant_content: bool = proto.Field(
            proto.BOOL,
            number=7,
            optional=True,
        )
        ignore_jail_breaking_query: bool = proto.Field(
            proto.BOOL,
            number=8,
        )

    class SearchSpec(proto.Message):
        r"""Search specification.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            search_params (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchParams):
                Search parameters.

                This field is a member of `oneof`_ ``input``.
            search_result_list (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchResultList):
                Search result list.

                This field is a member of `oneof`_ ``input``.
        """

        class SearchParams(proto.Message):
            r"""Search parameters.

            Attributes:
                max_return_results (int):
                    Number of search results to return.
                    The default value is 10.
                filter (str):
                    The filter syntax consists of an expression language for
                    constructing a predicate from one or more fields of the
                    documents being filtered. Filter expression is
                    case-sensitive. This will be used to filter search results
                    which may affect the Answer response.

                    If this field is unrecognizable, an ``INVALID_ARGUMENT`` is
                    returned.

                    Filtering in Vertex AI Search is done by mapping the LHS
                    filter key to a key property defined in the Vertex AI Search
                    backend -- this mapping is defined by the customer in their
                    schema. For example a media customers might have a field
                    'name' in their schema. In this case the filter would look
                    like this: filter --> name:'ANY("king kong")'

                    For more information about filtering including syntax and
                    filter operators, see
                    `Filter <https://cloud.google.com/generative-ai-app-builder/docs/filter-search-metadata>`__
                boost_spec (google.cloud.discoveryengine_v1.types.SearchRequest.BoostSpec):
                    Boost specification to boost certain documents in search
                    results which may affect the answer query response. For more
                    information on boosting, see
                    `Boosting <https://cloud.google.com/retail/docs/boosting#boost>`__
                order_by (str):
                    The order in which documents are returned. Documents can be
                    ordered by a field in an
                    [Document][google.cloud.discoveryengine.v1.Document] object.
                    Leave it unset if ordered by relevance. ``order_by``
                    expression is case-sensitive. For more information on
                    ordering, see
                    `Ordering <https://cloud.google.com/retail/docs/filter-and-order#order>`__

                    If this field is unrecognizable, an ``INVALID_ARGUMENT`` is
                    returned.
                search_result_mode (google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec.SearchResultMode):
                    Specifies the search result mode. If unspecified, the search
                    result mode defaults to ``DOCUMENTS``. See `parse and chunk
                    documents <https://cloud.google.com/generative-ai-app-builder/docs/parse-chunk-documents>`__
                data_store_specs (MutableSequence[google.cloud.discoveryengine_v1.types.SearchRequest.DataStoreSpec]):
                    Specs defining dataStores to filter on in a
                    search call and configurations for those
                    dataStores. This is only considered for engines
                    with multiple dataStores use case. For single
                    dataStore within an engine, they should use the
                    specs at the top level.
            """

            max_return_results: int = proto.Field(
                proto.INT32,
                number=1,
            )
            filter: str = proto.Field(
                proto.STRING,
                number=2,
            )
            boost_spec: search_service.SearchRequest.BoostSpec = proto.Field(
                proto.MESSAGE,
                number=3,
                message=search_service.SearchRequest.BoostSpec,
            )
            order_by: str = proto.Field(
                proto.STRING,
                number=4,
            )
            search_result_mode: search_service.SearchRequest.ContentSearchSpec.SearchResultMode = proto.Field(
                proto.ENUM,
                number=5,
                enum=search_service.SearchRequest.ContentSearchSpec.SearchResultMode,
            )
            data_store_specs: MutableSequence[
                search_service.SearchRequest.DataStoreSpec
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=7,
                message=search_service.SearchRequest.DataStoreSpec,
            )

        class SearchResultList(proto.Message):
            r"""Search result list.

            Attributes:
                search_results (MutableSequence[google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult]):
                    Search results.
            """

            class SearchResult(proto.Message):
                r"""Search result.

                This message has `oneof`_ fields (mutually exclusive fields).
                For each oneof, at most one member field can be set at the same time.
                Setting any member of the oneof automatically clears all other
                members.

                .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                Attributes:
                    unstructured_document_info (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo):
                        Unstructured document information.

                        This field is a member of `oneof`_ ``content``.
                    chunk_info (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.ChunkInfo):
                        Chunk information.

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
                        document_contexts (MutableSequence[google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.DocumentContext]):
                            List of document contexts. The content will
                            be used for Answer Generation. This is supposed
                            to be the main content of the document that can
                            be long and comprehensive.
                        extractive_segments (MutableSequence[google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.ExtractiveSegment]):
                            List of extractive segments.
                        extractive_answers (MutableSequence[google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.ExtractiveAnswer]):
                            Deprecated: This field is deprecated and will have no effect
                            on the Answer generation. Please use document_contexts and
                            extractive_segments fields. List of extractive answers.
                    """

                    class DocumentContext(proto.Message):
                        r"""Document context.

                        Attributes:
                            page_identifier (str):
                                Page identifier.
                            content (str):
                                Document content to be used for answer
                                generation.
                        """

                        page_identifier: str = proto.Field(
                            proto.STRING,
                            number=1,
                        )
                        content: str = proto.Field(
                            proto.STRING,
                            number=2,
                        )

                    class ExtractiveSegment(proto.Message):
                        r"""Extractive segment.
                        `Guide <https://cloud.google.com/generative-ai-app-builder/docs/snippets#extractive-segments>`__
                        Answer generation will only use it if document_contexts is empty.
                        This is supposed to be shorter snippets.

                        Attributes:
                            page_identifier (str):
                                Page identifier.
                            content (str):
                                Extractive segment content.
                        """

                        page_identifier: str = proto.Field(
                            proto.STRING,
                            number=1,
                        )
                        content: str = proto.Field(
                            proto.STRING,
                            number=2,
                        )

                    class ExtractiveAnswer(proto.Message):
                        r"""Extractive answer.
                        `Guide <https://cloud.google.com/generative-ai-app-builder/docs/snippets#get-answers>`__

                        Attributes:
                            page_identifier (str):
                                Page identifier.
                            content (str):
                                Extractive answer content.
                        """

                        page_identifier: str = proto.Field(
                            proto.STRING,
                            number=1,
                        )
                        content: str = proto.Field(
                            proto.STRING,
                            number=2,
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
                    document_contexts: MutableSequence[
                        "AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.DocumentContext"
                    ] = proto.RepeatedField(
                        proto.MESSAGE,
                        number=4,
                        message="AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.DocumentContext",
                    )
                    extractive_segments: MutableSequence[
                        "AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.ExtractiveSegment"
                    ] = proto.RepeatedField(
                        proto.MESSAGE,
                        number=5,
                        message="AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.ExtractiveSegment",
                    )
                    extractive_answers: MutableSequence[
                        "AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.ExtractiveAnswer"
                    ] = proto.RepeatedField(
                        proto.MESSAGE,
                        number=6,
                        message="AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo.ExtractiveAnswer",
                    )

                class ChunkInfo(proto.Message):
                    r"""Chunk information.

                    Attributes:
                        chunk (str):
                            Chunk resource name.
                        content (str):
                            Chunk textual content.
                        document_metadata (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.ChunkInfo.DocumentMetadata):
                            Metadata of the document from the current
                            chunk.
                    """

                    class DocumentMetadata(proto.Message):
                        r"""Document metadata contains the information of the document of
                        the current chunk.

                        Attributes:
                            uri (str):
                                Uri of the document.
                            title (str):
                                Title of the document.
                        """

                        uri: str = proto.Field(
                            proto.STRING,
                            number=1,
                        )
                        title: str = proto.Field(
                            proto.STRING,
                            number=2,
                        )

                    chunk: str = proto.Field(
                        proto.STRING,
                        number=1,
                    )
                    content: str = proto.Field(
                        proto.STRING,
                        number=2,
                    )
                    document_metadata: "AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.ChunkInfo.DocumentMetadata" = proto.Field(
                        proto.MESSAGE,
                        number=4,
                        message="AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.ChunkInfo.DocumentMetadata",
                    )

                unstructured_document_info: "AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo" = proto.Field(
                    proto.MESSAGE,
                    number=1,
                    oneof="content",
                    message="AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.UnstructuredDocumentInfo",
                )
                chunk_info: "AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.ChunkInfo" = proto.Field(
                    proto.MESSAGE,
                    number=2,
                    oneof="content",
                    message="AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult.ChunkInfo",
                )

            search_results: MutableSequence[
                "AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult"
            ] = proto.RepeatedField(
                proto.MESSAGE,
                number=1,
                message="AnswerQueryRequest.SearchSpec.SearchResultList.SearchResult",
            )

        search_params: "AnswerQueryRequest.SearchSpec.SearchParams" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="input",
            message="AnswerQueryRequest.SearchSpec.SearchParams",
        )
        search_result_list: "AnswerQueryRequest.SearchSpec.SearchResultList" = (
            proto.Field(
                proto.MESSAGE,
                number=2,
                oneof="input",
                message="AnswerQueryRequest.SearchSpec.SearchResultList",
            )
        )

    class QueryUnderstandingSpec(proto.Message):
        r"""Query understanding specification.

        Attributes:
            query_classification_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec):
                Query classification specification.
            query_rephraser_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec):
                Query rephraser specification.
            disable_spell_correction (bool):
                Optional. Whether to disable spell correction. The default
                value is ``false``.
        """

        class QueryClassificationSpec(proto.Message):
            r"""Query classification specification.

            Attributes:
                types (MutableSequence[google.cloud.discoveryengine_v1.types.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type]):
                    Enabled query classification types.
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

            types: MutableSequence[
                "AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type"
            ] = proto.RepeatedField(
                proto.ENUM,
                number=1,
                enum="AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type",
            )

        class QueryRephraserSpec(proto.Message):
            r"""Query rephraser specification.

            Attributes:
                disable (bool):
                    Disable query rephraser.
                max_rephrase_steps (int):
                    Max rephrase steps.
                    The max number is 5 steps.
                    If not set or set to < 1, it will be set to 1 by
                    default.
                model_spec (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec.ModelSpec):
                    Optional. Query Rephraser Model
                    specification.
            """

            class ModelSpec(proto.Message):
                r"""Query Rephraser Model specification.

                Attributes:
                    model_type (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec.ModelSpec.ModelType):
                        Optional. Enabled query rephraser model type.
                        If not set, it will use LARGE by default.
                """

                class ModelType(proto.Enum):
                    r"""Query rephraser types. Currently only supports single-hop
                    (max_rephrase_steps = 1) model selections. For multi-hop
                    (max_rephrase_steps > 1), there is only one default model.

                    Values:
                        MODEL_TYPE_UNSPECIFIED (0):
                            Unspecified model type.
                        SMALL (1):
                            Small query rephraser model. Gemini 1.0 XS
                            model.
                        LARGE (2):
                            Large query rephraser model. Gemini 1.0 Pro
                            model.
                    """
                    MODEL_TYPE_UNSPECIFIED = 0
                    SMALL = 1
                    LARGE = 2

                model_type: "AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec.ModelSpec.ModelType" = proto.Field(
                    proto.ENUM,
                    number=1,
                    enum="AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec.ModelSpec.ModelType",
                )

            disable: bool = proto.Field(
                proto.BOOL,
                number=1,
            )
            max_rephrase_steps: int = proto.Field(
                proto.INT32,
                number=2,
            )
            model_spec: "AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec.ModelSpec" = proto.Field(
                proto.MESSAGE,
                number=3,
                message="AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec.ModelSpec",
            )

        query_classification_spec: "AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec",
        )
        query_rephraser_spec: "AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec" = proto.Field(
            proto.MESSAGE,
            number=2,
            message="AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec",
        )
        disable_spell_correction: bool = proto.Field(
            proto.BOOL,
            number=3,
        )

    class EndUserSpec(proto.Message):
        r"""End user specification.

        Attributes:
            end_user_metadata (MutableSequence[google.cloud.discoveryengine_v1.types.AnswerQueryRequest.EndUserSpec.EndUserMetaData]):
                Optional. End user metadata.
        """

        class EndUserMetaData(proto.Message):
            r"""End user metadata.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                chunk_info (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.EndUserSpec.EndUserMetaData.ChunkInfo):
                    Chunk information.

                    This field is a member of `oneof`_ ``content``.
            """

            class ChunkInfo(proto.Message):
                r"""Chunk information.

                Attributes:
                    content (str):
                        Chunk textual content. It is limited to 8000
                        characters.
                    document_metadata (google.cloud.discoveryengine_v1.types.AnswerQueryRequest.EndUserSpec.EndUserMetaData.ChunkInfo.DocumentMetadata):
                        Metadata of the document from the current
                        chunk.
                """

                class DocumentMetadata(proto.Message):
                    r"""Document metadata contains the information of the document of
                    the current chunk.

                    Attributes:
                        title (str):
                            Title of the document.
                    """

                    title: str = proto.Field(
                        proto.STRING,
                        number=1,
                    )

                content: str = proto.Field(
                    proto.STRING,
                    number=1,
                )
                document_metadata: "AnswerQueryRequest.EndUserSpec.EndUserMetaData.ChunkInfo.DocumentMetadata" = proto.Field(
                    proto.MESSAGE,
                    number=2,
                    message="AnswerQueryRequest.EndUserSpec.EndUserMetaData.ChunkInfo.DocumentMetadata",
                )

            chunk_info: "AnswerQueryRequest.EndUserSpec.EndUserMetaData.ChunkInfo" = (
                proto.Field(
                    proto.MESSAGE,
                    number=1,
                    oneof="content",
                    message="AnswerQueryRequest.EndUserSpec.EndUserMetaData.ChunkInfo",
                )
            )

        end_user_metadata: MutableSequence[
            "AnswerQueryRequest.EndUserSpec.EndUserMetaData"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="AnswerQueryRequest.EndUserSpec.EndUserMetaData",
        )

    serving_config: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: gcd_session.Query = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_session.Query,
    )
    session: str = proto.Field(
        proto.STRING,
        number=3,
    )
    safety_spec: SafetySpec = proto.Field(
        proto.MESSAGE,
        number=4,
        message=SafetySpec,
    )
    related_questions_spec: RelatedQuestionsSpec = proto.Field(
        proto.MESSAGE,
        number=5,
        message=RelatedQuestionsSpec,
    )
    grounding_spec: GroundingSpec = proto.Field(
        proto.MESSAGE,
        number=6,
        message=GroundingSpec,
    )
    answer_generation_spec: AnswerGenerationSpec = proto.Field(
        proto.MESSAGE,
        number=7,
        message=AnswerGenerationSpec,
    )
    search_spec: SearchSpec = proto.Field(
        proto.MESSAGE,
        number=8,
        message=SearchSpec,
    )
    query_understanding_spec: QueryUnderstandingSpec = proto.Field(
        proto.MESSAGE,
        number=9,
        message=QueryUnderstandingSpec,
    )
    asynchronous_mode: bool = proto.Field(
        proto.BOOL,
        number=10,
    )
    user_pseudo_id: str = proto.Field(
        proto.STRING,
        number=12,
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=13,
    )
    end_user_spec: EndUserSpec = proto.Field(
        proto.MESSAGE,
        number=14,
        message=EndUserSpec,
    )


class AnswerQueryResponse(proto.Message):
    r"""Response message for
    [ConversationalSearchService.AnswerQuery][google.cloud.discoveryengine.v1.ConversationalSearchService.AnswerQuery]
    method.

    Attributes:
        answer (google.cloud.discoveryengine_v1.types.Answer):
            Answer resource object. If
            [AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec.max_rephrase_steps][google.cloud.discoveryengine.v1.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec.max_rephrase_steps]
            is greater than 1, use
            [Answer.name][google.cloud.discoveryengine.v1.Answer.name]
            to fetch answer information using
            [ConversationalSearchService.GetAnswer][google.cloud.discoveryengine.v1.ConversationalSearchService.GetAnswer]
            API.
        session (google.cloud.discoveryengine_v1.types.Session):
            Session resource object. It will be only available when
            session field is set and valid in the
            [AnswerQueryRequest][google.cloud.discoveryengine.v1.AnswerQueryRequest]
            request.
        answer_query_token (str):
            A global unique ID used for logging.
    """

    answer: gcd_answer.Answer = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_answer.Answer,
    )
    session: gcd_session.Session = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_session.Session,
    )
    answer_query_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class GetAnswerRequest(proto.Message):
    r"""Request for GetAnswer method.

    Attributes:
        name (str):
            Required. The resource name of the Answer to get. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine_id}/sessions/{session_id}/answers/{answer_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CreateSessionRequest(proto.Message):
    r"""Request for CreateSession method.

    Attributes:
        parent (str):
            Required. Full resource name of parent data store. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}``
        session (google.cloud.discoveryengine_v1.types.Session):
            Required. The session to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    session: gcd_session.Session = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_session.Session,
    )


class UpdateSessionRequest(proto.Message):
    r"""Request for UpdateSession method.

    Attributes:
        session (google.cloud.discoveryengine_v1.types.Session):
            Required. The Session to update.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            [Session][google.cloud.discoveryengine.v1.Session] to
            update. The following are NOT supported:

            - [Session.name][google.cloud.discoveryengine.v1.Session.name]

            If not set or empty, all supported fields are updated.
    """

    session: gcd_session.Session = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gcd_session.Session,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


class DeleteSessionRequest(proto.Message):
    r"""Request for DeleteSession method.

    Attributes:
        name (str):
            Required. The resource name of the Session to delete.
            Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/sessions/{session_id}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GetSessionRequest(proto.Message):
    r"""Request for GetSession method.

    Attributes:
        name (str):
            Required. The resource name of the Session to get. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}/sessions/{session_id}``
        include_answer_details (bool):
            Optional. If set to true, the full session
            including all answer details will be returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    include_answer_details: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class ListSessionsRequest(proto.Message):
    r"""Request for ListSessions method.

    Attributes:
        parent (str):
            Required. The data store resource name. Format:
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store_id}``
        page_size (int):
            Maximum number of results to return. If
            unspecified, defaults to 50. Max allowed value
            is 1000.
        page_token (str):
            A page token, received from a previous ``ListSessions``
            call. Provide this to retrieve the subsequent page.
        filter (str):
            A comma-separated list of fields to filter by, in EBNF
            grammar. The supported fields are:

            - ``user_pseudo_id``
            - ``state``
            - ``display_name``
            - ``starred``
            - ``is_pinned``
            - ``labels``
            - ``create_time``
            - ``update_time``

            Examples: "user_pseudo_id = some_id" "display_name =
            "some_name"" "starred = true" "is_pinned=true AND (NOT
            labels:hidden)" "create_time > "1970-01-01T12:00:00Z"".
        order_by (str):
            A comma-separated list of fields to order by, sorted in
            ascending order. Use "desc" after a field name for
            descending. Supported fields:

            - ``update_time``
            - ``create_time``
            - ``session_name``
            - ``is_pinned``

            Example:

            - "update_time desc"
            - "create_time"
            - "is_pinned desc,update_time desc": list sessions by
              is_pinned first, then by update_time.
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
    r"""Response for ListSessions method.

    Attributes:
        sessions (MutableSequence[google.cloud.discoveryengine_v1.types.Session]):
            All the Sessions for a given data store.
        next_page_token (str):
            Pagination token, if not returned indicates
            the last page.
    """

    @property
    def raw_page(self):
        return self

    sessions: MutableSequence[gcd_session.Session] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_session.Session,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
