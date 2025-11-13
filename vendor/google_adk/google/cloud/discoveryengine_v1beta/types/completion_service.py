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

from google.cloud.discoveryengine_v1beta.types import common
from google.cloud.discoveryengine_v1beta.types import document as gcd_document

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "CompleteQueryRequest",
        "CompleteQueryResponse",
        "AdvancedCompleteQueryRequest",
        "AdvancedCompleteQueryResponse",
    },
)


class CompleteQueryRequest(proto.Message):
    r"""Request message for
    [CompletionService.CompleteQuery][google.cloud.discoveryengine.v1beta.CompletionService.CompleteQuery]
    method.

    Attributes:
        data_store (str):
            Required. The parent data store resource name for which the
            completion is performed, such as
            ``projects/*/locations/global/collections/default_collection/dataStores/default_data_store``.
        query (str):
            Required. The typeahead input used to fetch
            suggestions. Maximum length is 128 characters.
        query_model (str):
            Specifies the autocomplete data model. This overrides any
            model specified in the Configuration > Autocomplete section
            of the Cloud console. Currently supported values:

            - ``document`` - Using suggestions generated from
              user-imported documents.
            - ``search-history`` - Using suggestions generated from the
              past history of
              [SearchService.Search][google.cloud.discoveryengine.v1beta.SearchService.Search]
              API calls. Do not use it when there is no traffic for
              Search API.
            - ``user-event`` - Using suggestions generated from
              user-imported search events.
            - ``document-completable`` - Using suggestions taken
              directly from user-imported document fields marked as
              completable.

            Default values:

            - ``document`` is the default model for regular dataStores.
            - ``search-history`` is the default model for site search
              dataStores.
        user_pseudo_id (str):
            A unique identifier for tracking visitors. For example, this
            could be implemented with an HTTP cookie, which should be
            able to uniquely identify a visitor on a single device. This
            unique identifier should not change if the visitor logs in
            or out of the website.

            This field should NOT have a fixed value such as
            ``unknown_visitor``.

            This should be the same identifier as
            [UserEvent.user_pseudo_id][google.cloud.discoveryengine.v1beta.UserEvent.user_pseudo_id]
            and
            [SearchRequest.user_pseudo_id][google.cloud.discoveryengine.v1beta.SearchRequest.user_pseudo_id].

            The field must be a UTF-8 encoded string with a length limit
            of 128 characters. Otherwise, an ``INVALID_ARGUMENT`` error
            is returned.
        include_tail_suggestions (bool):
            Indicates if tail suggestions should be
            returned if there are no suggestions that match
            the full query. Even if set to true, if there
            are suggestions that match the full query, those
            are returned and no tail suggestions are
            returned.
    """

    data_store: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: str = proto.Field(
        proto.STRING,
        number=2,
    )
    query_model: str = proto.Field(
        proto.STRING,
        number=3,
    )
    user_pseudo_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    include_tail_suggestions: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


class CompleteQueryResponse(proto.Message):
    r"""Response message for
    [CompletionService.CompleteQuery][google.cloud.discoveryengine.v1beta.CompletionService.CompleteQuery]
    method.

    Attributes:
        query_suggestions (MutableSequence[google.cloud.discoveryengine_v1beta.types.CompleteQueryResponse.QuerySuggestion]):
            Results of the matched query suggestions. The
            result list is ordered and the first result is a
            top suggestion.
        tail_match_triggered (bool):
            True if the returned suggestions are all tail suggestions.

            For tail matching to be triggered, include_tail_suggestions
            in the request must be true and there must be no suggestions
            that match the full query.
    """

    class QuerySuggestion(proto.Message):
        r"""Suggestions as search queries.

        Attributes:
            suggestion (str):
                The suggestion for the query.
            completable_field_paths (MutableSequence[str]):
                The unique document field paths that serve as
                the source of this suggestion if it was
                generated from completable fields.

                This field is only populated for the
                document-completable model.
        """

        suggestion: str = proto.Field(
            proto.STRING,
            number=1,
        )
        completable_field_paths: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )

    query_suggestions: MutableSequence[QuerySuggestion] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=QuerySuggestion,
    )
    tail_match_triggered: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class AdvancedCompleteQueryRequest(proto.Message):
    r"""Request message for
    [CompletionService.AdvancedCompleteQuery][google.cloud.discoveryengine.v1beta.CompletionService.AdvancedCompleteQuery]
    method. .

    Attributes:
        completion_config (str):
            Required. The completion_config of the parent dataStore or
            engine resource name for which the completion is performed,
            such as
            ``projects/*/locations/global/collections/default_collection/dataStores/*/completionConfig``
            ``projects/*/locations/global/collections/default_collection/engines/*/completionConfig``.
        query (str):
            Required. The typeahead input used to fetch suggestions.
            Maximum length is 128 characters.

            The query can not be empty for most of the suggestion types.
            If it is empty, an ``INVALID_ARGUMENT`` error is returned.
            The exception is when the suggestion_types contains only the
            type ``RECENT_SEARCH``, the query can be an empty string.
            The is called "zero prefix" feature, which returns user's
            recently searched queries given the empty query.
        query_model (str):
            Specifies the autocomplete data model. This overrides any
            model specified in the Configuration > Autocomplete section
            of the Cloud console. Currently supported values:

            - ``document`` - Using suggestions generated from
              user-imported documents.
            - ``search-history`` - Using suggestions generated from the
              past history of
              [SearchService.Search][google.cloud.discoveryengine.v1beta.SearchService.Search]
              API calls. Do not use it when there is no traffic for
              Search API.
            - ``user-event`` - Using suggestions generated from
              user-imported search events.
            - ``document-completable`` - Using suggestions taken
              directly from user-imported document fields marked as
              completable.

            Default values:

            - ``document`` is the default model for regular dataStores.
            - ``search-history`` is the default model for site search
              dataStores.
        user_pseudo_id (str):
            A unique identifier for tracking visitors. For example, this
            could be implemented with an HTTP cookie, which should be
            able to uniquely identify a visitor on a single device. This
            unique identifier should not change if the visitor logs in
            or out of the website.

            This field should NOT have a fixed value such as
            ``unknown_visitor``.

            This should be the same identifier as
            [UserEvent.user_pseudo_id][google.cloud.discoveryengine.v1beta.UserEvent.user_pseudo_id]
            and
            [SearchRequest.user_pseudo_id][google.cloud.discoveryengine.v1beta.SearchRequest.user_pseudo_id].

            The field must be a UTF-8 encoded string with a length limit
            of 128
        user_info (google.cloud.discoveryengine_v1beta.types.UserInfo):
            Optional. Information about the end user.

            This should be the same identifier information as
            [UserEvent.user_info][google.cloud.discoveryengine.v1beta.UserEvent.user_info]
            and
            [SearchRequest.user_info][google.cloud.discoveryengine.v1beta.SearchRequest.user_info].
        include_tail_suggestions (bool):
            Indicates if tail suggestions should be
            returned if there are no suggestions that match
            the full query. Even if set to true, if there
            are suggestions that match the full query, those
            are returned and no tail suggestions are
            returned.
        boost_spec (google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryRequest.BoostSpec):
            Optional. Specification to boost suggestions
            matching the condition.
        suggestion_types (MutableSequence[google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryRequest.SuggestionType]):
            Optional. Suggestion types to return. If
            empty or unspecified, query suggestions are
            returned. Only one suggestion type is supported
            at the moment.
    """

    class SuggestionType(proto.Enum):
        r"""Suggestion type to return.

        Values:
            SUGGESTION_TYPE_UNSPECIFIED (0):
                Default value.
            QUERY (1):
                Returns query suggestions.
            PEOPLE (2):
                Returns people suggestions.
            CONTENT (3):
                Returns content suggestions.
            RECENT_SEARCH (4):
                Returns recent search suggestions.
            GOOGLE_WORKSPACE (5):
                Returns Google Workspace suggestions.
        """
        SUGGESTION_TYPE_UNSPECIFIED = 0
        QUERY = 1
        PEOPLE = 2
        CONTENT = 3
        RECENT_SEARCH = 4
        GOOGLE_WORKSPACE = 5

    class BoostSpec(proto.Message):
        r"""Specification to boost suggestions based on the condtion of
        the suggestion.

        Attributes:
            condition_boost_specs (MutableSequence[google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryRequest.BoostSpec.ConditionBoostSpec]):
                Condition boost specifications. If a
                suggestion matches multiple conditions in the
                specifictions, boost values from these
                specifications are all applied and combined in a
                non-linear way. Maximum number of specifications
                is 20.

                Note: Currently only support language condition
                boost.
        """

        class ConditionBoostSpec(proto.Message):
            r"""Boost applies to suggestions which match a condition.

            Attributes:
                condition (str):
                    An expression which specifies a boost condition. The syntax
                    is the same as `filter expression
                    syntax <https://cloud.google.com/generative-ai-app-builder/docs/filter-search-metadata#filter-expression-syntax>`__.
                    Currently, the only supported condition is a list of BCP-47
                    lang codes.

                    Example:

                    - To boost suggestions in languages ``en`` or ``fr``:
                      ``(lang_code: ANY("en", "fr"))``
                boost (float):
                    Strength of the boost, which should be in [-1, 1]. Negative
                    boost means demotion. Default is 0.0.

                    Setting to 1.0 gives the suggestions a big promotion.
                    However, it does not necessarily mean that the top result
                    will be a boosted suggestion.

                    Setting to -1.0 gives the suggestions a big demotion.
                    However, other suggestions that are relevant might still be
                    shown.

                    Setting to 0.0 means no boost applied. The boosting
                    condition is ignored.
            """

            condition: str = proto.Field(
                proto.STRING,
                number=1,
            )
            boost: float = proto.Field(
                proto.FLOAT,
                number=2,
            )

        condition_boost_specs: MutableSequence[
            "AdvancedCompleteQueryRequest.BoostSpec.ConditionBoostSpec"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="AdvancedCompleteQueryRequest.BoostSpec.ConditionBoostSpec",
        )

    completion_config: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: str = proto.Field(
        proto.STRING,
        number=2,
    )
    query_model: str = proto.Field(
        proto.STRING,
        number=3,
    )
    user_pseudo_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    user_info: common.UserInfo = proto.Field(
        proto.MESSAGE,
        number=9,
        message=common.UserInfo,
    )
    include_tail_suggestions: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    boost_spec: BoostSpec = proto.Field(
        proto.MESSAGE,
        number=6,
        message=BoostSpec,
    )
    suggestion_types: MutableSequence[SuggestionType] = proto.RepeatedField(
        proto.ENUM,
        number=7,
        enum=SuggestionType,
    )


class AdvancedCompleteQueryResponse(proto.Message):
    r"""Response message for
    [CompletionService.AdvancedCompleteQuery][google.cloud.discoveryengine.v1beta.CompletionService.AdvancedCompleteQuery]
    method.

    Attributes:
        query_suggestions (MutableSequence[google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryResponse.QuerySuggestion]):
            Results of the matched query suggestions. The
            result list is ordered and the first result is a
            top suggestion.
        tail_match_triggered (bool):
            True if the returned suggestions are all tail suggestions.

            For tail matching to be triggered, include_tail_suggestions
            in the request must be true and there must be no suggestions
            that match the full query.
        people_suggestions (MutableSequence[google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryResponse.PersonSuggestion]):
            Results of the matched people suggestions.
            The result list is ordered and the first result
            is the top suggestion.
        content_suggestions (MutableSequence[google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryResponse.ContentSuggestion]):
            Results of the matched content suggestions.
            The result list is ordered and the first result
            is the top suggestion.
        recent_search_suggestions (MutableSequence[google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryResponse.RecentSearchSuggestion]):
            Results of the matched "recent search"
            suggestions. The result list is ordered and the
            first result is the top suggestion.
    """

    class QuerySuggestion(proto.Message):
        r"""Suggestions as search queries.

        Attributes:
            suggestion (str):
                The suggestion for the query.
            completable_field_paths (MutableSequence[str]):
                The unique document field paths that serve as
                the source of this suggestion if it was
                generated from completable fields.

                This field is only populated for the
                document-completable model.
            data_store (MutableSequence[str]):
                The name of the dataStore that this
                suggestion belongs to.
        """

        suggestion: str = proto.Field(
            proto.STRING,
            number=1,
        )
        completable_field_paths: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )
        data_store: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=3,
        )

    class PersonSuggestion(proto.Message):
        r"""Suggestions as people.

        Attributes:
            suggestion (str):
                The suggestion for the query.
            person_type (google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryResponse.PersonSuggestion.PersonType):
                The type of the person.
            document (google.cloud.discoveryengine_v1beta.types.Document):
                The document data snippet in the suggestion.
                Only a subset of fields is populated.
            data_store (str):
                The name of the dataStore that this
                suggestion belongs to.
        """

        class PersonType(proto.Enum):
            r"""The type of the person based on the source.

            Values:
                PERSON_TYPE_UNSPECIFIED (0):
                    Default value.
                CLOUD_IDENTITY (1):
                    The suggestion is from a GOOGLE_IDENTITY source.
                THIRD_PARTY_IDENTITY (2):
                    The suggestion is from a THIRD_PARTY_IDENTITY source.
            """
            PERSON_TYPE_UNSPECIFIED = 0
            CLOUD_IDENTITY = 1
            THIRD_PARTY_IDENTITY = 2

        suggestion: str = proto.Field(
            proto.STRING,
            number=1,
        )
        person_type: "AdvancedCompleteQueryResponse.PersonSuggestion.PersonType" = (
            proto.Field(
                proto.ENUM,
                number=2,
                enum="AdvancedCompleteQueryResponse.PersonSuggestion.PersonType",
            )
        )
        document: gcd_document.Document = proto.Field(
            proto.MESSAGE,
            number=4,
            message=gcd_document.Document,
        )
        data_store: str = proto.Field(
            proto.STRING,
            number=5,
        )

    class ContentSuggestion(proto.Message):
        r"""Suggestions as content.

        Attributes:
            suggestion (str):
                The suggestion for the query.
            content_type (google.cloud.discoveryengine_v1beta.types.AdvancedCompleteQueryResponse.ContentSuggestion.ContentType):
                The type of the content suggestion.
            document (google.cloud.discoveryengine_v1beta.types.Document):
                The document data snippet in the suggestion.
                Only a subset of fields will be populated.
            data_store (str):
                The name of the dataStore that this
                suggestion belongs to.
        """

        class ContentType(proto.Enum):
            r"""The type of the content returned for content suggestions.

            Values:
                CONTENT_TYPE_UNSPECIFIED (0):
                    Default value.
                GOOGLE_WORKSPACE (1):
                    The suggestion is from a Google Workspace
                    source.
                THIRD_PARTY (2):
                    The suggestion is from a third party source.
            """
            CONTENT_TYPE_UNSPECIFIED = 0
            GOOGLE_WORKSPACE = 1
            THIRD_PARTY = 2

        suggestion: str = proto.Field(
            proto.STRING,
            number=1,
        )
        content_type: "AdvancedCompleteQueryResponse.ContentSuggestion.ContentType" = (
            proto.Field(
                proto.ENUM,
                number=2,
                enum="AdvancedCompleteQueryResponse.ContentSuggestion.ContentType",
            )
        )
        document: gcd_document.Document = proto.Field(
            proto.MESSAGE,
            number=4,
            message=gcd_document.Document,
        )
        data_store: str = proto.Field(
            proto.STRING,
            number=5,
        )

    class RecentSearchSuggestion(proto.Message):
        r"""Suggestions from recent search history.

        Attributes:
            suggestion (str):
                The suggestion for the query.
            recent_search_time (google.protobuf.timestamp_pb2.Timestamp):
                The time when this recent rearch happened.
        """

        suggestion: str = proto.Field(
            proto.STRING,
            number=1,
        )
        recent_search_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=2,
            message=timestamp_pb2.Timestamp,
        )

    query_suggestions: MutableSequence[QuerySuggestion] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=QuerySuggestion,
    )
    tail_match_triggered: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    people_suggestions: MutableSequence[PersonSuggestion] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=PersonSuggestion,
    )
    content_suggestions: MutableSequence[ContentSuggestion] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=ContentSuggestion,
    )
    recent_search_suggestions: MutableSequence[
        RecentSearchSuggestion
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message=RecentSearchSuggestion,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
