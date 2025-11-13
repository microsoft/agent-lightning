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

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "CompleteQueryRequest",
        "CompleteQueryResponse",
    },
)


class CompleteQueryRequest(proto.Message):
    r"""Request message for
    [CompletionService.CompleteQuery][google.cloud.discoveryengine.v1alpha.CompletionService.CompleteQuery]
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
              [SearchService.Search][google.cloud.discoveryengine.v1alpha.SearchService.Search]
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
            [UserEvent.user_pseudo_id][google.cloud.discoveryengine.v1alpha.UserEvent.user_pseudo_id]
            and
            [SearchRequest.user_pseudo_id][google.cloud.discoveryengine.v1alpha.SearchRequest.user_pseudo_id].

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
    [CompletionService.CompleteQuery][google.cloud.discoveryengine.v1alpha.CompletionService.CompleteQuery]
    method.

    Attributes:
        query_suggestions (MutableSequence[google.cloud.discoveryengine_v1alpha.types.CompleteQueryResponse.QuerySuggestion]):
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


__all__ = tuple(sorted(__protobuf__.manifest))
