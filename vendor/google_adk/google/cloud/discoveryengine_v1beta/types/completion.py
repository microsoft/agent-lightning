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
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "SuggestionDenyListEntry",
        "CompletionSuggestion",
    },
)


class SuggestionDenyListEntry(proto.Message):
    r"""Suggestion deny list entry identifying the phrase to block
    from suggestions and the applied operation for the phrase.

    Attributes:
        block_phrase (str):
            Required. Phrase to block from suggestions
            served. Can be maximum 125 characters.
        match_operator (google.cloud.discoveryengine_v1beta.types.SuggestionDenyListEntry.MatchOperator):
            Required. The match operator to apply for
            this phrase. Whether to block the exact phrase,
            or block any suggestions containing this phrase.
    """

    class MatchOperator(proto.Enum):
        r"""Operator for matching with the generated suggestions.

        Values:
            MATCH_OPERATOR_UNSPECIFIED (0):
                Default value. Should not be used
            EXACT_MATCH (1):
                If the suggestion is an exact match to the block_phrase,
                then block it.
            CONTAINS (2):
                If the suggestion contains the block_phrase, then block it.
        """
        MATCH_OPERATOR_UNSPECIFIED = 0
        EXACT_MATCH = 1
        CONTAINS = 2

    block_phrase: str = proto.Field(
        proto.STRING,
        number=1,
    )
    match_operator: MatchOperator = proto.Field(
        proto.ENUM,
        number=2,
        enum=MatchOperator,
    )


class CompletionSuggestion(proto.Message):
    r"""Autocomplete suggestions that are imported from Customer.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        global_score (float):
            Global score of this suggestion. Control how
            this suggestion would be scored / ranked.

            This field is a member of `oneof`_ ``ranking_info``.
        frequency (int):
            Frequency of this suggestion. Will be used to
            rank suggestions when score is not available.

            This field is a member of `oneof`_ ``ranking_info``.
        suggestion (str):
            Required. The suggestion text.
        language_code (str):
            BCP-47 language code of this suggestion.
        group_id (str):
            If two suggestions have the same groupId,
            they will not be returned together. Instead the
            one ranked higher will be returned. This can be
            used to deduplicate semantically identical
            suggestions.
        group_score (float):
            The score of this suggestion within its
            group.
        alternative_phrases (MutableSequence[str]):
            Alternative matching phrases for this
            suggestion.
    """

    global_score: float = proto.Field(
        proto.DOUBLE,
        number=2,
        oneof="ranking_info",
    )
    frequency: int = proto.Field(
        proto.INT64,
        number=3,
        oneof="ranking_info",
    )
    suggestion: str = proto.Field(
        proto.STRING,
        number=1,
    )
    language_code: str = proto.Field(
        proto.STRING,
        number=4,
    )
    group_id: str = proto.Field(
        proto.STRING,
        number=5,
    )
    group_score: float = proto.Field(
        proto.DOUBLE,
        number=6,
    )
    alternative_phrases: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=7,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
