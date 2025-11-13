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
    package="google.cloud.discoveryengine.v1",
    manifest={
        "IndustryVertical",
        "SolutionType",
        "SearchUseCase",
        "SearchTier",
        "SearchAddOn",
        "Interval",
        "CustomAttribute",
        "UserInfo",
        "DoubleList",
        "Principal",
        "HealthcareFhirConfig",
        "SearchLinkPromotion",
    },
)


class IndustryVertical(proto.Enum):
    r"""The industry vertical associated with the
    [DataStore][google.cloud.discoveryengine.v1.DataStore].

    Values:
        INDUSTRY_VERTICAL_UNSPECIFIED (0):
            Value used when unset.
        GENERIC (1):
            The generic vertical for documents that are
            not specific to any industry vertical.
        MEDIA (2):
            The media industry vertical.
        HEALTHCARE_FHIR (7):
            The healthcare FHIR vertical.
    """
    INDUSTRY_VERTICAL_UNSPECIFIED = 0
    GENERIC = 1
    MEDIA = 2
    HEALTHCARE_FHIR = 7


class SolutionType(proto.Enum):
    r"""The type of solution.

    Values:
        SOLUTION_TYPE_UNSPECIFIED (0):
            Default value.
        SOLUTION_TYPE_RECOMMENDATION (1):
            Used for Recommendations AI.
        SOLUTION_TYPE_SEARCH (2):
            Used for Discovery Search.
        SOLUTION_TYPE_CHAT (3):
            Used for use cases related to the Generative
            AI agent.
        SOLUTION_TYPE_GENERATIVE_CHAT (4):
            Used for use cases related to the Generative Chat agent.
            It's used for Generative chat engine only, the associated
            data stores must enrolled with ``SOLUTION_TYPE_CHAT``
            solution.
    """
    SOLUTION_TYPE_UNSPECIFIED = 0
    SOLUTION_TYPE_RECOMMENDATION = 1
    SOLUTION_TYPE_SEARCH = 2
    SOLUTION_TYPE_CHAT = 3
    SOLUTION_TYPE_GENERATIVE_CHAT = 4


class SearchUseCase(proto.Enum):
    r"""Defines a further subdivision of ``SolutionType``. Specifically
    applies to
    [SOLUTION_TYPE_SEARCH][google.cloud.discoveryengine.v1.SolutionType.SOLUTION_TYPE_SEARCH].

    Values:
        SEARCH_USE_CASE_UNSPECIFIED (0):
            Value used when unset. Will not occur in CSS.
        SEARCH_USE_CASE_SEARCH (1):
            Search use case. Expects the traffic has a non-empty
            [query][google.cloud.discoveryengine.v1.SearchRequest.query].
        SEARCH_USE_CASE_BROWSE (2):
            Browse use case. Expects the traffic has an empty
            [query][google.cloud.discoveryengine.v1.SearchRequest.query].
    """
    SEARCH_USE_CASE_UNSPECIFIED = 0
    SEARCH_USE_CASE_SEARCH = 1
    SEARCH_USE_CASE_BROWSE = 2


class SearchTier(proto.Enum):
    r"""Tiers of search features. Different tiers might have
    different pricing. To learn more, check the pricing
    documentation.

    Values:
        SEARCH_TIER_UNSPECIFIED (0):
            Default value when the enum is unspecified.
            This is invalid to use.
        SEARCH_TIER_STANDARD (1):
            Standard tier.
        SEARCH_TIER_ENTERPRISE (2):
            Enterprise tier.
    """
    SEARCH_TIER_UNSPECIFIED = 0
    SEARCH_TIER_STANDARD = 1
    SEARCH_TIER_ENTERPRISE = 2


class SearchAddOn(proto.Enum):
    r"""Add-on that provides additional functionality for search.

    Values:
        SEARCH_ADD_ON_UNSPECIFIED (0):
            Default value when the enum is unspecified.
            This is invalid to use.
        SEARCH_ADD_ON_LLM (1):
            Large language model add-on.
    """
    SEARCH_ADD_ON_UNSPECIFIED = 0
    SEARCH_ADD_ON_LLM = 1


class Interval(proto.Message):
    r"""A floating point interval.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        minimum (float):
            Inclusive lower bound.

            This field is a member of `oneof`_ ``min``.
        exclusive_minimum (float):
            Exclusive lower bound.

            This field is a member of `oneof`_ ``min``.
        maximum (float):
            Inclusive upper bound.

            This field is a member of `oneof`_ ``max``.
        exclusive_maximum (float):
            Exclusive upper bound.

            This field is a member of `oneof`_ ``max``.
    """

    minimum: float = proto.Field(
        proto.DOUBLE,
        number=1,
        oneof="min",
    )
    exclusive_minimum: float = proto.Field(
        proto.DOUBLE,
        number=2,
        oneof="min",
    )
    maximum: float = proto.Field(
        proto.DOUBLE,
        number=3,
        oneof="max",
    )
    exclusive_maximum: float = proto.Field(
        proto.DOUBLE,
        number=4,
        oneof="max",
    )


class CustomAttribute(proto.Message):
    r"""A custom attribute that is not explicitly modeled in a resource,
    e.g. [UserEvent][google.cloud.discoveryengine.v1.UserEvent].

    Attributes:
        text (MutableSequence[str]):
            The textual values of this custom attribute. For example,
            ``["yellow", "green"]`` when the key is "color".

            Empty string is not allowed. Otherwise, an
            ``INVALID_ARGUMENT`` error is returned.

            Exactly one of
            [CustomAttribute.text][google.cloud.discoveryengine.v1.CustomAttribute.text]
            or
            [CustomAttribute.numbers][google.cloud.discoveryengine.v1.CustomAttribute.numbers]
            should be set. Otherwise, an ``INVALID_ARGUMENT`` error is
            returned.
        numbers (MutableSequence[float]):
            The numerical values of this custom attribute. For example,
            ``[2.3, 15.4]`` when the key is "lengths_cm".

            Exactly one of
            [CustomAttribute.text][google.cloud.discoveryengine.v1.CustomAttribute.text]
            or
            [CustomAttribute.numbers][google.cloud.discoveryengine.v1.CustomAttribute.numbers]
            should be set. Otherwise, an ``INVALID_ARGUMENT`` error is
            returned.
    """

    text: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    numbers: MutableSequence[float] = proto.RepeatedField(
        proto.DOUBLE,
        number=2,
    )


class UserInfo(proto.Message):
    r"""Information of an end user.

    Attributes:
        user_id (str):
            Highly recommended for logged-in users. Unique identifier
            for logged-in user, such as a user name. Don't set for
            anonymous users.

            Always use a hashed value for this ID.

            Don't set the field to the same fixed ID for different
            users. This mixes the event history of those users together,
            which results in degraded model quality.

            The field must be a UTF-8 encoded string with a length limit
            of 128 characters. Otherwise, an ``INVALID_ARGUMENT`` error
            is returned.
        user_agent (str):
            User agent as included in the HTTP header.

            The field must be a UTF-8 encoded string with a length limit
            of 1,000 characters. Otherwise, an ``INVALID_ARGUMENT``
            error is returned.

            This should not be set when using the client side event
            reporting with GTM or JavaScript tag in
            [UserEventService.CollectUserEvent][google.cloud.discoveryengine.v1.UserEventService.CollectUserEvent]
            or if
            [UserEvent.direct_user_request][google.cloud.discoveryengine.v1.UserEvent.direct_user_request]
            is set.
        time_zone (str):
            Optional. IANA time zone, e.g.
            Europe/Budapest.
    """

    user_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    user_agent: str = proto.Field(
        proto.STRING,
        number=2,
    )
    time_zone: str = proto.Field(
        proto.STRING,
        number=3,
    )


class DoubleList(proto.Message):
    r"""Double list.

    Attributes:
        values (MutableSequence[float]):
            Double values.
    """

    values: MutableSequence[float] = proto.RepeatedField(
        proto.DOUBLE,
        number=1,
    )


class Principal(proto.Message):
    r"""Principal identifier of a user or a group.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        user_id (str):
            User identifier. For Google Workspace user account, user_id
            should be the google workspace user email. For non-google
            identity provider user account, user_id is the mapped user
            identifier configured during the workforcepool config.

            This field is a member of `oneof`_ ``principal``.
        group_id (str):
            Group identifier. For Google Workspace user account,
            group_id should be the google workspace group email. For
            non-google identity provider user account, group_id is the
            mapped group identifier configured during the workforcepool
            config.

            This field is a member of `oneof`_ ``principal``.
        external_entity_id (str):
            For 3P application identities which are not
            present in the customer identity provider.

            This field is a member of `oneof`_ ``principal``.
    """

    user_id: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="principal",
    )
    group_id: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="principal",
    )
    external_entity_id: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="principal",
    )


class HealthcareFhirConfig(proto.Message):
    r"""Config to data store for ``HEALTHCARE_FHIR`` vertical.

    Attributes:
        enable_configurable_schema (bool):
            Whether to enable configurable schema for
            ``HEALTHCARE_FHIR`` vertical.

            If set to ``true``, the predefined healthcare fhir schema
            can be extended for more customized searching and filtering.
        enable_static_indexing_for_batch_ingestion (bool):
            Whether to enable static indexing for ``HEALTHCARE_FHIR``
            batch ingestion.

            If set to ``true``, the batch ingestion will be processed in
            a static indexing mode which is slower but more capable of
            handling larger volume.
    """

    enable_configurable_schema: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    enable_static_indexing_for_batch_ingestion: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class SearchLinkPromotion(proto.Message):
    r"""Promotion proto includes uri and other helping information to
    display the promotion.

    Attributes:
        title (str):
            Required. The title of the promotion.
            Maximum length: 160 characters.
        uri (str):
            Optional. The URL for the page the user wants
            to promote. Must be set for site search. For
            other verticals, this is optional.
        document (str):
            Optional. The
            [Document][google.cloud.discoveryengine.v1.Document] the
            user wants to promote. For site search, leave unset and only
            populate uri. Can be set along with uri.
        image_uri (str):
            Optional. The promotion thumbnail image url.
        description (str):
            Optional. The Promotion description.
            Maximum length: 200 characters.
        enabled (bool):
            Optional. The enabled promotion will be
            returned for any serving configs associated with
            the parent of the control this promotion is
            attached to.

            This flag is used for basic site search only.
    """

    title: str = proto.Field(
        proto.STRING,
        number=1,
    )
    uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    document: str = proto.Field(
        proto.STRING,
        number=6,
    )
    image_uri: str = proto.Field(
        proto.STRING,
        number=3,
    )
    description: str = proto.Field(
        proto.STRING,
        number=4,
    )
    enabled: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
