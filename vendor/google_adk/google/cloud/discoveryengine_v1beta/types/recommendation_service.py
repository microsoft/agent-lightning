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
import proto  # type: ignore

from google.cloud.discoveryengine_v1beta.types import user_event as gcd_user_event
from google.cloud.discoveryengine_v1beta.types import document as gcd_document

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "RecommendRequest",
        "RecommendResponse",
    },
)


class RecommendRequest(proto.Message):
    r"""Request message for Recommend method.

    Attributes:
        serving_config (str):
            Required. Full resource name of a
            [ServingConfig][google.cloud.discoveryengine.v1beta.ServingConfig]:
            ``projects/*/locations/global/collections/*/engines/*/servingConfigs/*``,
            or
            ``projects/*/locations/global/collections/*/dataStores/*/servingConfigs/*``

            One default serving config is created along with your
            recommendation engine creation. The engine ID is used as the
            ID of the default serving config. For example, for Engine
            ``projects/*/locations/global/collections/*/engines/my-engine``,
            you can use
            ``projects/*/locations/global/collections/*/engines/my-engine/servingConfigs/my-engine``
            for your
            [RecommendationService.Recommend][google.cloud.discoveryengine.v1beta.RecommendationService.Recommend]
            requests.
        user_event (google.cloud.discoveryengine_v1beta.types.UserEvent):
            Required. Context about the user, what they are looking at
            and what action they took to trigger the Recommend request.
            Note that this user event detail won't be ingested to
            userEvent logs. Thus, a separate userEvent write request is
            required for event logging.

            Don't set
            [UserEvent.user_pseudo_id][google.cloud.discoveryengine.v1beta.UserEvent.user_pseudo_id]
            or
            [UserEvent.user_info.user_id][google.cloud.discoveryengine.v1beta.UserInfo.user_id]
            to the same fixed ID for different users. If you are trying
            to receive non-personalized recommendations (not
            recommended; this can negatively impact model performance),
            instead set
            [UserEvent.user_pseudo_id][google.cloud.discoveryengine.v1beta.UserEvent.user_pseudo_id]
            to a random unique ID and leave
            [UserEvent.user_info.user_id][google.cloud.discoveryengine.v1beta.UserInfo.user_id]
            unset.
        page_size (int):
            Maximum number of results to return. Set this
            property to the number of recommendation results
            needed. If zero, the service chooses a
            reasonable default. The maximum allowed value is
            100. Values above 100 are set to 100.
        filter (str):
            Filter for restricting recommendation results with a length
            limit of 5,000 characters. Currently, only filter
            expressions on the ``filter_tags`` attribute is supported.

            Examples:

            - ``(filter_tags: ANY("Red", "Blue") OR filter_tags: ANY("Hot", "Cold"))``
            - ``(filter_tags: ANY("Red", "Blue")) AND NOT (filter_tags: ANY("Green"))``

            If ``attributeFilteringSyntax`` is set to true under the
            ``params`` field, then attribute-based expressions are
            expected instead of the above described tag-based syntax.
            Examples:

            - (launguage: ANY("en", "es")) AND NOT (categories:
              ANY("Movie"))
            - (available: true) AND (launguage: ANY("en", "es")) OR
              (categories: ANY("Movie"))

            If your filter blocks all results, the API returns generic
            (unfiltered) popular Documents. If you only want results
            strictly matching the filters, set ``strictFiltering`` to
            ``true`` in
            [RecommendRequest.params][google.cloud.discoveryengine.v1beta.RecommendRequest.params]
            to receive empty results instead.

            Note that the API never returns
            [Document][google.cloud.discoveryengine.v1beta.Document]s
            with ``storageStatus`` as ``EXPIRED`` or ``DELETED``
            regardless of filter choices.
        validate_only (bool):
            Use validate only mode for this recommendation query. If set
            to ``true``, a fake model is used that returns arbitrary
            Document IDs. Note that the validate only mode should only
            be used for testing the API, or if the model is not ready.
        params (MutableMapping[str, google.protobuf.struct_pb2.Value]):
            Additional domain specific parameters for the
            recommendations.

            Allowed values:

            - ``returnDocument``: Boolean. If set to ``true``, the
              associated Document object is returned in
              [RecommendResponse.RecommendationResult.document][google.cloud.discoveryengine.v1beta.RecommendResponse.RecommendationResult.document].
            - ``returnScore``: Boolean. If set to true, the
              recommendation score corresponding to each returned
              Document is set in
              [RecommendResponse.RecommendationResult.metadata][google.cloud.discoveryengine.v1beta.RecommendResponse.RecommendationResult.metadata].
              The given score indicates the probability of a Document
              conversion given the user's context and history.
            - ``strictFiltering``: Boolean. True by default. If set to
              ``false``, the service returns generic (unfiltered)
              popular Documents instead of empty if your filter blocks
              all recommendation results.
            - ``diversityLevel``: String. Default empty. If set to be
              non-empty, then it needs to be one of:

              - ``no-diversity``
              - ``low-diversity``
              - ``medium-diversity``
              - ``high-diversity``
              - ``auto-diversity`` This gives request-level control and
                adjusts recommendation results based on Document
                category.

            - ``attributeFilteringSyntax``: Boolean. False by default.
              If set to true, the ``filter`` field is interpreted
              according to the new, attribute-based syntax.
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

            See `Requirements for
            labels <https://cloud.google.com/resource-manager/docs/creating-managing-labels#requirements>`__
            for more details.
    """

    serving_config: str = proto.Field(
        proto.STRING,
        number=1,
    )
    user_event: gcd_user_event.UserEvent = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_user_event.UserEvent,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=4,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    params: MutableMapping[str, struct_pb2.Value] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=6,
        message=struct_pb2.Value,
    )
    user_labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=8,
    )


class RecommendResponse(proto.Message):
    r"""Response message for Recommend method.

    Attributes:
        results (MutableSequence[google.cloud.discoveryengine_v1beta.types.RecommendResponse.RecommendationResult]):
            A list of recommended Documents. The order
            represents the ranking (from the most relevant
            Document to the least).
        attribution_token (str):
            A unique attribution token. This should be included in the
            [UserEvent][google.cloud.discoveryengine.v1beta.UserEvent]
            logs resulting from this recommendation, which enables
            accurate attribution of recommendation model performance.
        missing_ids (MutableSequence[str]):
            IDs of documents in the request that were
            missing from the default Branch associated with
            the requested ServingConfig.
        validate_only (bool):
            True if
            [RecommendRequest.validate_only][google.cloud.discoveryengine.v1beta.RecommendRequest.validate_only]
            was set.
    """

    class RecommendationResult(proto.Message):
        r"""RecommendationResult represents a generic recommendation
        result with associated metadata.

        Attributes:
            id (str):
                Resource ID of the recommended Document.
            document (google.cloud.discoveryengine_v1beta.types.Document):
                Set if ``returnDocument`` is set to true in
                [RecommendRequest.params][google.cloud.discoveryengine.v1beta.RecommendRequest.params].
            metadata (MutableMapping[str, google.protobuf.struct_pb2.Value]):
                Additional Document metadata or annotations.

                Possible values:

                - ``score``: Recommendation score in double value. Is set if
                  ``returnScore`` is set to true in
                  [RecommendRequest.params][google.cloud.discoveryengine.v1beta.RecommendRequest.params].
        """

        id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        document: gcd_document.Document = proto.Field(
            proto.MESSAGE,
            number=2,
            message=gcd_document.Document,
        )
        metadata: MutableMapping[str, struct_pb2.Value] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=3,
            message=struct_pb2.Value,
        )

    results: MutableSequence[RecommendationResult] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=RecommendationResult,
    )
    attribution_token: str = proto.Field(
        proto.STRING,
        number=2,
    )
    missing_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )
    validate_only: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
