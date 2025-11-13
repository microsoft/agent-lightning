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

from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1alpha.types import common

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "UserEvent",
        "PageInfo",
        "SearchInfo",
        "CompletionInfo",
        "TransactionInfo",
        "DocumentInfo",
        "PanelInfo",
        "MediaInfo",
    },
)


class UserEvent(proto.Message):
    r"""UserEvent captures all metadata information Discovery Engine
    API needs to know about how end users interact with your
    website.

    Attributes:
        event_type (str):
            Required. User event type. Allowed values are:

            Generic values:

            - ``search``: Search for Documents.
            - ``view-item``: Detailed page view of a Document.
            - ``view-item-list``: View of a panel or ordered list of
              Documents.
            - ``view-home-page``: View of the home page.
            - ``view-category-page``: View of a category page, e.g. Home
              > Men > Jeans

            Retail-related values:

            - ``add-to-cart``: Add an item(s) to cart, e.g. in Retail
              online shopping
            - ``purchase``: Purchase an item(s)

            Media-related values:

            - ``media-play``: Start/resume watching a video, playing a
              song, etc.
            - ``media-complete``: Finished or stopped midway through a
              video, song, etc.
        user_pseudo_id (str):
            Required. A unique identifier for tracking visitors.

            For example, this could be implemented with an HTTP cookie,
            which should be able to uniquely identify a visitor on a
            single device. This unique identifier should not change if
            the visitor log in/out of the website.

            Do not set the field to the same fixed ID for different
            users. This mixes the event history of those users together,
            which results in degraded model quality.

            The field must be a UTF-8 encoded string with a length limit
            of 128 characters. Otherwise, an ``INVALID_ARGUMENT`` error
            is returned.

            The field should not contain PII or user-data. We recommend
            to use Google Analytics `Client
            ID <https://developers.google.com/analytics/devguides/collection/analyticsjs/field-reference#clientId>`__
            for this field.
        engine (str):
            The [Engine][google.cloud.discoveryengine.v1alpha.Engine]
            resource name, in the form of
            ``projects/{project}/locations/{location}/collections/{collection_id}/engines/{engine_id}``.

            Optional. Only required for
            [Engine][google.cloud.discoveryengine.v1alpha.Engine]
            produced user events. For example, user events from blended
            search.
        data_store (str):
            The
            [DataStore][google.cloud.discoveryengine.v1alpha.DataStore]
            resource full name, of the form
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}``.

            Optional. Only required for user events whose data store
            can't by determined by
            [UserEvent.engine][google.cloud.discoveryengine.v1alpha.UserEvent.engine]
            or
            [UserEvent.documents][google.cloud.discoveryengine.v1alpha.UserEvent.documents].
            If data store is set in the parent of write/import/collect
            user event requests, this field can be omitted.
        event_time (google.protobuf.timestamp_pb2.Timestamp):
            Only required for
            [UserEventService.ImportUserEvents][google.cloud.discoveryengine.v1alpha.UserEventService.ImportUserEvents]
            method. Timestamp of when the user event happened.
        user_info (google.cloud.discoveryengine_v1alpha.types.UserInfo):
            Information about the end user.
        direct_user_request (bool):
            Should set to true if the request is made directly from the
            end user, in which case the
            [UserEvent.user_info.user_agent][google.cloud.discoveryengine.v1alpha.UserInfo.user_agent]
            can be populated from the HTTP request.

            This flag should be set only if the API request is made
            directly from the end user such as a mobile app (and not if
            a gateway or a server is processing and pushing the user
            events).

            This should not be set when using the JavaScript tag in
            [UserEventService.CollectUserEvent][google.cloud.discoveryengine.v1alpha.UserEventService.CollectUserEvent].
        session_id (str):
            A unique identifier for tracking a visitor session with a
            length limit of 128 bytes. A session is an aggregation of an
            end user behavior in a time span.

            A general guideline to populate the session_id:

            1. If user has no activity for 30 min, a new session_id
               should be assigned.
            2. The session_id should be unique across users, suggest use
               uuid or add
               [UserEvent.user_pseudo_id][google.cloud.discoveryengine.v1alpha.UserEvent.user_pseudo_id]
               as prefix.
        page_info (google.cloud.discoveryengine_v1alpha.types.PageInfo):
            Page metadata such as categories and other critical
            information for certain event types such as
            ``view-category-page``.
        attribution_token (str):
            Token to attribute an API response to user action(s) to
            trigger the event.

            Highly recommended for user events that are the result of
            [RecommendationService.Recommend][google.cloud.discoveryengine.v1alpha.RecommendationService.Recommend].
            This field enables accurate attribution of recommendation
            model performance.

            The value must be one of:

            - [RecommendResponse.attribution_token][google.cloud.discoveryengine.v1alpha.RecommendResponse.attribution_token]
              for events that are the result of
              [RecommendationService.Recommend][google.cloud.discoveryengine.v1alpha.RecommendationService.Recommend].
            - [SearchResponse.attribution_token][google.cloud.discoveryengine.v1alpha.SearchResponse.attribution_token]
              for events that are the result of
              [SearchService.Search][google.cloud.discoveryengine.v1alpha.SearchService.Search].

            This token enables us to accurately attribute page view or
            conversion completion back to the event and the particular
            predict response containing this clicked/purchased product.
            If user clicks on product K in the recommendation results,
            pass
            [RecommendResponse.attribution_token][google.cloud.discoveryengine.v1alpha.RecommendResponse.attribution_token]
            as a URL parameter to product K's page. When recording
            events on product K's page, log the
            [RecommendResponse.attribution_token][google.cloud.discoveryengine.v1alpha.RecommendResponse.attribution_token]
            to this field.
        filter (str):
            The filter syntax consists of an expression language for
            constructing a predicate from one or more fields of the
            documents being filtered.

            One example is for ``search`` events, the associated
            [SearchRequest][google.cloud.discoveryengine.v1alpha.SearchRequest]
            may contain a filter expression in
            [SearchRequest.filter][google.cloud.discoveryengine.v1alpha.SearchRequest.filter]
            conforming to https://google.aip.dev/160#filtering.

            Similarly, for ``view-item-list`` events that are generated
            from a
            [RecommendRequest][google.cloud.discoveryengine.v1alpha.RecommendRequest],
            this field may be populated directly from
            [RecommendRequest.filter][google.cloud.discoveryengine.v1alpha.RecommendRequest.filter]
            conforming to https://google.aip.dev/160#filtering.

            The value must be a UTF-8 encoded string with a length limit
            of 1,000 characters. Otherwise, an ``INVALID_ARGUMENT``
            error is returned.
        documents (MutableSequence[google.cloud.discoveryengine_v1alpha.types.DocumentInfo]):
            List of
            [Document][google.cloud.discoveryengine.v1alpha.Document]s
            associated with this user event.

            This field is optional except for the following event types:

            - ``view-item``
            - ``add-to-cart``
            - ``purchase``
            - ``media-play``
            - ``media-complete``

            In a ``search`` event, this field represents the documents
            returned to the end user on the current page (the end user
            may have not finished browsing the whole page yet). When a
            new page is returned to the end user, after
            pagination/filtering/ordering even for the same query, a new
            ``search`` event with different
            [UserEvent.documents][google.cloud.discoveryengine.v1alpha.UserEvent.documents]
            is desired.
        panel (google.cloud.discoveryengine_v1alpha.types.PanelInfo):
            Panel metadata associated with this user
            event.
        search_info (google.cloud.discoveryengine_v1alpha.types.SearchInfo):
            [SearchService.Search][google.cloud.discoveryengine.v1alpha.SearchService.Search]
            details related to the event.

            This field should be set for ``search`` event.
        completion_info (google.cloud.discoveryengine_v1alpha.types.CompletionInfo):
            [CompletionService.CompleteQuery][google.cloud.discoveryengine.v1alpha.CompletionService.CompleteQuery]
            details related to the event.

            This field should be set for ``search`` event when
            autocomplete function is enabled and the user clicks a
            suggestion for search.
        transaction_info (google.cloud.discoveryengine_v1alpha.types.TransactionInfo):
            The transaction metadata (if any) associated
            with this user event.
        tag_ids (MutableSequence[str]):
            A list of identifiers for the independent
            experiment groups this user event belongs to.
            This is used to distinguish between user events
            associated with different experiment setups.
        promotion_ids (MutableSequence[str]):
            The promotion IDs if this is an event
            associated with promotions. Currently, this
            field is restricted to at most one ID.
        attributes (MutableMapping[str, google.cloud.discoveryengine_v1alpha.types.CustomAttribute]):
            Extra user event features to include in the recommendation
            model. These attributes must NOT contain data that needs to
            be parsed or processed further, e.g. JSON or other
            encodings.

            If you provide custom attributes for ingested user events,
            also include them in the user events that you associate with
            prediction requests. Custom attribute formatting must be
            consistent between imported events and events provided with
            prediction requests. This lets the Discovery Engine API use
            those custom attributes when training models and serving
            predictions, which helps improve recommendation quality.

            This field needs to pass all below criteria, otherwise an
            ``INVALID_ARGUMENT`` error is returned:

            - The key must be a UTF-8 encoded string with a length limit
              of 5,000 characters.
            - For text attributes, at most 400 values are allowed. Empty
              values are not allowed. Each value must be a UTF-8 encoded
              string with a length limit of 256 characters.
            - For number attributes, at most 400 values are allowed.

            For product recommendations, an example of extra user
            information is ``traffic_channel``, which is how a user
            arrives at the site. Users can arrive at the site by coming
            to the site directly, coming through Google search, or in
            other ways.
        media_info (google.cloud.discoveryengine_v1alpha.types.MediaInfo):
            Media-specific info.
    """

    event_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    user_pseudo_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    engine: str = proto.Field(
        proto.STRING,
        number=19,
    )
    data_store: str = proto.Field(
        proto.STRING,
        number=20,
    )
    event_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    user_info: common.UserInfo = proto.Field(
        proto.MESSAGE,
        number=4,
        message=common.UserInfo,
    )
    direct_user_request: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    session_id: str = proto.Field(
        proto.STRING,
        number=6,
    )
    page_info: "PageInfo" = proto.Field(
        proto.MESSAGE,
        number=7,
        message="PageInfo",
    )
    attribution_token: str = proto.Field(
        proto.STRING,
        number=8,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=9,
    )
    documents: MutableSequence["DocumentInfo"] = proto.RepeatedField(
        proto.MESSAGE,
        number=10,
        message="DocumentInfo",
    )
    panel: "PanelInfo" = proto.Field(
        proto.MESSAGE,
        number=11,
        message="PanelInfo",
    )
    search_info: "SearchInfo" = proto.Field(
        proto.MESSAGE,
        number=12,
        message="SearchInfo",
    )
    completion_info: "CompletionInfo" = proto.Field(
        proto.MESSAGE,
        number=13,
        message="CompletionInfo",
    )
    transaction_info: "TransactionInfo" = proto.Field(
        proto.MESSAGE,
        number=14,
        message="TransactionInfo",
    )
    tag_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=15,
    )
    promotion_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=16,
    )
    attributes: MutableMapping[str, common.CustomAttribute] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=17,
        message=common.CustomAttribute,
    )
    media_info: "MediaInfo" = proto.Field(
        proto.MESSAGE,
        number=18,
        message="MediaInfo",
    )


class PageInfo(proto.Message):
    r"""Detailed page information.

    Attributes:
        pageview_id (str):
            A unique ID of a web page view.

            This should be kept the same for all user events triggered
            from the same pageview. For example, an item detail page
            view could trigger multiple events as the user is browsing
            the page. The ``pageview_id`` property should be kept the
            same for all these events so that they can be grouped
            together properly.

            When using the client side event reporting with JavaScript
            pixel and Google Tag Manager, this value is filled in
            automatically.
        page_category (str):
            The most specific category associated with a category page.

            To represent full path of category, use '>' sign to separate
            different hierarchies. If '>' is part of the category name,
            replace it with other character(s).

            Category pages include special pages such as sales or
            promotions. For instance, a special sale page may have the
            category hierarchy:
            ``"pageCategory" : "Sales > 2017 Black Friday Deals"``.

            Required for ``view-category-page`` events. Other event
            types should not set this field. Otherwise, an
            ``INVALID_ARGUMENT`` error is returned.
        uri (str):
            Complete URL (window.location.href) of the
            user's current page.
            When using the client side event reporting with
            JavaScript pixel and Google Tag Manager, this
            value is filled in automatically. Maximum length
            5,000 characters.
        referrer_uri (str):
            The referrer URL of the current page.

            When using the client side event reporting with
            JavaScript pixel and Google Tag Manager, this
            value is filled in automatically. However, some
            browser privacy restrictions may cause this
            field to be empty.
    """

    pageview_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_category: str = proto.Field(
        proto.STRING,
        number=2,
    )
    uri: str = proto.Field(
        proto.STRING,
        number=3,
    )
    referrer_uri: str = proto.Field(
        proto.STRING,
        number=4,
    )


class SearchInfo(proto.Message):
    r"""Detailed search information.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        search_query (str):
            The user's search query.

            See
            [SearchRequest.query][google.cloud.discoveryengine.v1alpha.SearchRequest.query]
            for definition.

            The value must be a UTF-8 encoded string with a length limit
            of 5,000 characters. Otherwise, an ``INVALID_ARGUMENT``
            error is returned.

            At least one of
            [search_query][google.cloud.discoveryengine.v1alpha.SearchInfo.search_query]
            or
            [PageInfo.page_category][google.cloud.discoveryengine.v1alpha.PageInfo.page_category]
            is required for ``search`` events. Other event types should
            not set this field. Otherwise, an ``INVALID_ARGUMENT`` error
            is returned.
        order_by (str):
            The order in which products are returned, if applicable.

            See
            [SearchRequest.order_by][google.cloud.discoveryengine.v1alpha.SearchRequest.order_by]
            for definition and syntax.

            The value must be a UTF-8 encoded string with a length limit
            of 1,000 characters. Otherwise, an ``INVALID_ARGUMENT``
            error is returned.

            This can only be set for ``search`` events. Other event
            types should not set this field. Otherwise, an
            ``INVALID_ARGUMENT`` error is returned.
        offset (int):
            An integer that specifies the current offset for pagination
            (the 0-indexed starting location, amongst the products
            deemed by the API as relevant).

            See
            [SearchRequest.offset][google.cloud.discoveryengine.v1alpha.SearchRequest.offset]
            for definition.

            If this field is negative, an ``INVALID_ARGUMENT`` is
            returned.

            This can only be set for ``search`` events. Other event
            types should not set this field. Otherwise, an
            ``INVALID_ARGUMENT`` error is returned.

            This field is a member of `oneof`_ ``_offset``.
    """

    search_query: str = proto.Field(
        proto.STRING,
        number=1,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=2,
    )
    offset: int = proto.Field(
        proto.INT32,
        number=3,
        optional=True,
    )


class CompletionInfo(proto.Message):
    r"""Detailed completion information including completion
    attribution token and clicked completion info.

    Attributes:
        selected_suggestion (str):
            End user selected
            [CompleteQueryResponse.QuerySuggestion.suggestion][google.cloud.discoveryengine.v1alpha.CompleteQueryResponse.QuerySuggestion.suggestion].
        selected_position (int):
            End user selected
            [CompleteQueryResponse.QuerySuggestion.suggestion][google.cloud.discoveryengine.v1alpha.CompleteQueryResponse.QuerySuggestion.suggestion]
            position, starting from 0.
    """

    selected_suggestion: str = proto.Field(
        proto.STRING,
        number=1,
    )
    selected_position: int = proto.Field(
        proto.INT32,
        number=2,
    )


class TransactionInfo(proto.Message):
    r"""A transaction represents the entire purchase transaction.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        value (float):
            Required. Total non-zero value associated
            with the transaction. This value may include
            shipping, tax, or other adjustments to the total
            value that you want to include.

            This field is a member of `oneof`_ ``_value``.
        currency (str):
            Required. Currency code. Use three-character
            ISO-4217 code.
        transaction_id (str):
            The transaction ID with a length limit of 128
            characters.
        tax (float):
            All the taxes associated with the
            transaction.

            This field is a member of `oneof`_ ``_tax``.
        cost (float):
            All the costs associated with the products. These can be
            manufacturing costs, shipping expenses not borne by the end
            user, or any other costs, such that:

            - Profit =
              [value][google.cloud.discoveryengine.v1alpha.TransactionInfo.value]
              -
              [tax][google.cloud.discoveryengine.v1alpha.TransactionInfo.tax]
              -
              [cost][google.cloud.discoveryengine.v1alpha.TransactionInfo.cost]

            This field is a member of `oneof`_ ``_cost``.
        discount_value (float):
            The total discount(s) value applied to this transaction.
            This figure should be excluded from
            [TransactionInfo.value][google.cloud.discoveryengine.v1alpha.TransactionInfo.value]

            For example, if a user paid
            [TransactionInfo.value][google.cloud.discoveryengine.v1alpha.TransactionInfo.value]
            amount, then nominal (pre-discount) value of the transaction
            is the sum of
            [TransactionInfo.value][google.cloud.discoveryengine.v1alpha.TransactionInfo.value]
            and
            [TransactionInfo.discount_value][google.cloud.discoveryengine.v1alpha.TransactionInfo.discount_value]

            This means that profit is calculated the same way,
            regardless of the discount value, and that
            [TransactionInfo.discount_value][google.cloud.discoveryengine.v1alpha.TransactionInfo.discount_value]
            can be larger than
            [TransactionInfo.value][google.cloud.discoveryengine.v1alpha.TransactionInfo.value]:

            - Profit =
              [value][google.cloud.discoveryengine.v1alpha.TransactionInfo.value]
              -
              [tax][google.cloud.discoveryengine.v1alpha.TransactionInfo.tax]
              -
              [cost][google.cloud.discoveryengine.v1alpha.TransactionInfo.cost]

            This field is a member of `oneof`_ ``_discount_value``.
    """

    value: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    currency: str = proto.Field(
        proto.STRING,
        number=2,
    )
    transaction_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    tax: float = proto.Field(
        proto.FLOAT,
        number=4,
        optional=True,
    )
    cost: float = proto.Field(
        proto.FLOAT,
        number=5,
        optional=True,
    )
    discount_value: float = proto.Field(
        proto.FLOAT,
        number=6,
        optional=True,
    )


class DocumentInfo(proto.Message):
    r"""Detailed document information associated with a user event.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        id (str):
            The
            [Document][google.cloud.discoveryengine.v1alpha.Document]
            resource ID.

            This field is a member of `oneof`_ ``document_descriptor``.
        name (str):
            The
            [Document][google.cloud.discoveryengine.v1alpha.Document]
            resource full name, of the form:
            ``projects/{project_id}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}/branches/{branch_id}/documents/{document_id}``

            This field is a member of `oneof`_ ``document_descriptor``.
        uri (str):
            The
            [Document][google.cloud.discoveryengine.v1alpha.Document]
            URI - only allowed for website data stores.

            This field is a member of `oneof`_ ``document_descriptor``.
        quantity (int):
            Quantity of the Document associated with the user event.
            Defaults to 1.

            For example, this field is 2 if two quantities of the same
            Document are involved in a ``add-to-cart`` event.

            Required for events of the following event types:

            - ``add-to-cart``
            - ``purchase``

            This field is a member of `oneof`_ ``_quantity``.
        promotion_ids (MutableSequence[str]):
            The promotion IDs associated with this
            Document. Currently, this field is restricted to
            at most one ID.
        joined (bool):
            Output only. Whether the referenced Document
            can be found in the data store.
    """

    id: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="document_descriptor",
    )
    name: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="document_descriptor",
    )
    uri: str = proto.Field(
        proto.STRING,
        number=6,
        oneof="document_descriptor",
    )
    quantity: int = proto.Field(
        proto.INT32,
        number=3,
        optional=True,
    )
    promotion_ids: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=4,
    )
    joined: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


class PanelInfo(proto.Message):
    r"""Detailed panel information associated with a user event.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        panel_id (str):
            Required. The panel ID.
        display_name (str):
            The display name of the panel.
        panel_position (int):
            The ordered position of the panel, if shown to the user with
            other panels. If set, then
            [total_panels][google.cloud.discoveryengine.v1alpha.PanelInfo.total_panels]
            must also be set.

            This field is a member of `oneof`_ ``_panel_position``.
        total_panels (int):
            The total number of panels, including this one, shown to the
            user. Must be set if
            [panel_position][google.cloud.discoveryengine.v1alpha.PanelInfo.panel_position]
            is set.

            This field is a member of `oneof`_ ``_total_panels``.
    """

    panel_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    panel_position: int = proto.Field(
        proto.INT32,
        number=4,
        optional=True,
    )
    total_panels: int = proto.Field(
        proto.INT32,
        number=5,
        optional=True,
    )


class MediaInfo(proto.Message):
    r"""Media-specific user event information.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        media_progress_duration (google.protobuf.duration_pb2.Duration):
            The media progress time in seconds, if applicable. For
            example, if the end user has finished 90 seconds of a
            playback video, then
            [MediaInfo.media_progress_duration.seconds][google.protobuf.Duration.seconds]
            should be set to 90.
        media_progress_percentage (float):
            Media progress should be computed using only the
            [media_progress_duration][google.cloud.discoveryengine.v1alpha.MediaInfo.media_progress_duration]
            relative to the media total length.

            This value must be between ``[0, 1.0]`` inclusive.

            If this is not a playback or the progress cannot be computed
            (e.g. ongoing livestream), this field should be unset.

            This field is a member of `oneof`_ ``_media_progress_percentage``.
    """

    media_progress_duration: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )
    media_progress_percentage: float = proto.Field(
        proto.FLOAT,
        number=2,
        optional=True,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
