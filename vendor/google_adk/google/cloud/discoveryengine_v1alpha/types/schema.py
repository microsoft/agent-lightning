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

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "Schema",
        "FieldConfig",
    },
)


class Schema(proto.Message):
    r"""Defines the structure and layout of a type of document data.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        struct_schema (google.protobuf.struct_pb2.Struct):
            The structured representation of the schema.

            This field is a member of `oneof`_ ``schema``.
        json_schema (str):
            The JSON representation of the schema.

            This field is a member of `oneof`_ ``schema``.
        name (str):
            Immutable. The full resource name of the schema, in the
            format of
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/schemas/{schema}``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        field_configs (MutableSequence[google.cloud.discoveryengine_v1alpha.types.FieldConfig]):
            Output only. Configurations for fields of the
            schema.
    """

    struct_schema: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="schema",
        message=struct_pb2.Struct,
    )
    json_schema: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="schema",
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    field_configs: MutableSequence["FieldConfig"] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message="FieldConfig",
    )


class FieldConfig(proto.Message):
    r"""Configurations for fields of a schema. For example,
    configuring a field is indexable, or searchable.

    Attributes:
        field_path (str):
            Required. Field path of the schema field. For example:
            ``title``, ``description``, ``release_info.release_year``.
        field_type (google.cloud.discoveryengine_v1alpha.types.FieldConfig.FieldType):
            Output only. Raw type of the field.
        indexable_option (google.cloud.discoveryengine_v1alpha.types.FieldConfig.IndexableOption):
            If
            [indexable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.indexable_option]
            is
            [INDEXABLE_ENABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.IndexableOption.INDEXABLE_ENABLED],
            field values are indexed so that it can be filtered or
            faceted in
            [SearchService.Search][google.cloud.discoveryengine.v1alpha.SearchService.Search].

            If
            [indexable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.indexable_option]
            is unset, the server behavior defaults to
            [INDEXABLE_DISABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.IndexableOption.INDEXABLE_DISABLED]
            for fields that support setting indexable options. For those
            fields that do not support setting indexable options, such
            as ``object`` and ``boolean`` and key properties, the server
            will skip
            [indexable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.indexable_option]
            setting, and setting
            [indexable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.indexable_option]
            for those fields will throw ``INVALID_ARGUMENT`` error.
        dynamic_facetable_option (google.cloud.discoveryengine_v1alpha.types.FieldConfig.DynamicFacetableOption):
            If
            [dynamic_facetable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.dynamic_facetable_option]
            is
            [DYNAMIC_FACETABLE_ENABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.DynamicFacetableOption.DYNAMIC_FACETABLE_ENABLED],
            field values are available for dynamic facet. Could only be
            [DYNAMIC_FACETABLE_DISABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.DynamicFacetableOption.DYNAMIC_FACETABLE_DISABLED]
            if
            [FieldConfig.indexable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.indexable_option]
            is
            [INDEXABLE_DISABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.IndexableOption.INDEXABLE_DISABLED].
            Otherwise, an ``INVALID_ARGUMENT`` error will be returned.

            If
            [dynamic_facetable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.dynamic_facetable_option]
            is unset, the server behavior defaults to
            [DYNAMIC_FACETABLE_DISABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.DynamicFacetableOption.DYNAMIC_FACETABLE_DISABLED]
            for fields that support setting dynamic facetable options.
            For those fields that do not support setting dynamic
            facetable options, such as ``object`` and ``boolean``, the
            server will skip dynamic facetable option setting, and
            setting
            [dynamic_facetable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.dynamic_facetable_option]
            for those fields will throw ``INVALID_ARGUMENT`` error.
        searchable_option (google.cloud.discoveryengine_v1alpha.types.FieldConfig.SearchableOption):
            If
            [searchable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.searchable_option]
            is
            [SEARCHABLE_ENABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.SearchableOption.SEARCHABLE_ENABLED],
            field values are searchable by text queries in
            [SearchService.Search][google.cloud.discoveryengine.v1alpha.SearchService.Search].

            If
            [SEARCHABLE_ENABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.SearchableOption.SEARCHABLE_ENABLED]
            but field type is numerical, field values will not be
            searchable by text queries in
            [SearchService.Search][google.cloud.discoveryengine.v1alpha.SearchService.Search],
            as there are no text values associated to numerical fields.

            If
            [searchable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.searchable_option]
            is unset, the server behavior defaults to
            [SEARCHABLE_DISABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.SearchableOption.SEARCHABLE_DISABLED]
            for fields that support setting searchable options. Only
            ``string`` fields that have no key property mapping support
            setting
            [searchable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.searchable_option].

            For those fields that do not support setting searchable
            options, the server will skip searchable option setting, and
            setting
            [searchable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.searchable_option]
            for those fields will throw ``INVALID_ARGUMENT`` error.
        retrievable_option (google.cloud.discoveryengine_v1alpha.types.FieldConfig.RetrievableOption):
            If
            [retrievable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.retrievable_option]
            is
            [RETRIEVABLE_ENABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.RetrievableOption.RETRIEVABLE_ENABLED],
            field values are included in the search results.

            If
            [retrievable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.retrievable_option]
            is unset, the server behavior defaults to
            [RETRIEVABLE_DISABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.RetrievableOption.RETRIEVABLE_DISABLED]
            for fields that support setting retrievable options. For
            those fields that do not support setting retrievable
            options, such as ``object`` and ``boolean``, the server will
            skip retrievable option setting, and setting
            [retrievable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.retrievable_option]
            for those fields will throw ``INVALID_ARGUMENT`` error.
        completable_option (google.cloud.discoveryengine_v1alpha.types.FieldConfig.CompletableOption):
            If
            [completable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.completable_option]
            is
            [COMPLETABLE_ENABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.CompletableOption.COMPLETABLE_ENABLED],
            field values are directly used and returned as suggestions
            for Autocomplete in
            [CompletionService.CompleteQuery][google.cloud.discoveryengine.v1alpha.CompletionService.CompleteQuery].

            If
            [completable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.completable_option]
            is unset, the server behavior defaults to
            [COMPLETABLE_DISABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.CompletableOption.COMPLETABLE_DISABLED]
            for fields that support setting completable options, which
            are just ``string`` fields. For those fields that do not
            support setting completable options, the server will skip
            completable option setting, and setting
            [completable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.completable_option]
            for those fields will throw ``INVALID_ARGUMENT`` error.
        recs_filterable_option (google.cloud.discoveryengine_v1alpha.types.FieldConfig.FilterableOption):
            If
            [recs_filterable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.recs_filterable_option]
            is
            [FILTERABLE_ENABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.FilterableOption.FILTERABLE_ENABLED],
            field values are filterable by filter expression in
            [RecommendationService.Recommend][google.cloud.discoveryengine.v1alpha.RecommendationService.Recommend].

            If
            [FILTERABLE_ENABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.FilterableOption.FILTERABLE_ENABLED]
            but the field type is numerical, field values are not
            filterable by text queries in
            [RecommendationService.Recommend][google.cloud.discoveryengine.v1alpha.RecommendationService.Recommend].
            Only textual fields are supported.

            If
            [recs_filterable_option][google.cloud.discoveryengine.v1alpha.FieldConfig.recs_filterable_option]
            is unset, the default setting is
            [FILTERABLE_DISABLED][google.cloud.discoveryengine.v1alpha.FieldConfig.FilterableOption.FILTERABLE_DISABLED]
            for fields that support setting filterable options.

            When a field set to [FILTERABLE_DISABLED] is filtered, a
            warning is generated and an empty result is returned.
        key_property_type (str):
            Output only. Type of the key property that this field is
            mapped to. Empty string if this is not annotated as mapped
            to a key property.

            Example types are ``title``, ``description``. Full list is
            defined by ``keyPropertyMapping`` in the schema field
            annotation.

            If the schema field has a ``KeyPropertyMapping`` annotation,
            ``indexable_option`` and ``searchable_option`` of this field
            cannot be modified.
        advanced_site_search_data_sources (MutableSequence[google.cloud.discoveryengine_v1alpha.types.FieldConfig.AdvancedSiteSearchDataSource]):
            If this field is set, only the corresponding source will be
            indexed for this field. Otherwise, the values from different
            sources are merged.

            Assuming a page with ``<author, a>`` in meta tag, and
            ``<author, b>`` in page map: if this enum is set to
            METATAGS, we will only index ``<author, a>``; if this enum
            is not set, we will merge them and index
            ``<author, [a, b]>``.
        schema_org_paths (MutableSequence[str]):
            Field paths for indexing custom attribute from schema.org
            data. More details of schema.org and its defined types can
            be found at `schema.org <https://schema.org>`__.

            It is only used on advanced site search schema.

            Currently only support full path from root. The full path to
            a field is constructed by concatenating field names,
            starting from ``_root``, with a period ``.`` as the
            delimiter. Examples:

            - Publish date of the root: \_root.datePublished
            - Publish date of the reviews: \_root.review.datePublished
    """

    class FieldType(proto.Enum):
        r"""Field value type in the Schema.

        Values:
            FIELD_TYPE_UNSPECIFIED (0):
                Field type is unspecified.
            OBJECT (1):
                Field value type is Object.
            STRING (2):
                Field value type is String.
            NUMBER (3):
                Field value type is Number.
            INTEGER (4):
                Field value type is Integer.
            BOOLEAN (5):
                Field value type is Boolean.
            GEOLOCATION (6):
                Field value type is Geolocation. Geolocation is expressed as
                an object with the following keys:

                - ``id``: a string representing the location id
                - ``longitude``: a number representing the longitude
                  coordinate of the location
                - ``latitude``: a number repesenting the latitude coordinate
                  of the location
                - ``address``: a string representing the full address of the
                  location

                ``latitude`` and ``longitude`` must always be provided
                together. At least one of a) ``address`` or b)
                ``latitude``-``longitude`` pair must be provided.
            DATETIME (7):
                Field value type is Datetime. Datetime can be expressed as
                either:

                - a number representing milliseconds-since-the-epoch
                - a string representing milliseconds-since-the-epoch. e.g.
                  ``"1420070400001"``
                - a string representing the `ISO
                  8601 <https://en.wikipedia.org/wiki/ISO_8601>`__ date or
                  date and time. e.g. ``"2015-01-01"`` or
                  ``"2015-01-01T12:10:30Z"``
        """
        FIELD_TYPE_UNSPECIFIED = 0
        OBJECT = 1
        STRING = 2
        NUMBER = 3
        INTEGER = 4
        BOOLEAN = 5
        GEOLOCATION = 6
        DATETIME = 7

    class IndexableOption(proto.Enum):
        r"""The setting of Indexable options in schema.

        Values:
            INDEXABLE_OPTION_UNSPECIFIED (0):
                Value used when unset.
            INDEXABLE_ENABLED (1):
                Indexable option enabled for a schema field.
            INDEXABLE_DISABLED (2):
                Indexable option disabled for a schema field.
        """
        INDEXABLE_OPTION_UNSPECIFIED = 0
        INDEXABLE_ENABLED = 1
        INDEXABLE_DISABLED = 2

    class DynamicFacetableOption(proto.Enum):
        r"""The status of the dynamic facetable option of a schema field.

        Values:
            DYNAMIC_FACETABLE_OPTION_UNSPECIFIED (0):
                Value used when unset.
            DYNAMIC_FACETABLE_ENABLED (1):
                Dynamic facetable option enabled for a schema
                field.
            DYNAMIC_FACETABLE_DISABLED (2):
                Dynamic facetable option disabled for a
                schema field.
        """
        DYNAMIC_FACETABLE_OPTION_UNSPECIFIED = 0
        DYNAMIC_FACETABLE_ENABLED = 1
        DYNAMIC_FACETABLE_DISABLED = 2

    class SearchableOption(proto.Enum):
        r"""The setting of Searchable options in schema.

        Values:
            SEARCHABLE_OPTION_UNSPECIFIED (0):
                Value used when unset.
            SEARCHABLE_ENABLED (1):
                Searchable option enabled for a schema field.
            SEARCHABLE_DISABLED (2):
                Searchable option disabled for a schema
                field.
        """
        SEARCHABLE_OPTION_UNSPECIFIED = 0
        SEARCHABLE_ENABLED = 1
        SEARCHABLE_DISABLED = 2

    class RetrievableOption(proto.Enum):
        r"""The setting of Retrievable options in schema.

        Values:
            RETRIEVABLE_OPTION_UNSPECIFIED (0):
                Value used when unset.
            RETRIEVABLE_ENABLED (1):
                Retrievable option enabled for a schema
                field.
            RETRIEVABLE_DISABLED (2):
                Retrievable option disabled for a schema
                field.
        """
        RETRIEVABLE_OPTION_UNSPECIFIED = 0
        RETRIEVABLE_ENABLED = 1
        RETRIEVABLE_DISABLED = 2

    class CompletableOption(proto.Enum):
        r"""The setting of Completable options in schema.

        Values:
            COMPLETABLE_OPTION_UNSPECIFIED (0):
                Value used when unset.
            COMPLETABLE_ENABLED (1):
                Completable option enabled for a schema
                field.
            COMPLETABLE_DISABLED (2):
                Completable option disabled for a schema
                field.
        """
        COMPLETABLE_OPTION_UNSPECIFIED = 0
        COMPLETABLE_ENABLED = 1
        COMPLETABLE_DISABLED = 2

    class FilterableOption(proto.Enum):
        r"""Sets the filterable option for schema fields.

        Values:
            FILTERABLE_OPTION_UNSPECIFIED (0):
                Value used when unset.
            FILTERABLE_ENABLED (1):
                Filterable option enabled for a schema field.
            FILTERABLE_DISABLED (2):
                Filterable option disabled for a schema
                field.
        """
        FILTERABLE_OPTION_UNSPECIFIED = 0
        FILTERABLE_ENABLED = 1
        FILTERABLE_DISABLED = 2

    class AdvancedSiteSearchDataSource(proto.Enum):
        r"""Optional source of the advanced site search field.

        Values:
            ADVANCED_SITE_SEARCH_DATA_SOURCE_UNSPECIFIED (0):
                Value used when unset.
            METATAGS (1):
                Retrieve value from meta tag.
            PAGEMAP (2):
                Retrieve value from page map.
            URI_PATTERN_MAPPING (3):
                Retrieve value from the attributes set by
                [SiteSearchEngineService.SetUriPatternDocumentData][google.cloud.discoveryengine.v1alpha.SiteSearchEngineService.SetUriPatternDocumentData]
                API.
            SCHEMA_ORG (4):
                Retrieve value from schema.org data.
        """
        ADVANCED_SITE_SEARCH_DATA_SOURCE_UNSPECIFIED = 0
        METATAGS = 1
        PAGEMAP = 2
        URI_PATTERN_MAPPING = 3
        SCHEMA_ORG = 4

    field_path: str = proto.Field(
        proto.STRING,
        number=1,
    )
    field_type: FieldType = proto.Field(
        proto.ENUM,
        number=2,
        enum=FieldType,
    )
    indexable_option: IndexableOption = proto.Field(
        proto.ENUM,
        number=3,
        enum=IndexableOption,
    )
    dynamic_facetable_option: DynamicFacetableOption = proto.Field(
        proto.ENUM,
        number=4,
        enum=DynamicFacetableOption,
    )
    searchable_option: SearchableOption = proto.Field(
        proto.ENUM,
        number=5,
        enum=SearchableOption,
    )
    retrievable_option: RetrievableOption = proto.Field(
        proto.ENUM,
        number=6,
        enum=RetrievableOption,
    )
    completable_option: CompletableOption = proto.Field(
        proto.ENUM,
        number=8,
        enum=CompletableOption,
    )
    recs_filterable_option: FilterableOption = proto.Field(
        proto.ENUM,
        number=9,
        enum=FilterableOption,
    )
    key_property_type: str = proto.Field(
        proto.STRING,
        number=7,
    )
    advanced_site_search_data_sources: MutableSequence[
        AdvancedSiteSearchDataSource
    ] = proto.RepeatedField(
        proto.ENUM,
        number=10,
        enum=AdvancedSiteSearchDataSource,
    )
    schema_org_paths: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=11,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
