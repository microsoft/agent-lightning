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

from google.cloud.discoveryengine_v1alpha.types import (
    document_processing_config as gcd_document_processing_config,
)
from google.cloud.discoveryengine_v1alpha.types import common
from google.cloud.discoveryengine_v1alpha.types import schema

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "DataStore",
        "LanguageInfo",
        "WorkspaceConfig",
    },
)


class DataStore(proto.Message):
    r"""DataStore captures global settings and configs at the
    DataStore level.

    Attributes:
        name (str):
            Immutable. The full resource name of the data store. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        display_name (str):
            Required. The data store display name.

            This field must be a UTF-8 encoded string with a length
            limit of 128 characters. Otherwise, an INVALID_ARGUMENT
            error is returned.
        industry_vertical (google.cloud.discoveryengine_v1alpha.types.IndustryVertical):
            Immutable. The industry vertical that the
            data store registers.
        solution_types (MutableSequence[google.cloud.discoveryengine_v1alpha.types.SolutionType]):
            The solutions that the data store enrolls. Available
            solutions for each
            [industry_vertical][google.cloud.discoveryengine.v1alpha.DataStore.industry_vertical]:

            - ``MEDIA``: ``SOLUTION_TYPE_RECOMMENDATION`` and
              ``SOLUTION_TYPE_SEARCH``.
            - ``SITE_SEARCH``: ``SOLUTION_TYPE_SEARCH`` is automatically
              enrolled. Other solutions cannot be enrolled.
        default_schema_id (str):
            Output only. The id of the default
            [Schema][google.cloud.discoveryengine.v1alpha.Schema]
            asscociated to this data store.
        content_config (google.cloud.discoveryengine_v1alpha.types.DataStore.ContentConfig):
            Immutable. The content config of the data store. If this
            field is unset, the server behavior defaults to
            [ContentConfig.NO_CONTENT][google.cloud.discoveryengine.v1alpha.DataStore.ContentConfig.NO_CONTENT].
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp the
            [DataStore][google.cloud.discoveryengine.v1alpha.DataStore]
            was created at.
        language_info (google.cloud.discoveryengine_v1alpha.types.LanguageInfo):
            Language info for DataStore.
        idp_config (google.cloud.discoveryengine_v1alpha.types.IdpConfig):
            Output only. Data store level identity
            provider config.
        acl_enabled (bool):
            Immutable. Whether data in the
            [DataStore][google.cloud.discoveryengine.v1alpha.DataStore]
            has ACL information. If set to ``true``, the source data
            must have ACL. ACL will be ingested when data is ingested by
            [DocumentService.ImportDocuments][google.cloud.discoveryengine.v1alpha.DocumentService.ImportDocuments]
            methods.

            When ACL is enabled for the
            [DataStore][google.cloud.discoveryengine.v1alpha.DataStore],
            [Document][google.cloud.discoveryengine.v1alpha.Document]
            can't be accessed by calling
            [DocumentService.GetDocument][google.cloud.discoveryengine.v1alpha.DocumentService.GetDocument]
            or
            [DocumentService.ListDocuments][google.cloud.discoveryengine.v1alpha.DocumentService.ListDocuments].

            Currently ACL is only supported in ``GENERIC`` industry
            vertical with non-``PUBLIC_WEBSITE`` content config.
        workspace_config (google.cloud.discoveryengine_v1alpha.types.WorkspaceConfig):
            Config to store data store type configuration for workspace
            data. This must be set when
            [DataStore.content_config][google.cloud.discoveryengine.v1alpha.DataStore.content_config]
            is set as
            [DataStore.ContentConfig.GOOGLE_WORKSPACE][google.cloud.discoveryengine.v1alpha.DataStore.ContentConfig.GOOGLE_WORKSPACE].
        document_processing_config (google.cloud.discoveryengine_v1alpha.types.DocumentProcessingConfig):
            Configuration for Document understanding and
            enrichment.
        starting_schema (google.cloud.discoveryengine_v1alpha.types.Schema):
            The start schema to use for this
            [DataStore][google.cloud.discoveryengine.v1alpha.DataStore]
            when provisioning it. If unset, a default vertical
            specialized schema will be used.

            This field is only used by [CreateDataStore][] API, and will
            be ignored if used in other APIs. This field will be omitted
            from all API responses including [CreateDataStore][] API. To
            retrieve a schema of a
            [DataStore][google.cloud.discoveryengine.v1alpha.DataStore],
            use
            [SchemaService.GetSchema][google.cloud.discoveryengine.v1alpha.SchemaService.GetSchema]
            API instead.

            The provided schema will be validated against certain rules
            on schema. Learn more from `this
            doc <https://cloud.google.com/generative-ai-app-builder/docs/provide-schema>`__.
    """

    class ContentConfig(proto.Enum):
        r"""Content config of the data store.

        Values:
            CONTENT_CONFIG_UNSPECIFIED (0):
                Default value.
            NO_CONTENT (1):
                Only contains documents without any
                [Document.content][google.cloud.discoveryengine.v1alpha.Document.content].
            CONTENT_REQUIRED (2):
                Only contains documents with
                [Document.content][google.cloud.discoveryengine.v1alpha.Document.content].
            PUBLIC_WEBSITE (3):
                The data store is used for public website
                search.
            GOOGLE_WORKSPACE (4):
                The data store is used for workspace search. Details of
                workspace data store are specified in the
                [WorkspaceConfig][google.cloud.discoveryengine.v1alpha.WorkspaceConfig].
        """
        CONTENT_CONFIG_UNSPECIFIED = 0
        NO_CONTENT = 1
        CONTENT_REQUIRED = 2
        PUBLIC_WEBSITE = 3
        GOOGLE_WORKSPACE = 4

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    industry_vertical: common.IndustryVertical = proto.Field(
        proto.ENUM,
        number=3,
        enum=common.IndustryVertical,
    )
    solution_types: MutableSequence[common.SolutionType] = proto.RepeatedField(
        proto.ENUM,
        number=5,
        enum=common.SolutionType,
    )
    default_schema_id: str = proto.Field(
        proto.STRING,
        number=7,
    )
    content_config: ContentConfig = proto.Field(
        proto.ENUM,
        number=6,
        enum=ContentConfig,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    language_info: "LanguageInfo" = proto.Field(
        proto.MESSAGE,
        number=14,
        message="LanguageInfo",
    )
    idp_config: common.IdpConfig = proto.Field(
        proto.MESSAGE,
        number=21,
        message=common.IdpConfig,
    )
    acl_enabled: bool = proto.Field(
        proto.BOOL,
        number=24,
    )
    workspace_config: "WorkspaceConfig" = proto.Field(
        proto.MESSAGE,
        number=25,
        message="WorkspaceConfig",
    )
    document_processing_config: gcd_document_processing_config.DocumentProcessingConfig = proto.Field(
        proto.MESSAGE,
        number=27,
        message=gcd_document_processing_config.DocumentProcessingConfig,
    )
    starting_schema: schema.Schema = proto.Field(
        proto.MESSAGE,
        number=28,
        message=schema.Schema,
    )


class LanguageInfo(proto.Message):
    r"""Language info for DataStore.

    Attributes:
        language_code (str):
            The language code for the DataStore.
        normalized_language_code (str):
            Output only. This is the normalized form of language_code.
            E.g.: language_code of ``en-GB``, ``en_GB``, ``en-UK`` or
            ``en-gb`` will have normalized_language_code of ``en-GB``.
        language (str):
            Output only. Language part of normalized_language_code.
            E.g.: ``en-US`` -> ``en``, ``zh-Hans-HK`` -> ``zh``, ``en``
            -> ``en``.
        region (str):
            Output only. Region part of normalized_language_code, if
            present. E.g.: ``en-US`` -> ``US``, ``zh-Hans-HK`` ->
            ``HK``, ``en`` -> \`\`.
    """

    language_code: str = proto.Field(
        proto.STRING,
        number=1,
    )
    normalized_language_code: str = proto.Field(
        proto.STRING,
        number=2,
    )
    language: str = proto.Field(
        proto.STRING,
        number=3,
    )
    region: str = proto.Field(
        proto.STRING,
        number=4,
    )


class WorkspaceConfig(proto.Message):
    r"""Config to store data store type configuration for workspace
    data

    Attributes:
        type_ (google.cloud.discoveryengine_v1alpha.types.WorkspaceConfig.Type):
            The Google Workspace data source.
        dasher_customer_id (str):
            Obfuscated Dasher customer ID.
    """

    class Type(proto.Enum):
        r"""Specifies the type of Workspace App supported by this
        DataStore

        Values:
            TYPE_UNSPECIFIED (0):
                Defaults to an unspecified Workspace type.
            GOOGLE_DRIVE (1):
                Workspace Data Store contains Drive data
            GOOGLE_MAIL (2):
                Workspace Data Store contains Mail data
            GOOGLE_SITES (3):
                Workspace Data Store contains Sites data
            GOOGLE_CALENDAR (4):
                Workspace Data Store contains Calendar data
            GOOGLE_CHAT (5):
                Workspace Data Store contains Chat data
            GOOGLE_GROUPS (6):
                Workspace Data Store contains Groups data
            GOOGLE_KEEP (7):
                Workspace Data Store contains Keep data
        """
        TYPE_UNSPECIFIED = 0
        GOOGLE_DRIVE = 1
        GOOGLE_MAIL = 2
        GOOGLE_SITES = 3
        GOOGLE_CALENDAR = 4
        GOOGLE_CHAT = 5
        GOOGLE_GROUPS = 6
        GOOGLE_KEEP = 7

    type_: Type = proto.Field(
        proto.ENUM,
        number=1,
        enum=Type,
    )
    dasher_customer_id: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
