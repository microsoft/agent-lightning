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

from google.cloud.discoveryengine_v1.types import (
    document_processing_config as gcd_document_processing_config,
)
from google.cloud.discoveryengine_v1.types import cmek_config_service, common
from google.cloud.discoveryengine_v1.types import schema

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "DataStore",
        "AdvancedSiteSearchConfig",
        "WorkspaceConfig",
    },
)


class DataStore(proto.Message):
    r"""DataStore captures global settings and configs at the
    DataStore level.

    Attributes:
        name (str):
            Immutable. Identifier. The full resource name of the data
            store. Format:
            ``projects/{project}/locations/{location}/collections/{collection_id}/dataStores/{data_store_id}``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        display_name (str):
            Required. The data store display name.

            This field must be a UTF-8 encoded string with a length
            limit of 128 characters. Otherwise, an INVALID_ARGUMENT
            error is returned.
        industry_vertical (google.cloud.discoveryengine_v1.types.IndustryVertical):
            Immutable. The industry vertical that the
            data store registers.
        solution_types (MutableSequence[google.cloud.discoveryengine_v1.types.SolutionType]):
            The solutions that the data store enrolls. Available
            solutions for each
            [industry_vertical][google.cloud.discoveryengine.v1.DataStore.industry_vertical]:

            - ``MEDIA``: ``SOLUTION_TYPE_RECOMMENDATION`` and
              ``SOLUTION_TYPE_SEARCH``.
            - ``SITE_SEARCH``: ``SOLUTION_TYPE_SEARCH`` is automatically
              enrolled. Other solutions cannot be enrolled.
        default_schema_id (str):
            Output only. The id of the default
            [Schema][google.cloud.discoveryengine.v1.Schema] associated
            to this data store.
        content_config (google.cloud.discoveryengine_v1.types.DataStore.ContentConfig):
            Immutable. The content config of the data store. If this
            field is unset, the server behavior defaults to
            [ContentConfig.NO_CONTENT][google.cloud.discoveryengine.v1.DataStore.ContentConfig.NO_CONTENT].
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp the
            [DataStore][google.cloud.discoveryengine.v1.DataStore] was
            created at.
        advanced_site_search_config (google.cloud.discoveryengine_v1.types.AdvancedSiteSearchConfig):
            Optional. Configuration for advanced site
            search.
        kms_key_name (str):
            Input only. The KMS key to be used to protect this DataStore
            at creation time.

            Must be set for requests that need to comply with CMEK Org
            Policy protections.

            If this field is set and processed successfully, the
            DataStore will be protected by the KMS key, as indicated in
            the cmek_config field.
        cmek_config (google.cloud.discoveryengine_v1.types.CmekConfig):
            Output only. CMEK-related information for the
            DataStore.
        billing_estimation (google.cloud.discoveryengine_v1.types.DataStore.BillingEstimation):
            Output only. Data size estimation for
            billing.
        acl_enabled (bool):
            Immutable. Whether data in the
            [DataStore][google.cloud.discoveryengine.v1.DataStore] has
            ACL information. If set to ``true``, the source data must
            have ACL. ACL will be ingested when data is ingested by
            [DocumentService.ImportDocuments][google.cloud.discoveryengine.v1.DocumentService.ImportDocuments]
            methods.

            When ACL is enabled for the
            [DataStore][google.cloud.discoveryengine.v1.DataStore],
            [Document][google.cloud.discoveryengine.v1.Document] can't
            be accessed by calling
            [DocumentService.GetDocument][google.cloud.discoveryengine.v1.DocumentService.GetDocument]
            or
            [DocumentService.ListDocuments][google.cloud.discoveryengine.v1.DocumentService.ListDocuments].

            Currently ACL is only supported in ``GENERIC`` industry
            vertical with non-``PUBLIC_WEBSITE`` content config.
        workspace_config (google.cloud.discoveryengine_v1.types.WorkspaceConfig):
            Config to store data store type configuration for workspace
            data. This must be set when
            [DataStore.content_config][google.cloud.discoveryengine.v1.DataStore.content_config]
            is set as
            [DataStore.ContentConfig.GOOGLE_WORKSPACE][google.cloud.discoveryengine.v1.DataStore.ContentConfig.GOOGLE_WORKSPACE].
        document_processing_config (google.cloud.discoveryengine_v1.types.DocumentProcessingConfig):
            Configuration for Document understanding and
            enrichment.
        starting_schema (google.cloud.discoveryengine_v1.types.Schema):
            The start schema to use for this
            [DataStore][google.cloud.discoveryengine.v1.DataStore] when
            provisioning it. If unset, a default vertical specialized
            schema will be used.

            This field is only used by
            [CreateDataStore][google.cloud.discoveryengine.v1.DataStoreService.CreateDataStore]
            API, and will be ignored if used in other APIs. This field
            will be omitted from all API responses including
            [CreateDataStore][google.cloud.discoveryengine.v1.DataStoreService.CreateDataStore]
            API. To retrieve a schema of a
            [DataStore][google.cloud.discoveryengine.v1.DataStore], use
            [SchemaService.GetSchema][google.cloud.discoveryengine.v1.SchemaService.GetSchema]
            API instead.

            The provided schema will be validated against certain rules
            on schema. Learn more from `this
            doc <https://cloud.google.com/generative-ai-app-builder/docs/provide-schema>`__.
        healthcare_fhir_config (google.cloud.discoveryengine_v1.types.HealthcareFhirConfig):
            Optional. Configuration for ``HEALTHCARE_FHIR`` vertical.
        identity_mapping_store (str):
            Immutable. The fully qualified resource name of the
            associated
            [IdentityMappingStore][google.cloud.discoveryengine.v1.IdentityMappingStore].
            This field can only be set for acl_enabled DataStores with
            ``THIRD_PARTY`` or ``GSUITE`` IdP. Format:
            ``projects/{project}/locations/{location}/identityMappingStores/{identity_mapping_store}``.
    """

    class ContentConfig(proto.Enum):
        r"""Content config of the data store.

        Values:
            CONTENT_CONFIG_UNSPECIFIED (0):
                Default value.
            NO_CONTENT (1):
                Only contains documents without any
                [Document.content][google.cloud.discoveryengine.v1.Document.content].
            CONTENT_REQUIRED (2):
                Only contains documents with
                [Document.content][google.cloud.discoveryengine.v1.Document.content].
            PUBLIC_WEBSITE (3):
                The data store is used for public website
                search.
            GOOGLE_WORKSPACE (4):
                The data store is used for workspace search. Details of
                workspace data store are specified in the
                [WorkspaceConfig][google.cloud.discoveryengine.v1.WorkspaceConfig].
        """
        CONTENT_CONFIG_UNSPECIFIED = 0
        NO_CONTENT = 1
        CONTENT_REQUIRED = 2
        PUBLIC_WEBSITE = 3
        GOOGLE_WORKSPACE = 4

    class BillingEstimation(proto.Message):
        r"""Estimation of data size per data store.

        Attributes:
            structured_data_size (int):
                Data size for structured data in terms of
                bytes.
            unstructured_data_size (int):
                Data size for unstructured data in terms of
                bytes.
            website_data_size (int):
                Data size for websites in terms of bytes.
            structured_data_update_time (google.protobuf.timestamp_pb2.Timestamp):
                Last updated timestamp for structured data.
            unstructured_data_update_time (google.protobuf.timestamp_pb2.Timestamp):
                Last updated timestamp for unstructured data.
            website_data_update_time (google.protobuf.timestamp_pb2.Timestamp):
                Last updated timestamp for websites.
        """

        structured_data_size: int = proto.Field(
            proto.INT64,
            number=1,
        )
        unstructured_data_size: int = proto.Field(
            proto.INT64,
            number=2,
        )
        website_data_size: int = proto.Field(
            proto.INT64,
            number=3,
        )
        structured_data_update_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=4,
            message=timestamp_pb2.Timestamp,
        )
        unstructured_data_update_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=5,
            message=timestamp_pb2.Timestamp,
        )
        website_data_update_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=6,
            message=timestamp_pb2.Timestamp,
        )

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
    advanced_site_search_config: "AdvancedSiteSearchConfig" = proto.Field(
        proto.MESSAGE,
        number=12,
        message="AdvancedSiteSearchConfig",
    )
    kms_key_name: str = proto.Field(
        proto.STRING,
        number=32,
    )
    cmek_config: cmek_config_service.CmekConfig = proto.Field(
        proto.MESSAGE,
        number=18,
        message=cmek_config_service.CmekConfig,
    )
    billing_estimation: BillingEstimation = proto.Field(
        proto.MESSAGE,
        number=23,
        message=BillingEstimation,
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
    healthcare_fhir_config: common.HealthcareFhirConfig = proto.Field(
        proto.MESSAGE,
        number=29,
        message=common.HealthcareFhirConfig,
    )
    identity_mapping_store: str = proto.Field(
        proto.STRING,
        number=31,
    )


class AdvancedSiteSearchConfig(proto.Message):
    r"""Configuration data for advance site search.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        disable_initial_index (bool):
            If set true, initial indexing is disabled for
            the DataStore.

            This field is a member of `oneof`_ ``_disable_initial_index``.
        disable_automatic_refresh (bool):
            If set true, automatic refresh is disabled
            for the DataStore.

            This field is a member of `oneof`_ ``_disable_automatic_refresh``.
    """

    disable_initial_index: bool = proto.Field(
        proto.BOOL,
        number=3,
        optional=True,
    )
    disable_automatic_refresh: bool = proto.Field(
        proto.BOOL,
        number=4,
        optional=True,
    )


class WorkspaceConfig(proto.Message):
    r"""Config to store data store type configuration for workspace
    data

    Attributes:
        type_ (google.cloud.discoveryengine_v1.types.WorkspaceConfig.Type):
            The Google Workspace data source.
        dasher_customer_id (str):
            Obfuscated Dasher customer ID.
        super_admin_service_account (str):
            Optional. The super admin service account for
            the workspace that will be used for access token
            generation. For now we only use it for Native
            Google Drive connector data ingestion.
        super_admin_email_address (str):
            Optional. The super admin email address for
            the workspace that will be used for access token
            generation. For now we only use it for Native
            Google Drive connector data ingestion.
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
            GOOGLE_PEOPLE (8):
                Workspace Data Store contains People data
        """
        TYPE_UNSPECIFIED = 0
        GOOGLE_DRIVE = 1
        GOOGLE_MAIL = 2
        GOOGLE_SITES = 3
        GOOGLE_CALENDAR = 4
        GOOGLE_CHAT = 5
        GOOGLE_GROUPS = 6
        GOOGLE_KEEP = 7
        GOOGLE_PEOPLE = 8

    type_: Type = proto.Field(
        proto.ENUM,
        number=1,
        enum=Type,
    )
    dasher_customer_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    super_admin_service_account: str = proto.Field(
        proto.STRING,
        number=4,
    )
    super_admin_email_address: str = proto.Field(
        proto.STRING,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
