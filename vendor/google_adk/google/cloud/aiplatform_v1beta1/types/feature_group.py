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

from google.cloud.aiplatform_v1beta1.types import io
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "FeatureGroup",
    },
)


class FeatureGroup(proto.Message):
    r"""Vertex AI Feature Group.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        big_query (google.cloud.aiplatform_v1beta1.types.FeatureGroup.BigQuery):
            Indicates that features for this group come from BigQuery
            Table/View. By default treats the source as a sparse time
            series source. The BigQuery source table or view must have
            at least one entity ID column and a column named
            ``feature_timestamp``.

            This field is a member of `oneof`_ ``source``.
        name (str):
            Identifier. Name of the FeatureGroup. Format:
            ``projects/{project}/locations/{location}/featureGroups/{featureGroup}``
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this FeatureGroup
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this FeatureGroup
            was last updated.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize your FeatureGroup.

            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            on and examples of labels. No more than 64 user
            labels can be associated with one
            FeatureGroup(System labels are excluded)."
            System reserved label keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable.
        description (str):
            Optional. Description of the FeatureGroup.
        service_agent_type (google.cloud.aiplatform_v1beta1.types.FeatureGroup.ServiceAgentType):
            Optional. Service agent type used during jobs under a
            FeatureGroup. By default, the Vertex AI Service Agent is
            used. When using an IAM Policy to isolate this FeatureGroup
            within a project, a separate service account should be
            provisioned by setting this field to
            ``SERVICE_AGENT_TYPE_FEATURE_GROUP``. This will generate a
            separate service account to access the BigQuery source
            table.
        service_account_email (str):
            Output only. A Service Account unique to this
            FeatureGroup. The role bigquery.dataViewer
            should be granted to this service account to
            allow Vertex AI Feature Store to access source
            data while running jobs under this FeatureGroup.
    """

    class ServiceAgentType(proto.Enum):
        r"""Service agent type used during jobs under a FeatureGroup.

        Values:
            SERVICE_AGENT_TYPE_UNSPECIFIED (0):
                By default, the project-level Vertex AI
                Service Agent is enabled.
            SERVICE_AGENT_TYPE_PROJECT (1):
                Specifies the project-level Vertex AI Service
                Agent
                (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents).
            SERVICE_AGENT_TYPE_FEATURE_GROUP (2):
                Enable a FeatureGroup service account to be created by
                Vertex AI and output in the field ``service_account_email``.
                This service account will be used to read from the source
                BigQuery table during jobs under a FeatureGroup.
        """

        SERVICE_AGENT_TYPE_UNSPECIFIED = 0
        SERVICE_AGENT_TYPE_PROJECT = 1
        SERVICE_AGENT_TYPE_FEATURE_GROUP = 2

    class BigQuery(proto.Message):
        r"""Input source type for BigQuery Tables and Views.

        Attributes:
            big_query_source (google.cloud.aiplatform_v1beta1.types.BigQuerySource):
                Required. Immutable. The BigQuery source URI
                that points to either a BigQuery Table or View.
            entity_id_columns (MutableSequence[str]):
                Optional. Columns to construct entity_id / row keys. If not
                provided defaults to ``entity_id``.
            static_data_source (bool):
                Optional. Set if the data source is not a
                time-series.
            time_series (google.cloud.aiplatform_v1beta1.types.FeatureGroup.BigQuery.TimeSeries):
                Optional. If the source is a time-series source, this can be
                set to control how downstream sources (ex:
                [FeatureView][google.cloud.aiplatform.v1beta1.FeatureView] )
                will treat time-series sources. If not set, will treat the
                source as a time-series source with ``feature_timestamp`` as
                timestamp column and no scan boundary.
            dense (bool):
                Optional. If set, all feature values will be fetched from a
                single row per unique entityId including nulls. If not set,
                will collapse all rows for each unique entityId into a singe
                row with any non-null values if present, if no non-null
                values are present will sync null. ex: If source has schema
                ``(entity_id, feature_timestamp, f0, f1)`` and the following
                rows: ``(e1, 2020-01-01T10:00:00.123Z, 10, 15)``
                ``(e1, 2020-02-01T10:00:00.123Z, 20, null)`` If dense is
                set, ``(e1, 20, null)`` is synced to online stores. If dense
                is not set, ``(e1, 20, 15)`` is synced to online stores.
        """

        class TimeSeries(proto.Message):
            r"""

            Attributes:
                timestamp_column (str):
                    Optional. Column hosting timestamp values for a time-series
                    source. Will be used to determine the latest
                    ``feature_values`` for each entity. Optional. If not
                    provided, column named ``feature_timestamp`` of type
                    ``TIMESTAMP`` will be used.
            """

            timestamp_column: str = proto.Field(
                proto.STRING,
                number=1,
            )

        big_query_source: io.BigQuerySource = proto.Field(
            proto.MESSAGE,
            number=1,
            message=io.BigQuerySource,
        )
        entity_id_columns: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )
        static_data_source: bool = proto.Field(
            proto.BOOL,
            number=3,
        )
        time_series: "FeatureGroup.BigQuery.TimeSeries" = proto.Field(
            proto.MESSAGE,
            number=4,
            message="FeatureGroup.BigQuery.TimeSeries",
        )
        dense: bool = proto.Field(
            proto.BOOL,
            number=5,
        )

    big_query: BigQuery = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="source",
        message=BigQuery,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=4,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=5,
    )
    description: str = proto.Field(
        proto.STRING,
        number=6,
    )
    service_agent_type: ServiceAgentType = proto.Field(
        proto.ENUM,
        number=8,
        enum=ServiceAgentType,
    )
    service_account_email: str = proto.Field(
        proto.STRING,
        number=9,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
