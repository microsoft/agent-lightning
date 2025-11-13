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

from google.cloud.discoveryengine_v1alpha.types import import_config

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "EstimateDataSizeRequest",
        "EstimateDataSizeResponse",
        "EstimateDataSizeMetadata",
    },
)


class EstimateDataSizeRequest(proto.Message):
    r"""Request message for
    [EstimateBillingService.EstimateDataSize][google.cloud.discoveryengine.v1alpha.EstimateBillingService.EstimateDataSize]
    method

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        website_data_source (google.cloud.discoveryengine_v1alpha.types.EstimateDataSizeRequest.WebsiteDataSource):
            Website data.

            This field is a member of `oneof`_ ``data_source``.
        file_data_source (google.cloud.discoveryengine_v1alpha.types.EstimateDataSizeRequest.FileDataSource):
            Structured or unstructured data.

            This field is a member of `oneof`_ ``data_source``.
        location (str):
            Required. Full resource name of the location, such as
            ``projects/{project}/locations/{location}``.
    """

    class WebsiteDataSource(proto.Message):
        r"""Data source is a set of website patterns that we crawl to get
        the total number of websites.

        Attributes:
            estimator_uri_patterns (MutableSequence[google.cloud.discoveryengine_v1alpha.types.EstimateDataSizeRequest.WebsiteDataSource.EstimatorUriPattern]):
                Required. The URI patterns to estimate the data sizes. At
                most 10 patterns are allowed, otherwise an INVALID_ARGUMENT
                error is thrown.
        """

        class EstimatorUriPattern(proto.Message):
            r"""URI patterns that we use to crawl.

            Attributes:
                provided_uri_pattern (str):
                    User provided URI pattern. For example, ``foo.com/bar/*``.
                exact_match (bool):
                    Whether we infer the generated URI or use the
                    exact provided one.
                exclusive (bool):
                    Whether the pattern is exclusive or not. If
                    set to true, the pattern is considered
                    exclusive. If unset or set to false, the pattern
                    is considered inclusive by default.
            """

            provided_uri_pattern: str = proto.Field(
                proto.STRING,
                number=1,
            )
            exact_match: bool = proto.Field(
                proto.BOOL,
                number=2,
            )
            exclusive: bool = proto.Field(
                proto.BOOL,
                number=3,
            )

        estimator_uri_patterns: MutableSequence[
            "EstimateDataSizeRequest.WebsiteDataSource.EstimatorUriPattern"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="EstimateDataSizeRequest.WebsiteDataSource.EstimatorUriPattern",
        )

    class FileDataSource(proto.Message):
        r"""Data source contains files either in Cloud Storage or
        BigQuery.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            gcs_source (google.cloud.discoveryengine_v1alpha.types.GcsSource):
                Cloud Storage location for the input content.

                This field is a member of `oneof`_ ``source``.
            bigquery_source (google.cloud.discoveryengine_v1alpha.types.BigQuerySource):
                BigQuery input source.

                This field is a member of `oneof`_ ``source``.
        """

        gcs_source: import_config.GcsSource = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="source",
            message=import_config.GcsSource,
        )
        bigquery_source: import_config.BigQuerySource = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="source",
            message=import_config.BigQuerySource,
        )

    website_data_source: WebsiteDataSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="data_source",
        message=WebsiteDataSource,
    )
    file_data_source: FileDataSource = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="data_source",
        message=FileDataSource,
    )
    location: str = proto.Field(
        proto.STRING,
        number=1,
    )


class EstimateDataSizeResponse(proto.Message):
    r"""Response of the EstimateDataSize request. If the long running
    operation was successful, then this message is returned by the
    google.longrunning.Operations.response field if the operation
    was successful.

    Attributes:
        data_size_bytes (int):
            Data size in terms of bytes.
        document_count (int):
            Total number of documents.
    """

    data_size_bytes: int = proto.Field(
        proto.INT64,
        number=1,
    )
    document_count: int = proto.Field(
        proto.INT64,
        number=2,
    )


class EstimateDataSizeMetadata(proto.Message):
    r"""Metadata related to the progress of the EstimateDataSize
    operation. This is returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
