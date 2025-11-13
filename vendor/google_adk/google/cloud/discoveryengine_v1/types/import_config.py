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

from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1.types import completion, document, user_event

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1",
    manifest={
        "GcsSource",
        "BigQuerySource",
        "SpannerSource",
        "BigtableOptions",
        "BigtableSource",
        "FhirStoreSource",
        "CloudSqlSource",
        "AlloyDbSource",
        "FirestoreSource",
        "ImportErrorConfig",
        "ImportUserEventsRequest",
        "ImportUserEventsResponse",
        "ImportUserEventsMetadata",
        "ImportDocumentsMetadata",
        "ImportDocumentsRequest",
        "ImportDocumentsResponse",
        "ImportSuggestionDenyListEntriesRequest",
        "ImportSuggestionDenyListEntriesResponse",
        "ImportSuggestionDenyListEntriesMetadata",
        "ImportCompletionSuggestionsRequest",
        "ImportCompletionSuggestionsResponse",
        "ImportCompletionSuggestionsMetadata",
    },
)


class GcsSource(proto.Message):
    r"""Cloud Storage location for input content.

    Attributes:
        input_uris (MutableSequence[str]):
            Required. Cloud Storage URIs to input files. Each URI can be
            up to 2000 characters long. URIs can match the full object
            path (for example, ``gs://bucket/directory/object.json``) or
            a pattern matching one or more files, such as
            ``gs://bucket/directory/*.json``.

            A request can contain at most 100 files (or 100,000 files if
            ``data_schema`` is ``content``). Each file can be up to 2 GB
            (or 100 MB if ``data_schema`` is ``content``).
        data_schema (str):
            The schema to use when parsing the data from the source.

            Supported values for document imports:

            - ``document`` (default): One JSON
              [Document][google.cloud.discoveryengine.v1.Document] per
              line. Each document must have a valid
              [Document.id][google.cloud.discoveryengine.v1.Document.id].
            - ``content``: Unstructured data (e.g. PDF, HTML). Each file
              matched by ``input_uris`` becomes a document, with the ID
              set to the first 128 bits of SHA256(URI) encoded as a hex
              string.
            - ``custom``: One custom data JSON per row in arbitrary
              format that conforms to the defined
              [Schema][google.cloud.discoveryengine.v1.Schema] of the
              data store. This can only be used by the GENERIC Data
              Store vertical.
            - ``csv``: A CSV file with header conforming to the defined
              [Schema][google.cloud.discoveryengine.v1.Schema] of the
              data store. Each entry after the header is imported as a
              Document. This can only be used by the GENERIC Data Store
              vertical.

            Supported values for user event imports:

            - ``user_event`` (default): One JSON
              [UserEvent][google.cloud.discoveryengine.v1.UserEvent] per
              line.
    """

    input_uris: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )
    data_schema: str = proto.Field(
        proto.STRING,
        number=2,
    )


class BigQuerySource(proto.Message):
    r"""BigQuery source import data from.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        partition_date (google.type.date_pb2.Date):
            BigQuery time partitioned table's \_PARTITIONDATE in
            YYYY-MM-DD format.

            This field is a member of `oneof`_ ``partition``.
        project_id (str):
            The project ID or the project number that
            contains the BigQuery source. Has a length limit
            of 128 characters. If not specified, inherits
            the project ID from the parent request.
        dataset_id (str):
            Required. The BigQuery data set to copy the
            data from with a length limit of 1,024
            characters.
        table_id (str):
            Required. The BigQuery table to copy the data
            from with a length limit of 1,024 characters.
        gcs_staging_dir (str):
            Intermediate Cloud Storage directory used for
            the import with a length limit of 2,000
            characters. Can be specified if one wants to
            have the BigQuery export to a specific Cloud
            Storage directory.
        data_schema (str):
            The schema to use when parsing the data from the source.

            Supported values for user event imports:

            - ``user_event`` (default): One
              [UserEvent][google.cloud.discoveryengine.v1.UserEvent] per
              row.

            Supported values for document imports:

            - ``document`` (default): One
              [Document][google.cloud.discoveryengine.v1.Document]
              format per row. Each document must have a valid
              [Document.id][google.cloud.discoveryengine.v1.Document.id]
              and one of
              [Document.json_data][google.cloud.discoveryengine.v1.Document.json_data]
              or
              [Document.struct_data][google.cloud.discoveryengine.v1.Document.struct_data].
            - ``custom``: One custom data per row in arbitrary format
              that conforms to the defined
              [Schema][google.cloud.discoveryengine.v1.Schema] of the
              data store. This can only be used by the GENERIC Data
              Store vertical.
    """

    partition_date: date_pb2.Date = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="partition",
        message=date_pb2.Date,
    )
    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    dataset_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    table_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    gcs_staging_dir: str = proto.Field(
        proto.STRING,
        number=4,
    )
    data_schema: str = proto.Field(
        proto.STRING,
        number=6,
    )


class SpannerSource(proto.Message):
    r"""The Spanner source for importing data

    Attributes:
        project_id (str):
            The project ID that contains the Spanner
            source. Has a length limit of 128 characters. If
            not specified, inherits the project ID from the
            parent request.
        instance_id (str):
            Required. The instance ID of the source
            Spanner table.
        database_id (str):
            Required. The database ID of the source
            Spanner table.
        table_id (str):
            Required. The table name of the Spanner
            database that needs to be imported.
        enable_data_boost (bool):
            Whether to apply data boost on Spanner export. Enabling this
            option will incur additional cost. More info can be found
            `here <https://cloud.google.com/spanner/docs/databoost/databoost-overview#billing_and_quotas>`__.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    instance_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    database_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    table_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    enable_data_boost: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


class BigtableOptions(proto.Message):
    r"""The Bigtable Options object that contains information to
    support the import.

    Attributes:
        key_field_name (str):
            The field name used for saving row key value in the
            document. The name has to match the pattern
            ``[a-zA-Z0-9][a-zA-Z0-9-_]*``.
        families (MutableMapping[str, google.cloud.discoveryengine_v1.types.BigtableOptions.BigtableColumnFamily]):
            The mapping from family names to an object
            that contains column families level information
            for the given column family. If a family is not
            present in this map it will be ignored.
    """

    class Type(proto.Enum):
        r"""The type of values in a Bigtable column or column family. The values
        are expected to be encoded using `HBase
        Bytes.toBytes <https://hbase.apache.org/1.4/apidocs/org/apache/hadoop/hbase/util/Bytes.html>`__
        function when the encoding value is set to ``BINARY``.

        Values:
            TYPE_UNSPECIFIED (0):
                The type is unspecified.
            STRING (1):
                String type.
            NUMBER (2):
                Numerical type.
            INTEGER (3):
                Integer type.
            VAR_INTEGER (4):
                Variable length integer type.
            BIG_NUMERIC (5):
                BigDecimal type.
            BOOLEAN (6):
                Boolean type.
            JSON (7):
                JSON type.
        """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        NUMBER = 2
        INTEGER = 3
        VAR_INTEGER = 4
        BIG_NUMERIC = 5
        BOOLEAN = 6
        JSON = 7

    class Encoding(proto.Enum):
        r"""The encoding mode of a Bigtable column or column family.

        Values:
            ENCODING_UNSPECIFIED (0):
                The encoding is unspecified.
            TEXT (1):
                Text encoding.
            BINARY (2):
                Binary encoding.
        """
        ENCODING_UNSPECIFIED = 0
        TEXT = 1
        BINARY = 2

    class BigtableColumnFamily(proto.Message):
        r"""The column family of the Bigtable.

        Attributes:
            field_name (str):
                The field name to use for this column family in the
                document. The name has to match the pattern
                ``[a-zA-Z0-9][a-zA-Z0-9-_]*``. If not set, it is parsed from
                the family name with best effort. However, due to different
                naming patterns, field name collisions could happen, where
                parsing behavior is undefined.
            encoding (google.cloud.discoveryengine_v1.types.BigtableOptions.Encoding):
                The encoding mode of the values when the type is not STRING.
                Acceptable encoding values are:

                - ``TEXT``: indicates values are alphanumeric text strings.
                - ``BINARY``: indicates values are encoded using
                  ``HBase Bytes.toBytes`` family of functions. This can be
                  overridden for a specific column by listing that column in
                  ``columns`` and specifying an encoding for it.
            type_ (google.cloud.discoveryengine_v1.types.BigtableOptions.Type):
                The type of values in this column family. The values are
                expected to be encoded using ``HBase Bytes.toBytes``
                function when the encoding value is set to ``BINARY``.
            columns (MutableSequence[google.cloud.discoveryengine_v1.types.BigtableOptions.BigtableColumn]):
                The list of objects that contains column
                level information for each column. If a column
                is not present in this list it will be ignored.
        """

        field_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        encoding: "BigtableOptions.Encoding" = proto.Field(
            proto.ENUM,
            number=2,
            enum="BigtableOptions.Encoding",
        )
        type_: "BigtableOptions.Type" = proto.Field(
            proto.ENUM,
            number=3,
            enum="BigtableOptions.Type",
        )
        columns: MutableSequence[
            "BigtableOptions.BigtableColumn"
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=4,
            message="BigtableOptions.BigtableColumn",
        )

    class BigtableColumn(proto.Message):
        r"""The column of the Bigtable.

        Attributes:
            qualifier (bytes):
                Required. Qualifier of the column. If it
                cannot be decoded with utf-8, use a base-64
                encoded string instead.
            field_name (str):
                The field name to use for this column in the document. The
                name has to match the pattern ``[a-zA-Z0-9][a-zA-Z0-9-_]*``.
                If not set, it is parsed from the qualifier bytes with best
                effort. However, due to different naming patterns, field
                name collisions could happen, where parsing behavior is
                undefined.
            encoding (google.cloud.discoveryengine_v1.types.BigtableOptions.Encoding):
                The encoding mode of the values when the type is not
                ``STRING``. Acceptable encoding values are:

                - ``TEXT``: indicates values are alphanumeric text strings.
                - ``BINARY``: indicates values are encoded using
                  ``HBase Bytes.toBytes`` family of functions. This can be
                  overridden for a specific column by listing that column in
                  ``columns`` and specifying an encoding for it.
            type_ (google.cloud.discoveryengine_v1.types.BigtableOptions.Type):
                The type of values in this column family. The values are
                expected to be encoded using ``HBase Bytes.toBytes``
                function when the encoding value is set to ``BINARY``.
        """

        qualifier: bytes = proto.Field(
            proto.BYTES,
            number=1,
        )
        field_name: str = proto.Field(
            proto.STRING,
            number=2,
        )
        encoding: "BigtableOptions.Encoding" = proto.Field(
            proto.ENUM,
            number=3,
            enum="BigtableOptions.Encoding",
        )
        type_: "BigtableOptions.Type" = proto.Field(
            proto.ENUM,
            number=4,
            enum="BigtableOptions.Type",
        )

    key_field_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    families: MutableMapping[str, BigtableColumnFamily] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=2,
        message=BigtableColumnFamily,
    )


class BigtableSource(proto.Message):
    r"""The Cloud Bigtable source for importing data.

    Attributes:
        project_id (str):
            The project ID that contains the Bigtable
            source. Has a length limit of 128 characters. If
            not specified, inherits the project ID from the
            parent request.
        instance_id (str):
            Required. The instance ID of the Cloud
            Bigtable that needs to be imported.
        table_id (str):
            Required. The table ID of the Cloud Bigtable
            that needs to be imported.
        bigtable_options (google.cloud.discoveryengine_v1.types.BigtableOptions):
            Required. Bigtable options that contains
            information needed when parsing data into typed
            structures. For example, column type
            annotations.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    instance_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    table_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    bigtable_options: "BigtableOptions" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="BigtableOptions",
    )


class FhirStoreSource(proto.Message):
    r"""Cloud FhirStore source import data from.

    Attributes:
        fhir_store (str):
            Required. The full resource name of the FHIR store to import
            data from, in the format of
            ``projects/{project}/locations/{location}/datasets/{dataset}/fhirStores/{fhir_store}``.
        gcs_staging_dir (str):
            Intermediate Cloud Storage directory used for
            the import with a length limit of 2,000
            characters. Can be specified if one wants to
            have the FhirStore export to a specific Cloud
            Storage directory.
        resource_types (MutableSequence[str]):
            The FHIR resource types to import. The resource types should
            be a subset of all `supported FHIR resource
            types <https://cloud.google.com/generative-ai-app-builder/docs/fhir-schema-reference#resource-level-specification>`__.
            Default to all supported FHIR resource types if empty.
        update_from_latest_predefined_schema (bool):
            Optional. Whether to update the DataStore schema to the
            latest predefined schema.

            If true, the DataStore schema will be updated to include any
            FHIR fields or resource types that have been added since the
            last import and corresponding FHIR resources will be
            imported from the FHIR store.

            Note this field cannot be used in conjunction with
            ``resource_types``. It should be used after initial import.
    """

    fhir_store: str = proto.Field(
        proto.STRING,
        number=1,
    )
    gcs_staging_dir: str = proto.Field(
        proto.STRING,
        number=2,
    )
    resource_types: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )
    update_from_latest_predefined_schema: bool = proto.Field(
        proto.BOOL,
        number=4,
    )


class CloudSqlSource(proto.Message):
    r"""Cloud SQL source import data from.

    Attributes:
        project_id (str):
            The project ID that contains the Cloud SQL
            source. Has a length limit of 128 characters. If
            not specified, inherits the project ID from the
            parent request.
        instance_id (str):
            Required. The Cloud SQL instance to copy the
            data from with a length limit of 256 characters.
        database_id (str):
            Required. The Cloud SQL database to copy the
            data from with a length limit of 256 characters.
        table_id (str):
            Required. The Cloud SQL table to copy the
            data from with a length limit of 256 characters.
        gcs_staging_dir (str):
            Intermediate Cloud Storage directory used for
            the import with a length limit of 2,000
            characters. Can be specified if one wants to
            have the Cloud SQL export to a specific Cloud
            Storage directory.

            Ensure that the Cloud SQL service account has
            the necessary Cloud Storage Admin permissions to
            access the specified Cloud Storage directory.
        offload (bool):
            Option for serverless export. Enabling this option will
            incur additional cost. More info can be found
            `here <https://cloud.google.com/sql/pricing#serverless>`__.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    instance_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    database_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    table_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    gcs_staging_dir: str = proto.Field(
        proto.STRING,
        number=5,
    )
    offload: bool = proto.Field(
        proto.BOOL,
        number=6,
    )


class AlloyDbSource(proto.Message):
    r"""AlloyDB source import data from.

    Attributes:
        project_id (str):
            The project ID that contains the AlloyDB
            source. Has a length limit of 128 characters. If
            not specified, inherits the project ID from the
            parent request.
        location_id (str):
            Required. The AlloyDB location to copy the
            data from with a length limit of 256 characters.
        cluster_id (str):
            Required. The AlloyDB cluster to copy the
            data from with a length limit of 256 characters.
        database_id (str):
            Required. The AlloyDB database to copy the
            data from with a length limit of 256 characters.
        table_id (str):
            Required. The AlloyDB table to copy the data
            from with a length limit of 256 characters.
        gcs_staging_dir (str):
            Intermediate Cloud Storage directory used for
            the import with a length limit of 2,000
            characters. Can be specified if one wants to
            have the AlloyDB export to a specific Cloud
            Storage directory.

            Ensure that the AlloyDB service account has the
            necessary Cloud Storage Admin permissions to
            access the specified Cloud Storage directory.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    location_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    cluster_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    database_id: str = proto.Field(
        proto.STRING,
        number=4,
    )
    table_id: str = proto.Field(
        proto.STRING,
        number=5,
    )
    gcs_staging_dir: str = proto.Field(
        proto.STRING,
        number=6,
    )


class FirestoreSource(proto.Message):
    r"""Firestore source import data from.

    Attributes:
        project_id (str):
            The project ID that the Cloud SQL source is
            in with a length limit of 128 characters. If not
            specified, inherits the project ID from the
            parent request.
        database_id (str):
            Required. The Firestore database to copy the
            data from with a length limit of 256 characters.
        collection_id (str):
            Required. The Firestore collection (or
            entity) to copy the data from with a length
            limit of 1,500 characters.
        gcs_staging_dir (str):
            Intermediate Cloud Storage directory used for
            the import with a length limit of 2,000
            characters. Can be specified if one wants to
            have the Firestore export to a specific Cloud
            Storage directory.

            Ensure that the Firestore service account has
            the necessary Cloud Storage Admin permissions to
            access the specified Cloud Storage directory.
    """

    project_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    database_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    collection_id: str = proto.Field(
        proto.STRING,
        number=3,
    )
    gcs_staging_dir: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ImportErrorConfig(proto.Message):
    r"""Configuration of destination for Import related errors.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_prefix (str):
            Cloud Storage prefix for import errors. This must be an
            empty, existing Cloud Storage directory. Import errors are
            written to sharded files in this directory, one per line, as
            a JSON-encoded ``google.rpc.Status`` message.

            This field is a member of `oneof`_ ``destination``.
    """

    gcs_prefix: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="destination",
    )


class ImportUserEventsRequest(proto.Message):
    r"""Request message for the ImportUserEvents request.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        inline_source (google.cloud.discoveryengine_v1.types.ImportUserEventsRequest.InlineSource):
            The Inline source for the input content for
            UserEvents.

            This field is a member of `oneof`_ ``source``.
        gcs_source (google.cloud.discoveryengine_v1.types.GcsSource):
            Cloud Storage location for the input content.

            This field is a member of `oneof`_ ``source``.
        bigquery_source (google.cloud.discoveryengine_v1.types.BigQuerySource):
            BigQuery input source.

            This field is a member of `oneof`_ ``source``.
        parent (str):
            Required. Parent DataStore resource name, of the form
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}``
        error_config (google.cloud.discoveryengine_v1.types.ImportErrorConfig):
            The desired location of errors incurred
            during the Import. Cannot be set for inline user
            event imports.
    """

    class InlineSource(proto.Message):
        r"""The inline source for the input config for ImportUserEvents
        method.

        Attributes:
            user_events (MutableSequence[google.cloud.discoveryengine_v1.types.UserEvent]):
                Required. A list of user events to import.
                Recommended max of 10k items.
        """

        user_events: MutableSequence[user_event.UserEvent] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=user_event.UserEvent,
        )

    inline_source: InlineSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=InlineSource,
    )
    gcs_source: "GcsSource" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="source",
        message="GcsSource",
    )
    bigquery_source: "BigQuerySource" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="source",
        message="BigQuerySource",
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    error_config: "ImportErrorConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="ImportErrorConfig",
    )


class ImportUserEventsResponse(proto.Message):
    r"""Response of the ImportUserEventsRequest. If the long running
    operation was successful, then this message is returned by the
    google.longrunning.Operations.response field if the operation
    was successful.

    Attributes:
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the request.
        error_config (google.cloud.discoveryengine_v1.types.ImportErrorConfig):
            Echoes the destination for the complete
            errors if this field was set in the request.
        joined_events_count (int):
            Count of user events imported with complete
            existing Documents.
        unjoined_events_count (int):
            Count of user events imported, but with
            Document information not found in the existing
            Branch.
    """

    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )
    error_config: "ImportErrorConfig" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ImportErrorConfig",
    )
    joined_events_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    unjoined_events_count: int = proto.Field(
        proto.INT64,
        number=4,
    )


class ImportUserEventsMetadata(proto.Message):
    r"""Metadata related to the progress of the Import operation.
    This is returned by the google.longrunning.Operation.metadata
    field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
        success_count (int):
            Count of entries that were processed
            successfully.
        failure_count (int):
            Count of entries that encountered errors
            while processing.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    success_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    failure_count: int = proto.Field(
        proto.INT64,
        number=4,
    )


class ImportDocumentsMetadata(proto.Message):
    r"""Metadata related to the progress of the ImportDocuments
    operation. This is returned by the
    google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
        success_count (int):
            Count of entries that were processed
            successfully.
        failure_count (int):
            Count of entries that encountered errors
            while processing.
        total_count (int):
            Total count of entries that were processed.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    success_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    failure_count: int = proto.Field(
        proto.INT64,
        number=4,
    )
    total_count: int = proto.Field(
        proto.INT64,
        number=5,
    )


class ImportDocumentsRequest(proto.Message):
    r"""Request message for Import methods.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        inline_source (google.cloud.discoveryengine_v1.types.ImportDocumentsRequest.InlineSource):
            The Inline source for the input content for
            documents.

            This field is a member of `oneof`_ ``source``.
        gcs_source (google.cloud.discoveryengine_v1.types.GcsSource):
            Cloud Storage location for the input content.

            This field is a member of `oneof`_ ``source``.
        bigquery_source (google.cloud.discoveryengine_v1.types.BigQuerySource):
            BigQuery input source.

            This field is a member of `oneof`_ ``source``.
        fhir_store_source (google.cloud.discoveryengine_v1.types.FhirStoreSource):
            FhirStore input source.

            This field is a member of `oneof`_ ``source``.
        spanner_source (google.cloud.discoveryengine_v1.types.SpannerSource):
            Spanner input source.

            This field is a member of `oneof`_ ``source``.
        cloud_sql_source (google.cloud.discoveryengine_v1.types.CloudSqlSource):
            Cloud SQL input source.

            This field is a member of `oneof`_ ``source``.
        firestore_source (google.cloud.discoveryengine_v1.types.FirestoreSource):
            Firestore input source.

            This field is a member of `oneof`_ ``source``.
        alloy_db_source (google.cloud.discoveryengine_v1.types.AlloyDbSource):
            AlloyDB input source.

            This field is a member of `oneof`_ ``source``.
        bigtable_source (google.cloud.discoveryengine_v1.types.BigtableSource):
            Cloud Bigtable input source.

            This field is a member of `oneof`_ ``source``.
        parent (str):
            Required. The parent branch resource name, such as
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/branches/{branch}``.
            Requires create/update permission.
        error_config (google.cloud.discoveryengine_v1.types.ImportErrorConfig):
            The desired location of errors incurred
            during the Import.
        reconciliation_mode (google.cloud.discoveryengine_v1.types.ImportDocumentsRequest.ReconciliationMode):
            The mode of reconciliation between existing documents and
            the documents to be imported. Defaults to
            [ReconciliationMode.INCREMENTAL][google.cloud.discoveryengine.v1.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL].
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Indicates which fields in the provided
            imported documents to update. If not set, the
            default is to update all fields.
        auto_generate_ids (bool):
            Whether to automatically generate IDs for the documents if
            absent.

            If set to ``true``,
            [Document.id][google.cloud.discoveryengine.v1.Document.id]s
            are automatically generated based on the hash of the
            payload, where IDs may not be consistent during multiple
            imports. In which case
            [ReconciliationMode.FULL][google.cloud.discoveryengine.v1.ImportDocumentsRequest.ReconciliationMode.FULL]
            is highly recommended to avoid duplicate contents. If unset
            or set to ``false``,
            [Document.id][google.cloud.discoveryengine.v1.Document.id]s
            have to be specified using
            [id_field][google.cloud.discoveryengine.v1.ImportDocumentsRequest.id_field],
            otherwise, documents without IDs fail to be imported.

            Supported data sources:

            - [GcsSource][google.cloud.discoveryengine.v1.GcsSource].
              [GcsSource.data_schema][google.cloud.discoveryengine.v1.GcsSource.data_schema]
              must be ``custom`` or ``csv``. Otherwise, an
              INVALID_ARGUMENT error is thrown.
            - [BigQuerySource][google.cloud.discoveryengine.v1.BigQuerySource].
              [BigQuerySource.data_schema][google.cloud.discoveryengine.v1.BigQuerySource.data_schema]
              must be ``custom`` or ``csv``. Otherwise, an
              INVALID_ARGUMENT error is thrown.
            - [SpannerSource][google.cloud.discoveryengine.v1.SpannerSource].
            - [CloudSqlSource][google.cloud.discoveryengine.v1.CloudSqlSource].
            - [FirestoreSource][google.cloud.discoveryengine.v1.FirestoreSource].
            - [BigtableSource][google.cloud.discoveryengine.v1.BigtableSource].
        id_field (str):
            The field indicates the ID field or column to be used as
            unique IDs of the documents.

            For [GcsSource][google.cloud.discoveryengine.v1.GcsSource]
            it is the key of the JSON field. For instance, ``my_id`` for
            JSON ``{"my_id": "some_uuid"}``. For others, it may be the
            column name of the table where the unique ids are stored.

            The values of the JSON field or the table column are used as
            the
            [Document.id][google.cloud.discoveryengine.v1.Document.id]s.
            The JSON field or the table column must be of string type,
            and the values must be set as valid strings conform to
            `RFC-1034 <https://tools.ietf.org/html/rfc1034>`__ with 1-63
            characters. Otherwise, documents without valid IDs fail to
            be imported.

            Only set this field when
            [auto_generate_ids][google.cloud.discoveryengine.v1.ImportDocumentsRequest.auto_generate_ids]
            is unset or set as ``false``. Otherwise, an INVALID_ARGUMENT
            error is thrown.

            If it is unset, a default value ``_id`` is used when
            importing from the allowed data sources.

            Supported data sources:

            - [GcsSource][google.cloud.discoveryengine.v1.GcsSource].
              [GcsSource.data_schema][google.cloud.discoveryengine.v1.GcsSource.data_schema]
              must be ``custom`` or ``csv``. Otherwise, an
              INVALID_ARGUMENT error is thrown.
            - [BigQuerySource][google.cloud.discoveryengine.v1.BigQuerySource].
              [BigQuerySource.data_schema][google.cloud.discoveryengine.v1.BigQuerySource.data_schema]
              must be ``custom`` or ``csv``. Otherwise, an
              INVALID_ARGUMENT error is thrown.
            - [SpannerSource][google.cloud.discoveryengine.v1.SpannerSource].
            - [CloudSqlSource][google.cloud.discoveryengine.v1.CloudSqlSource].
            - [FirestoreSource][google.cloud.discoveryengine.v1.FirestoreSource].
            - [BigtableSource][google.cloud.discoveryengine.v1.BigtableSource].
        force_refresh_content (bool):
            Optional. Whether to force refresh the unstructured content
            of the documents.

            If set to ``true``, the content part of the documents will
            be refreshed regardless of the update status of the
            referencing content.
    """

    class ReconciliationMode(proto.Enum):
        r"""Indicates how imported documents are reconciled with the
        existing documents created or imported before.

        Values:
            RECONCILIATION_MODE_UNSPECIFIED (0):
                Defaults to ``INCREMENTAL``.
            INCREMENTAL (1):
                Inserts new documents or updates existing
                documents.
            FULL (2):
                Calculates diff and replaces the entire
                document dataset. Existing documents may be
                deleted if they are not present in the source
                location.
        """
        RECONCILIATION_MODE_UNSPECIFIED = 0
        INCREMENTAL = 1
        FULL = 2

    class InlineSource(proto.Message):
        r"""The inline source for the input config for ImportDocuments
        method.

        Attributes:
            documents (MutableSequence[google.cloud.discoveryengine_v1.types.Document]):
                Required. A list of documents to update/create. Each
                document must have a valid
                [Document.id][google.cloud.discoveryengine.v1.Document.id].
                Recommended max of 100 items.
        """

        documents: MutableSequence[document.Document] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=document.Document,
        )

    inline_source: InlineSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=InlineSource,
    )
    gcs_source: "GcsSource" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="source",
        message="GcsSource",
    )
    bigquery_source: "BigQuerySource" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="source",
        message="BigQuerySource",
    )
    fhir_store_source: "FhirStoreSource" = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="source",
        message="FhirStoreSource",
    )
    spanner_source: "SpannerSource" = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="source",
        message="SpannerSource",
    )
    cloud_sql_source: "CloudSqlSource" = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="source",
        message="CloudSqlSource",
    )
    firestore_source: "FirestoreSource" = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="source",
        message="FirestoreSource",
    )
    alloy_db_source: "AlloyDbSource" = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="source",
        message="AlloyDbSource",
    )
    bigtable_source: "BigtableSource" = proto.Field(
        proto.MESSAGE,
        number=15,
        oneof="source",
        message="BigtableSource",
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    error_config: "ImportErrorConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="ImportErrorConfig",
    )
    reconciliation_mode: ReconciliationMode = proto.Field(
        proto.ENUM,
        number=6,
        enum=ReconciliationMode,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=7,
        message=field_mask_pb2.FieldMask,
    )
    auto_generate_ids: bool = proto.Field(
        proto.BOOL,
        number=8,
    )
    id_field: str = proto.Field(
        proto.STRING,
        number=9,
    )
    force_refresh_content: bool = proto.Field(
        proto.BOOL,
        number=16,
    )


class ImportDocumentsResponse(proto.Message):
    r"""Response of the
    [ImportDocumentsRequest][google.cloud.discoveryengine.v1.ImportDocumentsRequest].
    If the long running operation is done, then this message is returned
    by the google.longrunning.Operations.response field if the operation
    was successful.

    Attributes:
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the request.
        error_config (google.cloud.discoveryengine_v1.types.ImportErrorConfig):
            Echoes the destination for the complete
            errors in the request if set.
    """

    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )
    error_config: "ImportErrorConfig" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ImportErrorConfig",
    )


class ImportSuggestionDenyListEntriesRequest(proto.Message):
    r"""Request message for
    [CompletionService.ImportSuggestionDenyListEntries][google.cloud.discoveryengine.v1.CompletionService.ImportSuggestionDenyListEntries]
    method.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        inline_source (google.cloud.discoveryengine_v1.types.ImportSuggestionDenyListEntriesRequest.InlineSource):
            The Inline source for the input content for
            suggestion deny list entries.

            This field is a member of `oneof`_ ``source``.
        gcs_source (google.cloud.discoveryengine_v1.types.GcsSource):
            Cloud Storage location for the input content.

            Only 1 file can be specified that contains all entries to
            import. Supported values ``gcs_source.schema`` for
            autocomplete suggestion deny list entry imports:

            - ``suggestion_deny_list`` (default): One JSON
              [SuggestionDenyListEntry] per line.

            This field is a member of `oneof`_ ``source``.
        parent (str):
            Required. The parent data store resource name for which to
            import denylist entries. Follows pattern
            projects/*/locations/*/collections/*/dataStores/*.
    """

    class InlineSource(proto.Message):
        r"""The inline source for SuggestionDenyListEntry.

        Attributes:
            entries (MutableSequence[google.cloud.discoveryengine_v1.types.SuggestionDenyListEntry]):
                Required. A list of all denylist entries to
                import. Max of 1000 items.
        """

        entries: MutableSequence[
            completion.SuggestionDenyListEntry
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=completion.SuggestionDenyListEntry,
        )

    inline_source: InlineSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=InlineSource,
    )
    gcs_source: "GcsSource" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="source",
        message="GcsSource",
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ImportSuggestionDenyListEntriesResponse(proto.Message):
    r"""Response message for
    [CompletionService.ImportSuggestionDenyListEntries][google.cloud.discoveryengine.v1.CompletionService.ImportSuggestionDenyListEntries]
    method.

    Attributes:
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the request.
        imported_entries_count (int):
            Count of deny list entries successfully
            imported.
        failed_entries_count (int):
            Count of deny list entries that failed to be
            imported.
    """

    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )
    imported_entries_count: int = proto.Field(
        proto.INT64,
        number=2,
    )
    failed_entries_count: int = proto.Field(
        proto.INT64,
        number=3,
    )


class ImportSuggestionDenyListEntriesMetadata(proto.Message):
    r"""Metadata related to the progress of the
    ImportSuggestionDenyListEntries operation. This is returned by
    the google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )


class ImportCompletionSuggestionsRequest(proto.Message):
    r"""Request message for
    [CompletionService.ImportCompletionSuggestions][google.cloud.discoveryengine.v1.CompletionService.ImportCompletionSuggestions]
    method.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        inline_source (google.cloud.discoveryengine_v1.types.ImportCompletionSuggestionsRequest.InlineSource):
            The Inline source for suggestion entries.

            This field is a member of `oneof`_ ``source``.
        gcs_source (google.cloud.discoveryengine_v1.types.GcsSource):
            Cloud Storage location for the input content.

            This field is a member of `oneof`_ ``source``.
        bigquery_source (google.cloud.discoveryengine_v1.types.BigQuerySource):
            BigQuery input source.

            This field is a member of `oneof`_ ``source``.
        parent (str):
            Required. The parent data store resource name for which to
            import customer autocomplete suggestions.

            Follows pattern
            ``projects/*/locations/*/collections/*/dataStores/*``
        error_config (google.cloud.discoveryengine_v1.types.ImportErrorConfig):
            The desired location of errors incurred
            during the Import.
    """

    class InlineSource(proto.Message):
        r"""The inline source for CompletionSuggestions.

        Attributes:
            suggestions (MutableSequence[google.cloud.discoveryengine_v1.types.CompletionSuggestion]):
                Required. A list of all denylist entries to
                import. Max of 1000 items.
        """

        suggestions: MutableSequence[
            completion.CompletionSuggestion
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=completion.CompletionSuggestion,
        )

    inline_source: InlineSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=InlineSource,
    )
    gcs_source: "GcsSource" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="source",
        message="GcsSource",
    )
    bigquery_source: "BigQuerySource" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="source",
        message="BigQuerySource",
    )
    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    error_config: "ImportErrorConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="ImportErrorConfig",
    )


class ImportCompletionSuggestionsResponse(proto.Message):
    r"""Response of the
    [CompletionService.ImportCompletionSuggestions][google.cloud.discoveryengine.v1.CompletionService.ImportCompletionSuggestions]
    method. If the long running operation is done, this message is
    returned by the google.longrunning.Operations.response field if the
    operation is successful.

    Attributes:
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            A sample of errors encountered while
            processing the request.
        error_config (google.cloud.discoveryengine_v1.types.ImportErrorConfig):
            The desired location of errors incurred
            during the Import.
    """

    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )
    error_config: "ImportErrorConfig" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ImportErrorConfig",
    )


class ImportCompletionSuggestionsMetadata(proto.Message):
    r"""Metadata related to the progress of the
    ImportCompletionSuggestions operation. This will be returned by
    the google.longrunning.Operation.metadata field.

    Attributes:
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation create time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Operation last update time. If the operation
            is done, this is also the finish time.
        success_count (int):
            Count of
            [CompletionSuggestion][google.cloud.discoveryengine.v1.CompletionSuggestion]s
            successfully imported.
        failure_count (int):
            Count of
            [CompletionSuggestion][google.cloud.discoveryengine.v1.CompletionSuggestion]s
            that failed to be imported.
    """

    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=1,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    success_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    failure_count: int = proto.Field(
        proto.INT64,
        number=4,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
