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

from google.cloud.aiplatform_v1.types import api_auth as gca_api_auth
from google.cloud.aiplatform_v1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1.types import io
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "RagEmbeddingModelConfig",
        "RagVectorDbConfig",
        "FileStatus",
        "VertexAiSearchConfig",
        "CorpusStatus",
        "RagCorpus",
        "RagFile",
        "RagChunk",
        "RagFileChunkingConfig",
        "RagFileTransformationConfig",
        "RagFileParsingConfig",
        "UploadRagFileConfig",
        "ImportRagFilesConfig",
        "RagManagedDbConfig",
        "RagEngineConfig",
    },
)


class RagEmbeddingModelConfig(proto.Message):
    r"""Config for the embedding model to use for RAG.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        vertex_prediction_endpoint (google.cloud.aiplatform_v1.types.RagEmbeddingModelConfig.VertexPredictionEndpoint):
            The Vertex AI Prediction Endpoint that either
            refers to a publisher model or an endpoint that
            is hosting a 1P fine-tuned text embedding model.
            Endpoints hosting non-1P fine-tuned text
            embedding models are currently not supported.
            This is used for dense vector search.

            This field is a member of `oneof`_ ``model_config``.
    """

    class VertexPredictionEndpoint(proto.Message):
        r"""Config representing a model hosted on Vertex Prediction
        Endpoint.

        Attributes:
            endpoint (str):
                Required. The endpoint resource name. Format:
                ``projects/{project}/locations/{location}/publishers/{publisher}/models/{model}``
                or
                ``projects/{project}/locations/{location}/endpoints/{endpoint}``
            model (str):
                Output only. The resource name of the model that is deployed
                on the endpoint. Present only when the endpoint is not a
                publisher model. Pattern:
                ``projects/{project}/locations/{location}/models/{model}``
            model_version_id (str):
                Output only. Version ID of the model that is
                deployed on the endpoint. Present only when the
                endpoint is not a publisher model.
        """

        endpoint: str = proto.Field(
            proto.STRING,
            number=1,
        )
        model: str = proto.Field(
            proto.STRING,
            number=2,
        )
        model_version_id: str = proto.Field(
            proto.STRING,
            number=3,
        )

    vertex_prediction_endpoint: VertexPredictionEndpoint = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="model_config",
        message=VertexPredictionEndpoint,
    )


class RagVectorDbConfig(proto.Message):
    r"""Config for the Vector DB to use for RAG.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        rag_managed_db (google.cloud.aiplatform_v1.types.RagVectorDbConfig.RagManagedDb):
            The config for the RAG-managed Vector DB.

            This field is a member of `oneof`_ ``vector_db``.
        pinecone (google.cloud.aiplatform_v1.types.RagVectorDbConfig.Pinecone):
            The config for the Pinecone.

            This field is a member of `oneof`_ ``vector_db``.
        vertex_vector_search (google.cloud.aiplatform_v1.types.RagVectorDbConfig.VertexVectorSearch):
            The config for the Vertex Vector Search.

            This field is a member of `oneof`_ ``vector_db``.
        api_auth (google.cloud.aiplatform_v1.types.ApiAuth):
            Authentication config for the chosen Vector
            DB.
        rag_embedding_model_config (google.cloud.aiplatform_v1.types.RagEmbeddingModelConfig):
            Optional. Immutable. The embedding model
            config of the Vector DB.
    """

    class RagManagedDb(proto.Message):
        r"""The config for the default RAG-managed Vector DB.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            knn (google.cloud.aiplatform_v1.types.RagVectorDbConfig.RagManagedDb.KNN):
                Performs a KNN search on RagCorpus.
                Default choice if not specified.

                This field is a member of `oneof`_ ``retrieval_strategy``.
            ann (google.cloud.aiplatform_v1.types.RagVectorDbConfig.RagManagedDb.ANN):
                Performs an ANN search on RagCorpus. Use this
                if you have a lot of files (> 10K) in your
                RagCorpus and want to reduce the search latency.

                This field is a member of `oneof`_ ``retrieval_strategy``.
        """

        class KNN(proto.Message):
            r"""Config for KNN search."""

        class ANN(proto.Message):
            r"""Config for ANN search.

            RagManagedDb uses a tree-based structure to partition data and
            facilitate faster searches. As a tradeoff, it requires longer
            indexing time and manual triggering of index rebuild via the
            ImportRagFiles and UpdateRagCorpus API.

            Attributes:
                tree_depth (int):
                    The depth of the tree-based structure. Only
                    depth values of 2 and 3 are supported.

                    Recommended value is 2 if you have if you have
                    O(10K) files in the RagCorpus and set this to 3
                    if more than that.

                    Default value is 2.
                leaf_count (int):
                    Number of leaf nodes in the tree-based structure. Each leaf
                    node contains groups of closely related vectors along with
                    their corresponding centroid.

                    Recommended value is 10 \* sqrt(num of RagFiles in your
                    RagCorpus).

                    Default value is 500.
            """

            tree_depth: int = proto.Field(
                proto.INT32,
                number=1,
            )
            leaf_count: int = proto.Field(
                proto.INT32,
                number=2,
            )

        knn: "RagVectorDbConfig.RagManagedDb.KNN" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="retrieval_strategy",
            message="RagVectorDbConfig.RagManagedDb.KNN",
        )
        ann: "RagVectorDbConfig.RagManagedDb.ANN" = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="retrieval_strategy",
            message="RagVectorDbConfig.RagManagedDb.ANN",
        )

    class Pinecone(proto.Message):
        r"""The config for the Pinecone.

        Attributes:
            index_name (str):
                Pinecone index name.
                This value cannot be changed after it's set.
        """

        index_name: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class VertexVectorSearch(proto.Message):
        r"""The config for the Vertex Vector Search.

        Attributes:
            index_endpoint (str):
                The resource name of the Index Endpoint. Format:
                ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
            index (str):
                The resource name of the Index. Format:
                ``projects/{project}/locations/{location}/indexes/{index}``
        """

        index_endpoint: str = proto.Field(
            proto.STRING,
            number=1,
        )
        index: str = proto.Field(
            proto.STRING,
            number=2,
        )

    rag_managed_db: RagManagedDb = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="vector_db",
        message=RagManagedDb,
    )
    pinecone: Pinecone = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="vector_db",
        message=Pinecone,
    )
    vertex_vector_search: VertexVectorSearch = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="vector_db",
        message=VertexVectorSearch,
    )
    api_auth: gca_api_auth.ApiAuth = proto.Field(
        proto.MESSAGE,
        number=5,
        message=gca_api_auth.ApiAuth,
    )
    rag_embedding_model_config: "RagEmbeddingModelConfig" = proto.Field(
        proto.MESSAGE,
        number=7,
        message="RagEmbeddingModelConfig",
    )


class FileStatus(proto.Message):
    r"""RagFile status.

    Attributes:
        state (google.cloud.aiplatform_v1.types.FileStatus.State):
            Output only. RagFile state.
        error_status (str):
            Output only. Only when the ``state`` field is ERROR.
    """

    class State(proto.Enum):
        r"""RagFile state.

        Values:
            STATE_UNSPECIFIED (0):
                RagFile state is unspecified.
            ACTIVE (1):
                RagFile resource has been created and indexed
                successfully.
            ERROR (2):
                RagFile resource is in a problematic state. See
                ``error_message`` field for details.
        """

        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        ERROR = 2

    state: State = proto.Field(
        proto.ENUM,
        number=1,
        enum=State,
    )
    error_status: str = proto.Field(
        proto.STRING,
        number=2,
    )


class VertexAiSearchConfig(proto.Message):
    r"""Config for the Vertex AI Search.

    Attributes:
        serving_config (str):
            Vertex AI Search Serving Config resource full name. For
            example,
            ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/servingConfigs/{serving_config}``
            or
            ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/servingConfigs/{serving_config}``.
    """

    serving_config: str = proto.Field(
        proto.STRING,
        number=1,
    )


class CorpusStatus(proto.Message):
    r"""RagCorpus status.

    Attributes:
        state (google.cloud.aiplatform_v1.types.CorpusStatus.State):
            Output only. RagCorpus life state.
        error_status (str):
            Output only. Only when the ``state`` field is ERROR.
    """

    class State(proto.Enum):
        r"""RagCorpus life state.

        Values:
            UNKNOWN (0):
                This state is not supposed to happen.
            INITIALIZED (1):
                RagCorpus resource entry is initialized, but
                hasn't done validation.
            ACTIVE (2):
                RagCorpus is provisioned successfully and is
                ready to serve.
            ERROR (3):
                RagCorpus is in a problematic situation. See
                ``error_message`` field for details.
        """

        UNKNOWN = 0
        INITIALIZED = 1
        ACTIVE = 2
        ERROR = 3

    state: State = proto.Field(
        proto.ENUM,
        number=1,
        enum=State,
    )
    error_status: str = proto.Field(
        proto.STRING,
        number=2,
    )


class RagCorpus(proto.Message):
    r"""A RagCorpus is a RagFile container and a project can have
    multiple RagCorpora.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        vector_db_config (google.cloud.aiplatform_v1.types.RagVectorDbConfig):
            Optional. Immutable. The config for the
            Vector DBs.

            This field is a member of `oneof`_ ``backend_config``.
        vertex_ai_search_config (google.cloud.aiplatform_v1.types.VertexAiSearchConfig):
            Optional. Immutable. The config for the
            Vertex AI Search.

            This field is a member of `oneof`_ ``backend_config``.
        name (str):
            Output only. The resource name of the
            RagCorpus.
        display_name (str):
            Required. The display name of the RagCorpus.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        description (str):
            Optional. The description of the RagCorpus.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this RagCorpus
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this RagCorpus
            was last updated.
        corpus_status (google.cloud.aiplatform_v1.types.CorpusStatus):
            Output only. RagCorpus state.
        encryption_spec (google.cloud.aiplatform_v1.types.EncryptionSpec):
            Optional. Immutable. The CMEK key name used
            to encrypt at-rest data related to this Corpus.
            Only applicable to RagManagedDb option for
            Vector DB. This field can only be set at corpus
            creation time, and cannot be updated or deleted.
    """

    vector_db_config: "RagVectorDbConfig" = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="backend_config",
        message="RagVectorDbConfig",
    )
    vertex_ai_search_config: "VertexAiSearchConfig" = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="backend_config",
        message="VertexAiSearchConfig",
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    corpus_status: "CorpusStatus" = proto.Field(
        proto.MESSAGE,
        number=8,
        message="CorpusStatus",
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=12,
        message=gca_encryption_spec.EncryptionSpec,
    )


class RagFile(proto.Message):
    r"""A RagFile contains user data for chunking, embedding and
    indexing.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_source (google.cloud.aiplatform_v1.types.GcsSource):
            Output only. Google Cloud Storage location of
            the RagFile. It does not support wildcards in
            the Cloud Storage uri for now.

            This field is a member of `oneof`_ ``rag_file_source``.
        google_drive_source (google.cloud.aiplatform_v1.types.GoogleDriveSource):
            Output only. Google Drive location. Supports
            importing individual files as well as Google
            Drive folders.

            This field is a member of `oneof`_ ``rag_file_source``.
        direct_upload_source (google.cloud.aiplatform_v1.types.DirectUploadSource):
            Output only. The RagFile is encapsulated and
            uploaded in the UploadRagFile request.

            This field is a member of `oneof`_ ``rag_file_source``.
        slack_source (google.cloud.aiplatform_v1.types.SlackSource):
            The RagFile is imported from a Slack channel.

            This field is a member of `oneof`_ ``rag_file_source``.
        jira_source (google.cloud.aiplatform_v1.types.JiraSource):
            The RagFile is imported from a Jira query.

            This field is a member of `oneof`_ ``rag_file_source``.
        share_point_sources (google.cloud.aiplatform_v1.types.SharePointSources):
            The RagFile is imported from a SharePoint
            source.

            This field is a member of `oneof`_ ``rag_file_source``.
        name (str):
            Output only. The resource name of the
            RagFile.
        display_name (str):
            Required. The display name of the RagFile.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        description (str):
            Optional. The description of the RagFile.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this RagFile was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this RagFile was
            last updated.
        file_status (google.cloud.aiplatform_v1.types.FileStatus):
            Output only. State of the RagFile.
    """

    gcs_source: io.GcsSource = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="rag_file_source",
        message=io.GcsSource,
    )
    google_drive_source: io.GoogleDriveSource = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="rag_file_source",
        message=io.GoogleDriveSource,
    )
    direct_upload_source: io.DirectUploadSource = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="rag_file_source",
        message=io.DirectUploadSource,
    )
    slack_source: io.SlackSource = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="rag_file_source",
        message=io.SlackSource,
    )
    jira_source: io.JiraSource = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="rag_file_source",
        message=io.JiraSource,
    )
    share_point_sources: io.SharePointSources = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="rag_file_source",
        message=io.SharePointSources,
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    file_status: "FileStatus" = proto.Field(
        proto.MESSAGE,
        number=13,
        message="FileStatus",
    )


class RagChunk(proto.Message):
    r"""A RagChunk includes the content of a chunk of a RagFile, and
    associated metadata.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        text (str):
            The content of the chunk.
        page_span (google.cloud.aiplatform_v1.types.RagChunk.PageSpan):
            If populated, represents where the chunk
            starts and ends in the document.

            This field is a member of `oneof`_ ``_page_span``.
    """

    class PageSpan(proto.Message):
        r"""Represents where the chunk starts and ends in the document.

        Attributes:
            first_page (int):
                Page where chunk starts in the document.
                Inclusive. 1-indexed.
            last_page (int):
                Page where chunk ends in the document.
                Inclusive. 1-indexed.
        """

        first_page: int = proto.Field(
            proto.INT32,
            number=1,
        )
        last_page: int = proto.Field(
            proto.INT32,
            number=2,
        )

    text: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_span: PageSpan = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message=PageSpan,
    )


class RagFileChunkingConfig(proto.Message):
    r"""Specifies the size and overlap of chunks for RagFiles.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        fixed_length_chunking (google.cloud.aiplatform_v1.types.RagFileChunkingConfig.FixedLengthChunking):
            Specifies the fixed length chunking config.

            This field is a member of `oneof`_ ``chunking_config``.
    """

    class FixedLengthChunking(proto.Message):
        r"""Specifies the fixed length chunking config.

        Attributes:
            chunk_size (int):
                The size of the chunks.
            chunk_overlap (int):
                The overlap between chunks.
        """

        chunk_size: int = proto.Field(
            proto.INT32,
            number=1,
        )
        chunk_overlap: int = proto.Field(
            proto.INT32,
            number=2,
        )

    fixed_length_chunking: FixedLengthChunking = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="chunking_config",
        message=FixedLengthChunking,
    )


class RagFileTransformationConfig(proto.Message):
    r"""Specifies the transformation config for RagFiles.

    Attributes:
        rag_file_chunking_config (google.cloud.aiplatform_v1.types.RagFileChunkingConfig):
            Specifies the chunking config for RagFiles.
    """

    rag_file_chunking_config: "RagFileChunkingConfig" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RagFileChunkingConfig",
    )


class RagFileParsingConfig(proto.Message):
    r"""Specifies the parsing config for RagFiles.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        layout_parser (google.cloud.aiplatform_v1.types.RagFileParsingConfig.LayoutParser):
            The Layout Parser to use for RagFiles.

            This field is a member of `oneof`_ ``parser``.
        llm_parser (google.cloud.aiplatform_v1.types.RagFileParsingConfig.LlmParser):
            The LLM Parser to use for RagFiles.

            This field is a member of `oneof`_ ``parser``.
    """

    class LayoutParser(proto.Message):
        r"""Document AI Layout Parser config.

        Attributes:
            processor_name (str):
                The full resource name of a Document AI processor or
                processor version. The processor must have type
                ``LAYOUT_PARSER_PROCESSOR``. If specified, the
                ``additional_config.parse_as_scanned_pdf`` field must be
                false. Format:

                - ``projects/{project_id}/locations/{location}/processors/{processor_id}``
                - ``projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}``
            max_parsing_requests_per_min (int):
                The maximum number of requests the job is
                allowed to make to the Document AI processor per
                minute. Consult
                https://cloud.google.com/document-ai/quotas and
                the Quota page for your project to set an
                appropriate value here. If unspecified, a
                default value of 120 QPM would be used.
        """

        processor_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        max_parsing_requests_per_min: int = proto.Field(
            proto.INT32,
            number=2,
        )

    class LlmParser(proto.Message):
        r"""Specifies the advanced parsing for RagFiles.

        Attributes:
            model_name (str):
                The name of a LLM model used for parsing. Format:

                - ``projects/{project_id}/locations/{location}/publishers/{publisher}/models/{model}``
            max_parsing_requests_per_min (int):
                The maximum number of requests the job is
                allowed to make to the LLM model per minute.
                Consult
                https://cloud.google.com/vertex-ai/generative-ai/docs/quotas
                and your document size to set an appropriate
                value here. If unspecified, a default value of
                5000 QPM would be used.
            custom_parsing_prompt (str):
                The prompt to use for parsing. If not
                specified, a default prompt will be used.
        """

        model_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        max_parsing_requests_per_min: int = proto.Field(
            proto.INT32,
            number=2,
        )
        custom_parsing_prompt: str = proto.Field(
            proto.STRING,
            number=3,
        )

    layout_parser: LayoutParser = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="parser",
        message=LayoutParser,
    )
    llm_parser: LlmParser = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="parser",
        message=LlmParser,
    )


class UploadRagFileConfig(proto.Message):
    r"""Config for uploading RagFile.

    Attributes:
        rag_file_transformation_config (google.cloud.aiplatform_v1.types.RagFileTransformationConfig):
            Specifies the transformation config for
            RagFiles.
    """

    rag_file_transformation_config: "RagFileTransformationConfig" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="RagFileTransformationConfig",
    )


class ImportRagFilesConfig(proto.Message):
    r"""Config for importing RagFiles.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_source (google.cloud.aiplatform_v1.types.GcsSource):
            Google Cloud Storage location. Supports importing individual
            files as well as entire Google Cloud Storage directories.
            Sample formats:

            - ``gs://bucket_name/my_directory/object_name/my_file.txt``
            - ``gs://bucket_name/my_directory``

            This field is a member of `oneof`_ ``import_source``.
        google_drive_source (google.cloud.aiplatform_v1.types.GoogleDriveSource):
            Google Drive location. Supports importing
            individual files as well as Google Drive
            folders.

            This field is a member of `oneof`_ ``import_source``.
        slack_source (google.cloud.aiplatform_v1.types.SlackSource):
            Slack channels with their corresponding
            access tokens.

            This field is a member of `oneof`_ ``import_source``.
        jira_source (google.cloud.aiplatform_v1.types.JiraSource):
            Jira queries with their corresponding
            authentication.

            This field is a member of `oneof`_ ``import_source``.
        share_point_sources (google.cloud.aiplatform_v1.types.SharePointSources):
            SharePoint sources.

            This field is a member of `oneof`_ ``import_source``.
        partial_failure_gcs_sink (google.cloud.aiplatform_v1.types.GcsDestination):
            The Cloud Storage path to write partial failures to.
            Deprecated. Prefer to use ``import_result_gcs_sink``.

            This field is a member of `oneof`_ ``partial_failure_sink``.
        partial_failure_bigquery_sink (google.cloud.aiplatform_v1.types.BigQueryDestination):
            The BigQuery destination to write partial failures to. It
            should be a bigquery table resource name (e.g.
            "bq://projectId.bqDatasetId.bqTableId"). The dataset must
            exist. If the table does not exist, it will be created with
            the expected schema. If the table exists, the schema will be
            validated and data will be added to this existing table.
            Deprecated. Prefer to use ``import_result_bq_sink``.

            This field is a member of `oneof`_ ``partial_failure_sink``.
        import_result_gcs_sink (google.cloud.aiplatform_v1.types.GcsDestination):
            The Cloud Storage path to write import result
            to.

            This field is a member of `oneof`_ ``import_result_sink``.
        import_result_bigquery_sink (google.cloud.aiplatform_v1.types.BigQueryDestination):
            The BigQuery destination to write import
            result to. It should be a bigquery table
            resource name (e.g.
            "bq://projectId.bqDatasetId.bqTableId"). The
            dataset must exist. If the table does not exist,
            it will be created with the expected schema. If
            the table exists, the schema will be validated
            and data will be added to this existing table.

            This field is a member of `oneof`_ ``import_result_sink``.
        rag_file_transformation_config (google.cloud.aiplatform_v1.types.RagFileTransformationConfig):
            Specifies the transformation config for
            RagFiles.
        rag_file_parsing_config (google.cloud.aiplatform_v1.types.RagFileParsingConfig):
            Optional. Specifies the parsing config for
            RagFiles. RAG will use the default parser if
            this field is not set.
        max_embedding_requests_per_min (int):
            Optional. The max number of queries per
            minute that this job is allowed to make to the
            embedding model specified on the corpus. This
            value is specific to this job and not shared
            across other import jobs. Consult the Quotas
            page on the project to set an appropriate value
            here. If unspecified, a default value of 1,000
            QPM would be used.
        rebuild_ann_index (bool):
            Rebuilds the ANN index to optimize for recall on the
            imported data. Only applicable for RagCorpora running on
            RagManagedDb with ``retrieval_strategy`` set to ``ANN``. The
            rebuild will be performed using the existing ANN config set
            on the RagCorpus. To change the ANN config, please use the
            UpdateRagCorpus API.

            Default is false, i.e., index is not rebuilt.
    """

    gcs_source: io.GcsSource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="import_source",
        message=io.GcsSource,
    )
    google_drive_source: io.GoogleDriveSource = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="import_source",
        message=io.GoogleDriveSource,
    )
    slack_source: io.SlackSource = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="import_source",
        message=io.SlackSource,
    )
    jira_source: io.JiraSource = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="import_source",
        message=io.JiraSource,
    )
    share_point_sources: io.SharePointSources = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="import_source",
        message=io.SharePointSources,
    )
    partial_failure_gcs_sink: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="partial_failure_sink",
        message=io.GcsDestination,
    )
    partial_failure_bigquery_sink: io.BigQueryDestination = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="partial_failure_sink",
        message=io.BigQueryDestination,
    )
    import_result_gcs_sink: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="import_result_sink",
        message=io.GcsDestination,
    )
    import_result_bigquery_sink: io.BigQueryDestination = proto.Field(
        proto.MESSAGE,
        number=15,
        oneof="import_result_sink",
        message=io.BigQueryDestination,
    )
    rag_file_transformation_config: "RagFileTransformationConfig" = proto.Field(
        proto.MESSAGE,
        number=16,
        message="RagFileTransformationConfig",
    )
    rag_file_parsing_config: "RagFileParsingConfig" = proto.Field(
        proto.MESSAGE,
        number=8,
        message="RagFileParsingConfig",
    )
    max_embedding_requests_per_min: int = proto.Field(
        proto.INT32,
        number=5,
    )
    rebuild_ann_index: bool = proto.Field(
        proto.BOOL,
        number=19,
    )


class RagManagedDbConfig(proto.Message):
    r"""Configuration message for RagManagedDb used by RagEngine.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        scaled (google.cloud.aiplatform_v1.types.RagManagedDbConfig.Scaled):
            Sets the RagManagedDb to the Scaled tier.

            This field is a member of `oneof`_ ``tier``.
        basic (google.cloud.aiplatform_v1.types.RagManagedDbConfig.Basic):
            Sets the RagManagedDb to the Basic tier.

            This field is a member of `oneof`_ ``tier``.
        unprovisioned (google.cloud.aiplatform_v1.types.RagManagedDbConfig.Unprovisioned):
            Sets the RagManagedDb to the Unprovisioned
            tier.

            This field is a member of `oneof`_ ``tier``.
    """

    class Scaled(proto.Message):
        r"""Scaled tier offers production grade performance along with
        autoscaling functionality. It is suitable for customers with
        large amounts of data or performance sensitive workloads.

        """

    class Basic(proto.Message):
        r"""Basic tier is a cost-effective and low compute tier suitable for the
        following cases:

        - Experimenting with RagManagedDb.
        - Small data size.
        - Latency insensitive workload.
        - Only using RAG Engine with external vector DBs.

        NOTE: This is the default tier if not explicitly chosen.

        """

    class Unprovisioned(proto.Message):
        r"""Disables the RAG Engine service and deletes all your data
        held within this service. This will halt the billing of the
        service.

        NOTE: Once deleted the data cannot be recovered. To start using
        RAG Engine again, you will need to update the tier by calling
        the UpdateRagEngineConfig API.

        """

    scaled: Scaled = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="tier",
        message=Scaled,
    )
    basic: Basic = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="tier",
        message=Basic,
    )
    unprovisioned: Unprovisioned = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="tier",
        message=Unprovisioned,
    )


class RagEngineConfig(proto.Message):
    r"""Config for RagEngine.

    Attributes:
        name (str):
            Identifier. The name of the RagEngineConfig. Format:
            ``projects/{project}/locations/{location}/ragEngineConfig``
        rag_managed_db_config (google.cloud.aiplatform_v1.types.RagManagedDbConfig):
            The config of the RagManagedDb used by
            RagEngine.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    rag_managed_db_config: "RagManagedDbConfig" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="RagManagedDbConfig",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
