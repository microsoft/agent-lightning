# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
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

import dataclasses
from typing import List, Optional, Sequence, Union

from google.cloud.aiplatform_v1beta1.types import EncryptionSpec

from google.protobuf import timestamp_pb2

DEPRECATION_DATE = "June 2025"


@dataclasses.dataclass
class RagFile:
    """RAG file (output only).

    Attributes:
        name: Generated resource name. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}/ragFiles/{rag_file}``
        display_name: Display name that was configured at client side.
        description: The description of the RagFile.
    """

    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None


@dataclasses.dataclass
class EmbeddingModelConfig:
    """EmbeddingModelConfig.

    The representation of the embedding model config. Users input a 1P embedding
    model as a Publisher model resource, or a 1P fine tuned embedding model
    as an Endpoint resource.

    Attributes:
        publisher_model: 1P publisher model resource name. Format:
            ``publishers/google/models/{model}`` or
            ``projects/{project}/locations/{location}/publishers/google/models/{model}``
        endpoint: 1P fine tuned embedding model resource name. Format:
            ``endpoints/{endpoint}`` or
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``.
        model:
            Output only. The resource name of the model that is deployed
            on the endpoint. Present only when the endpoint is not a
            publisher model. Pattern:
            ``projects/{project}/locations/{location}/models/{model}``
        model_version_id:
            Output only. Version ID of the model that is
            deployed on the endpoint. Present only when the
            endpoint is not a publisher model.
    """

    publisher_model: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    model_version_id: Optional[str] = None


@dataclasses.dataclass
class VertexPredictionEndpoint:
    """VertexPredictionEndpoint.

    Attributes:
        publisher_model: 1P publisher model resource name. Format:
            ``publishers/google/models/{model}`` or
            ``projects/{project}/locations/{location}/publishers/google/models/{model}``
        endpoint: 1P fine tuned embedding model resource name. Format:
            ``endpoints/{endpoint}`` or
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``.
        model:
            Output only. The resource name of the model that is deployed
            on the endpoint. Present only when the endpoint is not a
            publisher model. Pattern:
            ``projects/{project}/locations/{location}/models/{model}``
        model_version_id:
            Output only. Version ID of the model that is
            deployed on the endpoint. Present only when the
            endpoint is not a publisher model.
    """

    endpoint: Optional[str] = None
    publisher_model: Optional[str] = None
    model: Optional[str] = None
    model_version_id: Optional[str] = None


@dataclasses.dataclass
class RagEmbeddingModelConfig:
    """RagEmbeddingModelConfig.

    Attributes:
        vertex_prediction_endpoint: The Vertex AI Prediction Endpoint resource
            name. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
    """

    vertex_prediction_endpoint: Optional[VertexPredictionEndpoint] = None


@dataclasses.dataclass
class Weaviate:
    """Weaviate.

    Attributes:
        weaviate_http_endpoint: The Weaviate DB instance HTTP endpoint
        collection_name: The corresponding Weaviate collection this corpus maps to
        api_key: The SecretManager resource name for the Weaviate DB API token. Format:
            ``projects/{project}/secrets/{secret}/versions/{version}``
    """

    weaviate_http_endpoint: Optional[str] = None
    collection_name: Optional[str] = None
    api_key: Optional[str] = None


@dataclasses.dataclass
class VertexFeatureStore:
    """VertexFeatureStore.

    Attributes:
        resource_name: The resource name of the FeatureView. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/
              {feature_online_store}/featureViews/{feature_view}``
    """

    resource_name: Optional[str] = None


@dataclasses.dataclass
class VertexVectorSearch:
    """VertexVectorSearch.

    Attributes:
        index_endpoint (str):
            The resource name of the Index Endpoint. Format:
            ``projects/{project}/locations/{location}/indexEndpoints/{index_endpoint}``
        index (str):
            The resource name of the Index. Format:
            ``projects/{project}/locations/{location}/indexes/{index}``
    """

    index_endpoint: Optional[str] = None
    index: Optional[str] = None


@dataclasses.dataclass
class KNN:
    """Config for KNN search."""


@dataclasses.dataclass
class ANN:
    """Config for ANN search.

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

            Recommended value is 10 * sqrt(num of RagFiles in your
            RagCorpus).

            Default value is 500.
    """

    tree_depth: Optional[int] = None
    leaf_count: Optional[int] = None


@dataclasses.dataclass
class RagManagedDb:
    """RagManagedDb.

    Attributes:
        retrieval_strategy: Performs a KNN or ANN search on RagCorpus.
            Default choice is KNN if not specified.
    """

    retrieval_strategy: Optional[Union[KNN, ANN]] = None


@dataclasses.dataclass
class Pinecone:
    """Pinecone.

    Attributes:
        index_name: The Pinecone index name.
        api_key: The SecretManager resource name for the Pinecone DB API token. Format:
            ``projects/{project}/secrets/{secret}/versions/{version}``
    """

    index_name: Optional[str] = None
    api_key: Optional[str] = None


@dataclasses.dataclass
class VertexAiSearchConfig:
    """VertexAiSearchConfig.

    Attributes:
        serving_config: The resource name of the Vertex AI Search serving config.
            Format:
                ``projects/{project}/locations/{location}/collections/{collection}/engines/{engine}/servingConfigs/{serving_config}``
            or
                ``projects/{project}/locations/{location}/collections/{collection}/dataStores/{data_store}/servingConfigs/{serving_config}``
    """

    serving_config: Optional[str] = None


@dataclasses.dataclass
class RagVectorDbConfig:
    """RagVectorDbConfig.

    Attributes:
        vector_db: Can be one of the following: Weaviate, VertexFeatureStore,
            VertexVectorSearch, Pinecone, RagManagedDb.
        rag_embedding_model_config: The embedding model config of the Vector DB.
    """

    vector_db: Optional[
        Union[Weaviate, VertexFeatureStore, VertexVectorSearch, Pinecone, RagManagedDb]
    ] = None
    rag_embedding_model_config: Optional[RagEmbeddingModelConfig] = None


@dataclasses.dataclass
class RagResource:
    """RagResource.

    The representation of the rag source. It can be used to specify corpus only
    or ragfiles. Currently only support one corpus or multiple files from one
    corpus. In the future we may open up multiple corpora support.

    Attributes:
        rag_corpus: A Rag corpus resource name or corpus id. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}``
            or ``{rag_corpus_id}``.
        rag_files_id: List of Rag file resource name or file ids in the same corpus. Format:
            ``{rag_file}``.
    """

    rag_corpus: Optional[str] = None
    rag_file_ids: Optional[List[str]] = None


@dataclasses.dataclass
class SlackChannel:
    """SlackChannel.

    Attributes:
        channel_id: The Slack channel ID.
        api_key: The SecretManager resource name for the Slack API token. Format:
            ``projects/{project}/secrets/{secret}/versions/{version}``
            See: https://api.slack.com/tutorials/tracks/getting-a-token.
        start_time: The starting timestamp for messages to import.
        end_time: The ending timestamp for messages to import.
    """

    channel_id: str
    api_key: str
    start_time: Optional[timestamp_pb2.Timestamp] = None
    end_time: Optional[timestamp_pb2.Timestamp] = None


@dataclasses.dataclass
class SlackChannelsSource:
    """SlackChannelsSource.

    Attributes:
        channels: The Slack channels.
    """

    channels: Sequence[SlackChannel]


@dataclasses.dataclass
class JiraQuery:
    """JiraQuery.

    Attributes:
        email: The Jira email address.
        jira_projects: A list of Jira projects to import in their entirety.
        custom_queries: A list of custom JQL Jira queries to import.
        api_key: The SecretManager version resource name for Jira API access. Format:
            ``projects/{project}/secrets/{secret}/versions/{version}``
            See: https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/
        server_uri: The Jira server URI. Format:
            ``{server}.atlassian.net``
    """

    email: str
    jira_projects: Sequence[str]
    custom_queries: Sequence[str]
    api_key: str
    server_uri: str


@dataclasses.dataclass
class JiraSource:
    """JiraSource.

    Attributes:
        queries: The Jira queries.
    """

    queries: Sequence[JiraQuery]


@dataclasses.dataclass
class SharePointSource:
    """SharePointSource.

    Attributes:
        sharepoint_folder_path: The path of the SharePoint folder to download
            from.
        sharepoint_folder_id: The ID of the SharePoint folder to download
            from.
        drive_name: The name of the drive to download from.
        drive_id: The ID of the drive to download from.
        client_id: The Application ID for the app registered in
            Microsoft Azure Portal. The application must
            also be configured with MS Graph permissions
            "Files.ReadAll", "Sites.ReadAll" and
            BrowserSiteLists.Read.All.
        client_secret: The application secret for the app registered
            in Azure.
        tenant_id: Unique identifier of the Azure Active
            Directory Instance.
        sharepoint_site_name: The name of the SharePoint site to download
            from. This can be the site name or the site id.
    """

    sharepoint_folder_path: Optional[str] = None
    sharepoint_folder_id: Optional[str] = None
    drive_name: Optional[str] = None
    drive_id: Optional[str] = None
    client_id: str = None
    client_secret: str = None
    tenant_id: str = None
    sharepoint_site_name: str = None


@dataclasses.dataclass
class SharePointSources:
    """SharePointSources.

    Attributes:
        share_point_sources: The SharePoint sources.
    """

    share_point_sources: Sequence[SharePointSource]


@dataclasses.dataclass
class Filter:
    """Filter.

    Attributes:
        vector_distance_threshold: Only returns contexts with vector
            distance smaller than the threshold.
        vector_similarity_threshold: Only returns contexts with vector
            similarity larger than the threshold.
        metadata_filter: String for metadata filtering.
    """

    vector_distance_threshold: Optional[float] = None
    vector_similarity_threshold: Optional[float] = None
    metadata_filter: Optional[str] = None


@dataclasses.dataclass
class HybridSearch:
    """HybridSearch.

    Attributes:
        alpha: Alpha value controls the weight between dense and
            sparse vector search results. The range is [0, 1], while 0
            means sparse vector search only and 1 means dense vector
            search only. The default value is 0.5 which balances sparse
            and dense vector search equally.
    """

    alpha: Optional[float] = None


@dataclasses.dataclass
class LlmRanker:
    """LlmRanker.

    Attributes:
        model_name: The model name used for ranking. Only Gemini models are
            supported for now.
    """

    model_name: Optional[str] = None


@dataclasses.dataclass
class RankService:
    """RankService.

    Attributes:
        model_name: The model name of the rank service. Format:
            ``semantic-ranker-512@latest``
    """

    model_name: Optional[str] = None


@dataclasses.dataclass
class Ranking:
    """Ranking.

    Attributes:
        rank_service: (google.cloud.aiplatform_v1beta1.types.RagRetrievalConfig.Ranking.RankService)
                Config for Rank Service.
        llm_ranker (google.cloud.aiplatform_v1beta1.types.RagRetrievalConfig.Ranking.LlmRanker):
                Config for LlmRanker.
    """

    rank_service: Optional[RankService] = None
    llm_ranker: Optional[LlmRanker] = None


@dataclasses.dataclass
class RagRetrievalConfig:
    """RagRetrievalConfig.

    Attributes:
        top_k: The number of contexts to retrieve.
        filter: Config for filters.
        hybrid_search (google.cloud.aiplatform_v1beta1.types.RagRetrievalConfig.HybridSearch):
            Config for Hybrid Search.
        ranking (google.cloud.aiplatform_v1beta1.types.RagRetrievalConfig.Ranking):
            Config for ranking and reranking.
    """

    top_k: Optional[int] = None
    filter: Optional[Filter] = None
    hybrid_search: Optional[HybridSearch] = None
    ranking: Optional[Ranking] = None


@dataclasses.dataclass
class ChunkingConfig:
    """ChunkingConfig.

    Attributes:
        chunk_size: The size of each chunk.
        chunk_overlap: The size of the overlap between chunks.
    """

    chunk_size: int
    chunk_overlap: int


@dataclasses.dataclass
class TransformationConfig:
    """TransformationConfig.

    Attributes:
        chunking_config: The chunking config.
    """

    chunking_config: Optional[ChunkingConfig] = None


@dataclasses.dataclass
class LayoutParserConfig:
    """Configuration for the Document AI Layout Parser Processor.

    Attributes:
        processor_name (str):
            The full resource name of a Document AI processor or processor
            version. The processor must have type `LAYOUT_PARSER_PROCESSOR`.
            Format:
            -  `projects/{project_id}/locations/{location}/processors/{processor_id}`
            -  `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
        max_parsing_requests_per_min (int):
            The maximum number of requests the job is allowed to make to the
            Document AI processor per minute. Consult
            https://cloud.google.com/document-ai/quotas and the Quota page for
            your project to set an appropriate value here. If unspecified, a
            default value of 120 QPM will be used.
        global_max_parsing_requests_per_min (int):
            The maximum number of requests the job is allowed to make to
            the Document AI processor per minute in this project.
            Consult https://cloud.google.com/document-ai/quotas and the
            Quota page for your project to set an appropriate value
            here. If this value is not specified,
            max_parsing_requests_per_min will be used by indexing
            pipeline as the global limit.
    """

    processor_name: str
    max_parsing_requests_per_min: Optional[int] = None
    global_max_parsing_requests_per_min: Optional[int] = None


@dataclasses.dataclass
class LlmParserConfig:
    """Configuration for the LLM Parser Processor.

    Attributes:
        model_name (str):
            The full resource name of a Vertex AI model. Format:
            -  `projects/{project_id}/locations/{location}/publishers/google/models/{model_id}`
            -  `projects/{project_id}/locations/{location}/models/{model_id}`
        max_parsing_requests_per_min (int):
            The maximum number of requests the job is allowed to make to the
            Vertex AI model per minute. Consult
            https://cloud.google.com/vertex-ai/generative-ai/docs/quotas and
            the Quota page for your project to set an appropriate value here.
            If unspecified, a default value of 5000 QPM will be used.
        global_max_parsing_requests_per_min (int):
            The maximum number of requests the job is allowed to make to
            the LLM model per minute in this project. Consult
            https://cloud.google.com/vertex-ai/generative-ai/docs/quotas
            and your document size to set an appropriate value here. If
            this value is not specified, max_parsing_requests_per_min
            will be used by indexing pipeline job as the global limit.
        custom_parsing_prompt (str):
            A custom prompt to use for parsing.
    """

    model_name: str
    max_parsing_requests_per_min: Optional[int] = None
    global_max_parsing_requests_per_min: Optional[int] = None
    custom_parsing_prompt: Optional[str] = None


@dataclasses.dataclass
class Enterprise:
    """Enterprise tier offers production grade performance along with

    autoscaling functionality. It is suitable for customers with large
    amounts of data or performance sensitive workloads.

    NOTE: This is deprecated. Use Scaled tier instead.
    """


@dataclasses.dataclass
class Scaled:
    """Scaled tier offers production grade performance along with

    autoscaling functionality. It is suitable for customers with large
    amounts of data or performance sensitive workloads.
    """


@dataclasses.dataclass
class Basic:
    """Basic tier is a cost-effective and low compute tier suitable for the following cases:

    * Experimenting with RagManagedDb.
    * Small data size.
    * Latency insensitive workload.
    * Only using RAG Engine with external vector DBs.

    NOTE: This is the default tier if not explicitly chosen.
    """


@dataclasses.dataclass
class Unprovisioned:
    """Disables the RAG Engine service and deletes all your data held within
    this service. This will halt the billing of the service.

    NOTE: Once deleted the data cannot be recovered. To start using
    RAG Engine again, you will need to update the tier by calling the
    UpdateRagEngineConfig API.
    """


@dataclasses.dataclass
class RagManagedDbConfig:
    """RagManagedDbConfig.

    The config of the RagManagedDb used by RagEngine.

    Attributes:
        tier: The tier of the RagManagedDb. The default tier is Basic.
    """

    tier: Optional[Union[Enterprise, Basic, Scaled, Unprovisioned]] = None


@dataclasses.dataclass
class RagEngineConfig:
    """RagEngineConfig.

    Attributes:
        name: Generated resource name for singleton resource. Format:
          ``projects/{project}/locations/{location}/ragEngineConfig``
        rag_managed_db_config: The config of the RagManagedDb used by RagEngine.
          The default tier is Basic.
    """

    name: str
    rag_managed_db_config: Optional[RagManagedDbConfig] = None


@dataclasses.dataclass
class DocumentCorpus:
    """DocumentCorpus."""


@dataclasses.dataclass
class MemoryCorpus:
    """MemoryCorpus.

    Attributes:
        llm_parser: The LLM parser to use for the memory corpus.
    """

    llm_parser: Optional[LlmParserConfig] = None


@dataclasses.dataclass
class RagCorpusTypeConfig:
    """CorpusTypeConfig.

    Attributes:
        corpus_type_config: Can be one of the following: DocumentCorpus,
            MemoryCorpus.
    """

    corpus_type_config: Optional[Union[DocumentCorpus, MemoryCorpus]] = None


@dataclasses.dataclass
class RagCorpus:
    """RAG corpus(output only).

    Attributes:
        name: Generated resource name. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}``
        display_name: Display name that was configured at client side.
        description: The description of the RagCorpus.
        corpus_type_config: The corpus type config of the RagCorpus.
        embedding_model_config: The embedding model config of the RagCorpus.
            Note: Deprecated. Use backend_config instead.
        vector_db: The Vector DB of the RagCorpus.
            Note: Deprecated. Use backend_config instead.
        vertex_ai_search_config: The Vertex AI Search config of the RagCorpus.
        backend_config: The backend config of the RagCorpus. It can specify a
            Vector DB and/or the embedding model config.
        encryption_spec: The encryption spec of the RagCorpus. Immutable.
    """

    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    corpus_type_config: Optional[RagCorpusTypeConfig] = None
    embedding_model_config: Optional[EmbeddingModelConfig] = None
    vector_db: Optional[
        Union[Weaviate, VertexFeatureStore, VertexVectorSearch, Pinecone, RagManagedDb]
    ] = None
    vertex_ai_search_config: Optional[VertexAiSearchConfig] = None
    backend_config: Optional[RagVectorDbConfig] = None
    encryption_spec: Optional[EncryptionSpec] = None
