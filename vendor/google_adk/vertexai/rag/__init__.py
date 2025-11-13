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

from vertexai.rag.rag_data import (
    add_inline_citations_and_references,
    create_corpus,
    delete_corpus,
    delete_file,
    get_corpus,
    get_file,
    get_rag_engine_config,
    import_files,
    import_files_async,
    list_corpora,
    list_files,
    update_corpus,
    update_rag_engine_config,
    upload_file,
)
from vertexai.rag.rag_retrieval import (
    retrieval_query,
)
from vertexai.rag.rag_store import (
    Retrieval,
    VertexRagStore,
)
from vertexai.rag.utils.resources import (
    Basic,
    ChunkingConfig,
    Filter,
    JiraQuery,
    JiraSource,
    LayoutParserConfig,
    LlmParserConfig,
    LlmRanker,
    Pinecone,
    RagCitedGenerationResponse,
    RagCorpus,
    RagEmbeddingModelConfig,
    RagEngineConfig,
    RagFile,
    RagManagedDb,
    RagManagedDbConfig,
    RagResource,
    RagRetrievalConfig,
    RagVectorDbConfig,
    RankService,
    Ranking,
    Scaled,
    SharePointSource,
    SharePointSources,
    SlackChannel,
    SlackChannelsSource,
    TransformationConfig,
    Unprovisioned,
    VertexAiSearchConfig,
    VertexPredictionEndpoint,
    VertexVectorSearch,
)


__all__ = (
    "Basic",
    "ChunkingConfig",
    "Filter",
    "JiraQuery",
    "JiraSource",
    "LayoutParserConfig",
    "LlmParserConfig",
    "LlmRanker",
    "Pinecone",
    "RagCorpus",
    "RagEmbeddingModelConfig",
    "RagEngineConfig",
    "RagFile",
    "RagCitedGenerationResponse",
    "RagManagedDb",
    "RagManagedDbConfig",
    "RagResource",
    "RagRetrievalConfig",
    "RagVectorDbConfig",
    "Ranking",
    "RankService",
    "Retrieval",
    "Scaled",
    "SharePointSource",
    "SharePointSources",
    "SlackChannel",
    "SlackChannelsSource",
    "TransformationConfig",
    "Unprovisioned",
    "VertexAiSearchConfig",
    "VertexRagStore",
    "VertexPredictionEndpoint",
    "VertexVectorSearch",
    "create_corpus",
    "delete_corpus",
    "delete_file",
    "get_corpus",
    "get_rag_engine_config",
    "get_file",
    "import_files",
    "import_files_async",
    "list_corpora",
    "list_files",
    "retrieval_query",
    "upload_file",
    "update_corpus",
    "update_rag_engine_config",
    "add_inline_citations_and_references",
)
