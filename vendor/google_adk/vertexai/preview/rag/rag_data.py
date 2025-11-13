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
"""RAG data management SDK."""

from typing import Optional, Sequence, Union
from google import auth
from google.api_core import operation_async
from google.auth.transport import requests as google_auth_requests
from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform_v1beta1 import (
    CreateRagCorpusRequest,
    DeleteRagCorpusRequest,
    DeleteRagFileRequest,
    GetRagCorpusRequest,
    GetRagEngineConfigRequest,
    GetRagFileRequest,
    ImportRagFilesResponse,
    ListRagCorporaRequest,
    ListRagFilesRequest,
    RagCorpus as GapicRagCorpus,
    UpdateRagCorpusRequest,
    UpdateRagEngineConfigRequest,
)
from google.cloud.aiplatform_v1beta1.services.vertex_rag_data_service.pagers import (
    ListRagCorporaPager,
    ListRagFilesPager,
)
from google.cloud.aiplatform_v1beta1.types import EncryptionSpec
from vertexai.preview.rag.utils import (
    _gapic_utils,
)
from vertexai.preview.rag.utils.resources import (
    EmbeddingModelConfig,
    JiraSource,
    LayoutParserConfig,
    LlmParserConfig,
    Pinecone,
    RagCorpus,
    RagCorpusTypeConfig,
    RagEngineConfig,
    RagFile,
    RagManagedDb,
    RagVectorDbConfig,
    SharePointSources,
    SlackChannelsSource,
    TransformationConfig,
    VertexAiSearchConfig,
    VertexFeatureStore,
    VertexVectorSearch,
    Weaviate,
)


def create_corpus(
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    corpus_type_config: Optional[RagCorpusTypeConfig] = None,
    embedding_model_config: Optional[EmbeddingModelConfig] = None,
    vector_db: Optional[
        Union[
            Weaviate,
            VertexFeatureStore,
            VertexVectorSearch,
            Pinecone,
            RagManagedDb,
        ]
    ] = None,
    vertex_ai_search_config: Optional[VertexAiSearchConfig] = None,
    backend_config: Optional[RagVectorDbConfig] = None,
    encryption_spec: Optional[EncryptionSpec] = None,
    timeout: int = 600,
) -> RagCorpus:
    """Creates a new RagCorpus resource.

    Example usage:
    ```
    import vertexai
    from vertexai.preview import rag

    vertexai.init(project="my-project")

    rag_corpus = rag.create_corpus(
        display_name="my-corpus-1",
    )
    ```

    Args:
        display_name: If not provided, SDK will create one. The display name of
          the RagCorpus. The name can be up to 128 characters long and can consist
          of any UTF-8 characters.
        description: The description of the RagCorpus.
        corpus_type_config: The corpus type config of the RagCorpus.
        embedding_model_config: The embedding model config.
            Note: Deprecated. Use backend_config instead.
        vector_db: The vector db config of the RagCorpus. If unspecified, the
          default database Spanner is used.
            Note: Deprecated. Use backend_config instead.
        vertex_ai_search_config: The Vertex AI Search config of the RagCorpus.
            Note: embedding_model_config or vector_db cannot be set if
              vertex_ai_search_config is specified.
        backend_config: The backend config of the RagCorpus. It can specify a
          Vector DB and/or the embedding model config.
        encryption_spec: The encryption spec of the RagCorpus.
        timeout: Default is 600 seconds.

    Returns:
        RagCorpus.
    Raises:
        RuntimeError: Failed in RagCorpus creation due to exception.
        RuntimeError: Failed in RagCorpus creation due to operation error.
    """
    if not display_name:
        display_name = "vertex-" + utils.timestamped_unique_name()
    parent = initializer.global_config.common_location_path(project=None, location=None)

    rag_corpus = GapicRagCorpus(display_name=display_name, description=description)

    if corpus_type_config:
        _gapic_utils.set_corpus_type_config(
            corpus_type_config=corpus_type_config,
            rag_corpus=rag_corpus,
        )

    if embedding_model_config:
        _gapic_utils.set_embedding_model_config(
            embedding_model_config=embedding_model_config,
            rag_corpus=rag_corpus,
        )

    if vertex_ai_search_config and embedding_model_config:
        raise ValueError(
            "Only one of vertex_ai_search_config or embedding_model_config can be set."
        )

    if vertex_ai_search_config and backend_config:
        raise ValueError(
            "Only one of vertex_ai_search_config or backend_config can be set."
        )

    if backend_config and (embedding_model_config or vector_db):
        raise ValueError(
            "Only one of backend_config or embedding_model_config and vector_db can be set. embedding_model_config and vector_db are deprecated, use backend_config instead."
        )

    if backend_config:
        _gapic_utils.set_backend_config(
            backend_config=backend_config,
            rag_corpus=rag_corpus,
        )

    if vertex_ai_search_config and vector_db:
        raise ValueError("Only one of vertex_ai_search_config or vector_db can be set.")

    if vertex_ai_search_config:
        _gapic_utils.set_vertex_ai_search_config(
            vertex_ai_search_config=vertex_ai_search_config,
            rag_corpus=rag_corpus,
        )
    else:
        _gapic_utils.set_vector_db(
            vector_db=vector_db,
            rag_corpus=rag_corpus,
        )

    if encryption_spec:
        _gapic_utils.set_encryption_spec(
            encryption_spec=encryption_spec,
            rag_corpus=rag_corpus,
        )

    request = CreateRagCorpusRequest(
        parent=parent,
        rag_corpus=rag_corpus,
    )
    client = _gapic_utils.create_rag_data_service_client()

    try:
        response = client.create_rag_corpus(request=request)
    except Exception as e:
        raise RuntimeError("Failed in RagCorpus creation due to: ", e) from e
    return _gapic_utils.convert_gapic_to_rag_corpus(response.result(timeout=timeout))


def update_corpus(
    corpus_name: str,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    vector_db: Optional[
        Union[
            Weaviate,
            VertexFeatureStore,
            VertexVectorSearch,
            Pinecone,
            RagManagedDb,
        ]
    ] = None,
    vertex_ai_search_config: Optional[VertexAiSearchConfig] = None,
    backend_config: Optional[RagVectorDbConfig] = None,
    timeout: int = 600,
) -> RagCorpus:
    """Updates a RagCorpus resource.

    Example usage:
    ```
    import vertexai
    from vertexai.preview import rag

    vertexai.init(project="my-project")

    rag_corpus = rag.update_corpus(
        corpus_name="projects/my-project/locations/us-central1/ragCorpora/my-corpus-1",
        display_name="my-corpus-1",
    )
    ```

    Args:
        corpus_name: The name of the RagCorpus resource to update. Format:
          ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`` or
          ``{rag_corpus}``.
        display_name: If not provided, the display name will not be updated. The
          display name of the RagCorpus. The name can be up to 128 characters long
          and can consist of any UTF-8 characters.
        description: The description of the RagCorpus. If not provided, the
          description will not be updated.
        vector_db: The vector db config of the RagCorpus. If not provided, the
          vector db will not be updated.
        vertex_ai_search_config: The Vertex AI Search config of the RagCorpus. If
          not provided, the Vertex AI Search config will not be updated.
          Note: embedding_model_config or vector_db cannot be set if
            vertex_ai_search_config is specified.
        backend_config: The backend config of the RagCorpus. Specifies a Vector DB
          and/or the embedding model config.
        timeout: Default is 600 seconds.

    Returns:
        RagCorpus.
    Raises:
        RuntimeError: Failed in RagCorpus update due to exception.
        RuntimeError: Failed in RagCorpus update due to operation error.
    """
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    if display_name and description:
        rag_corpus = GapicRagCorpus(
            name=corpus_name, display_name=display_name, description=description
        )
    elif display_name:
        rag_corpus = GapicRagCorpus(name=corpus_name, display_name=display_name)
    elif description:
        rag_corpus = GapicRagCorpus(name=corpus_name, description=description)
    else:
        rag_corpus = GapicRagCorpus(name=corpus_name)

    if vertex_ai_search_config and vector_db:
        raise ValueError("Only one of vertex_ai_search_config or vector_db can be set.")

    if backend_config:
        _gapic_utils.set_backend_config(
            backend_config=backend_config,
            rag_corpus=rag_corpus,
        )

    if vertex_ai_search_config:
        _gapic_utils.set_vertex_ai_search_config(
            vertex_ai_search_config=vertex_ai_search_config,
            rag_corpus=rag_corpus,
        )
    else:
        _gapic_utils.set_vector_db(
            vector_db=vector_db,
            rag_corpus=rag_corpus,
        )

    request = UpdateRagCorpusRequest(
        rag_corpus=rag_corpus,
    )
    client = _gapic_utils.create_rag_data_service_client()

    try:
        response = client.update_rag_corpus(request=request)
    except Exception as e:
        raise RuntimeError("Failed in RagCorpus update due to: ", e) from e
    return _gapic_utils.convert_gapic_to_rag_corpus_no_embedding_model_config(
        response.result(timeout=timeout)
    )


def get_corpus(name: str) -> RagCorpus:
    """
    Get an existing RagCorpus.

    Args:
        name: An existing RagCorpus resource name. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
            or ``{rag_corpus}``.
    Returns:
        RagCorpus.
    """
    corpus_name = _gapic_utils.get_corpus_name(name)
    request = GetRagCorpusRequest(name=corpus_name)
    client = _gapic_utils.create_rag_data_service_client()
    try:
        response = client.get_rag_corpus(request=request)
    except Exception as e:
        raise RuntimeError("Failed in getting the RagCorpus due to: ", e) from e
    return _gapic_utils.convert_gapic_to_rag_corpus(response)


def list_corpora(
    page_size: Optional[int] = None, page_token: Optional[str] = None
) -> ListRagCorporaPager:
    """
    List all RagCorpora in the same project and location.

    Example usage:
    ```
    import vertexai
    from vertexai.preview import rag

    vertexai.init(project="my-project")

    # List all corpora.
    rag_corpora = list(rag.list_corpora())

    # Alternatively, return a ListRagCorporaPager.
    pager_1 = rag.list_corpora(page_size=10)
    # Then get the next page, use the generated next_page_token from the last pager.
    pager_2 = rag.list_corpora(page_size=10, page_token=pager_1.next_page_token)

    ```
    Args:
        page_size: The standard list page size. Leaving out the page_size
            causes all of the results to be returned.
        page_token: The standard list page token.

    Returns:
        ListRagCorporaPager.
    """
    parent = initializer.global_config.common_location_path(project=None, location=None)
    request = ListRagCorporaRequest(
        parent=parent,
        page_size=page_size,
        page_token=page_token,
    )
    client = _gapic_utils.create_rag_data_service_client()
    try:
        pager = client.list_rag_corpora(request=request)
    except Exception as e:
        raise RuntimeError("Failed in listing the RagCorpora due to: ", e) from e

    return pager


def delete_corpus(name: str) -> None:
    """
    Delete an existing RagCorpus.

    Args:
        name: An existing RagCorpus resource name. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
            or ``{rag_corpus}``.
    """
    corpus_name = _gapic_utils.get_corpus_name(name)
    request = DeleteRagCorpusRequest(name=corpus_name)

    client = _gapic_utils.create_rag_data_service_client()
    try:
        client.delete_rag_corpus(request=request)
        print("Successfully deleted the RagCorpus.")
    except Exception as e:
        raise RuntimeError("Failed in RagCorpus deletion due to: ", e) from e
    return None


def upload_file(
    corpus_name: str,
    path: Union[str, Sequence[str]],
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    transformation_config: Optional[TransformationConfig] = None,
) -> RagFile:
    """
    Synchronous file upload to an existing RagCorpus.

    Example usage:

    ```
    import vertexai
    from vertexai.preview import rag

    vertexai.init(project="my-project")

    # Optional.
    transformation_config = TransformationConfig(
        chunking_config=ChunkingConfig(
            chunk_size=1024,
            chunk_overlap=200,
        ),
    )

    rag_file = rag.upload_file(
        corpus_name="projects/my-project/locations/us-central1/ragCorpora/my-corpus-1",
        display_name="my_file.txt",
        path="usr/home/my_file.txt",
        transformation_config=transformation_config,
    )
    ```

    Args:
        corpus_name: The name of the RagCorpus resource into which to upload the file.
            Format: ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
            or ``{rag_corpus}``.
        path: A local file path. For example,
            "usr/home/my_file.txt".
        display_name: The display name of the data file.
        description: The description of the RagFile.
        transformation_config: The config for transforming the RagFile, such as chunking.
    Returns:
        RagFile.
    Raises:
        RuntimeError: Failed in RagFile upload.
        ValueError: RagCorpus is not found.
        RuntimeError: Failed in indexing the RagFile.
    """
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    location = initializer.global_config.location
    # GAPIC doesn't expose a path (scotty). Use requests API instead
    if display_name is None:
        display_name = "vertex-" + utils.timestamped_unique_name()
    headers = {"X-Goog-Upload-Protocol": "multipart"}
    if not initializer.global_config.api_endpoint:
        request_endpoint = "{}-{}".format(
            location, aiplatform.constants.base.API_BASE_PATH
        )
    else:
        request_endpoint = initializer.global_config.api_endpoint
    upload_request_uri = "https://{}/upload/v1beta1/{}/ragFiles:upload".format(
        request_endpoint,
        corpus_name,
    )
    js_rag_file = {"rag_file": {"display_name": display_name}}

    if description:
        js_rag_file["rag_file"]["description"] = description

    if transformation_config and transformation_config.chunking_config:
        chunk_size = transformation_config.chunking_config.chunk_size
        chunk_overlap = transformation_config.chunking_config.chunk_overlap
        js_rag_file["upload_rag_file_config"] = {
            "rag_file_transformation_config": {
                "rag_file_chunking_config": {
                    "fixed_length_chunking": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    }
                }
            }
        }
    files = {
        "metadata": (None, str(js_rag_file)),
        "file": open(path, "rb"),
    }
    credentials, _ = auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    authorized_session = google_auth_requests.AuthorizedSession(credentials=credentials)
    try:
        response = authorized_session.post(
            url=upload_request_uri,
            files=files,
            headers=headers,
        )
    except Exception as e:
        raise RuntimeError("Failed in uploading the RagFile due to: ", e) from e

    if response.status_code == 404:
        raise ValueError(
            "RagCorpus '%s' is not found: %s", corpus_name, upload_request_uri
        )
    if response.json().get("error"):
        raise RuntimeError(
            "Failed in indexing the RagFile due to: ", response.json().get("error")
        )
    return _gapic_utils.convert_json_to_rag_file(response.json())


def import_files(
    corpus_name: str,
    paths: Optional[Sequence[str]] = None,
    source: Optional[Union[SlackChannelsSource, JiraSource, SharePointSources]] = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    transformation_config: Optional[TransformationConfig] = None,
    timeout: int = 600,
    max_embedding_requests_per_min: int = 1000,
    global_max_embedding_requests_per_min: Optional[int] = None,
    use_advanced_pdf_parsing: Optional[bool] = False,
    partial_failures_sink: Optional[str] = None,
    layout_parser: Optional[LayoutParserConfig] = None,
    llm_parser: Optional[LlmParserConfig] = None,
    rebuild_ann_index: Optional[bool] = False,
) -> ImportRagFilesResponse:
    """
    Import files to an existing RagCorpus, wait until completion.

    Example usage:

    ```
    import vertexai
    from vertexai.preview import rag
    from google.protobuf import timestamp_pb2

    vertexai.init(project="my-project")
    # Google Drive example
    paths = [
        "https://drive.google.com/file/d/123",
        "https://drive.google.com/drive/folders/456"
    ]
    # Google Cloud Storage example
    paths = ["gs://my_bucket/my_files_dir", ...]

    transformation_config = TransformationConfig(
        chunking_config=ChunkingConfig(
            chunk_size=1024,
            chunk_overlap=200,
        ),
    )

    response = rag.import_files(
        corpus_name="projects/my-project/locations/us-central1/ragCorpora/my-corpus-1",
        paths=paths,
        transformation_config=transformation_config,
    )

    # Slack example
    start_time = timestamp_pb2.Timestamp()
    start_time.FromJsonString('2020-12-31T21:33:44Z')
    end_time = timestamp_pb2.Timestamp()
    end_time.GetCurrentTime()
    source = rag.SlackChannelsSource(
        channels = [
            SlackChannel("channel1", "api_key1"),
            SlackChannel("channel2", "api_key2", start_time, end_time)
        ],
    )
    # Jira Example
    jira_query = rag.JiraQuery(
        email="xxx@yyy.com",
        jira_projects=["project1", "project2"],
        custom_queries=["query1", "query2"],
        api_key="api_key",
        server_uri="server.atlassian.net"
    )
    source = rag.JiraSource(
        queries=[jira_query],
    )

    response = rag.import_files(
        corpus_name="projects/my-project/locations/us-central1/ragCorpora/my-corpus-1",
        source=source,
        transformation_config=transformation_config,
    )

    # SharePoint Example.
    sharepoint_query = rag.SharePointSource(
        sharepoint_folder_path="https://my-sharepoint-site.com/my-folder",
        sharepoint_site_name="my-sharepoint-site.com",
        client_id="my-client-id",
        client_secret="my-client-secret",
        tenant_id="my-tenant-id",
        drive_id="my-drive-id",
    )
    source = rag.SharePointSources(
        share_point_sources=[sharepoint_query],
    )

    # Return the number of imported RagFiles after completion.
    print(response.imported_rag_files_count)

    ```
    Args:
        corpus_name: The name of the RagCorpus resource into which to import files.
            Format: ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
            or ``{rag_corpus}``.
        paths: A list of uris. Eligible uris will be Google Cloud Storage
            directory ("gs://my-bucket/my_dir") or a Google Drive url for file
            (https://drive.google.com/file/... or folder
            "https://drive.google.com/corp/drive/folders/...").
        source: The source of the Slack or Jira import.
            Must be either a SlackChannelsSource or JiraSource.
        chunk_size: The size of the chunks. This field is deprecated. Please use
            transformation_config instead.
        chunk_overlap: The overlap between chunks. This field is deprecated. Please use
            transformation_config instead.
        transformation_config: The config for transforming the imported
            RagFiles.
        max_embedding_requests_per_min:
            Optional. The max number of queries per
            minute that this job is allowed to make to the
            embedding model specified on the corpus. This
            value is specific to this job and not shared
            across other import jobs. Consult the Quotas
            page on the project to set an appropriate value
            here. If unspecified, a default value of 1,000
            QPM would be used.
        global_max_embedding_requests_per_min:
            Optional. The max number of queries per minute that the indexing
            pipeline job is allowed to make to the embedding model specified in
            the project. Please follow the quota usage guideline of the embedding
            model you use to set the value properly. If this value is not specified,
            max_embedding_requests_per_min will be used by indexing pipeline job
            as the global limit and this means parallel import jobs are not allowed.
        timeout: Default is 600 seconds.
        use_advanced_pdf_parsing: Whether to use advanced PDF
            parsing on uploaded files. This field is deprecated.
        partial_failures_sink: Either a GCS path to store partial failures or a
            BigQuery table to store partial failures. The format is
            "gs://my-bucket/my/object.ndjson" for GCS or
            "bq://my-project.my-dataset.my-table" for BigQuery. An existing GCS
            object cannot be used. However, the BigQuery table may or may not
            exist - if it does not exist, it will be created. If it does exist,
            the schema will be checked and the partial failures will be appended
            to the table.
        layout_parser: Configuration for the Document AI Layout Parser Processor
            to use for document parsing. Optional.
            If not None, the other parser configs must be None.
        llm_parser: Configuration for the LLM Parser to use for document parsing.
            Optional.
            If not None, the other parser configs must be None.
        rebuild_ann_index: Rebuilds the ANN index to optimize for recall on the
            imported data. Only applicable for RagCorpora running on
            RagManagedDb with ``retrieval_strategy`` set to ``ANN``. The
            rebuild will be performed using the existing ANN config set
            on the RagCorpus. To change the ANN config, please use the
            UpdateRagCorpus API. Optional.Default is false, i.e., index is not
            rebuilt.
    Returns:
        ImportRagFilesResponse.
    """
    if source is not None and paths is not None:
        raise ValueError("Only one of source or paths must be passed in at a time")
    if source is None and paths is None:
        raise ValueError("One of source or paths must be passed in")
    if use_advanced_pdf_parsing and layout_parser is not None:
        raise ValueError(
            "Only one of use_advanced_pdf_parsing or layout_parser may be "
            "passed in at a time"
        )
    if use_advanced_pdf_parsing and llm_parser is not None:
        raise ValueError(
            "Only one of use_advanced_pdf_parsing or llm_parser may be "
            "passed in at a time"
        )
    if layout_parser is not None and llm_parser is not None:
        raise ValueError(
            "Only one of layout_parser or llm_parser may be passed in at a time"
        )

    rebuild_ann_index_request = (
        rebuild_ann_index if rebuild_ann_index is not None else False
    )
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    request = _gapic_utils.prepare_import_files_request(
        corpus_name=corpus_name,
        paths=paths,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        transformation_config=transformation_config,
        max_embedding_requests_per_min=max_embedding_requests_per_min,
        global_max_embedding_requests_per_min=global_max_embedding_requests_per_min,
        use_advanced_pdf_parsing=use_advanced_pdf_parsing,
        partial_failures_sink=partial_failures_sink,
        layout_parser=layout_parser,
        llm_parser=llm_parser,
        rebuild_ann_index=rebuild_ann_index_request,
    )
    client = _gapic_utils.create_rag_data_service_client()
    try:
        response = client.import_rag_files(request=request)
    except Exception as e:
        raise RuntimeError("Failed in importing the RagFiles due to: ", e) from e

    return response.result(timeout=timeout)


async def import_files_async(
    corpus_name: str,
    paths: Optional[Sequence[str]] = None,
    source: Optional[Union[SlackChannelsSource, JiraSource, SharePointSources]] = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    transformation_config: Optional[TransformationConfig] = None,
    max_embedding_requests_per_min: int = 1000,
    global_max_embedding_requests_per_min: Optional[int] = None,
    use_advanced_pdf_parsing: Optional[bool] = False,
    partial_failures_sink: Optional[str] = None,
    layout_parser: Optional[LayoutParserConfig] = None,
    llm_parser: Optional[LlmParserConfig] = None,
    rebuild_ann_index: Optional[bool] = False,
) -> operation_async.AsyncOperation:
    """
    Import files to an existing RagCorpus asynchronously.

    Example usage:

    ```
    import vertexai
    from vertexai.preview import rag
    from google.protobuf import timestamp_pb2

    vertexai.init(project="my-project")

    # Google Drive example
    paths = [
        "https://drive.google.com/file/d/123",
        "https://drive.google.com/drive/folders/456"
    ]
    # Google Cloud Storage example
    paths = ["gs://my_bucket/my_files_dir", ...]

    transformation_config = TransformationConfig(
        chunking_config=ChunkingConfig(
            chunk_size=1024,
            chunk_overlap=200,
        ),
    )

    response = await rag.import_files_async(
        corpus_name="projects/my-project/locations/us-central1/ragCorpora/my-corpus-1",
        paths=paths,
        transformation_config=transformation_config,
    )

    # Slack example
    start_time = timestamp_pb2.Timestamp()
    start_time.FromJsonString('2020-12-31T21:33:44Z')
    end_time = timestamp_pb2.Timestamp()
    end_time.GetCurrentTime()
    source = rag.SlackChannelsSource(
        channels = [
            SlackChannel("channel1", "api_key1"),
            SlackChannel("channel2", "api_key2", start_time, end_time)
        ],
    )
    # Jira Example
    jira_query = rag.JiraQuery(
        email="xxx@yyy.com",
        jira_projects=["project1", "project2"],
        custom_queries=["query1", "query2"],
        api_key="api_key",
        server_uri="server.atlassian.net"
    )
    source = rag.JiraSource(
        queries=[jira_query],
    )

    response = await rag.import_files_async(
        corpus_name="projects/my-project/locations/us-central1/ragCorpora/my-corpus-1",
        source=source,
        transformation_config=transformation_config,
    )

    # SharePoint Example.
    sharepoint_query = rag.SharePointSource(
        sharepoint_folder_path="https://my-sharepoint-site.com/my-folder",
        sharepoint_site_name="my-sharepoint-site.com",
        client_id="my-client-id",
        client_secret="my-client-secret",
        tenant_id="my-tenant-id",
        drive_id="my-drive-id",
    )
    source = rag.SharePointSources(
        share_point_sources=[sharepoint_query],
    )

    # Get the result.
    await response.result()

    ```
    Args:
        corpus_name: The name of the RagCorpus resource into which to import files.
            Format: ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
            or ``{rag_corpus}``.
        paths: A list of uris. Eligible uris will be Google Cloud Storage
            directory ("gs://my-bucket/my_dir") or a Google Drive url for file
            (https://drive.google.com/file/... or folder
            "https://drive.google.com/corp/drive/folders/...").
        source: The source of the Slack or Jira import.
            Must be either a SlackChannelsSource or JiraSource.
        chunk_size: The size of the chunks. This field is deprecated. Please use
            transformation_config instead.
        chunk_overlap: The overlap between chunks. This field is deprecated. Please use
            transformation_config instead.
        transformation_config: The config for transforming the imported
            RagFiles.
        max_embedding_requests_per_min:
            Optional. The max number of queries per
            minute that this job is allowed to make to the
            embedding model specified on the corpus. This
            value is specific to this job and not shared
            across other import jobs. Consult the Quotas
            page on the project to set an appropriate value
            here. If unspecified, a default value of 1,000
            QPM would be used.
        global_max_embedding_requests_per_min:
            Optional. The max number of queries per minute that the indexing
            pipeline job is allowed to make to the embedding model specified in
            the project. Please follow the quota usage guideline of the embedding
            model you use to set the value properly. If this value is not specified,
            max_embedding_requests_per_min will be used by indexing pipeline job
            as the global limit and this means parallel import jobs are not allowed.
        use_advanced_pdf_parsing: Whether to use advanced PDF
            parsing on uploaded files.
        partial_failures_sink: Either a GCS path to store partial failures or a
            BigQuery table to store partial failures. The format is
            "gs://my-bucket/my/object.ndjson" for GCS or
            "bq://my-project.my-dataset.my-table" for BigQuery. An existing GCS
            object cannot be used. However, the BigQuery table may or may not
            exist - if it does not exist, it will be created. If it does exist,
            the schema will be checked and the partial failures will be appended
            to the table.
        layout_parser: Configuration for the Document AI Layout Parser Processor
            to use for document parsing. Optional.
            If not None, the other parser configs must be None.
        llm_parser: Configuration for the LLM Parser to use for document parsing.
            Optional.
            If not None, the other parser configs must be None.
        rebuild_ann_index: Rebuilds the ANN index to optimize for recall on the
            imported data. Only applicable for RagCorpora running on
            RagManagedDb with ``retrieval_strategy`` set to ``ANN``. The
            rebuild will be performed using the existing ANN config set
            on the RagCorpus. To change the ANN config, please use the
            UpdateRagCorpus API. Optional.Default is false, i.e., index is not
            rebuilt.
    Returns:
        operation_async.AsyncOperation.
    """
    if source is not None and paths is not None:
        raise ValueError("Only one of source or paths must be passed in at a time")
    if source is None and paths is None:
        raise ValueError("One of source or paths must be passed in")
    if use_advanced_pdf_parsing and layout_parser is not None:
        raise ValueError(
            "Only one of use_advanced_pdf_parsing or layout_parser may be "
            "passed in at a time"
        )
    if use_advanced_pdf_parsing and llm_parser is not None:
        raise ValueError(
            "Only one of use_advanced_pdf_parsing or llm_parser may be "
            "passed in at a time"
        )
    if layout_parser is not None and llm_parser is not None:
        raise ValueError(
            "Only one of layout_parser or llm_parser may be passed in at a time"
        )
    rebuild_ann_index_request = (
        rebuild_ann_index if rebuild_ann_index is not None else False
    )
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    request = _gapic_utils.prepare_import_files_request(
        corpus_name=corpus_name,
        paths=paths,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        transformation_config=transformation_config,
        max_embedding_requests_per_min=max_embedding_requests_per_min,
        global_max_embedding_requests_per_min=global_max_embedding_requests_per_min,
        use_advanced_pdf_parsing=use_advanced_pdf_parsing,
        partial_failures_sink=partial_failures_sink,
        layout_parser=layout_parser,
        llm_parser=llm_parser,
        rebuild_ann_index=rebuild_ann_index_request,
    )
    async_client = _gapic_utils.create_rag_data_service_async_client()
    try:
        response = await async_client.import_rag_files(request=request)
    except Exception as e:
        raise RuntimeError("Failed in importing the RagFiles due to: ", e) from e
    return response


def get_file(name: str, corpus_name: Optional[str] = None) -> RagFile:
    """
    Get an existing RagFile.

    Args:
        name: Either a full RagFile resource name must be provided, or a RagCorpus
            name and a RagFile name must be provided. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}/ragFiles/{rag_file}``
            or ``{rag_file}``.
        corpus_name: If `name` is not a full resource name, an existing RagCorpus
            name must be provided. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
            or ``{rag_corpus}``.
    Returns:
        RagFile.
    """
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    name = _gapic_utils.get_file_name(name, corpus_name)
    request = GetRagFileRequest(name=name)
    client = _gapic_utils.create_rag_data_service_client()
    try:
        response = client.get_rag_file(request=request)
    except Exception as e:
        raise RuntimeError("Failed in getting the RagFile due to: ", e) from e
    return _gapic_utils.convert_gapic_to_rag_file(response)


def list_files(
    corpus_name: str, page_size: Optional[int] = None, page_token: Optional[str] = None
) -> ListRagFilesPager:
    """
    List all RagFiles in an existing RagCorpus.

    Example usage:
    ```
    import vertexai

    vertexai.init(project="my-project")
    # List all corpora.
    rag_corpora = list(rag.list_corpora())

    # List all files of the first corpus.
    rag_files = list(rag.list_files(corpus_name=rag_corpora[0].name))

    # Alternatively, return a ListRagFilesPager.
    pager_1 = rag.list_files(
        corpus_name=rag_corpora[0].name,
        page_size=10
    )
    # Then get the next page, use the generated next_page_token from the last pager.
    pager_2 = rag.list_files(
        corpus_name=rag_corpora[0].name,
        page_size=10,
        page_token=pager_1.next_page_token
    )

    ```

    Args:
        corpus_name: An existing RagCorpus name. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
            or ``{rag_corpus}``.
        page_size: The standard list page size. Leaving out the page_size
            causes all of the results to be returned.
        page_token: The standard list page token.
    Returns:
        ListRagFilesPager.
    """
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    request = ListRagFilesRequest(
        parent=corpus_name,
        page_size=page_size,
        page_token=page_token,
    )
    client = _gapic_utils.create_rag_data_service_client()
    try:
        pager = client.list_rag_files(request=request)
    except Exception as e:
        raise RuntimeError("Failed in listing the RagFiles due to: ", e) from e

    return pager


def delete_file(name: str, corpus_name: Optional[str] = None) -> None:
    """
    Delete RagFile from an existing RagCorpus.

    Args:
        name: Either a full RagFile resource name must be provided, or a RagCorpus
            name and a RagFile name must be provided. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}/ragFiles/{rag_file}``
            or ``{rag_file}``.
        corpus_name: If `name` is not a full resource name, an existing RagCorpus
            name must be provided. Format:
            ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus}``
            or ``{rag_corpus}``.
    """
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    name = _gapic_utils.get_file_name(name, corpus_name)
    request = DeleteRagFileRequest(name=name)

    client = _gapic_utils.create_rag_data_service_client()
    try:
        client.delete_rag_file(request=request)
        print("Successfully deleted the RagFile.")
    except Exception as e:
        raise RuntimeError("Failed in RagFile deletion due to: ", e) from e
    return None


def update_rag_engine_config(
    rag_engine_config: RagEngineConfig,
    timeout: int = 600,
) -> RagEngineConfig:
    """Update RagEngineConfig.

    Example usage:
    ```
    import vertexai
    from vertexai.preview import rag
    vertexai.init(project="my-project")
    rag_engine_config = rag.RagEngineConfig(
        rag_managed_db_config=rag.RagManagedDbConfig(
            rag_managed_db=rag.RagManagedDb(
                db_basic_tier=rag.Basic(),
            ),
        )
        ),
    )
    rag.update_rag_engine_config(rag_engine_config=rag_engine_config)
    ```

    Args:
        rag_engine_config: The RagEngineConfig to update.
        timeout: Default is 600 seconds.

    Raises:
        RuntimeError: Failed in RagEngineConfig update due to exception.
    """
    gapic_rag_engine_config = _gapic_utils.convert_rag_engine_config_to_gapic(
        rag_engine_config
    )
    request = UpdateRagEngineConfigRequest(
        rag_engine_config=gapic_rag_engine_config,
    )
    client = _gapic_utils.create_rag_data_service_client()
    try:
        response = client.update_rag_engine_config(request=request)
    except Exception as e:
        raise RuntimeError("Failed in RagEngineConfig update due to: ", e) from e
    return _gapic_utils.convert_gapic_to_rag_engine_config(
        response.result(timeout=timeout)
    )


def get_rag_engine_config(name: str) -> RagEngineConfig:
    """Get an existing RagEngineConfig.

    Example usage:
    ```
    import vertexai
    from vertexai.preview import rag
    vertexai.init(project="my-project")
    rag_engine_config = rag.get_rag_engine_config(
        name="projects/my-project/locations/us-central1/ragEngineConfig"
    )
    ```
    Args:
        name: The RagEngineConfig resource name pattern of the singleton resource.

    Returns:
        RagEngineConfig.
    Raises:
        RuntimeError: Failed in getting the RagEngineConfig.
    """
    request = GetRagEngineConfigRequest(name=name)
    client = _gapic_utils.create_rag_data_service_client()
    try:
        response = client.get_rag_engine_config(request=request)
    except Exception as e:
        raise RuntimeError("Failed in getting the RagEngineConfig due to: ", e) from e
    return _gapic_utils.convert_gapic_to_rag_engine_config(response)
