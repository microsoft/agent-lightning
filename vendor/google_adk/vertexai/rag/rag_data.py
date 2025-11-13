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
from google.cloud.aiplatform_v1 import (
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
from google.cloud.aiplatform_v1.services.vertex_rag_data_service.pagers import (
    ListRagCorporaPager,
    ListRagFilesPager,
)
from vertexai.rag.rag_inline_citations import (
    format_bibliography,
    populate_cited_chunk_references,
)
from vertexai.rag.utils import (
    _gapic_utils,
)
from google.cloud.aiplatform_v1.types import EncryptionSpec
from vertexai.rag.utils.resources import (
    JiraSource,
    LayoutParserConfig,
    LlmParserConfig,
    RagCitedGenerationResponse,
    RagCorpus,
    RagEngineConfig,
    RagFile,
    RagVectorDbConfig,
    SharePointSources,
    SlackChannelsSource,
    VertexAiSearchConfig,
    TransformationConfig,
)


def create_corpus(
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    vertex_ai_search_config: Optional[VertexAiSearchConfig] = None,
    backend_config: Optional[
        Union[
            RagVectorDbConfig,
            None,
        ]
    ] = None,
    encryption_spec: Optional[EncryptionSpec] = None,
    timeout: int = 600,
) -> RagCorpus:
    """Creates a new RagCorpus resource.

    Example usage:
    ```
    import vertexai
    from vertexai import rag

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
        vertex_ai_search_config: The Vertex AI Search config of the RagCorpus.
            Note: backend_config cannot be set if vertex_ai_search_config is
              specified.
        backend_config: The backend config of the RagCorpus, specifying a data
          store and/or embedding model.
        encryption_spec: The encryption spec of the RagCorpus.
        timeout: Default is 600 seconds.

    Returns:
        RagCorpus.
    Raises:
        RuntimeError: Failed in RagCorpus creation due to exception.
        RuntimeError: Failed in RagCorpus creation due to operation error.
    """
    if vertex_ai_search_config and backend_config:
        raise ValueError(
            "Only one of vertex_ai_search_config or backend_config can be set."
        )

    if not display_name:
        display_name = "vertex-" + utils.timestamped_unique_name()
    parent = initializer.global_config.common_location_path(project=None, location=None)

    rag_corpus = GapicRagCorpus(display_name=display_name, description=description)

    if backend_config:
        _gapic_utils.set_backend_config(
            backend_config=backend_config,
            rag_corpus=rag_corpus,
        )
    elif vertex_ai_search_config:
        _gapic_utils.set_vertex_ai_search_config(
            vertex_ai_search_config=vertex_ai_search_config,
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
    vertex_ai_search_config: Optional[VertexAiSearchConfig] = None,
    backend_config: Optional[
        Union[
            RagVectorDbConfig,
            None,
        ]
    ] = None,
    timeout: int = 600,
) -> RagCorpus:
    """Updates a RagCorpus resource.

    It is intended to update 3rd party vector DBs (Vector Search, Vertex AI
    Feature Store, Weaviate, Pinecone) but not Vertex RagManagedDb.

    Example usage:
    ```
    import vertexai
    from vertexai import rag

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
        vertex_ai_search_config: The Vertex AI Search config of the RagCorpus. If
          not provided, the Vertex AI Search config will not be updated.
          Note: backend_config cannot be set if vertex_ai_search_config is
            specified.
        backend_config: The backend config of the RagCorpus, specifying a data
          store and/or embedding model.
        timeout: Default is 600 seconds.

    Returns:
        RagCorpus.
    Raises:
        RuntimeError: Failed in RagCorpus update due to exception.
        RuntimeError: Failed in RagCorpus update due to operation error.
    """
    if vertex_ai_search_config and backend_config:
        raise ValueError(
            "Only one of vertex_ai_search_config or backend_config can be set."
        )

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
    from vertexai import rag

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
    timeout: int = 600,
) -> RagFile:
    """
    Synchronous file upload to an existing RagCorpus.

    Example usage:

    ```
    import vertexai
    from vertexai import rag

    vertexai.init(project="my-project")

    // Optional.
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
        transformation_config: The config for transforming the RagFile, like chunking.
        timeout: Default is 600 seconds.

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
    upload_request_uri = "https://{}/upload/v1/{}/ragFiles:upload".format(
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
    credentials, _ = auth.default()
    authorized_session = google_auth_requests.AuthorizedSession(credentials=credentials)
    try:
        response = authorized_session.post(
            url=upload_request_uri,
            files=files,
            headers=headers,
            timeout=timeout,
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
    transformation_config: Optional[TransformationConfig] = None,
    timeout: int = 600,
    max_embedding_requests_per_min: int = 1000,
    import_result_sink: Optional[str] = None,
    partial_failures_sink: Optional[str] = None,
    layout_parser: Optional[LayoutParserConfig] = None,
    llm_parser: Optional[LlmParserConfig] = None,
) -> ImportRagFilesResponse:
    """
    Import files to an existing RagCorpus, wait until completion.

    Example usage:

    ```
    import vertexai
    from vertexai import rag
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

    # Document AI Layout Parser example.
    parser = LayoutParserConfig(
        processor_name="projects/my-project/locations/us-central1/processors/my-processor-id",
        max_parsing_requests_per_min=120,
    )
    response = rag.import_files(
        corpus_name="projects/my-project/locations/us-central1/ragCorpora/my-corpus-1",
        paths=paths,
        parser=parser,
    )

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
        timeout: Default is 600 seconds.
        import_result_sink: Either a GCS path to store import results or a
            BigQuery table to store import results. The format is
            "gs://my-bucket/my/object.ndjson" for GCS or
            "bq://my-project.my-dataset.my-table" for BigQuery. An existing GCS
            object cannot be used. However, the BigQuery table may or may not
            exist - if it does not exist, it will be created. If it does exist,
            the schema will be checked and the import results will be appended
            to the table.
        partial_failures_sink: Deprecated. Prefer to use `import_result_sink`.
            Either a GCS path to store partial failures or a BigQuery table to
            store partial failures. The format is
            "gs://my-bucket/my/object.ndjson" for GCS or
            "bq://my-project.my-dataset.my-table" for BigQuery. An existing GCS
            object cannot be used. However, the BigQuery table may or may not
            exist - if it does not exist, it will be created. If it does exist,
            the schema will be checked and the partial failures will be appended
            to the table.
        parser: Document parser to use. Should be either None (default parser),
            or a LayoutParserConfig (to parse documents using a Document AI
            Layout Parser processor).
    Returns:
        ImportRagFilesResponse.
    """
    if source is not None and paths is not None:
        raise ValueError("Only one of source or paths must be passed in at a time")
    if source is None and paths is None:
        raise ValueError("One of source or paths must be passed in")
    if layout_parser is not None and llm_parser is not None:
        raise ValueError(
            "Only one of layout_parser or llm_parser may be passed in at a time"
        )
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    request = _gapic_utils.prepare_import_files_request(
        corpus_name=corpus_name,
        paths=paths,
        source=source,
        transformation_config=transformation_config,
        max_embedding_requests_per_min=max_embedding_requests_per_min,
        import_result_sink=import_result_sink,
        partial_failures_sink=partial_failures_sink,
        layout_parser=layout_parser,
        llm_parser=llm_parser,
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
    transformation_config: Optional[TransformationConfig] = None,
    max_embedding_requests_per_min: int = 1000,
    import_result_sink: Optional[str] = None,
    partial_failures_sink: Optional[str] = None,
    layout_parser: Optional[LayoutParserConfig] = None,
    llm_parser: Optional[LlmParserConfig] = None,
) -> operation_async.AsyncOperation:
    """
    Import files to an existing RagCorpus asynchronously.

    Example usage:

    ```
    import vertexai
    from vertexai import rag
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

    # Document AI Layout Parser example.
    parser = LayoutParserConfig(
        processor_name="projects/my-project/locations/us-central1/processors/my-processor-id",
        max_parsing_requests_per_min=120,
    )
    response = rag.import_files_async(
        corpus_name="projects/my-project/locations/us-central1/ragCorpora/my-corpus-1",
        paths=paths,
        parser=parser,
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
        import_result_sink: Either a GCS path to store import results or a
            BigQuery table to store import results. The format is
            "gs://my-bucket/my/object.ndjson" for GCS or
            "bq://my-project.my-dataset.my-table" for BigQuery. An existing GCS
            object cannot be used. However, the BigQuery table may or may not
            exist - if it does not exist, it will be created. If it does exist,
            the schema will be checked and the import results will be appended
            to the table.
        partial_failures_sink: Deprecated. Prefer to use `import_result_sink`.
            Either a GCS path to store partial failures or a BigQuery table to
            store partial failures. The format is
            "gs://my-bucket/my/object.ndjson" for GCS or
            "bq://my-project.my-dataset.my-table" for BigQuery. An existing GCS
            object cannot be used. However, the BigQuery table may or may not
            exist - if it does not exist, it will be created. If it does exist,
            the schema will be checked and the partial failures will be appended
            to the table.
        parser: Document parser to use. Should be either None (default parser),
            or a LayoutParserConfig (to parse documents using a Document AI
            Layout Parser processor).
    Returns:
        operation_async.AsyncOperation.
    """
    if source is not None and paths is not None:
        raise ValueError("Only one of source or paths must be passed in at a time")
    if source is None and paths is None:
        raise ValueError("One of source or paths must be passed in")
    if layout_parser is not None and llm_parser is not None:
        raise ValueError(
            "Only one of layout_parser or llm_parser may be passed in at a time"
        )
    corpus_name = _gapic_utils.get_corpus_name(corpus_name)
    request = _gapic_utils.prepare_import_files_request(
        corpus_name=corpus_name,
        paths=paths,
        source=source,
        transformation_config=transformation_config,
        max_embedding_requests_per_min=max_embedding_requests_per_min,
        import_result_sink=import_result_sink,
        partial_failures_sink=partial_failures_sink,
        layout_parser=layout_parser,
        llm_parser=llm_parser,
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
    from vertexai import rag
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
    from vertexai import rag
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


def add_inline_citations_and_references(
    original_text_str, grounding_supports, grounding_chunks
) -> RagCitedGenerationResponse:
    """Adds inline citations to a text string based on grounding_supports and grounding_chunks.

    Args:
        original_text_str (str): The text (as a Unicode string) to which citations
          will be added.
        grounding_supports (list): A list of objects, where each object represents
          a grounding support and has attributes: - segment: An object with
          'end_index' (byte offset relative to UTF-8). - grounding_chunk_indices:
          A list of integers.
        grounding_chunks (list): A list of objects, where each object is a source
          chunk wrapper. To get URI: obj.retrieved_context.uri. To get page
          span: obj.retrieved_context.rag_chunk.page_span.

    Returns:
        RagCitedGenerationResponse: An object containing the text with inline
                                   citations and a formatted bibliography string.

    Raises:
        TypeError: If original_text_str is not a string, or if grounding_supports,
                   grounding_chunks, or internally generated reference dictionaries
                   are of an unexpected type (raised by this function or helpers).
        ValueError: If original_text_str has encoding/decoding issues,
                    if segment data in grounding_supports is invalid,
                    if calculated insertion indices are out of bounds,
                    or if essential data within grounding_chunks (like the chunks
                    themselves, retrieved_context, or URI) is None when a value
                    is expected (raised by this function or helpers).
        IndexError: If chunk indices used for citation or bibliography generation
                    are out of bounds for the provided grounding_chunks (raised by helpers).
        AttributeError: If expected attributes (e.g., 'retrieved_context', 'uri')
                        are missing from the data structures within grounding_chunks
                        (raised by helpers).
      Example usage:
      ```
      import vertexai

      vertexai.init(project="my-project")

      rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=corpus_name,
                        # Optional: supply IDs from `rag.list_files()`.
                        # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                    )
                ],
                rag_retrieval_config=rag.RagRetrievalConfig(
                    top_k=10,
                    filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
                ),
            ),
        )
      )

    rag_model = GenerativeModel(
        model_name="gemini-2.5-pro-preview-05-06", tools=[rag_retrieval_tool]
    )
    response = rag_model.generate_content("Why is the sky blue?")

    rag_cited_generation_response = rag.add_inline_citations_and_references(
        original_text_str=response.candidates[0].content.parts[0].text,
        grounding_supports=list(response.candidates[0].grounding_metadata.grounding_supports),
        grounding_chunks=list(response.candidates[0].grounding_metadata.grounding_chunks),
    )
    print(rag_cited_generation_response.cited_text)
    print(rag_cited_generation_response.final_bibliography)

    """
    if not isinstance(original_text_str, str):
        raise TypeError("original_text_str must be a string.")
    if not isinstance(grounding_supports, list):
        raise TypeError("grounding_supports must be a list.")
    if not isinstance(grounding_chunks, list):
        raise TypeError("grounding_chunks must be a list.")

    cited_chunk_references = {}

    populate_cited_chunk_references(
        grounding_supports, grounding_chunks, cited_chunk_references
    )

    # If there are no grounding supports, return original text and empty bibliography.
    if not grounding_supports:
        return _gapic_utils.convert_tuple_to_rag_cited_generation_response(
            original_text_str, ""
        )

    try:
        original_text_bytes_equivalent = original_text_str.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(
            "Could not encode original_text_str to UTF-8 for index"
            f" conversion: {e}. Cannot process citations."
        ) from e

    insertions = []

    for support_idx, support in enumerate(grounding_supports):
        current_support_chunk_indices = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices is not None
        ):
            # Ensure indices are integers and unique for the citation string
            valid_indices = [
                idx for idx in support.grounding_chunk_indices if isinstance(idx, int)
            ]
            current_support_chunk_indices = sorted(list(set(valid_indices)))

        # Validate segment and end_index
        if not (
            hasattr(support, "segment")
            and support.segment
            and hasattr(support.segment, "end_index")
            and isinstance(support.segment.end_index, int)
            and support.segment.end_index >= 0  # Ensure end_index is not negative
        ):
            raise ValueError(f"Invalid segment data for support at index {support_idx}")

        byte_end_idx = support.segment.end_index

        if byte_end_idx > len(original_text_bytes_equivalent):
            byte_end_idx = len(original_text_bytes_equivalent)

        char_end_idx = -1  # Initialize before try block
        try:
            # Slice the byte string up to the (potentially clamped) byte_end_idx
            prefix_bytes = original_text_bytes_equivalent[:byte_end_idx]
            # Decode this prefix to find the corresponding character length
            char_end_idx = len(prefix_bytes.decode("utf-8"))
        except UnicodeDecodeError as e:
            # If decoding the prefix fails, we cannot reliably determine char_end_idx.
            raise ValueError(
                "Could not decode prefix of original_text_str (up to byte"
                f" {byte_end_idx}) from UTF-8 for support at index {support_idx}:"
                f" {e}. Cannot accurately place citation."
            ) from e
        except IndexError as e:  # Should be less common with slice clamping
            raise ValueError(
                "Index error during byte-to-char conversion for support at index"
                f" {support_idx}: {e}. This might indicate an issue with byte_end_idx"
                " calculation."
            ) from e

        if char_end_idx > len(original_text_str):
            char_end_idx = len(original_text_str)

        citation_str = "".join([f"[{idx}]" for idx in current_support_chunk_indices])

        if citation_str:
            insertions.append(
                {
                    "char_index": char_end_idx,
                    "citation": citation_str,
                    "original_byte_idx": byte_end_idx,
                }
            )

    # Sort insertions by character index in reverse order to insert from end to start,
    # preserving earlier indices.
    insertions.sort(
        key=lambda x: (x["char_index"], x["original_byte_idx"]), reverse=True
    )

    modified_text_list = list(original_text_str)
    for insertion_info in insertions:
        idx = insertion_info["char_index"]
        citation = insertion_info["citation"]
        if 0 <= idx <= len(modified_text_list):
            modified_text_list.insert(idx, citation)
        else:
            raise ValueError(
                f"Calculated insertion index {idx} is out of bounds for"
                f" modified_text_list (length {len(modified_text_list)}). Citation:"
                f" '{citation}'"
            )

    cited_text_output = "".join(modified_text_list)
    references_output_string = format_bibliography(
        cited_chunk_references, grounding_chunks
    )
    return _gapic_utils.convert_tuple_to_rag_cited_generation_response(
        cited_text_output, references_output_string
    )
