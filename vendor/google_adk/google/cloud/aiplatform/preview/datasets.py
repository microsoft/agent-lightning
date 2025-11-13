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

import dataclasses
import io
import json
from typing import Dict, List, Optional, Tuple
import uuid

from google.auth import credentials as auth_credentials
from google.cloud import storage
from google.cloud.aiplatform import base
from google.cloud.aiplatform import compat
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    dataset_v1beta1 as gca_dataset,
    dataset_service_v1beta1 as gca_dataset_service,
)
from vertexai import generative_models
from vertexai.generative_models import _generative_models
from vertexai.preview import prompts

from google.protobuf import field_mask_pb2
from google.protobuf import struct_pb2
from google.protobuf import json_format


_MULTIMODAL_METADATA_SCHEMA_URI = (
    "gs://google-cloud-aiplatform/schema/dataset/metadata/multimodal_1.0.0.yaml"
)
_DEFAULT_BQ_DATASET_PREFIX = "vertex_datasets"
_DEFAULT_BQ_TABLE_PREFIX = "multimodal_dataset"
_INPUT_CONFIG_FIELD = "inputConfig"
_BIGQUERY_SOURCE_FIELD = "bigquerySource"
_URI_FIELD = "uri"
_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD = "geminiTemplateConfigSource"
_GEMINI_TEMPLATE_CONFIG_FIELD = "geminiTemplateConfig"
_PROMPT_URI_FIELD = "promptUri"
_REQUEST_COLUMN_NAME_FIELD = "requestColumnName"
_BQ_MULTIREGIONS = {"us", "eu"}

_LOGGER = base.Logger(__name__)


def _try_import_bigframes():
    """Try to import `bigframes` and return it if successful - otherwise raise an import error."""
    try:
        import bigframes
        import bigframes.pandas
        import bigframes.bigquery

        return bigframes
    except ImportError as exc:
        raise ImportError(
            "`bigframes` is not installed but required for this functionality."
        ) from exc


def _get_metadata_for_bq(
    *,
    bq_uri: str,
    template_config: Optional[gca_dataset_service.GeminiTemplateConfig] = None,
    prompt_uri: Optional[str] = None,
    request_column_name: Optional[str] = None,
) -> struct_pb2.Value:
    if (
        sum(
            1
            for param in (template_config, prompt_uri, request_column_name)
            if param is not None
        )
        > 1
    ):
        raise ValueError(
            "Only one of template_config, prompt_uri, request_column_name can be specified."
        )

    input_config = {_INPUT_CONFIG_FIELD: {_BIGQUERY_SOURCE_FIELD: {_URI_FIELD: bq_uri}}}
    if template_config is not None:
        template_config_dict = gca_dataset_service.GeminiTemplateConfig.to_dict(
            template_config
        )
        input_config[_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD] = {
            _GEMINI_TEMPLATE_CONFIG_FIELD: template_config_dict
        }
    if prompt_uri is not None:
        input_config[_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD] = {
            _PROMPT_URI_FIELD: prompt_uri
        }
    if request_column_name is not None:
        input_config[_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD] = {
            _REQUEST_COLUMN_NAME_FIELD: request_column_name
        }
    return json_format.ParseDict(input_config, struct_pb2.Value())


def _bq_dataset_location_allowed(
    vertex_location: str, bq_dataset_location: str
) -> bool:
    if bq_dataset_location == vertex_location:
        return True
    if bq_dataset_location in _BQ_MULTIREGIONS:
        return vertex_location.startswith(bq_dataset_location)
    return False


def _normalize_and_validate_table_id(
    *,
    table_id: str,
    project: Optional[str] = None,
    vertex_location: Optional[str] = None,
    credentials: Optional[auth_credentials.Credentials] = None,
):
    from google.cloud import bigquery  # pylint: disable=g-import-not-at-top

    if not project:
        project = initializer.global_config.project
    if not vertex_location:
        vertex_location = initializer.global_config.location
    if not credentials:
        credentials = initializer.global_config.credentials

    table_ref = bigquery.TableReference.from_string(table_id, default_project=project)
    if table_ref.project != project:
        raise ValueError(
            f"The BigQuery table "
            f"`{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`"
            " must be in the same project as the multimodal dataset."
            f" The multimodal dataset is in `{project}`, but the BigQuery table"
            f" is in `{table_ref.project}`."
        )

    dataset_ref = bigquery.DatasetReference(
        project=table_ref.project, dataset_id=table_ref.dataset_id
    )
    client = bigquery.Client(project=project, credentials=credentials)
    bq_dataset = client.get_dataset(dataset_ref=dataset_ref)
    if not _bq_dataset_location_allowed(vertex_location, bq_dataset.location):
        raise ValueError(
            f"The BigQuery dataset"
            f" `{dataset_ref.project}.{dataset_ref.dataset_id}` must be in the"
            " same location as the multimodal dataset. The multimodal dataset"
            f" is in `{vertex_location}`, but the BigQuery dataset is in"
            f" `{bq_dataset.location}`."
        )
    return f"{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"


def _create_default_bigquery_dataset_if_not_exists(
    *,
    project: Optional[str] = None,
    location: Optional[str] = None,
    credentials: Optional[auth_credentials.Credentials] = None,
) -> str:
    # Loading bigquery lazily to avoid auto-loading it when importing vertexai
    from google.cloud import bigquery  # pylint: disable=g-import-not-at-top

    if not project:
        project = initializer.global_config.project
    if not location:
        location = initializer.global_config.location
    if not credentials:
        credentials = initializer.global_config.credentials

    bigquery_client = bigquery.Client(project=project, credentials=credentials)
    location_str = location.lower().replace("-", "_")
    dataset_id = bigquery.DatasetReference(
        project, f"{_DEFAULT_BQ_DATASET_PREFIX}_{location_str}"
    )
    dataset = bigquery.Dataset(dataset_ref=dataset_id)
    dataset.location = location
    bigquery_client.create_dataset(dataset, exists_ok=True)
    return f"{dataset_id.project}.{dataset_id.dataset_id}"


def _generate_target_table_id(dataset_id: str):
    return f"{dataset_id}.{_DEFAULT_BQ_TABLE_PREFIX}_{str(uuid.uuid4())}"


def construct_single_turn_template(
    *,
    prompt: str = None,
    response: Optional[str] = None,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    cached_content: Optional[str] = None,
    tools: Optional[List[generative_models.Tool]] = None,
    tool_config: Optional[generative_models.ToolConfig] = None,
    safety_settings: Optional[List[generative_models.SafetySetting]] = None,
    generation_config: Optional[generative_models.GenerationConfig] = None,
    field_mapping: List[Dict[str, str]] = None,
) -> "GeminiTemplateConfig":
    """Constructs a GeminiTemplateConfig object for single-turn cases.

    Example:
        template_config = dataset.construct_single_turn_template(
                prompt = "Which flower is this {flower_image} ?",
                response="This is a {label}.",
                system_instruction="You are a botanical classifier."
        )

    Args:

        prompt (str):
            Required. User input.
        response (str):
            Optional. Model response to user input.
        system_instruction (str):
            Optional. System instructions for the model.
        model (str):
            Optional. The model to use for the GeminiExample.
        cached_content (str):
            Optional. The cached content to use for the GeminiExample.
        tools (List[Tool]):
            Optional. The tools to use for the GeminiExample.
        tool_config (ToolConfig):
            Optional. The tool config to use for the GeminiExample.
        safety_settings (List[SafetySetting]):
            Optional. The safety settings to use for the GeminiExample.
        generation_config (GenerationConfig):
            Optional. The generation config to use for the GeminiExample.
        field_mapping (List[Dict[str, str]]):
            Optional. Mapping of placeholders to dataset columns.

    Returns:
        A GeminiTemplateConfig object.
    """
    contents = []
    contents.append(
        generative_models.Content(
            role="user",
            parts=[
                generative_models.Part.from_text(prompt),
            ],
        )
    )
    if response:
        contents.append(
            generative_models.Content(
                role="model",
                parts=[
                    generative_models.Part.from_text(response),
                ],
            )
        )
    if system_instruction:
        system_instruction = generative_models.Content(
            parts=[
                generative_models.Part.from_text(system_instruction),
            ],
        )

    # Set up GeminiExample.
    gemini_example = GeminiExample(
        model=model,
        contents=contents,
        system_instruction=system_instruction,
        cached_content=cached_content,
        tools=tools,
        tool_config=tool_config,
        safety_settings=safety_settings,
        generation_config=generation_config,
    )
    return GeminiTemplateConfig(
        gemini_example=gemini_example, field_mapping=field_mapping
    )


class GeminiExample:
    """A class representing a Gemini example."""

    Content = generative_models.Content
    Part = generative_models.Part
    Tool = generative_models.Tool
    ToolConfig = generative_models.ToolConfig
    SafetySetting = generative_models.SafetySetting
    GenerationConfig = generative_models.GenerationConfig

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        contents: Optional[List[Content]] = None,
        system_instruction: Optional[Content] = None,
        cached_content: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_config: Optional[ToolConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """Initializes a GeminiExample.

        Args:
            model (str):
                Optional. The model to use for the GeminiExample.
            contents (List[Content]):
                Optional. The contents to use for the GeminiExample.
            system_instruction (Content):
                Optional. The system instruction to use for the GeminiExample.
            cached_content (str):
                Optional. The cached content to use for the GeminiExample.
            tools (List[Tool]):
                Optional. The tools to use for the GeminiExample.
            tool_config (ToolConfig):
                Optional. The tool config to use for the GeminiExample.
            safety_settings (List[SafetySetting]):
                Optional. The safety settings to use for the GeminiExample.
            generation_config (GenerationConfig):
                Optional. The generation config to use for the GeminiExample.
        """
        self._raw_gemini_example = gca_dataset_service.GeminiExample()
        self.model = model
        self.contents = contents
        self.system_instruction = system_instruction
        self.cached_content = cached_content
        self.tools = tools
        self.tool_config = tool_config
        self.safety_settings = safety_settings
        self.generation_config = generation_config

    @property
    def model(self) -> Optional[str]:
        """The model to use for the GeminiExample."""
        if not self._raw_gemini_example.model:
            return None
        return self._raw_gemini_example.model

    @model.setter
    def model(self, model: str):
        """Setter for the model."""
        self._raw_gemini_example.model = model

    @property
    def contents(self) -> Optional[List[Content]]:
        """The contents of the GeminiExample."""
        if not self._raw_gemini_example.contents:
            return None
        return [
            generative_models.Content._from_gapic(content)
            for content in self._raw_gemini_example.contents
        ]

    @contents.setter
    def contents(self, contents: Optional[List[Content]]):
        """Setter for the contents."""
        if contents is None:
            self._raw_gemini_example.contents = None
        else:
            self._raw_gemini_example.contents = [
                content._raw_content for content in contents
            ]

    @property
    def system_instruction(self) -> Optional[Content]:
        """The system instruction of the GeminiExample."""
        if not self._raw_gemini_example.system_instruction:
            return None
        return generative_models.Content._from_gapic(
            self._raw_gemini_example.system_instruction
        )

    @system_instruction.setter
    def system_instruction(self, system_instruction: Optional[Content]):
        """Setter for the system instruction."""
        if system_instruction is None:
            self._raw_gemini_example.system_instruction = None
        else:
            self._raw_gemini_example.system_instruction = (
                system_instruction._raw_content
            )

    @property
    def cached_content(self) -> Optional[str]:
        """The cached content of the GeminiExample."""
        if not self._raw_gemini_example.cached_content:
            return None
        return self._raw_gemini_example.cached_content

    @cached_content.setter
    def cached_content(self, cached_content: Optional[str]):
        """Setter for the cached content."""
        self._raw_gemini_example.cached_content = cached_content

    @property
    def tools(self) -> Optional[List[Tool]]:
        """The tools of the GeminiExample."""
        if not self._raw_gemini_example.tools:
            return None
        return [
            generative_models.Tool._from_gapic(tool)
            for tool in self._raw_gemini_example.tools
        ]

    @tools.setter
    def tools(self, tools: Optional[List[Tool]]):
        """Setter for the tools."""
        if tools is None:
            self._raw_gemini_example.tools = None
        else:
            self._raw_gemini_example.tools = [tool._raw_tool for tool in tools]

    @property
    def tool_config(self) -> Optional[ToolConfig]:
        """The tool config of the GeminiExample."""
        if not self._raw_gemini_example.tool_config:
            return None
        return generative_models.ToolConfig._from_gapic(
            self._raw_gemini_example.tool_config
        )

    @tool_config.setter
    def tool_config(self, tool_config: Optional[ToolConfig]):
        """Setter for the tool config."""
        if tool_config is None:
            self._raw_gemini_example.tool_config = None
        else:
            self._raw_gemini_example.tool_config = tool_config._gapic_tool_config

    @property
    def safety_settings(self) -> Optional[List[SafetySetting]]:
        """The safety settings of the GeminiExample."""
        if not self._raw_gemini_example.safety_settings:
            return None
        return [
            generative_models.SafetySetting._from_gapic(safety_setting)
            for safety_setting in self._raw_gemini_example.safety_settings
        ]

    @safety_settings.setter
    def safety_settings(self, safety_settings: Optional[List[SafetySetting]]):
        """Setter for the safety settings."""
        if safety_settings is None:
            self._raw_gemini_example.safety_settings = None
        else:
            self._raw_gemini_example.safety_settings = [
                safety_setting._raw_safety_setting for safety_setting in safety_settings
            ]

    @property
    def generation_config(self) -> Optional[GenerationConfig]:
        """The generation config of the GeminiExample."""
        if not self._raw_gemini_example.generation_config:
            return None
        return generative_models.GenerationConfig._from_gapic(
            self._raw_gemini_example.generation_config
        )

    @generation_config.setter
    def generation_config(self, generation_config: Optional[GenerationConfig]):
        """Setter for the generation config."""
        if generation_config is None:
            self._raw_gemini_example.generation_config = None
        else:
            self._raw_gemini_example.generation_config = (
                generation_config._raw_generation_config
            )

    @classmethod
    def _from_gapic(
        cls, raw_gemini_example: gca_dataset_service.GeminiExample
    ) -> "GeminiExample":
        example = cls()
        example._raw_gemini_example = raw_gemini_example
        return example

    @classmethod
    def from_prompt(cls, prompt: prompts.Prompt) -> "GeminiExample":
        """Creates a GeminiExample from a Prompt.

        Args:
            prompt (prompts.Prompt):
                The Prompt to use for the GeminiExample.
        Returns:
            A GeminiExample created from the Prompt.
        """
        contents = prompt.assemble_contents()
        if prompt.system_instruction:
            system_instructions = generative_models.Content._from_gapic(
                _generative_models._to_content(prompt.system_instruction)
            )
        else:
            system_instructions = None
        # TODO(b/404208669): Prompt management SDK has a wrong type annotation
        # for safety_settings: It's annotated as SafetySetting, but the
        # validation assumes it's a sequence.
        if isinstance(prompt.safety_settings, generative_models.SafetySetting):
            safety_settings = [prompt.safety_settings]
        else:
            safety_settings = prompt.safety_settings

        return cls(
            model=prompt.model_name,
            contents=contents,
            system_instruction=system_instructions,
            tools=prompt.tools,
            tool_config=prompt.tool_config,
            safety_settings=safety_settings,
            generation_config=prompt.generation_config,
        )

    def __repr__(self) -> str:
        return self._raw_gemini_example.__repr__()


class GeminiTemplateConfig:
    """A class representing a Gemini template config.

    A Gemini template config contains a GeminiExample, which specifies the
    prompt including placeholders, and a field mapping, which specifies how to
    map the placeholders to the corresponding column in the BigQuery table of
    the dataset. If no field mapping is provided, the default behavior is to
    use the placeholder name as the column name.
    """

    def __init__(
        self,
        *,
        gemini_example: Optional[GeminiExample] = None,
        field_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initializes a GeminiTemplateConfig.

        Args:
            gemini_example (GeminiExample):
                Optional. The GeminiExample to use for the template config. If
                not provided, a default GeminiExample will be used.
            field_mapping (Dict[str, str]):
                Optional. The field mapping to use for the template config. If
                not provided, all placeholders in the GeminiExample will be
                mapped to the corresponding column in the BigQuery table of the
                dataset.
        """
        raw_gemini_example = (
            gemini_example._raw_gemini_example if gemini_example is not None else None
        )
        self._raw_gemini_template_config = gca_dataset_service.GeminiTemplateConfig(
            gemini_example=raw_gemini_example, field_mapping=field_mapping
        )

    @classmethod
    def _from_gapic(
        cls, raw_gemini_template_config: gca_dataset_service.GeminiTemplateConfig
    ) -> None:
        template_config = cls()
        template_config._raw_gemini_template_config = raw_gemini_template_config
        return template_config

    @property
    def gemini_example(self) -> Optional[GeminiExample]:
        """The GeminiExample of this template config."""
        return GeminiExample._from_gapic(
            self._raw_gemini_template_config.gemini_example
        )

    @property
    def field_mapping(self) -> Optional[Dict[str, str]]:
        """The field mapping of this template config."""
        return dict(self._raw_gemini_template_config.field_mapping)

    def __repr__(self) -> str:
        return self._raw_gemini_template_config.__repr__()


@dataclasses.dataclass(frozen=True)
class TuningResourceUsageAssessmentResult:
    """The result of a tuning resource usage assessment.

    Attributes:
        token_count (int):
            The number of tokens in the dataset.
        billable_character_count (int):
            The number of billable characters in the dataset.
    """

    token_count: int
    billable_character_count: int


@dataclasses.dataclass(frozen=True)
class TuningValidationAssessmentResult:
    """The result of a tuning validation assessment.

    Attributes:
        errors (List[str]):
            The list of errors found in the dataset.
    """

    errors: List[str]


@dataclasses.dataclass(frozen=True)
class BatchPredictionResourceUsageAssessmentResult:
    """The result of a batch prediction resource usage assessment.

    Attributes:
        token_count (int):
            Number of tokens in the dataset.
        audio_token_count (int):
            Number of audio tokens in the dataset.
    """

    token_count: int
    audio_token_count: int


class MultimodalDataset(base.VertexAiResourceNounWithFutureManager):
    """A class representing a unified multimodal dataset."""

    client_class = utils.DatasetClientWithOverride
    _resource_noun = "datasets"
    _getter_method = "get_dataset"
    _list_method = "list_datasets"
    _delete_method = "delete_dataset"
    _parse_resource_name_method = "parse_dataset_path"
    _format_resource_name_method = "dataset_path"
    _DEFAULT_REQUEST_COLUMN_NAME = "requests"

    def __init__(
        self,
        *,
        dataset_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves an existing multimodal dataset given a resource name.

        Args:
            dataset_name (str):
                Required. A fully-qualified dataset resource name or dataset ID.
                Example: "projects/123/locations/us-central1/datasets/456" or
                "456" when project and location are initialized or passed.
            project (str):
                Optional project to retrieve dataset from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional location to retrieve dataset from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Custom credentials to use to retrieve this Dataset. Overrides
                credentials set in aiplatform.init.
        """
        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=dataset_name,
        )
        self.api_client = self.api_client.select_version(compat.V1BETA1)
        self._gca_resource = self._get_gca_resource(resource_name=dataset_name)
        self._validate_metadata_schema_uri()

    @property
    def metadata_schema_uri(self) -> str:
        """The metadata schema uri of this dataset resource."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.metadata_schema_uri

    def _validate_metadata_schema_uri(self):
        if self.metadata_schema_uri != _MULTIMODAL_METADATA_SCHEMA_URI:

            raise ValueError(
                f"Dataset {self.resource_name} is not a multimodal dataset"
            )

    @property
    def bigquery_table(self) -> str:
        """The BigQuery table of this dataset resource, such as
        "bq://project.dataset.table"."""
        self._assert_gca_resource_is_available()
        return self._gca_resource.metadata[_INPUT_CONFIG_FIELD][_BIGQUERY_SOURCE_FIELD][
            _URI_FIELD
        ]

    @classmethod
    def from_bigquery(
        cls,
        *,
        bigquery_source: str,
        display_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "MultimodalDataset":
        """Creates a multimodal dataset from a BigQuery table.

        Args:
            bigquery_source (str):
                Required. The BigQuery table URI or ID to be used for the created
                dataset, which can be in the format of "bq://dataset.table",
                "bq://project.dataset.table" or "project.dataset.table".
            display_name (str):
                Optional. The user-defined name of the dataset. The name can be
                up to 128 characters long and can consist of any UTF-8
                characters.
            project (str):
                Optional. Project to create this dataset in. Overrides project
                set in aiplatform.init.
            location (str):
                Optional. Location to create this dataset in. Overrides location
                set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this dataset.
                Overrides credentials set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to organize your
                datasets. Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. See https://goo.gl/xmQnxf
                for more information on and examples of labels. No more than 64
                user labels can be associated with one dataset (System labels
                are excluded). System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            sync (bool):
                Optional. Whether to execute this method synchronously. If
                False, this method will be executed in concurrent Future and any
                downstream object will be immediately returned and synced when
                the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the dataset creation request.

        Returns:
            dataset (MultimodalDataset):
                The created multimodal dataset.
        """
        if not bigquery_source.startswith("bq://"):
            bigquery_source = f"bq://{bigquery_source}"
        return cls._create_from_bigquery(
            metadata=_get_metadata_for_bq(bq_uri=bigquery_source),
            display_name=display_name,
            project=project,
            location=location,
            credentials=credentials,
            labels=labels,
            sync=sync,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    def from_pandas(
        cls,
        *,
        dataframe: "pandas.DataFrame",  # type: ignore # noqa: F821
        target_table_id: Optional[str] = None,
        display_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "MultimodalDataset":
        """Creates a multimodal dataset from a pandas dataframe.

        Args:
            dataframe (pandas.DataFrame):
                The pandas dataframe to be used for the created dataset.
            target_table_id (str):
                Optional. The BigQuery table id where the dataframe will be
                uploaded. The table id can be in the format of "dataset.table"
                or "project.dataset.table". If a table already exists with the
                given table id, it will be overwritten. Note that the BigQuery
                dataset must already exist and be in the same location as the
                multimodal dataset. If not provided, a generated table id will
                be created in the `vertex_datasets` dataset (e.g.
                `project.vertex_datasets_us_central1.multimodal_dataset_4cbf7ffd`).
            display_name (str):
                Optional. The user-defined name of the dataset. The name can be
                up to 128 characters long and can consist of any UTF-8
                characters.
            project (str):
                Optional. Project to create this dataset in. Overrides project
                set in aiplatform.init.
            location (str):
                Optional. Location to create this dataset in. Overrides location
                set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this dataset.
                Overrides credentials set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to organize your
                datasets. Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. See https://goo.gl/xmQnxf
                for more information on and examples of labels. No more than 64
                user labels can be associated with one dataset (System labels
                are excluded). System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            sync (bool):
                Optional. Whether to execute this method synchronously. If
                False, this method will be executed in concurrent Future and any
                downstream object will be immediately returned and synced when
                the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the dataset creation request.

        Returns:
            dataset (MultimodalDataset):
                The created multimodal dataset.
        """
        bigframes = _try_import_bigframes()
        from google.cloud import bigquery  # pylint: disable=g-import-not-at-top

        if not project:
            project = initializer.global_config.project
        if not location:
            location = initializer.global_config.location
        if not credentials:
            credentials = initializer.global_config.credentials

        if target_table_id:
            target_table_id = _normalize_and_validate_table_id(
                table_id=target_table_id,
                project=project,
                vertex_location=location,
                credentials=credentials,
            )
        else:
            dataset_id = _create_default_bigquery_dataset_if_not_exists(
                project=project, location=location, credentials=credentials
            )
            target_table_id = _generate_target_table_id(dataset_id)

        session_options = bigframes.BigQueryOptions(
            credentials=credentials,
            project=project,
            location=location,
        )
        with bigframes.connect(session_options) as session:
            temp_bigframes_df = session.read_pandas(dataframe)
            temp_table_id = temp_bigframes_df.to_gbq()
        client = bigquery.Client(project=project, credentials=credentials)
        copy_job = client.copy_table(
            sources=temp_table_id,
            destination=target_table_id,
        )
        copy_job.result()

        bigquery_uri = f"bq://{target_table_id}"
        return cls._create_from_bigquery(
            metadata=_get_metadata_for_bq(bq_uri=bigquery_uri),
            display_name=display_name,
            project=project,
            location=location,
            credentials=credentials,
            labels=labels,
            sync=sync,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    def from_bigframes(
        cls,
        *,
        dataframe: "bigframes.pandas.DataFrame",  # type: ignore # noqa: F821
        target_table_id: Optional[str] = None,
        display_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "MultimodalDataset":
        """Creates a multimodal dataset from a bigframes dataframe.

        Args:
            dataframe (bigframes.pandas.DataFrame):
                The BigFrames dataframe that will be used for the created
                dataset.
            target_table_id (str):
                Optional. The BigQuery table id where the dataframe will be
                uploaded. The table id can be in the format of "dataset.table"
                or "project.dataset.table". If a table already exists with the
                given table id, it will be overwritten. Note that the BigQuery
                dataset must already exist and be in the same location as the
                multimodal dataset. If not provided, a generated table id will
                be created in the `vertex_datasets` dataset (e.g.
                `project.vertex_datasets_us_central1.multimodal_dataset_4cbf7ffd`).
            display_name (str):
                Optional. The user-defined name of the dataset. The name can be
                up to 128 characters long and can consist of any UTF-8
                characters.
            project (str):
                Optional. Project to create this dataset in. Overrides project
                set in aiplatform.init.
            location (str):
                Optional. Location to create this dataset in. Overrides location
                set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this dataset.
                Overrides credentials set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to organize your
                datasets. Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. See https://goo.gl/xmQnxf
                for more information on and examples of labels. No more than 64
                user labels can be associated with one dataset (System labels
                are excluded). System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            sync (bool):
                Optional. Whether to execute this method synchronously. If
                False, this method will be executed in concurrent Future and any
                downstream object will be immediately returned and synced when
                the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the dataset creation request.

        Returns:
            The created multimodal dataset.
        """
        from google.cloud import bigquery  # pylint: disable=g-import-not-at-top

        if target_table_id:
            target_table_id = _normalize_and_validate_table_id(
                table_id=target_table_id,
                project=project,
                vertex_location=location,
                credentials=credentials,
            )
        else:
            dataset_id = _create_default_bigquery_dataset_if_not_exists(
                project=project, location=location, credentials=credentials
            )
            target_table_id = _generate_target_table_id(dataset_id)

        if not project:
            project = initializer.global_config.project

        temp_table_id = dataframe.to_gbq()
        client = bigquery.Client(project=project, credentials=credentials)
        copy_job = client.copy_table(
            sources=temp_table_id,
            destination=target_table_id,
        )
        copy_job.result()

        bigquery_uri = f"bq://{target_table_id}"
        return cls._create_from_bigquery(
            metadata=_get_metadata_for_bq(bq_uri=bigquery_uri),
            display_name=display_name,
            project=project,
            location=location,
            credentials=credentials,
            labels=labels,
            sync=sync,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    def from_gemini_request_jsonl(
        cls,
        *,
        gcs_uri: str,
        target_table_id: Optional[str] = None,
        display_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "MultimodalDataset":
        """Creates a multimodal dataset from a JSONL file stored on GCS.

        The JSONL file should contain a instances of Gemini
        `GenerateContentRequest` on each line. The data will be stored in a
        BigQuery table with a single column called "requests". The
        request_column_name in the dataset metadata will be set to "requests".

        Args:
            gcs_uri (str):
                The Google Cloud Storage URI of the JSONL file to import.
                For example, 'gs://my-bucket/path/to/data.jsonl'
            target_table_id (str):
                Optional. The BigQuery table id where the dataframe will be
                uploaded. The table id can be in the format of "dataset.table"
                or "project.dataset.table". If a table already exists with the
                given table id, it will be overwritten. Note that the BigQuery
                dataset must already exist and be in the same location as the
                multimodal dataset. If not provided, a generated table id will
                be created in the `vertex_datasets` dataset (e.g.
                `project.vertex_datasets_us_central1.multimodal_dataset_4cbf7ffd`).
            display_name (str):
                Optional. The user-defined name of the dataset. The name can be
                up to 128 characters long and can consist of any UTF-8
                characters.
            project (str):
                Optional. Project to create this dataset in. Overrides project
                set in aiplatform.init.
            location (str):
                Optional. Location to create this dataset in. Overrides location
                set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this dataset.
                Overrides credentials set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to organize your
                datasets. Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. See https://goo.gl/xmQnxf
                for more information on and examples of labels. No more than 64
                user labels can be associated with one dataset (System labels
                are excluded). System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            sync (bool):
                Optional. Whether to execute this method synchronously. If
                False, this method will be executed in concurrent Future and any
                downstream object will be immediately returned and synced when
                the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the dataset creation request.

        Returns:
            The created multimodal dataset.
        """
        bigframes = _try_import_bigframes()
        from google.cloud import bigquery  # pylint: disable=g-import-not-at-top

        if not project:
            project = initializer.global_config.project
        if not location:
            location = initializer.global_config.location
        if not credentials:
            credentials = initializer.global_config.credentials

        if target_table_id:
            target_table_id = _normalize_and_validate_table_id(
                table_id=target_table_id,
                project=project,
                vertex_location=location,
                credentials=credentials,
            )
        else:
            dataset_id = _create_default_bigquery_dataset_if_not_exists(
                project=project, location=location, credentials=credentials
            )
            target_table_id = _generate_target_table_id(dataset_id)

        gcs_uri_prefix = "gs://"
        if gcs_uri.startswith(gcs_uri_prefix):
            gcs_uri = gcs_uri[len(gcs_uri_prefix) :]
        parts = gcs_uri.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                "Invalid GCS URI format. Expected: gs://bucket-name/object-path"
            )
        bucket_name = parts[0]
        blob_name = parts[1]

        storage_client = storage.Client(project=project)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        request_column_name = cls._DEFAULT_REQUEST_COLUMN_NAME

        jsonl_string = blob.download_as_text()
        lines = [line.strip() for line in jsonl_string.splitlines() if line.strip()]
        json_string = json.dumps({request_column_name: lines})

        session_options = bigframes.BigQueryOptions(
            credentials=credentials,
            project=project,
            location=location,
        )
        with bigframes.connect(session_options) as session:
            temp_bigframes_df = session.read_json(io.StringIO(json_string))
            temp_bigframes_df[request_column_name] = bigframes.bigquery.parse_json(
                temp_bigframes_df[request_column_name]
            )
            temp_table_id = temp_bigframes_df.to_gbq()
        client = bigquery.Client(project=project, credentials=credentials)
        copy_job = client.copy_table(
            sources=temp_table_id,
            destination=target_table_id,
        )
        copy_job.result()

        bigquery_uri = f"bq://{target_table_id}"
        return cls._create_from_bigquery(
            metadata=_get_metadata_for_bq(
                bq_uri=bigquery_uri, request_column_name=request_column_name
            ),
            display_name=display_name,
            project=project,
            location=location,
            credentials=credentials,
            labels=labels,
            sync=sync,
            create_request_timeout=create_request_timeout,
        )

    def to_bigframes(self) -> "bigframes.pandas.DataFrame":  # type: ignore # noqa: F821
        """Converts a multimodal dataset to a BigFrames dataframe.

        This is the preferred method to inspect the multimodal dataset in a
        notebook.

        Returns:
            A BigFrames dataframe.
        """
        bigframes = _try_import_bigframes()
        return bigframes.pandas.read_gbq_table(self.bigquery_table.lstrip("bq://"))

    @classmethod
    @base.optional_sync()
    def _create_from_bigquery(
        cls,
        *,
        metadata: struct_pb2.Value,
        display_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
    ) -> "MultimodalDataset":
        if not display_name:
            display_name = cls._generate_display_name()
        utils.validate_display_name(display_name)
        if labels:
            utils.validate_labels(labels)
        if not project:
            project = initializer.global_config.project
        if not location:
            location = initializer.global_config.location
        if not credentials:
            credentials = initializer.global_config.credentials

        dataset = gca_dataset.Dataset(
            display_name=display_name,
            metadata_schema_uri=_MULTIMODAL_METADATA_SCHEMA_URI,
            metadata=metadata,
            labels=labels,
        )
        parent = initializer.global_config.common_location_path(
            project=project, location=location
        )
        api_client = cls._instantiate_client(
            location=location, credentials=credentials
        ).select_version(compat.V1BETA1)
        create_lro = api_client.create_dataset(
            dataset=dataset, parent=parent, timeout=create_request_timeout
        )
        _LOGGER.log_create_with_lro(cls, create_lro)
        created_dataset = create_lro.result(timeout=None)
        _LOGGER.log_create_complete(cls, created_dataset, "ds")
        return cls(dataset_name=created_dataset.name)

    def update(
        self,
        *,
        display_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        update_request_timeout: Optional[float] = None,
    ):
        """Update the dataset.

        Updatable fields:
            -  ``display_name``
            -  ``labels``
            -  ``description``

        Args:
            display_name (str):
                Optional. The user-defined name of the Dataset. The name can be
                up to 128 characters long and can be consist of any UTF-8
                characters.
            labels (Dict[str, str]):
                Optional. Labels with user-defined metadata to organize your
                datasets. Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. No more than 64 user
                labels can be associated with one dataset (System labels are
                excluded). See https://goo.gl/xmQnxf for more information and
                examples of labels. System reserved label keys are prefixed with
                "aiplatform.googleapis.com/" and are immutable.
            description (str):
                Optional. The description of the Dataset.
            update_request_timeout (float):
                Optional. The timeout for the update request in seconds.

        Returns:
            dataset (MultimodalDataset):
                Updated dataset.
        """
        update_mask = field_mask_pb2.FieldMask()
        if display_name:
            update_mask.paths.append("display_name")

        if labels:
            update_mask.paths.append("labels")

        if description:
            update_mask.paths.append("description")

        update_dataset = gca_dataset.Dataset(
            name=self.resource_name,
            display_name=display_name,
            description=description,
            labels=labels,
        )

        self._gca_resource = self.api_client.update_dataset(
            dataset=update_dataset,
            update_mask=update_mask,
            timeout=update_request_timeout,
        )

        return self

    def attach_template_config(
        self,
        *,
        template_config: Optional[GeminiTemplateConfig] = None,
        prompt: Optional[prompts.Prompt] = None,
        update_request_timeout: Optional[float] = None,
    ):
        """Attach a template config or prompt to the dataset.

        Args:
            template_config (GeminiTemplateConfig):
                Optional. The template config to attach to the dataset.
            prompt (prompts.Prompt):
                Optional. The prompt to attach to the dataset.
            update_request_timeout (float):
                Optional. The timeout for the update request in seconds.

        Returns:
            MultimodalDataset - The updated dataset.
        """
        if not (template_config or prompt):
            raise ValueError("Either template_config or prompt must be provided.")
        if template_config and prompt:
            raise ValueError("Only one of template_config or prompt can be provided.")

        raw_template_config = None
        if template_config:
            raw_template_config = template_config._raw_gemini_template_config
        prompt_uri = None
        if prompt:
            if prompt.prompt_id:
                saved_prompt = prompt
            else:
                saved_prompt = prompts.create_version(prompt)
            location = initializer.global_config.location
            project = initializer.global_config.project
            # TODO(b/404208669): Support prompt versions.
            prompt_uri = f"projects/{project}/locations/{location}/datasets/{saved_prompt.prompt_id}"

        update_mask = field_mask_pb2.FieldMask(paths=["metadata"])
        update_dataset = gca_dataset.Dataset(
            name=self.resource_name,
            metadata=_get_metadata_for_bq(
                bq_uri=self.bigquery_table,
                template_config=raw_template_config,
                prompt_uri=prompt_uri,
            ),
        )
        self._gca_resource = self.api_client.update_dataset(
            dataset=update_dataset,
            update_mask=update_mask,
            timeout=update_request_timeout,
        )
        return self

    def has_template_config(self) -> bool:
        """Returns true if the dataset has a template config attached."""
        self._assert_gca_resource_is_available()
        return _GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD in self._gca_resource.metadata

    @property
    def template_config(self) -> Optional[GeminiTemplateConfig]:
        """Return a copy of the template config attached to this dataset."""
        self._assert_gca_resource_is_available()
        # Dataset has no attached template.
        if _GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD not in self._gca_resource.metadata:
            return None
        # Dataset has a template attached as a GeminiTemplateConfig.
        if (
            _GEMINI_TEMPLATE_CONFIG_FIELD
            in self._gca_resource.metadata[_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD]
        ):
            struct_proto_container = self._gca_resource.metadata[
                _GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD
            ].pb
            struct_proto = struct_proto_container.get(_GEMINI_TEMPLATE_CONFIG_FIELD)
            # Detour via json (instead of dict) to avoid field renaming mismatches.
            gapic_template_config = gca_dataset_service.GeminiTemplateConfig.from_json(
                json_format.MessageToJson(struct_proto)
            )
            return GeminiTemplateConfig._from_gapic(gapic_template_config)
        # Dataset has a template attached as a Prompt resource URI.
        if (
            _PROMPT_URI_FIELD
            in self._gca_resource.metadata[_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD]
        ):
            prompt_uri = (
                self._gca_resource.metadata[_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD]
                .pb.get(_PROMPT_URI_FIELD)
                .string_value
            )
            resource_name_prefix = f"projects/{initializer.global_config.project}/locations/{initializer.global_config.location}/datasets/"
            if not prompt_uri.startswith(resource_name_prefix):
                prompt_location = prompt_uri.split("/")[3]
                prompt_project = prompt_uri.split("/")[1]
                raise ValueError(
                    "Attached prompt is not in the currently configured global "
                    "project and/or location. (Configured project/location: "
                    f"{initializer.global_config.location}, "
                    f"{initializer.global_config.project}; Attached prompt: "
                    f"{prompt_location}, {prompt_project})"
                )
            prompt_id = prompt_uri
            if prompt_id.startswith(resource_name_prefix):
                prompt_id = prompt_id[len(resource_name_prefix) :]
            prompt = prompts.get(prompt_id)
            return GeminiTemplateConfig(
                gemini_example=GeminiExample.from_prompt(prompt), field_mapping={}
            )

        return None

    @property
    def request_column_name(self) -> Optional[str]:
        """Return the request column name if it is set in the dataset metadata.

        The request column name specifies a column in the dataset that contains
        assembled Gemini `GenerateContentRequest` instances.
        """

        self._assert_gca_resource_is_available()
        # Dataset has no attached template.
        if _GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD not in self._gca_resource.metadata:
            return None
        if (
            _REQUEST_COLUMN_NAME_FIELD
            not in self._gca_resource.metadata[_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD]
        ):
            return None
        return self._gca_resource.metadata[_GEMINI_TEMPLATE_CONFIG_SOURCE_FIELD][
            _REQUEST_COLUMN_NAME_FIELD
        ]

    def assemble(
        self,
        *,
        template_config: Optional[GeminiTemplateConfig] = None,
        load_dataframe: bool = True,
        assemble_request_timeout: Optional[float] = None,
    ) -> Tuple[str, "bigframes.pandas.DataFrame"]:  # type: ignore # noqa: F821
        """Assemble the dataset into a BigQuery table.

        Args:
            template_config (GeminiTemplateConfig):
                Optional. The template config to use to assemble the dataset.
                If not provided, the template config attached to the dataset
                will be used.
            load_dataframe (bool):
                Optional. Whether to load the assembled dataset into a BigFrames
                DataFrame.
            assemble_request_timeout (float):
                Optional. The timeout for the assemble request in seconds.

        Returns:
            Tuple[str, "bigframes.pandas.DataFrame"]:
                A tuple containing the ID of the assembled dataset in BigQuery
                and the assembled dataset as a BigFrames DataFrame if
                load_dataframe is True, otherwise None.
        """
        bigframes = _try_import_bigframes()
        request = gca_dataset_service.AssembleDataRequest(
            name=self.resource_name,
            gemini_request_read_config=self._build_gemini_request_read_config(
                template_config
            ),
        )

        assemble_lro = self.api_client.assemble_data(
            request=request, timeout=assemble_request_timeout
        )
        _LOGGER.log_action_started_against_resource_with_lro(
            "Assemble", "data", self.__class__, assemble_lro
        )
        result = assemble_lro.result(timeout=None)
        _LOGGER.log_action_completed_against_resource("data", "assembled", self)
        table_id = result.bigquery_destination.lstrip("bq://")
        if load_dataframe:
            session_options = bigframes.BigQueryOptions(
                credentials=initializer.global_config.credentials,
                project=initializer.global_config.project,
                location=initializer.global_config.location,
            )
            with bigframes.connect(session_options) as session:
                df = session.read_gbq(table_id)
        else:
            df = None

        return (table_id, df)

    def assess_tuning_resources(
        self,
        *,
        model_name: str,
        template_config: Optional[GeminiTemplateConfig] = None,
        assess_request_timeout: Optional[float] = None,
    ) -> TuningResourceUsageAssessmentResult:
        """Assess the tuning resources required for a given model.

        Args:
            model_name (str):
                Required. The name of the model to assess the tuning resources
                for.
            template_config (GeminiTemplateConfig):
                Optional. The template config used to assemble the dataset
                before assessing the tuning resources. If not provided, the
                template config attached to the dataset will be used. Required
                if no template config is attached to the dataset.
            assess_request_timeout (float):
                Optional. The timeout for the assess tuning resources request.
        Returns:
            A dict containing the tuning resource usage assessment result. The
            dict contains the following keys:
            - token_count: The number of tokens in the dataset.
            - billable_character_count: The number of billable characters in the
              dataset.

        """
        request = self._build_assess_data_request(template_config)
        request.tuning_resource_usage_assessment_config = (
            gca_dataset_service.AssessDataRequest.TuningResourceUsageAssessmentConfig(
                model_name=model_name
            )
        )

        assessment_result = (
            self.api_client.assess_data(request=request, timeout=assess_request_timeout)
            .result(timeout=None)
            .tuning_resource_usage_assessment_result
        )
        return TuningResourceUsageAssessmentResult(
            token_count=assessment_result.token_count,
            billable_character_count=assessment_result.billable_character_count,
        )

    def assess_tuning_validity(
        self,
        *,
        model_name: str,
        dataset_usage: str,
        template_config: Optional[GeminiTemplateConfig] = None,
        assess_request_timeout: Optional[float] = None,
    ) -> TuningValidationAssessmentResult:
        """Assess if the assembled dataset is valid in terms of tuning a given
        model.

        Args:
            model_name (str):
                Required. The name of the model to assess the tuning validity
                for.
            dataset_usage (str):
                Required. The dataset usage to assess the tuning validity for.
                Must be one of the following: SFT_TRAINING, SFT_VALIDATION.
            template_config (GeminiTemplateConfig):
                Optional. The template config used to assemble the dataset
                before assessing the tuning validity. If not provided, the
                template config attached to the dataset will be used. Required
                if no template config is attached to the dataset.
            assess_request_timeout (float):
                Optional. The timeout for the assess tuning validity request.
        Returns:
            A dict containing the tuning validity assessment result. The dict
            contains the following keys:
            - errors: A list of errors that occurred during the tuning validity
              assessment.
        """
        DatasetUsage = (
            gca_dataset_service.AssessDataRequest.TuningValidationAssessmentConfig.DatasetUsage
        )
        try:
            dataset_usage_enum = DatasetUsage[dataset_usage]
        except KeyError as e:
            valid_dataset_usage_names = [
                e.name for e in DatasetUsage if e.name != "DATASET_USAGE_UNSPECIFIED"
            ]
            raise ValueError(
                f"Argument 'dataset_usage' must be one of the following: "
                f"{', '.join(valid_dataset_usage_names)}."
            ) from e
        if dataset_usage_enum == DatasetUsage.DATASET_USAGE_UNSPECIFIED:
            raise ValueError("Dataset usage must be specified.")

        request = self._build_assess_data_request(template_config)
        request.tuning_validation_assessment_config = (
            gca_dataset_service.AssessDataRequest.TuningValidationAssessmentConfig(
                model_name=model_name,
                dataset_usage=dataset_usage_enum,
            )
        )
        assess_lro = self.api_client.assess_data(
            request=request, timeout=assess_request_timeout
        )
        assessment_result = assess_lro.result(timeout=None)
        return TuningValidationAssessmentResult(
            errors=assessment_result.tuning_validation_assessment_result.errors
        )

    def assess_batch_prediction_resources(
        self,
        *,
        model_name: str,
        template_config: Optional[GeminiTemplateConfig] = None,
        assess_request_timeout: Optional[float] = None,
    ) -> BatchPredictionResourceUsageAssessmentResult:
        """Assess the batch prediction resources required for a given model.

        Args:
            model_name (str):
                Required. The name of the model to assess the batch prediction resources
                for.
            template_config (GeminiTemplateConfig):
                Optional. The template config used to assemble the dataset
                before assessing the batch prediction resources. If not provided, the
                template config attached to the dataset will be used. Required
                if no template config is attached to the dataset.
            assess_request_timeout (float):
                Optional. The timeout for the assess batch prediction resources request.
        Returns:
            A dict containing the batch prediction resource usage assessment result. The
            dict contains the following keys:
            - token_count: The number of tokens in the dataset.
            - audio_token_count: The number of audio tokens in the dataset.

        """
        request = self._build_assess_data_request(template_config)
        request.batch_prediction_resource_usage_assessment_config = gca_dataset_service.AssessDataRequest.BatchPredictionResourceUsageAssessmentConfig(
            model_name=model_name
        )

        assessment_result = (
            self.api_client.assess_data(request=request, timeout=assess_request_timeout)
            .result(timeout=None)
            .batch_prediction_resource_usage_assessment_result
        )
        return BatchPredictionResourceUsageAssessmentResult(
            token_count=assessment_result.token_count,
            audio_token_count=assessment_result.audio_token_count,
        )

    def assess_batch_prediction_validity(
        self,
        *,
        model_name: str,
        template_config: Optional[GeminiTemplateConfig] = None,
        assess_request_timeout: Optional[float] = None,
    ) -> None:
        """Assess if the assembled dataset is valid in terms of batch prediction
        for a given model. Raises an error if the dataset is invalid, otherwise
        returns None.

        Args:
            model_name (str):
                Required. The name of the model to assess the batch prediction
                validity for.
            dataset_usage (str):
                Required. The dataset usage to assess the batch prediction
                validity for.
                Must be one of the following: SFT_TRAINING, SFT_VALIDATION.
            template_config (GeminiTemplateConfig):
                Optional. The template config used to assemble the dataset
                before assessing the batch prediction validity. If not provided, the
                template config attached to the dataset will be used. Required
                if no template config is attached to the dataset.
            assess_request_timeout (float):
                Optional. The timeout for the assess batch prediction validity request.
        """
        request = self._build_assess_data_request(template_config)
        request.batch_prediction_validation_assessment_config = gca_dataset_service.AssessDataRequest.BatchPredictionValidationAssessmentConfig(
            model_name=model_name,
        )
        assess_lro = self.api_client.assess_data(
            request=request, timeout=assess_request_timeout
        )
        assess_lro.result(timeout=None)

    def _build_assess_data_request(
        self,
        template_config: Optional[GeminiTemplateConfig] = None,
    ):
        return gca_dataset_service.AssessDataRequest(
            name=self.resource_name,
            gemini_request_read_config=self._build_gemini_request_read_config(
                template_config
            ),
        )

    def _build_gemini_request_read_config(
        self, provided_template_config: Optional[GeminiTemplateConfig] = None
    ) -> gca_dataset_service.GeminiRequestReadConfig:
        """Returns the provided template config wrapped in a read config if it
        is not None, otherwise returns the read config attached to the
        dataset."""
        if provided_template_config is not None:
            return gca_dataset_service.GeminiRequestReadConfig(
                template_config=provided_template_config._raw_gemini_template_config
            )
        elif self.template_config is not None:
            return gca_dataset_service.GeminiRequestReadConfig(
                template_config=self.template_config._raw_gemini_template_config
            )
        elif self.request_column_name is not None:
            return gca_dataset_service.GeminiRequestReadConfig(
                assembled_request_column_name=self.request_column_name
            )
        else:
            raise ValueError(
                "No template config was provided and no template config is attached to the dataset."
            )
