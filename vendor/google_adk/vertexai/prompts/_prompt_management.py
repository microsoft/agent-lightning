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

from __future__ import annotations

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer as aiplatform_initializer
from google.cloud.aiplatform.compat.types import dataset as gca_dataset
from google.cloud.aiplatform_v1.types import (
    dataset_version as gca_dataset_version,
)
from google.cloud.aiplatform_v1beta1.types import (
    prediction_service as gapic_prediction_service_types,
)
from vertexai.generative_models import (
    Part,
    Image,
    GenerativeModel,
    Tool,
    ToolConfig,
)
from vertexai.generative_models._generative_models import (
    _proto_to_dict,
    _dict_to_proto,
    _tool_types_to_gapic_tools,
    PartsType,
)
from vertexai.prompts._prompts import Prompt
from google.protobuf import field_mask_pb2 as field_mask

import dataclasses
from typing import (
    Any,
    Dict,
    Optional,
)

_LOGGER = base.Logger(__name__)
_dataset_client_value = None

DEFAULT_API_SCHEMA_VERSION = "1.0.0"
PROMPT_SCHEMA_URI = (
    "gs://google-cloud-aiplatform/schema/dataset/metadata/text_prompt_1.0.0.yaml"
)


def _format_function_declaration_parameters(obj: Any):
    """Recursively replaces type_ and format_ fields in-place."""
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, dict):
        new = obj.__class__()
        for key, value in obj.items():
            key = key.replace("type_", "type")
            key = key.replace("format_", "format")
            new[key] = _format_function_declaration_parameters(value)
    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(
            _format_function_declaration_parameters(value) for value in obj
        )
    else:
        return obj
    return new


@dataclasses.dataclass
class Arguments:
    """Arguments. Child of Execution.

    Attributes:
        variables: The arguments of the execution.
    """

    variables: dict[str, list[Part]]

    def to_dict(self) -> Dict[str, Any]:
        dct = {}
        for variable_name in self.variables:
            dct[variable_name] = {
                "partList": {
                    "parts": [part.to_dict() for part in self.variables[variable_name]]
                }
            }
        return dct

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "Arguments":
        variables = {}
        for variable_name in dct:
            variables[variable_name] = [
                Part.from_dict(part) for part in dct[variable_name]["partList"]["parts"]
            ]
        arguments = cls(variables=variables)
        return arguments


@dataclasses.dataclass
class Execution:
    """Execution. Child of MultimodalPrompt.

    Attributes:
        arguments: The arguments of the execution.
    """

    arguments: Arguments

    def __init__(self, arguments: dict[str, list[Part]]):
        self.arguments = Arguments(variables=arguments)

    def to_dict(self) -> Dict[str, Any]:
        dct = {}
        dct["arguments"] = self.arguments.to_dict()
        return dct

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "Execution":
        arguments = dct.get("arguments", None)
        execution = cls(arguments=arguments)
        return execution


@dataclasses.dataclass
class MultimodalPrompt:
    """MultimodalPrompt. Child of PromptDatasetMetadata.

    Attributes:
        prompt_message: The schema for the prompt. A subset of the GenerateContentRequest schema.
        api_schema_version: The api schema version of the prompt when it was last modified.
        executions: Contains data related to an execution of a prompt (ex. variables)
    """

    prompt_message: gapic_prediction_service_types.GenerateContentRequest
    api_schema_version: Optional[str] = DEFAULT_API_SCHEMA_VERSION
    executions: Optional[list[Execution]] = None

    def to_dict(self) -> Dict[str, Any]:
        dct = {"multimodalPrompt": {}}
        dct["apiSchemaVersion"] = self.api_schema_version
        dct["multimodalPrompt"]["promptMessage"] = _proto_to_dict(self.prompt_message)

        # Fix type_ and format_ fields
        if dct["multimodalPrompt"]["promptMessage"].get("tools", None):
            tools = dct["multimodalPrompt"]["promptMessage"]["tools"]
            for tool in tools:
                for function_declaration in tool.get("function_declarations", []):
                    function_declaration["parameters"] = (
                        _format_function_declaration_parameters(
                            function_declaration["parameters"]
                        )
                    )

        if self.executions and self.executions[0]:
            # Only add variable sets if they are non empty.
            execution_dcts = []
            for execution in self.executions:
                exeuction_dct = execution.to_dict()
                if exeuction_dct and exeuction_dct["arguments"]:
                    execution_dcts.append(exeuction_dct)
            if execution_dcts:
                dct["executions"] = execution_dcts
        return dct

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "MultimodalPrompt":
        api_schema_version = dct.get("apiSchemaVersion", DEFAULT_API_SCHEMA_VERSION)
        if int(api_schema_version.split(".")[0]) > int(
            DEFAULT_API_SCHEMA_VERSION.split(".")[0]
        ):
            # Disallow loading prompts with higher major schema version
            raise ValueError(
                "This prompt was saved with a newer schema version and cannot be loaded."
            )
        prompt_message_dct = dct.get("multimodalPrompt", {}).get("promptMessage", None)
        if not prompt_message_dct:
            raise ValueError("This prompt is not supported in the SDK.")
        # Tool function declaration will fail the proto conversion
        tools = prompt_message_dct.get("tools", None)
        if tools:
            tools = [Tool.from_dict(tool) for tool in tools]
            prompt_message_dct.pop("tools")
        prompt_message = _dict_to_proto(
            gapic_prediction_service_types.GenerateContentRequest, prompt_message_dct
        )
        if tools:
            # Convert Tools to gapic to store in the prompt_message
            prompt_message.tools = _tool_types_to_gapic_tools(tools)
        executions_dct = dct.get("executions", [])
        executions = [Execution.from_dict(execution) for execution in executions_dct]
        if not executions:
            executions = None
        multimodal_prompt = cls(
            prompt_message=prompt_message,
            api_schema_version=api_schema_version,
            executions=executions,
        )
        return multimodal_prompt


@dataclasses.dataclass
class PromptDatasetMetadata:
    """PromptDatasetMetadata.

    Attributes:
        prompt_type: Required. SDK only supports "freeform" or "multimodal_freeform"
        prompt_api_schema: Required. SDK only supports multimodalPrompt
    """

    prompt_type: str
    prompt_api_schema: MultimodalPrompt

    def to_dict(self) -> Dict[str, Any]:
        dct = {}
        dct["promptType"] = self.prompt_type
        dct["promptApiSchema"] = self.prompt_api_schema.to_dict()
        return dct

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "PromptDatasetMetadata":
        metadata = cls(
            prompt_type=dct.get("promptType", None),
            prompt_api_schema=MultimodalPrompt.from_dict(
                dct.get("promptApiSchema", None)
            ),
        )
        return metadata


@dataclasses.dataclass
class PromptMetadata:
    """Metadata containing the display name and prompt id of a prompt.

    Returned by the `list_prompts` method.

    Attributes:
        display_name: The display name of the prompt version.
        prompt_id: The id of the prompt.
    """

    display_name: str
    prompt_id: str


@dataclasses.dataclass
class PromptVersionMetadata:
    """Metadata containing the display name, prompt id, and version id of a prompt version.

    Returned by the `list_prompt_versions` method.

    Attributes:
        display_name: The display name of the prompt version.
        prompt_id: The id of the prompt.
        version_id: The version id of the prompt.
    """

    display_name: str
    prompt_id: str
    version_id: str


def create_version(
    prompt: Prompt,
    prompt_id: Optional[str] = None,
    version_name: Optional[str] = None,
) -> Prompt:
    """Creates a Prompt or Prompt Version in the online prompt store

    Args:
        prompt: The Prompt object to create a new version of.
        prompt_id: The id of the prompt resource to create a new version under.
            If it is not provided and the prompt has no prompt resource
            associated with it, a new prompt resource will be created.
        version_name: Optional display name of the new prompt version.
            If not specified, a default name including a timestamp will be used.

    Returns:
        A new Prompt object with a reference to the newly created or updated
            prompt resource. This new Prompt object is nearly identical to the
            original Prompt object, except it has references to the new
            prompt version.
    """
    if not (prompt_id or prompt._dataset):
        # Case 1: Neither prompt id nor prompt._dataset exists, so we
        # create a new prompt resource
        return _create_prompt_resource(prompt=prompt, version_name=version_name)

    # Case 2: No prompt_id override is given, so we update the existing prompt resource
    if not prompt_id:
        return _create_prompt_version_resource(prompt=prompt, version_name=version_name)

    # Case 3: Save a new version to the prompt_id provided as an arg
    # prompt_id is guaranteed to exist due to Cases 1 & 2 being handled before

    # Store the original prompt resource name, if it exists
    original_prompt_resource = None if not prompt._dataset else prompt._dataset.name

    # Create a gapic dataset object if it doesn't exist
    if not prompt._dataset:
        project = aiplatform_initializer.global_config.project
        location = aiplatform_initializer.global_config.location
        name = f"projects/{project}/locations/{location}/datasets/{prompt_id}"
        dataset_metadata = _format_dataset_metadata_dict(prompt=prompt)

        prompt._dataset = gca_dataset.Dataset(
            name=name,
            display_name=prompt.prompt_name or "Untitled Prompt",
            metadata_schema_uri=PROMPT_SCHEMA_URI,
            metadata=dataset_metadata,
            model_reference=prompt.model_name,
        )

    # Override the dataset prompt id with the new prompt id
    project = aiplatform_initializer.global_config.project
    location = aiplatform_initializer.global_config.location
    prompt._dataset.name = (
        f"projects/{project}/locations/{location}/datasets/{prompt_id}"
    )
    result = _create_prompt_version_resource(prompt=prompt, version_name=version_name)

    # Restore the original prompt resource name. This is a no-op if there
    # was no original prompt resource name.
    prompt._dataset.name = original_prompt_resource
    return result


def _check_multimodal_contents(prompt_data: PartsType):
    if isinstance(prompt_data, Image):
        return "multimodal_freeform"
    elif isinstance(prompt_data, list):
        for part in prompt_data:
            check = _check_multimodal_contents(part)
            if check == "multimodal_freeform":
                return "multimodal_freeform"
    elif isinstance(prompt_data, Part):
        if "text" not in prompt_data._raw_part:
            return "multimodal_freeform"
    return "freeform"


def _format_dataset_metadata_dict(prompt: Prompt) -> dict[str, Any]:
    """Helper function to convert the configs and prompt data stored in the Prompt object to a dataset metadata dict."""
    model = GenerativeModel(model_name=prompt.model_name)
    prompt_message = model._prepare_request(
        contents=prompt.prompt_data or "temporary data",
        model=prompt.model_name,
        system_instruction=prompt.system_instruction,
        tools=prompt.tools,
        tool_config=prompt.tool_config,
        safety_settings=prompt.safety_settings,
        generation_config=prompt.generation_config,
    )
    # Remove temporary contents
    if not prompt.prompt_data:
        prompt_message.contents = None

    # Stopgap solution to check for multimodal contents to set flag for UI
    if prompt.prompt_data:
        prompt_type = _check_multimodal_contents(prompt.prompt_data)
    else:
        prompt_type = "freeform"

    return PromptDatasetMetadata(
        prompt_type=prompt_type,
        prompt_api_schema=MultimodalPrompt(
            prompt_message=prompt_message,
            executions=[Execution(variable_set) for variable_set in prompt.variables],
        ),
    ).to_dict()


def _create_dataset(prompt: Prompt, parent: str) -> gca_dataset.Dataset:
    dataset_metadata = _format_dataset_metadata_dict(prompt=prompt)
    dataset = gca_dataset.Dataset(
        name=parent,
        display_name=prompt.prompt_name or "Untitled Prompt",
        metadata_schema_uri=PROMPT_SCHEMA_URI,
        metadata=dataset_metadata,
        model_reference=prompt.model_name,
    )
    operation = prompt._dataset_client.create_dataset(
        parent=parent,
        dataset=dataset,
    )
    dataset = operation.result()

    # Purge labels
    dataset.labels = None
    return dataset


def _create_dataset_version(
    prompt: Prompt, parent: str, version_name: Optional[str] = None
):
    dataset_version = gca_dataset_version.DatasetVersion(
        display_name=version_name,
    )

    dataset_version = prompt._dataset_client.create_dataset_version(
        parent=parent,
        dataset_version=dataset_version,
    )
    return dataset_version.result()


def _update_dataset(
    prompt: Prompt,
    dataset: gca_dataset.Dataset,
) -> gca_dataset_version.DatasetVersion:
    dataset.metadata = _format_dataset_metadata_dict(prompt=prompt)

    mask_paths = ["modelReference", "metadata"]
    if dataset.display_name != "Untitled Prompt":
        mask_paths.append("displayName")

    updated_dataset = prompt._dataset_client.update_dataset(
        dataset=dataset,
        update_mask=field_mask.FieldMask(paths=mask_paths),
    )
    # Remove etag to avoid error for repeated dataset updates
    updated_dataset.etag = None
    return updated_dataset


def _create_prompt_resource(
    prompt: Prompt, version_name: Optional[str] = None
) -> Prompt:
    project = aiplatform_initializer.global_config.project
    location = aiplatform_initializer.global_config.location

    # Step 1: Create prompt dataset API call
    parent = f"projects/{project}/locations/{location}"
    dataset = _create_dataset(prompt=prompt, parent=parent)

    # Step 2: Create prompt version API call
    dataset_version = _create_dataset_version(
        prompt=prompt,
        parent=dataset.name,
        version_name=version_name,
    )

    # Step 3: Create new Prompt object to return
    new_prompt = Prompt._clone(prompt=prompt)
    new_prompt._dataset = dataset
    new_prompt._version_id = dataset_version.name.split("/")[-1]
    new_prompt._version_name = dataset_version.display_name
    prompt_id = new_prompt._dataset.name.split("/")[5]

    _LOGGER.info(
        f"Created prompt resource with id {prompt_id} with version number {new_prompt._version_id}"
    )
    return new_prompt


def _create_prompt_version_resource(
    prompt: Prompt,
    version_name: Optional[str] = None,
) -> Prompt:
    # Step 1: Update prompt API call
    updated_dataset = _update_dataset(prompt=prompt, dataset=prompt._dataset)

    # Step 2: Create prompt version API call
    dataset_version = _create_dataset_version(
        prompt=prompt,
        parent=updated_dataset.name,
        version_name=version_name,
    )

    # Step 3: Create new Prompt object to return
    new_prompt = Prompt._clone(prompt=prompt)
    new_prompt._dataset = updated_dataset
    new_prompt._version_id = dataset_version.name.split("/")[-1]
    new_prompt._version_name = dataset_version.display_name
    prompt_id = prompt._dataset.name.split("/")[5]

    _LOGGER.info(
        f"Updated prompt resource with id {prompt_id} as version number {new_prompt._version_id}"
    )
    return new_prompt


def _get_prompt_resource(prompt: Prompt, prompt_id: str) -> gca_dataset.Dataset:
    """Helper function to get a prompt resource from a prompt id."""
    project = aiplatform_initializer.global_config.project
    location = aiplatform_initializer.global_config.location
    name = f"projects/{project}/locations/{location}/datasets/{prompt_id}"
    dataset = prompt._dataset_client.get_dataset(name=name)
    return dataset


def _get_prompt_resource_from_version(
    prompt: Prompt, prompt_id: str, version_id: str
) -> gca_dataset.Dataset:
    """Helper function to get a prompt resource from a prompt version id."""
    project = aiplatform_initializer.global_config.project
    location = aiplatform_initializer.global_config.location
    name = f"projects/{project}/locations/{location}/datasets/{prompt_id}/datasetVersions/{version_id}"

    # Step 1: Get dataset version object
    dataset_version = prompt._dataset_client.get_dataset_version(name=name)
    prompt._version_name = dataset_version.display_name

    # Step 2: Fetch dataset object to get the dataset display name
    name = f"projects/{project}/locations/{location}/datasets/{prompt_id}"
    dataset = prompt._dataset_client.get_dataset(name=name)

    # Step 3: Convert to DatasetVersion object to Dataset object
    dataset = gca_dataset.Dataset(
        name=name,
        display_name=dataset.display_name,
        metadata_schema_uri=PROMPT_SCHEMA_URI,
        metadata=dataset_version.metadata,
        model_reference=dataset_version.model_reference,
    )
    return dataset


def restore_version(prompt_id: str, version_id: str) -> PromptVersionMetadata:
    """Restores a previous version of the prompt resource and
    loads that version into the current Prompt object.

    Args:
        prompt_id: The id of the prompt resource to restore a version of.
        version_id: The version id of the online prompt resource.
    """

    # Step 1: Make restore dataset version API call
    project = aiplatform_initializer.global_config.project
    location = aiplatform_initializer.global_config.location
    name = f"projects/{project}/locations/{location}/datasets/{prompt_id}/datasetVersions/{version_id}"

    # Create a temporary Prompt object for a dataset client
    temp_prompt = Prompt()
    operation = temp_prompt._dataset_client.restore_dataset_version(name=name)
    result = operation.result()
    new_version_id = result.name.split("/")[-1]
    prompt_id = result.name.split("/")[5]

    _LOGGER.info(
        f"Restored prompt version {version_id} under prompt id {prompt_id} as version number {new_version_id}"
    )

    # Step 2: Create PromptVersionMetadata object to return
    return PromptVersionMetadata(
        display_name=result.display_name,
        prompt_id=result.name.split("/")[5],
        version_id=new_version_id,
    )


def get(prompt_id: str, version_id: Optional[str] = None) -> Prompt:
    """Creates a Prompt object from an online resource.

    Args:
        prompt_id: The id of the prompt resource.
        version_id: Optional version id of the prompt resource.
            If not specified, the latest version will be used.

    Returns:
        A prompt loaded from the online resource as a `Prompt` object.
    """
    prompt = Prompt()
    if version_id:
        dataset = _get_prompt_resource_from_version(
            prompt=prompt,
            prompt_id=prompt_id,
            version_id=version_id,
        )
    else:
        dataset = _get_prompt_resource(prompt=prompt, prompt_id=prompt_id)

    # Remove etag to avoid error for repeated dataset updates
    dataset.etag = None

    prompt._dataset = dataset
    prompt._version_id = version_id

    dataset_dict = _proto_to_dict(dataset)

    metadata = PromptDatasetMetadata.from_dict(dataset_dict["metadata"])
    _populate_fields_from_metadata(prompt=prompt, metadata=metadata)
    return prompt


def _populate_fields_from_metadata(
    prompt: Prompt, metadata: PromptDatasetMetadata
) -> None:
    """Helper function to populate Promptfields from metadata object"""
    # Populate model_name (Required, raw deserialized type is str)
    prompt.model_name = metadata.prompt_api_schema.prompt_message.model

    # Populate prompt_data (raw deserialized type is list[Content])
    contents = metadata.prompt_api_schema.prompt_message.contents
    if contents:
        if len(contents) > 1:
            raise ValueError("Multi-turn prompts are not supported yet.")
        prompt_data = [Part._from_gapic(part) for part in list(contents[0].parts)]

        # Unwrap single text part into str
        if len(prompt_data) == 1 and "text" in prompt_data[0]._raw_part:
            prompt.prompt_data = prompt_data[0].text
        else:
            prompt.prompt_data = prompt_data

    # Populate system_instruction (raw deserialized type is single Content)
    system_instruction = metadata.prompt_api_schema.prompt_message.system_instruction
    if system_instruction:
        system_instruction_parts = [
            Part._from_gapic(part) for part in list(system_instruction.parts)
        ]
        # Unwrap single text part into str
        if len(system_instruction_parts) == 1 and system_instruction_parts[0].text:
            prompt.system_instruction = system_instruction_parts[0].text
        else:
            prompt.system_instruction = system_instruction_parts

    # Populate variables
    executions = metadata.prompt_api_schema.executions
    variables = []
    if executions:
        for execution in executions:
            serialized_variable_set = execution.arguments
            variable_set = {}
            if serialized_variable_set:
                for name, value in serialized_variable_set.variables.items():
                    # Parts are dicts, not gapic objects for variables
                    variable_set[name] = [
                        Part.from_dict(part)
                        for part in list(value["partList"]["parts"])
                    ]
                variables.append(variable_set)

        # Unwrap variable single text part into str
        for variable_set in variables:
            for name, value in variable_set.items():
                if len(value) == 1 and "text" in value[0]._raw_part:
                    variable_set[name] = value[0].text
        prompt.variables = variables

    # Populate generation_config (raw deserialized type is GenerationConfig)
    generation_config = metadata.prompt_api_schema.prompt_message.generation_config
    if generation_config:
        prompt.generation_config = generation_config

    # Populate safety_settings (raw deserialized type is RepeatedComposite of SafetySetting)
    safety_settings = metadata.prompt_api_schema.prompt_message.safety_settings
    if safety_settings:
        prompt.safety_settings = list(safety_settings)

    # Populate tools (raw deserialized type is RepeatedComposite of Tool)
    tools = metadata.prompt_api_schema.prompt_message.tools
    if tools:
        prompt.tools = list(tools)

    # Populate tool_config (raw deserialized type is ToolConfig)
    tool_config = metadata.prompt_api_schema.prompt_message.tool_config
    if tool_config:
        prompt.tool_config = ToolConfig._from_gapic(tool_config)


def list_prompts() -> list[PromptMetadata]:
    """Lists all prompt resources in the online prompt store associated with the project."""
    project = aiplatform_initializer.global_config.project
    location = aiplatform_initializer.global_config.location
    parent = f"projects/{project}/locations/{location}"

    # Create a temporary Prompt object for a dataset client
    temp_prompt = Prompt()
    prompts_pager = temp_prompt._dataset_client.list_datasets(
        parent=parent,
    )
    prompts_list = []
    for prompt in prompts_pager:
        prompts_list.append(
            PromptMetadata(
                display_name=prompt.display_name,
                prompt_id=prompt.name.split("/")[5],
            )
        )
    return prompts_list


def list_versions(prompt_id: str) -> list[PromptVersionMetadata]:
    """Returns a list of PromptVersionMetadata objects for the prompt resource.

    Args:
        prompt_id: The id of the prompt resource to list versions of.

    Returns:
        A list of PromptVersionMetadata objects for the prompt resource.
    """
    # Create a temporary Prompt object for a dataset client
    temp_prompt = Prompt()
    project = aiplatform_initializer.global_config.project
    location = aiplatform_initializer.global_config.location
    parent = f"projects/{project}/locations/{location}/datasets/{prompt_id}"

    versions_pager = temp_prompt._dataset_client.list_dataset_versions(
        parent=parent,
    )
    version_history = []
    for version in versions_pager:
        version_history.append(
            PromptVersionMetadata(
                display_name=version.display_name,
                prompt_id=version.name.split("/")[5],
                version_id=version.name.split("/")[-1],
            )
        )
    return version_history


def delete(prompt_id: str) -> None:
    """Deletes the online prompt resource associated with the prompt id."""

    # Create a temporary Prompt object for a dataset client
    temp_prompt = Prompt()
    project = aiplatform_initializer.global_config.project
    location = aiplatform_initializer.global_config.location
    name = f"projects/{project}/locations/{location}/datasets/{prompt_id}"

    operation = temp_prompt._dataset_client.delete_dataset(
        name=name,
    )
    operation.result()

    _LOGGER.info(f"Deleted prompt resource with id {prompt_id}.")
