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
"""Classes for working with generative models."""
# pylint: disable=bad-continuation, line-too-long, protected-access

import copy
import functools
import io
import json
import pathlib
import re
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
    TYPE_CHECKING,
)

from google.cloud.aiplatform import initializer as aiplatform_initializer
from google.cloud.aiplatform import utils as aiplatform_utils
from google.cloud.aiplatform_v1 import types as types_v1
from google.cloud.aiplatform_v1.services import (
    prediction_service as prediction_service_v1,
    llm_utility_service as llm_utility_service_v1,
)
from google.cloud.aiplatform_v1beta1 import types as aiplatform_types
from google.cloud.aiplatform_v1beta1.services import endpoint_service
from google.cloud.aiplatform_v1beta1.services import prediction_service
from google.cloud.aiplatform_v1beta1.services import llm_utility_service
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)
from google.cloud.aiplatform_v1beta1.types import (
    prediction_service as gapic_prediction_service_types,
    llm_utility_service as gapic_llm_utility_service_types,
)
from google.cloud.aiplatform_v1beta1.types import tool as gapic_tool_types
from google.protobuf import json_format
from google.protobuf import field_mask_pb2
import warnings
from vertexai._utils import warning_logs

if TYPE_CHECKING:
    from vertexai.caching import CachedContent

try:
    from PIL import Image as PIL_Image  # pylint: disable=g-import-not-at-top
except ImportError:
    PIL_Image = None


T = TypeVar("T")


# Re-exporting some GAPIC types

# GAPIC types used in request
HarmCategory = gapic_content_types.HarmCategory
HarmBlockThreshold = gapic_content_types.SafetySetting.HarmBlockThreshold
# GAPIC types used in response
# We expose FinishReason to make it easier to check the response finish reason.
FinishReason = gapic_content_types.Candidate.FinishReason
# We expose SafetyRating to make it easier to check the response safety rating.
SafetyRating = gapic_content_types.SafetyRating


# These type defnitions are expanded to help the user see all the types
ContentDict = Dict[str, Any]
GenerationConfigDict = Dict[str, Any]

if TYPE_CHECKING:
    # Default to the current definitions if pytype is being used for type checks
    # because it does not support try-except for types.
    PartsType = Union[
        str,
        "Image",
        "Part",
        List[Union[str, "Image", "Part"]],
    ]
    ContentsType = Union[
        List["Content"],
        List[ContentDict],
        str,
        "Image",
        "Part",
        List[Union[str, "Image", "Part"]],
    ]
    GenerationConfigType = Union[
        "GenerationConfig",
        GenerationConfigDict,
    ]
    SafetySettingsType = Union[
        List["SafetySetting"],
        Dict[
            gapic_content_types.HarmCategory,
            gapic_content_types.SafetySetting.HarmBlockThreshold,
        ],
    ]
else:
    try:
        # For Pydantic to resolve the forward references inside these aliases.
        from typing_extensions import TypeAliasType

        PartsType = TypeAliasType(
            "PartsType",
            Union[
                str,
                "Image",
                "Part",
                List[Union[str, "Image", "Part"]],
            ],
        )
        ContentsType = TypeAliasType(
            "ContentsType",
            Union[
                List["Content"],
                List[ContentDict],
                str,
                "Image",
                "Part",
                List[Union[str, "Image", "Part"]],
            ],
        )
        GenerationConfigType = TypeAliasType(
            "GenerationConfigType",
            Union[
                "GenerationConfig",
                GenerationConfigDict,
            ],
        )
        SafetySettingsType = TypeAliasType(
            "SafetySettingsType",
            Union[
                List["SafetySetting"],
                Dict[
                    gapic_content_types.HarmCategory,
                    gapic_content_types.SafetySetting.HarmBlockThreshold,
                ],
            ],
        )
    except (ImportError, RuntimeError) as e:
        from google.cloud.aiplatform import base

        _LOGGER = base.Logger(__name__)
        _LOGGER.debug(f"Failed to import typing_extensions.TypeAliasType: {e}")
        # Use existing definitions if typing_extensions is not available.
        PartsType = Union[
            str,
            "Image",
            "Part",
            List[Union[str, "Image", "Part"]],
        ]
        ContentsType = Union[
            List["Content"],
            List[ContentDict],
            str,
            "Image",
            "Part",
            List[Union[str, "Image", "Part"]],
        ]
        GenerationConfigType = Union[
            "GenerationConfig",
            GenerationConfigDict,
        ]
        SafetySettingsType = Union[
            List["SafetySetting"],
            Dict[
                gapic_content_types.HarmCategory,
                gapic_content_types.SafetySetting.HarmBlockThreshold,
            ],
        ]


def _reconcile_model_name(model_name: str, project: str, location: str) -> str:
    """Returns a model name that's one of the following:
    1. A full resource name starting with projects/
    2. A partial resource name starting with publishers/
    """
    if "/" not in model_name:
        return f"publishers/google/models/{model_name}"
    elif model_name.startswith("models/"):
        return f"publishers/google/{model_name}"
    elif model_name.startswith("publishers/") or model_name.startswith("projects/"):
        return model_name
    else:
        raise ValueError(
            "model_name must be either a Model Garden model ID or a full resource name."
            f"recieved model_name {model_name}"
        )


def _get_resource_name_from_model_name(
    model_name: str, project: str, location: str
) -> str:
    """Returns the full resource name starting with projects/ given a model name."""
    if model_name.startswith("publishers/"):
        if not project:
            return model_name
        return f"projects/{project}/locations/{location}/{model_name}"
    elif model_name.startswith("projects/"):
        return model_name
    else:
        raise ValueError(
            "model_name must be either a Model Garden model ID or a full resource name."
        )


def _validate_generate_content_parameters(
    contents: ContentsType,
    *,
    generation_config: Optional[GenerationConfigType] = None,
    safety_settings: Optional[SafetySettingsType] = None,
    tools: Optional[List["Tool"]] = None,
    tool_config: Optional["ToolConfig"] = None,
    system_instruction: Optional[PartsType] = None,
    cached_content: Optional["CachedContent"] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Validates the parameters for a generate_content call."""
    if not contents:
        raise TypeError("contents must not be empty")

    _validate_contents_type_as_valid_sequence(contents)

    if cached_content and any([tools, tool_config, system_instruction]):
        raise ValueError(
            "When using cached_content, tools, tool_config, and system_instruction must be None."
        )

    if safety_settings:
        _validate_safety_settings_type_as_valid_sequence(safety_settings)

    if generation_config:
        if not isinstance(
            generation_config,
            (gapic_content_types.GenerationConfig, GenerationConfig, Dict),
        ):
            raise TypeError(
                "generation_config must either be a GenerationConfig object or a dictionary representation of it."
            )

    if tools:
        _validate_tools_type_as_valid_sequence(tools)

    if tool_config:
        _validate_tool_config_type(tool_config)

    if labels:
        if not isinstance(labels, Dict):
            raise TypeError("labels must be a dictionary.")


def _validate_contents_type_as_valid_sequence(contents: ContentsType) -> None:
    """Makes sure that individual elements of contents are of valid type."""
    # contents can either be a list of Content objects (most generic case)
    if isinstance(contents, Sequence) and any(
        isinstance(c, gapic_content_types.Content) for c in contents
    ):
        if not all(isinstance(c, gapic_content_types.Content) for c in contents):
            raise TypeError(
                "When passing a list with Content objects, every item in a "
                + "list must be a Content object."
            )
    elif isinstance(contents, Sequence) and any(
        isinstance(c, Content) for c in contents
    ):
        if not all(isinstance(c, Content) for c in contents):
            raise TypeError(
                "When passing a list with Content objects, every item in a "
                + "list must be a Content object."
            )
    elif isinstance(contents, Sequence) and any(isinstance(c, dict) for c in contents):
        if not all(isinstance(c, dict) for c in contents):
            raise TypeError(
                "When passing a list with Content dict objects, every item in "
                + "a list must be a Content dict object."
            )


def _validate_safety_settings_type_as_valid_sequence(
    safety_settings: SafetySettingsType,
) -> None:
    if not isinstance(safety_settings, (Sequence, Dict)):
        raise TypeError(
            "safety_settings must either be a SafetySetting object or a "
            + "dictionary mapping from HarmCategory to HarmBlockThreshold."
        )
    if isinstance(safety_settings, Sequence):
        for safety_setting in safety_settings:
            if not isinstance(
                safety_setting,
                (gapic_content_types.SafetySetting, SafetySetting),
            ):
                raise TypeError(
                    "When passing a list with SafetySettings objects, every "
                    + "item in a list must be a SafetySetting object."
                )


def _validate_tools_type_as_valid_sequence(tools: List["Tool"]):
    for tool in tools:
        if not isinstance(tool, (gapic_tool_types.Tool, Tool)):
            raise TypeError(f"Unexpected tool type: {tool}.")


def _validate_tool_config_type(tool_config: "ToolConfig"):
    if not isinstance(tool_config, ToolConfig):
        raise TypeError("tool_config must be a ToolConfig object.")


def _content_types_to_gapic_contents(
    contents: ContentsType,
) -> List[gapic_content_types.Content]:
    """Converts a list of Content objects to a list of gapic_content_types.Content objects."""
    if isinstance(contents, Sequence) and any(
        isinstance(c, gapic_content_types.Content) for c in contents
    ):
        return contents
    elif isinstance(contents, Sequence) and any(
        isinstance(c, Content) for c in contents
    ):
        return [content._raw_content for content in contents]
    elif isinstance(contents, Sequence) and any(isinstance(c, dict) for c in contents):
        return [gapic_content_types.Content(content_dict) for content_dict in contents]
    # or a value that can be converted to a *single* Content object
    else:
        return [_to_content(contents)]


def _tool_types_to_gapic_tools(
    tools: Optional[List["Tool"]],
) -> List[gapic_tool_types.Tool]:
    """Converts a list of Tool objects to a list of gapic_tool_types.Tool objects."""
    gapic_tools = []
    if tools:
        for tool in tools:
            if isinstance(tool, gapic_tool_types.Tool):
                gapic_tools.append(tool)
            elif isinstance(tool, Tool):
                gapic_tools.append(tool._raw_tool)
    return gapic_tools


class _GenerativeModel:
    r"""A model that can generate content.


    Usage:

        ```
        model = GenerativeModel("gemini-pro")
        response = model.generate_content(
            contents="Why is sky blue?",
            # Optional:
            generation_config=GenerationConfig(
                temperature=0.1,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                max_output_tokens=100,
                stop_sequences=["STOP!"],
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        ```

    """

    _USER_ROLE = "user"
    _MODEL_ROLE = "model"

    def __init__(
        self,
        model_name: str,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        system_instruction: Optional[PartsType] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        r"""Initializes GenerativeModel.

        Usage:

            ```
            model = GenerativeModel("gemini-pro")
            print(model.generate_content("Hello"))
            ```

        Args:
            model_name: Model Garden model resource name.
                Alternatively, a tuned model endpoint resource name can be provided.
            generation_config: Default generation config to use in generate_content.
            safety_settings: Default safety settings to use in generate_content.
            tools: Default tools to use in generate_content.
            tool_config: Default tool config to use in generate_content.
            system_instruction: Default system instruction to use in generate_content.
                Note: Only text should be used in parts.
                Content of each part will become a separate paragraph.
            labels: labels that will be passed to billing for cost tracking.
        """
        warning_logs.show_deprecation_warning()
        project = aiplatform_initializer.global_config.project
        location = aiplatform_initializer.global_config.location
        model_name = _reconcile_model_name(model_name, project, location)

        prediction_resource_name = _get_resource_name_from_model_name(
            model_name, project, location
        )

        location = aiplatform_utils.extract_project_and_location_from_parent(
            prediction_resource_name
        ).get("location")

        self._model_name = model_name
        self._prediction_resource_name = prediction_resource_name
        self._location = location
        self._generation_config = generation_config
        self._safety_settings = safety_settings
        self._tools = tools
        self._tool_config = tool_config
        self._system_instruction = system_instruction
        self._cached_content: Optional["CachedContent"] = None
        self._labels = labels

        # Validating the parameters
        _validate_generate_content_parameters(
            contents="test",
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
            labels=labels,
        )

    @functools.cached_property
    def _prediction_client(self) -> prediction_service.PredictionServiceClient:
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        return aiplatform_initializer.global_config.create_client(
            client_class=prediction_service.PredictionServiceClient,
            prediction_client=True,
            location_override=self._location if not api_key else None,
            api_key=api_key,
        )

    @functools.cached_property
    def _prediction_async_client(
        self,
    ) -> prediction_service.PredictionServiceAsyncClient:
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        if api_key:
            raise RuntimeError(
                "Using an api key is not supported yet for async clients."
            )
        return aiplatform_initializer.global_config.create_client(
            client_class=prediction_service.PredictionServiceAsyncClient,
            location_override=self._location if not api_key else None,
            prediction_client=True,
            api_key=api_key,
        )

    @functools.cached_property
    def _llm_utility_client(self) -> llm_utility_service.LlmUtilityServiceClient:
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        return aiplatform_initializer.global_config.create_client(
            client_class=prediction_service.PredictionServiceAsyncClient,
            location_override=self._location if not api_key else None,
            api_key=api_key,
        )

    @functools.cached_property
    def _endpoint_client(self) -> endpoint_service.EndpointServiceClient:
        # Note this doesn't work with GCP Express but it's better to set the
        # client correctly and allow the service to throw
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        return aiplatform_initializer.global_config.create_client(
            client_class=endpoint_service.EndpointServiceClient,
            location_override=self._location,
            api_key=api_key,
        )

    @functools.cached_property
    def _llm_utility_async_client(
        self,
    ) -> llm_utility_service.LlmUtilityServiceAsyncClient:
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        if api_key:
            raise RuntimeError(
                "Using an api key is not supported yet for async clients."
            )
        return aiplatform_initializer.global_config.create_client(
            client_class=llm_utility_service.LlmUtilityServiceAsyncClient,
            location_override=self._location if not api_key else None,
            api_key=api_key,
        )

    def _prepare_request(
        self,
        contents: ContentsType,
        *,
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        system_instruction: Optional[PartsType] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> gapic_prediction_service_types.GenerateContentRequest:
        """Prepares a GAPIC GenerateContentRequest."""
        if not contents:
            raise TypeError("contents must not be empty")

        model = model or self._prediction_resource_name
        generation_config = generation_config or self._generation_config
        safety_settings = safety_settings or self._safety_settings
        tools = tools or self._tools
        tool_config = tool_config or self._tool_config
        system_instruction = system_instruction or self._system_instruction
        cached_content = self._cached_content
        labels = labels or self._labels

        _validate_generate_content_parameters(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
            cached_content=cached_content,
            labels=labels,
        )

        contents = _content_types_to_gapic_contents(contents)

        gapic_system_instruction: Optional[gapic_content_types.Content] = None
        if system_instruction:
            gapic_system_instruction = _to_content(system_instruction)

        gapic_generation_config: Optional[gapic_content_types.GenerationConfig] = None
        if generation_config:
            if isinstance(generation_config, gapic_content_types.GenerationConfig):
                gapic_generation_config = generation_config
            elif isinstance(generation_config, GenerationConfig):
                gapic_generation_config = generation_config._raw_generation_config
            elif isinstance(generation_config, Dict):
                gapic_generation_config = gapic_content_types.GenerationConfig(
                    **generation_config
                )

        gapic_safety_settings = None
        if safety_settings:
            if isinstance(safety_settings, Sequence):
                gapic_safety_settings = []
                for safety_setting in safety_settings:
                    if isinstance(safety_setting, gapic_content_types.SafetySetting):
                        gapic_safety_settings.append(safety_setting)
                    elif isinstance(safety_setting, SafetySetting):
                        gapic_safety_settings.append(safety_setting._raw_safety_setting)
            elif isinstance(safety_settings, dict):
                gapic_safety_settings = [
                    gapic_content_types.SafetySetting(
                        category=gapic_content_types.HarmCategory(category),
                        threshold=gapic_content_types.SafetySetting.HarmBlockThreshold(
                            threshold
                        ),
                    )
                    for category, threshold in safety_settings.items()
                ]

        gapic_tools = None
        if tools:
            gapic_tools = _tool_types_to_gapic_tools(tools)

        gapic_tool_config = None
        if tool_config:
            gapic_tool_config = tool_config._gapic_tool_config

        return gapic_prediction_service_types.GenerateContentRequest(
            # The `model` parameter now needs to be set for the vision models.
            # Always need to pass the resource via the `model` parameter.
            # Even when resource is an endpoint.
            model=model,
            contents=contents,
            generation_config=gapic_generation_config,
            safety_settings=gapic_safety_settings,
            tools=gapic_tools,
            tool_config=gapic_tool_config,
            system_instruction=gapic_system_instruction,
            cached_content=cached_content.resource_name if cached_content else None,
            labels=labels,
        )

    def _parse_response(
        self,
        response: gapic_prediction_service_types.GenerateContentResponse,
    ) -> "GenerationResponse":
        return GenerationResponse._from_gapic(response)

    @overload
    def generate_content(  # noqa: E704
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: Literal[False] = False,
    ) -> "GenerationResponse": ...

    @overload
    def generate_content(  # noqa: E704
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: Literal[True],
    ) -> Iterable["GenerationResponse"]: ...

    def generate_content(
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union["GenerationResponse", Iterable["GenerationResponse"]]:
        """Generates content.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            tool_config: Config shared for all tools provided in the request.
            labels: Labels that will be passed to billing for cost tracking.
            stream: Whether to stream the response.

        Returns:
            A single GenerationResponse object if stream == False
            A stream of GenerationResponse objects if stream == True
        """
        if stream:
            # TODO(b/315810992): Surface prompt_feedback on the returned stream object
            return self._generate_content_streaming(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                tool_config=tool_config,
                labels=labels,
            )
        else:
            return self._generate_content(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                tool_config=tool_config,
                labels=labels,
            )

    @overload
    async def generate_content_async(  # noqa: E704
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: Literal[False] = False,
    ) -> "GenerationResponse": ...

    @overload
    async def generate_content_async(  # noqa: E704
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: Literal[True] = True,
    ) -> AsyncIterable["GenerationResponse"]: ...

    async def generate_content_async(
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union["GenerationResponse", AsyncIterable["GenerationResponse"]]:
        """Generates content asynchronously.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            tool_config: Config shared for all tools provided in the request.
            labels: Labels that will be passed to billing for cost tracking.
            stream: Whether to stream the response.

        Returns:
            An awaitable for a single GenerationResponse object if stream == False
            An awaitable for a stream of GenerationResponse objects if stream == True
        """
        if stream:
            return await self._generate_content_streaming_async(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                tool_config=tool_config,
                labels=labels,
            )
        else:
            return await self._generate_content_async(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                tool_config=tool_config,
                labels=labels,
            )

    def _generate_content(
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> "GenerationResponse":
        """Generates content.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            tool_config: Config shared for all tools provided in the request.
            labels: Labels that will be passed to billing for cost tracking.

        Returns:
            A single GenerationResponse object
        """
        request = self._prepare_request(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            labels=labels,
        )
        gapic_response = self._prediction_client.generate_content(request=request)
        return self._parse_response(gapic_response)

    async def _generate_content_async(
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> "GenerationResponse":
        """Generates content asynchronously.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            tool_config: Config shared for all tools provided in the request.
            labels: Labels that will be passed to billing for cost tracking.

        Returns:
            An awaitable for a single GenerationResponse object
        """
        request = self._prepare_request(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            labels=labels,
        )
        gapic_response = await self._prediction_async_client.generate_content(
            request=request
        )
        return self._parse_response(gapic_response)

    def _generate_content_streaming(
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Iterable["GenerationResponse"]:
        """Generates content.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            tool_config: Config shared for all tools provided in the request.
            labels: Labels that will be passed to billing for cost tracking.

        Yields:
            A stream of GenerationResponse objects
        """
        request = self._prepare_request(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            labels=labels,
        )
        response_stream = self._prediction_client.stream_generate_content(
            request=request
        )
        for chunk in response_stream:
            yield self._parse_response(chunk)

    async def _generate_content_streaming_async(
        self,
        contents: ContentsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> AsyncIterable["GenerationResponse"]:
        """Generates content asynchronously.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            tool_config: Config shared for all tools provided in the request.
            labels: Labels that will be passed to billing for cost tracking.

        Returns:
            An awaitable for a stream of GenerationResponse objects
        """
        request = self._prepare_request(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            labels=labels,
        )
        response_stream = await self._prediction_async_client.stream_generate_content(
            request=request
        )

        async def async_generator():
            async for chunk in response_stream:
                yield self._parse_response(chunk)

        return async_generator()

    async def _close_async_client(self) -> None:
        if self._prediction_async_client:
            return await self._prediction_async_client.transport.close()

    def count_tokens(
        self, contents: ContentsType, *, tools: Optional[List["Tool"]] = None
    ) -> gapic_prediction_service_types.CountTokensResponse:
        """Counts tokens.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]
            tools: A list of tools (functions) that the model can try calling.

        Returns:
            A CountTokensResponse object that has the following attributes:
                total_tokens: The total number of tokens counted across all instances from the request.
                total_billable_characters: The total number of billable characters counted across all instances from the request.
        """
        request = self._prepare_request(
            contents=contents,
            tools=tools,
        )
        return self._gapic_count_tokens(
            prediction_resource_name=self._prediction_resource_name,
            contents=request.contents,
            system_instruction=request.system_instruction,
            tools=request.tools,
        )

    async def count_tokens_async(
        self,
        contents: ContentsType,
        *,
        tools: Optional[List["Tool"]] = None,
    ) -> gapic_prediction_service_types.CountTokensResponse:
        """Counts tokens asynchronously.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]
            tools: A list of tools (functions) that the model can try calling.

        Returns:
            And awaitable for a CountTokensResponse object that has the following attributes:
                total_tokens: The total number of tokens counted across all instances from the request.
                total_billable_characters: The total number of billable characters counted across all instances from the request.
        """
        request = self._prepare_request(
            contents=contents,
            tools=tools,
        )
        return await self._gapic_count_tokens_async(
            prediction_resource_name=self._prediction_resource_name,
            contents=request.contents,
            system_instruction=request.system_instruction,
            tools=request.tools,
        )

    def _gapic_count_tokens(
        self,
        prediction_resource_name: str,
        contents: List[gapic_content_types.Content],
        system_instruction: Optional[gapic_content_types.Content] = None,
        tools: Optional[List[gapic_tool_types.Tool]] = None,
    ) -> gapic_prediction_service_types.CountTokensResponse:
        request = gapic_prediction_service_types.CountTokensRequest(
            endpoint=prediction_resource_name,
            model=prediction_resource_name,
            contents=contents,
            system_instruction=system_instruction,
            tools=tools,
        )
        return self._prediction_client.count_tokens(request=request)

    async def _gapic_count_tokens_async(
        self,
        prediction_resource_name: str,
        contents: List[gapic_content_types.Content],
        system_instruction: Optional[gapic_content_types.Content] = None,
        tools: Optional[List[gapic_tool_types.Tool]] = None,
    ) -> gapic_prediction_service_types.CountTokensResponse:
        request = gapic_prediction_service_types.CountTokensRequest(
            endpoint=prediction_resource_name,
            model=prediction_resource_name,
            contents=contents,
            system_instruction=system_instruction,
            tools=tools,
        )
        return await self._prediction_async_client.count_tokens(request=request)

    def compute_tokens(
        self, contents: ContentsType
    ) -> gapic_llm_utility_service_types.ComputeTokensResponse:
        """Computes tokens.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]

        Returns:
            A ComputeTokensResponse object that has the following attributes:
                tokens_info: Lists of tokens_info from the input.
                             The input `contents: ContentsType` could have
                             multiple string instances and each tokens_info
                             item represents each string instance. Each token
                             info consists tokens list, token_ids list and
                             a role.
        """
        return self._gapic_compute_tokens(
            prediction_resource_name=self._prediction_resource_name,
            contents=self._prepare_request(contents=contents).contents,
        )

    async def compute_tokens_async(
        self, contents: ContentsType
    ) -> gapic_llm_utility_service_types.ComputeTokensResponse:
        """Computes tokens asynchronously.

        Args:
            contents: Contents to send to the model.
                Supports either a list of Content objects (passing a multi-turn conversation)
                or a value that can be converted to a single Content object (passing a single message).
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
                * List[Content]

        Returns:
            And awaitable for a ComputeTokensResponse object that has the following attributes:
                tokens_info: Lists of tokens_info from the input.
                             The input `contents: ContentsType` could have
                             multiple string instances and each tokens_info
                             item represents each string instance. Each token
                             info consists tokens list, token_ids list and
                             a role.
        """
        return await self._gapic_compute_tokens_async(
            prediction_resource_name=self._prediction_resource_name,
            contents=self._prepare_request(contents=contents).contents,
        )

    def _gapic_compute_tokens(
        self,
        prediction_resource_name: str,
        contents: List[gapic_content_types.Content],
    ) -> gapic_prediction_service_types.CountTokensResponse:
        request = gapic_llm_utility_service_types.ComputeTokensRequest(
            endpoint=prediction_resource_name,
            model=prediction_resource_name,
            contents=contents,
        )
        return self._llm_utility_client.compute_tokens(request=request)

    async def _gapic_compute_tokens_async(
        self,
        prediction_resource_name: str,
        contents: List[gapic_content_types.Content],
    ) -> gapic_prediction_service_types.CountTokensResponse:
        request = gapic_llm_utility_service_types.ComputeTokensRequest(
            endpoint=prediction_resource_name,
            model=prediction_resource_name,
            contents=contents,
        )
        return await self._llm_utility_async_client.compute_tokens(request=request)

    def start_chat(
        self,
        *,
        history: Optional[List["Content"]] = None,
        response_validation: bool = True,
    ) -> "ChatSession":
        """Creates a stateful chat session.

        Args:
            history: Previous history to initialize the chat session.
            response_validation: Whether to validate responses before adding
                them to chat history. By default, `send_message` will raise
                error if the request or response is blocked or if the response
                is incomplete due to going over the max token limit.
                If set to `False`, the chat session history will always
                accumulate the request and response messages even if the
                reponse if blocked or incomplete. This can result in an unusable
                chat session state.

        Returns:
            A ChatSession object.
        """
        return ChatSession(
            model=self,
            history=history,
            response_validation=response_validation,
        )

    @classmethod
    def from_cached_content(
        cls,
        cached_content: Union[str, "CachedContent"],
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
    ) -> "_GenerativeModel":
        """Creates a model from cached content.

        Creates a model instance with an existing cached content. The cached
        content becomes the prefix of the requesting contents.

        Args:
            cached_content: The cached content resource name or object.
            generation_config: The generation config to use for this model.
            safety_settings: The safety settings to use for this model.

        Returns:
            A model instance with the cached content wtih cached content as
            prefix of all its requests.
        """
        if isinstance(cached_content, str):
            from vertexai.caching import CachedContent

            cached_content = CachedContent.get(cached_content)
        model_name = cached_content.model_name
        model = cls(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=None,
            tool_config=None,
            system_instruction=None,
        )
        model._cached_content = cached_content

        return model


_SUCCESSFUL_FINISH_REASONS = [
    gapic_content_types.Candidate.FinishReason.STOP,
    # Many responses have this finish reason
    gapic_content_types.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED,
]


def _validate_response(
    response: "GenerationResponse",
    request_contents: Optional[List["Content"]] = None,
    response_chunks: Optional[List["GenerationResponse"]] = None,
) -> None:
    message = ""
    if not response.candidates:
        message += (
            f"The model response was blocked due to {response._raw_response.prompt_feedback.block_reason}.\n"
            f"Block reason message: {response._raw_response.prompt_feedback.block_reason_message}.\n"
        )
    else:
        candidate = response.candidates[0]
        if candidate.finish_reason not in _SUCCESSFUL_FINISH_REASONS:
            message = (
                "The model response did not complete successfully.\n"
                f"Finish reason: {candidate.finish_reason}.\n"
                f"Finish message: {candidate.finish_message}.\n"
                f"Safety ratings: {candidate.safety_ratings}.\n"
            )
    if message:
        message += (
            "To protect the integrity of the chat session, the request and response were not added to chat history.\n"
            "To skip the response validation, specify `model.start_chat(response_validation=False)`.\n"
            "Note that letting blocked or otherwise incomplete responses into chat history might lead to future interactions being blocked by the service."
        )
        raise ResponseValidationError(
            message=message,
            request_contents=request_contents,
            responses=response_chunks,
        )


class ChatSession:
    """Chat session holds the chat history."""

    _USER_ROLE = "user"
    _MODEL_ROLE = "model"

    def __init__(
        self,
        model: _GenerativeModel,
        *,
        history: Optional[List["Content"]] = None,
        response_validation: bool = True,
    ):
        if history:
            if not all(isinstance(item, Content) for item in history):
                raise ValueError("history must be a list of Content objects.")

        self._model = model
        self._history = history or []
        self._response_validator = _validate_response if response_validation else None
        # _responder is currently only set by PreviewChatSession
        self._responder: Optional["AutomaticFunctionCallingResponder"] = None

    @property
    def history(self) -> List["Content"]:
        return self._history

    @overload
    def send_message(  # noqa: E704
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: Literal[False] = False,
    ) -> "GenerationResponse": ...

    @overload
    def send_message(  # noqa: E704
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: Literal[True] = True,
    ) -> Iterable["GenerationResponse"]: ...

    def send_message(
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union["GenerationResponse", Iterable["GenerationResponse"]]:
        """Generates content.

        Args:
            content: Content to send to the model.
                Supports a value that can be converted to a Part or a list of such values.
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            labels: Labels that will be passed to billing for cost tracking.
            stream: Whether to stream the response.

        Returns:
            A single GenerationResponse object if stream == False
            A stream of GenerationResponse objects if stream == True

        Raises:
            ResponseValidationError: If the response was blocked or is incomplete.
        """
        if stream:
            return self._send_message_streaming(
                content=content,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                labels=labels,
            )
        else:
            return self._send_message(
                content=content,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                labels=labels,
            )

    @overload
    def send_message_async(  # noqa: E704
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: Literal[False] = False,
    ) -> Awaitable["GenerationResponse"]: ...

    @overload
    def send_message_async(  # noqa: E704
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: Literal[True] = True,
    ) -> Awaitable[AsyncIterable["GenerationResponse"]]: ...

    def send_message_async(
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[
        Awaitable["GenerationResponse"],
        Awaitable[AsyncIterable["GenerationResponse"]],
    ]:
        """Generates content asynchronously.

        Args:
            content: Content to send to the model.
                Supports a value that can be converted to a Part or a list of such values.
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            labels: Labels that will be passed to billing for cost tracking.
            stream: Whether to stream the response.

        Returns:
            An awaitable for a single GenerationResponse object if stream == False
            An awaitable for a stream of GenerationResponse objects if stream == True

        Raises:
            ResponseValidationError: If the response was blocked or is incomplete.
        """
        if stream:
            return self._send_message_streaming_async(
                content=content,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                labels=labels,
            )
        else:
            return self._send_message_async(
                content=content,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                labels=labels,
            )

    def _send_message(
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> "GenerationResponse":
        """Generates content.

        Args:
            content: Content to send to the model.
                Supports a value that can be converted to a Part or a list of such values.
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            labels: Labels that will be passed to billing for cost tracking.

        Returns:
            A single GenerationResponse object

        Raises:
            ResponseValidationError: If the response was blocked or is incomplete.
        """
        # Preparing the message history to send
        request_message = Content._from_gapic(
            _to_content(value=content, role=self._USER_ROLE)
        )
        history_delta = [request_message]

        message_responder = (
            self._responder._create_responder_for_message(
                tools=tools or self._model._tools
            )
            if self._responder
            else None
        )

        while True:
            request_history = self._history + history_delta
            response = self._model._generate_content(
                contents=request_history,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                labels=labels,
            )
            # By default we're not adding incomplete interactions to history.
            if self._response_validator is not None:
                self._response_validator(
                    response=response,
                    request_contents=request_history,
                    response_chunks=[response],
                )

            # Adding the request and the first response candidate to history
            response_message = response.candidates[0].content
            # Response role is NOT set by the model.
            response_message.role = self._MODEL_ROLE
            history_delta.append(response_message)

            auto_responder_content = (
                message_responder.respond_to_model_response(response=response)
                if message_responder
                else None
            )
            if auto_responder_content:
                auto_responder_content.role = self._USER_ROLE
                history_delta.append(auto_responder_content)
            else:
                break

        self._history.extend(history_delta)
        return response

    async def _send_message_async(
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> "GenerationResponse":
        """Generates content asynchronously.

        Args:
            content: Content to send to the model.
                Supports a value that can be converted to a Part or a list of such values.
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            labels: Labels that will be passed to billing for cost tracking.

        Returns:
            An awaitable for a single GenerationResponse object

        Raises:
            ResponseValidationError: If the response was blocked or is incomplete.
        """

        # Preparing the message history to send
        request_message = Content._from_gapic(
            _to_content(value=content, role=self._USER_ROLE)
        )
        request_history = list(self._history)
        request_history.append(request_message)

        response = await self._model._generate_content_async(
            contents=request_history,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            labels=labels,
        )
        # By default we're not adding incomplete interactions to history.
        if self._response_validator is not None:
            self._response_validator(
                response=response,
                request_contents=request_history,
                response_chunks=[response],
            )

        # Adding the request and the first response candidate to history
        response_message = response.candidates[0].content
        # Response role is NOT set by the model.
        response_message.role = self._MODEL_ROLE
        self._history.append(request_message)
        self._history.append(response_message)
        return response

    def _send_message_streaming(
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Iterable["GenerationResponse"]:
        """Generates content.

        Args:
            content: Content to send to the model.
                Supports a value that can be converted to a Part or a list of such values.
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            labels: Labels that will be passed to billing for cost tracking.

        Yields:
            A stream of GenerationResponse objects

        Raises:
            ResponseValidationError: If the response was blocked or is incomplete.
        """

        # Preparing the message history to send
        request_message = Content._from_gapic(
            _to_content(value=content, role=self._USER_ROLE)
        )
        request_history = list(self._history)
        request_history.append(request_message)

        stream = self._model._generate_content_streaming(
            contents=request_history,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            labels=labels,
        )
        chunks = []
        full_response = None
        for chunk in stream:
            chunks.append(chunk)
            # By default we're not adding incomplete interactions to history.
            if self._response_validator is not None:
                self._response_validator(
                    response=chunk,
                    request_contents=request_history,
                    response_chunks=chunks,
                )
            if full_response:
                _append_response(full_response, chunk)
            else:
                full_response = chunk
            yield chunk
        if not full_response:
            return

        # Adding the request and the first response candidate to history
        response_message = full_response.candidates[0].content
        # Response role is NOT set by the model.
        response_message.role = self._MODEL_ROLE
        self._history.append(request_message)
        self._history.append(response_message)

    async def _send_message_streaming_async(
        self,
        content: PartsType,
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> AsyncIterable["GenerationResponse"]:
        """Generates content asynchronously.

        Args:
            content: Content to send to the model.
                Supports a value that can be converted to a Part or a list of such values.
                Supports
                * str, Image, Part,
                * List[Union[str, Image, Part]],
            generation_config: Parameters for the generation.
            safety_settings: Safety settings as a mapping from HarmCategory to HarmBlockThreshold.
            tools: A list of tools (functions) that the model can try calling.
            labels: Labels that will be passed to billing for cost tracking.

        Returns:
            An awaitable for a stream of GenerationResponse objects

        Raises:
            ResponseValidationError: If the response was blocked or is incomplete.
        """
        # Preparing the message history to send
        request_message = Content._from_gapic(
            _to_content(value=content, role=self._USER_ROLE)
        )
        request_history = list(self._history)
        request_history.append(request_message)

        stream = await self._model._generate_content_streaming_async(
            contents=request_history,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            labels=labels,
        )

        async def async_generator():
            chunks = []
            full_response = None
            async for chunk in stream:
                chunks.append(chunk)
                # By default we're not adding incomplete interactions to history.
                if self._response_validator is not None:
                    self._response_validator(
                        response=chunk,
                        request_contents=request_history,
                        response_chunks=chunks,
                    )
                if full_response:
                    _append_response(full_response, chunk)
                else:
                    full_response = chunk
                yield chunk
            if not full_response:
                return
            # Adding the request and the first response candidate to history
            response_message = full_response.candidates[0].content
            # Response role is NOT set by the model.
            response_message.role = self._MODEL_ROLE
            self._history.append(request_message)
            self._history.append(response_message)

        return async_generator()


class _PreviewChatSession(ChatSession):
    __doc__ = ChatSession.__doc__

    # This class preserves backwards compatibility with the `raise_on_blocked` parameter.

    def __init__(
        self,
        model: _GenerativeModel,
        *,
        history: Optional[List["Content"]] = None,
        response_validation: bool = True,
        # Preview features:
        responder: Optional["AutomaticFunctionCallingResponder"] = None,
        # Deprecated
        raise_on_blocked: Optional[bool] = None,
    ):
        if raise_on_blocked is not None:
            warnings.warn(
                message="Use `response_validation` instead of `raise_on_blocked`."
            )
            if response_validation is not None:
                raise ValueError(
                    "Cannot use `response_validation` when `raise_on_blocked` is set."
                )
            response_validation = raise_on_blocked
        super().__init__(
            model=model,
            history=history,
            response_validation=response_validation,
        )
        self._responder = responder


class ResponseBlockedError(Exception):
    def __init__(
        self,
        message: str,
        request_contents: List["Content"],
        responses: List["GenerationResponse"],
    ):
        super().__init__(message)
        self.request = request_contents
        self.responses = responses


class ResponseValidationError(ResponseBlockedError):
    pass


### Structures


class GenerationConfig:
    """Parameters for the generation."""

    Modality = gapic_content_types.GenerationConfig.Modality
    ModelConfig = gapic_content_types.GenerationConfig.ModelConfig

    def __init__(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        audio_timestamp: Optional[bool] = None,
        routing_config: Optional["RoutingConfig"] = None,
        logprobs: Optional[int] = None,
        response_logprobs: Optional[bool] = None,
        response_modalities: Optional[List["GenerationConfig.Modality"]] = None,
        model_config: Optional["GenerationConfig.ModelConfig"] = None,
    ):
        r"""Constructs a GenerationConfig object.

        Args:
            temperature: Controls the randomness of predictions. Range: [0.0, 1.0]
            top_p: If specified, nucleus sampling will be used. Range: (0.0, 1.0]
            top_k: If specified, top-k sampling will be used.
            candidate_count: Number of candidates to generate.
            seed: Random seed for the generation.
            max_output_tokens: The maximum number of output tokens to generate per message.
            stop_sequences: A list of stop sequences.
            presence_penalty: Positive values penalize tokens that have appeared in the generated text,
                thus increasing the possibility of generating more diversed topics. Range: [-2.0, 2.0]
            frequency_penalty: Positive values penalize tokens that repeatedly appear in the generated
                text, thus decreasing the possibility of repeating the same content. Range: [-2.0, 2.0]
            response_mime_type: Output response mimetype of the generated
                candidate text. Supported mimetypes:

                -  ``text/plain``: (default) Text output.
                -  ``application/json``: JSON response in the candidates.
                -  ``text/x.enum``: enum response in the candidates. Only valid when
                    response_schema is provided.

                The model needs to be prompted to output the appropriate
                response type, otherwise the behavior is undefined.
            response_schema: Output response schema of the genreated candidate text.
            audio_timestamp: If true, the timestamp of the audio will be included in the response.
            routing_config: Model routing preference set in the request. This field is deprecated,
                please use model_config field instead for model optimizer requests.
            logprobs: Logit probabilities.
            reponse_logprobs: If true, export the logprobs results in response.
            model_config: Sets cost vs quality preference for model routing requests.

        Usage:

            ```
            response = model.generate_content(
                "Why is sky blue?",
                generation_config=GenerationConfig(
                    temperature=0.1,
                    top_p=0.95,
                    top_k=20,
                    candidate_count=1,
                    max_output_tokens=100,
                    stop_sequences=["\n\n\n"],
                    seed=5,
                    audio_timestamp=True,
                )
            )
            ```
        """
        if response_schema is None:
            raw_schema = None
        else:
            raw_schema = FunctionDeclaration(
                name="tmp", parameters=response_schema
            )._raw_function_declaration.parameters
        self._raw_generation_config = gapic_content_types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_mime_type=response_mime_type,
            response_schema=raw_schema,
            seed=seed,
            audio_timestamp=audio_timestamp,
            logprobs=logprobs,
            response_logprobs=response_logprobs,
            response_modalities=response_modalities,
            model_config=model_config,
        )
        if routing_config is not None:
            self._raw_generation_config.routing_config = (
                routing_config._gapic_routing_config
            )

    @classmethod
    def _from_gapic(
        cls,
        raw_generation_config: gapic_content_types.GenerationConfig,
    ) -> "GenerationConfig":
        response = cls()
        response._raw_generation_config = raw_generation_config
        return response

    @classmethod
    def from_dict(cls, generation_config_dict: Dict[str, Any]) -> "GenerationConfig":
        generation_config_dict = copy.deepcopy(generation_config_dict)
        response_schema = generation_config_dict.get("response_schema")
        if response_schema:
            _fix_schema_dict_for_gapic_in_place(response_schema)
        raw_generation_config = _dict_to_proto(
            gapic_content_types.GenerationConfig, generation_config_dict
        )
        return cls._from_gapic(raw_generation_config=raw_generation_config)

    def to_dict(self) -> Dict[str, Any]:
        return _proto_to_dict(self._raw_generation_config)

    def __repr__(self) -> str:
        return self._raw_generation_config.__repr__()

    class RoutingConfig:
        r"""The configuration for model router requests.
        Deprecated, please use ModelConfig to set routing preference instead.

        The routing config is either one of the two nested classes:
        - AutoRoutingMode: Automated routing.
        - ManualRoutingMode: Manual routing.

        Usage:
        - AutoRoutingMode:

            ```
            routing_config=generative_models.RoutingConfig(
                routing_config=generative_models.RoutingConfig.AutoRoutingMode(
                    model_routing_preference=generative_models.RoutingConfig.AutoRoutingMode.ModelRoutingPreference.BALANCED,
                ),
            )
            ```
        - ManualRoutingMode:

            ```
            routing_config=generative_models.RoutingConfig(
                routing_config=generative_models.RoutingConfig.ManutalRoutingMode(
                    model_name="gemini-1.5-pro-001",
                ),
            )
            ```
        """

        def __init__(
            self,
            *,
            routing_config: Union[
                "GenerationConfig.RoutingConfig.AutoRoutingMode",
                "GenerationConfig.RoutingConfig.ManualRoutingMode",
            ],
        ):
            _LOGGER = base.Logger(__name__)
            _LOGGER.warning(
                "RoutingConfig is deprecated, please use ModelConfig to set routing preference instead."
            )

            if isinstance(routing_config, self.AutoRoutingMode):
                self._gapic_routing_config = (
                    gapic_content_types.GenerationConfig.RoutingConfig(
                        auto_mode=routing_config._gapic_auto_mode
                    )
                )
            else:
                self._gapic_routing_config = (
                    gapic_content_types.GenerationConfig.RoutingConfig(
                        manual_mode=routing_config._gapic_manual_mode
                    )
                )

        def __repr__(self):
            return self._gapic_routing_config.__repr__()

        class AutoRoutingMode:
            r"""When automated routing is specified, the routing will be
            determined by the routing model predicted quality and customer provided
            model routing preference.
            """

            ModelRoutingPreference = (
                gapic_content_types.GenerationConfig.RoutingConfig.AutoRoutingMode.ModelRoutingPreference
            )

            def __init__(
                self,
                *,
                model_routing_preference: "GenerationConfig.RoutingConfig.AutoRoutingMode.ModelRoutingPreference",
            ):
                r"""AutoRouingMode constructor

                Args:
                    model_routing_preference: Model routing preference for the routing request.
                """
                self._gapic_auto_mode = (
                    gapic_content_types.GenerationConfig.RoutingConfig.AutoRoutingMode(
                        model_routing_preference=model_routing_preference
                    )
                )

            def __repr__(self):
                return self._gapic_auto_mode.__repr__()

        class ManualRoutingMode:
            r"""When manual routing is set, the specified model will be used
            directly.
            """

            def __init__(
                self,
                *,
                model_name: str,
            ):
                r"""ManualRoutingMode constructor

                Args:
                    model_name: The model to use. Only public LLM model names and those that are supported by the router are allowed.
                """
                self._gapic_manual_mode = gapic_content_types.GenerationConfig.RoutingConfig.ManualRoutingMode(
                    model_name=model_name
                )

            def __repr__(self):
                return self._gapic_manual_mode.__repr__()


class Tool:
    r"""A collection of functions that the model may use to generate response.

    Usage:
        Create tool from function declarations:

        ```
        get_current_weather_func = generative_models.FunctionDeclaration(...)
        weather_tool = generative_models.Tool(
            function_declarations=[get_current_weather_func],
        )
        ```

        Use tool in `GenerativeModel.generate_content`:

        ```
        model = GenerativeModel("gemini-pro")
        print(model.generate_content(
            "What is the weather like in Boston?",
            # You can specify tools when creating a model to avoid having to send them with every request.
            tools=[weather_tool],
        ))
        ```

        Use tool in chat:

        ```
        model = GenerativeModel(
            "gemini-pro",
            # You can specify tools when creating a model to avoid having to send them with every request.
            tools=[weather_tool],
        )
        chat = model.start_chat()
        print(chat.send_message("What is the weather like in Boston?"))
        print(chat.send_message(
            Part.from_function_response(
                name="get_current_weather",
                response={
                    "content": {"weather_there": "super nice"},
                }
            ),
        ))
        ```
    """

    _raw_tool: gapic_tool_types.Tool

    def __init__(
        self,
        function_declarations: List["FunctionDeclaration"],
    ):
        gapic_function_declarations = [
            function_declaration._raw_function_declaration
            for function_declaration in function_declarations
        ]
        self._raw_tool = gapic_tool_types.Tool(
            function_declarations=gapic_function_declarations
        )
        callable_functions = {
            function_declaration._raw_function_declaration.name: function_declaration
            for function_declaration in function_declarations
            if isinstance(function_declaration, CallableFunctionDeclaration)
        }
        self._callable_functions = callable_functions

    @classmethod
    def from_function_declarations(
        cls,
        function_declarations: List["FunctionDeclaration"],
    ) -> "Tool":
        return Tool(function_declarations=function_declarations)

    @classmethod
    def from_retrieval(
        cls,
        retrieval: Union["grounding.Retrieval"],
    ) -> "Tool":
        raw_tool = gapic_tool_types.Tool(retrieval=retrieval._raw_retrieval)
        return cls._from_gapic(raw_tool=raw_tool)

    @classmethod
    def from_google_search_retrieval(
        cls,
        google_search_retrieval: "grounding.GoogleSearchRetrieval",
    ) -> "Tool":
        raw_tool = gapic_tool_types.Tool(
            google_search_retrieval=google_search_retrieval._raw_google_search_retrieval
        )
        return cls._from_gapic(raw_tool=raw_tool)

    @classmethod
    def _from_gapic(
        cls,
        raw_tool: gapic_tool_types.Tool,
    ) -> "Tool":
        response = cls([])
        response._raw_tool = raw_tool
        return response

    @classmethod
    def from_dict(cls, tool_dict: Dict[str, Any]) -> "Tool":
        tool_dict = copy.deepcopy(tool_dict)
        for function_declaration in tool_dict.get("function_declarations") or []:
            parameters = function_declaration.get("parameters")
            if parameters:
                _fix_schema_dict_for_gapic_in_place(parameters)
        raw_tool = _dict_to_proto(aiplatform_types.Tool, tool_dict)
        return cls._from_gapic(raw_tool=raw_tool)

    def to_dict(self) -> Dict[str, Any]:
        return _proto_to_dict(self._raw_tool)

    def __repr__(self) -> str:
        return self._raw_tool.__repr__()


class ToolConfig:
    r"""Config shared for all tools provided in the request.

    Usage:
        Create ToolConfig

        ```
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                allowed_function_names=["get_current_weather_func"],
        ))
        ```

        Use ToolConfig in `GenerativeModel.generate_content`:

        ```
        model = GenerativeModel("gemini-pro")
        print(model.generate_content(
            "What is the weather like in Boston?",
            # You can specify tools when creating a model to avoid having to send them with every request.
            tools=[weather_tool],
            tool_config=tool_config,
        ))
        ```

        Use ToolConfig in chat:

        ```
        model = GenerativeModel(
            "gemini-pro",
            # You can specify tools when creating a model to avoid having to send them with every request.
            tools=[weather_tool],
            tool_config=tool_config,
        )
        chat = model.start_chat()
        print(chat.send_message("What is the weather like in Boston?"))
        print(chat.send_message(
            Part.from_function_response(
                name="get_current_weather",
                response={
                    "content": {"weather_there": "super nice"},
                }
            ),
        ))
        ```
    """

    class FunctionCallingConfig:
        Mode = gapic_tool_types.FunctionCallingConfig.Mode

        def __init__(
            self,
            mode: "ToolConfig.FunctionCallingConfig.Mode",
            allowed_function_names: Optional[List[str]] = None,
        ):
            """Constructs FunctionCallingConfig.

            Args:
                mode: Enum describing the function calling mode
                allowed_function_names: A list of allowed function names
                    (must match from Tool). Only set when the Mode is ANY.
            """
            self._gapic_function_calling_config = (
                gapic_tool_types.FunctionCallingConfig(
                    mode=mode,
                    allowed_function_names=allowed_function_names,
                )
            )

        def __repr__(self) -> str:
            return self._gapic_function_calling_config.__repr__()

    def __init__(self, function_calling_config: "ToolConfig.FunctionCallingConfig"):
        self._gapic_tool_config = gapic_tool_types.ToolConfig(
            function_calling_config=function_calling_config._gapic_function_calling_config
        )

    @classmethod
    def _from_gapic(
        cls,
        gapic_tool_config: gapic_tool_types.ToolConfig,
    ) -> "ToolConfig":
        response = cls.__new__(cls)
        response._gapic_tool_config = gapic_tool_config
        return response

    def __repr__(self) -> str:
        return self._gapic_tool_config.__repr__()


class FunctionDeclaration:
    r"""A representation of a function declaration.

    Usage:
        Create function declaration and tool:

        ```
        get_current_weather_func = generative_models.FunctionDeclaration(
            name="get_current_weather",
            description="Get the current weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "celsius",
                            "fahrenheit",
                        ]
                    }
                },
                "required": [
                    "location"
                ]
            },
            # Optional:
            response={
                "type": "object",
                "properties": {
                    "weather": {
                        "type": "string",
                        "description": "The weather in the city"
                    },
                },
            },
        )
        weather_tool = generative_models.Tool(
            function_declarations=[get_current_weather_func],
        )
        ```

        Use tool in `GenerativeModel.generate_content`:

        ```
        model = GenerativeModel("gemini-pro")
        print(model.generate_content(
            "What is the weather like in Boston?",
            # You can specify tools when creating a model to avoid having to send them with every request.
            tools=[weather_tool],
        ))
        ```

        Use tool in chat:

        ```
        model = GenerativeModel(
            "gemini-pro",
            # You can specify tools when creating a model to avoid having to send them with every request.
            tools=[weather_tool],
        )
        chat = model.start_chat()
        print(chat.send_message("What is the weather like in Boston?"))
        print(chat.send_message(
            Part.from_function_response(
                name="get_current_weather",
                response={
                    "content": {"weather_there": "super nice"},
                }
            ),
        ))
        ```
    """

    def __init__(
        self,
        *,
        name: str,
        parameters: Dict[str, Any],
        description: Optional[str] = None,
        response: Dict[str, Any] = None,
    ):
        """Constructs a FunctionDeclaration.

        Args:
            name: The name of the function that the model can call.
            parameters: Describes the parameters to this function in JSON Schema Object format.
            description: Description and purpose of the function.
                Model uses it to decide how and whether to call the function.
            response: Describes the response type of this function in JSON Schema format.
        """
        parameters = copy.deepcopy(parameters)
        _fix_schema_dict_for_gapic_in_place(parameters)
        raw_schema = _dict_to_proto(aiplatform_types.Schema, parameters)

        if response:
            response = copy.deepcopy(response)
            _fix_schema_dict_for_gapic_in_place(response)
            raw_response_schema = _dict_to_proto(aiplatform_types.Schema, response)
        else:
            raw_response_schema = None

        self._raw_function_declaration = gapic_tool_types.FunctionDeclaration(
            name=name,
            description=description,
            parameters=raw_schema,
            response=raw_response_schema,
        )

    @classmethod
    def from_func(cls, func: Callable[..., Any]) -> "CallableFunctionDeclaration":
        return CallableFunctionDeclaration.from_func(func)

    def to_dict(self) -> Dict[str, Any]:
        return _proto_to_dict(self._raw_function_declaration)

    def __repr__(self) -> str:
        return self._raw_function_declaration.__repr__()


# TODO: Remove this function once Reasoning Engines moves away from it.
def _convert_schema_dict_to_gapic(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a JsonSchema to a dict that the GAPIC Schema class accepts."""
    gapic_schema_dict = copy.copy(schema_dict)
    if "type" in gapic_schema_dict:
        gapic_schema_dict["type_"] = gapic_schema_dict.pop("type").upper()
    if "format" in gapic_schema_dict:
        gapic_schema_dict["format_"] = gapic_schema_dict.pop("format")
    if "items" in gapic_schema_dict:
        gapic_schema_dict["items"] = _convert_schema_dict_to_gapic(
            gapic_schema_dict["items"]
        )
    properties = gapic_schema_dict.get("properties")
    if properties:
        for property_name, property_schema in properties.items():
            properties[property_name] = _convert_schema_dict_to_gapic(property_schema)
    return gapic_schema_dict


def _remove_dollar_from_keys(schema_dict: Dict[str, Any]) -> None:
    """Renames, e.g., "$defs" to "defs" in-place."""
    for dollar_name in ["$defs", "$ref"]:
        value = schema_dict.pop(dollar_name, None)
        if value is not None:
            schema_dict[dollar_name[1:]] = value


def _as_camel_case(name: str) -> str:
    """Returns the `name`, which may be in snake_case, in lowerCamelCase."""
    words = name.split("_")
    return words.pop(0) + "".join(word.title() for word in words)


def _rename_snake_to_camel_keys(schema_dict: Dict[str, Any]) -> None:
    """Renames, e.g., "max_items" to "maxItems" in-place."""
    keys = list(schema_dict)  # cache so we can update while iterating
    for key in keys:
        value = schema_dict.pop(key)
        schema_dict[_as_camel_case(key)] = value


def _fix_schema_dict_for_gapic_in_place(schema_dict: Dict[str, Any]) -> None:
    """Converts a JsonSchema to a dict that the Schema proto class accepts."""
    # Standardize keys so we don't have to consider multiple spellings below.
    _remove_dollar_from_keys(schema_dict)
    _rename_snake_to_camel_keys(schema_dict)

    if "type" in schema_dict:
        schema_dict["type"] = schema_dict["type"].upper()

    if items_schema := schema_dict.get("items"):
        _fix_schema_dict_for_gapic_in_place(items_schema)

    if prefixes := schema_dict.get("prefixItems"):
        for prefix_schema in prefixes:
            _fix_schema_dict_for_gapic_in_place(prefix_schema)

    if properties := schema_dict.get("properties"):
        for property_schema in properties.values():
            _fix_schema_dict_for_gapic_in_place(property_schema)
        if "propertyOrdering" not in schema_dict:
            schema_dict["propertyOrdering"] = list(properties.keys())

    # The "additionalProperties" field may be set to a sub-schema or a boolean.
    # To avoid this polymorphism, we eliminate boolean values as follows:
    #   False: This means that no additional properties are allowed, besides
    #          those listed in "properties".  We remove this because that is
    #          already our default behavior.
    #    True: This means that additional properties are allowed with any value
    #          type.  We replace this with an equivalent empty dict:
    #          https://screenshot.googleplex.com/yvgmAmZay5Dw7qY
    if (additional := schema_dict.get("additionalProperties")) is not None:
        if additional is False:
            del schema_dict["additionalProperties"]
        elif additional is True:
            schema_dict["additionalProperties"] = {}
        else:
            _fix_schema_dict_for_gapic_in_place(additional)

    if any_of := schema_dict.get("anyOf"):
        for any_of_schema in any_of:
            _fix_schema_dict_for_gapic_in_place(any_of_schema)

    if defs := schema_dict.get("defs"):
        for def_schema in defs.values():
            _fix_schema_dict_for_gapic_in_place(def_schema)


class CallableFunctionDeclaration(FunctionDeclaration):
    """A function declaration plus a function."""

    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        parameters: Dict[str, Any],
        description: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
        )
        self._function = function

    @classmethod
    def from_func(cls, func: Callable[..., Any]) -> "CallableFunctionDeclaration":
        """Automatically creates a CallableFunctionDeclaration from a Python function.

        The function parameter schema is automatically extracted.
        Args:
          func: The function from which to extract schema.

        Returns:
            CallableFunctionDeclaration.
        """
        from vertexai.generative_models import (
            _function_calling_utils,
        )

        function_schema = _function_calling_utils.generate_json_schema_from_function(
            func
        )
        # Getting out the description first since it will be removed from the schema.
        function_description = function_schema["description"]
        function_schema = (
            _function_calling_utils.adapt_json_schema_to_google_tool_schema(
                function_schema
            )
        )

        return CallableFunctionDeclaration(
            name=func.__name__,
            function=func,
            description=function_description,
            parameters=function_schema,
        )


class GenerationResponse:
    """The response from the model."""

    def __init__(self):
        raw_response = gapic_prediction_service_types.GenerateContentResponse()
        self._raw_response = raw_response

    @classmethod
    def _from_gapic(
        cls,
        raw_response: gapic_prediction_service_types.GenerateContentResponse,
    ) -> "GenerationResponse":
        response = cls()
        response._raw_response = raw_response
        return response

    @classmethod
    def from_dict(cls, response_dict: Dict[str, Any]) -> "GenerationResponse":
        raw_response = _dict_to_proto(
            gapic_prediction_service_types.GenerateContentResponse, response_dict
        )
        return cls._from_gapic(raw_response=raw_response)

    def to_dict(self) -> Dict[str, Any]:
        return _proto_to_dict(self._raw_response)

    def __repr__(self) -> str:
        return self._raw_response.__repr__()

    @property
    def candidates(self) -> List["Candidate"]:
        return [
            Candidate._from_gapic(raw_candidate=raw_candidate)
            for raw_candidate in self._raw_response.candidates
        ]

    # GenerationPart properties
    @property
    def text(self) -> str:
        if len(self.candidates) > 1:
            raise ValueError(
                "The response has multiple candidates."
                " Use `response.candidate[i].text` to get text of a particular candidate."
            )
        if not self.candidates:
            raise ValueError(
                "Response has no candidates (and thus no text)."
                " The response is likely blocked by the safety filters.\n"
                "Response:\n" + _dict_to_pretty_string(self.to_dict())
            )
        try:
            return self.candidates[0].text
        except (ValueError, AttributeError) as e:
            # Enrich the error message with the whole Response.
            # The Candidate object does not have full information.
            raise ValueError(
                "Cannot get the response text.\n"
                f"{e}\n"
                "Response:\n" + _dict_to_pretty_string(self.to_dict())
            ) from e

    @property
    def prompt_feedback(
        self,
    ) -> gapic_prediction_service_types.GenerateContentResponse.PromptFeedback:
        return self._raw_response.prompt_feedback

    @property
    def usage_metadata(
        self,
    ) -> gapic_prediction_service_types.GenerateContentResponse.UsageMetadata:
        return self._raw_response.usage_metadata


class Candidate:
    """A response candidate generated by the model."""

    def __init__(self):
        raw_candidate = gapic_content_types.Candidate()
        self._raw_candidate = raw_candidate

    @classmethod
    def _from_gapic(cls, raw_candidate: gapic_content_types.Candidate) -> "Candidate":
        candidate = cls()
        candidate._raw_candidate = raw_candidate
        return candidate

    @classmethod
    def from_dict(cls, candidate_dict: Dict[str, Any]) -> "Candidate":
        raw_candidate = _dict_to_proto(gapic_content_types.Candidate, candidate_dict)
        return cls._from_gapic(raw_candidate=raw_candidate)

    def to_dict(self) -> Dict[str, Any]:
        return _proto_to_dict(self._raw_candidate)

    def __repr__(self) -> str:
        return self._raw_candidate.__repr__()

    @property
    def content(self) -> "Content":
        return Content._from_gapic(
            raw_content=self._raw_candidate.content,
        )

    @property
    def avg_logprobs(self) -> float:
        return self._raw_candidate.avg_logprobs

    @property
    def logprobs_result(self) -> gapic_content_types.LogprobsResult:
        return self._raw_candidate.logprobs_result

    @property
    def finish_reason(self) -> gapic_content_types.Candidate.FinishReason:
        return self._raw_candidate.finish_reason

    @property
    def finish_message(self) -> str:
        return self._raw_candidate.finish_message

    @property
    def index(self) -> int:
        return self._raw_candidate.index

    @property
    def safety_ratings(self) -> Sequence[gapic_content_types.SafetyRating]:
        return self._raw_candidate.safety_ratings

    @property
    def citation_metadata(self) -> gapic_content_types.CitationMetadata:
        return self._raw_candidate.citation_metadata

    @property
    def grounding_metadata(self) -> gapic_content_types.GroundingMetadata:
        return self._raw_candidate.grounding_metadata

    # GenerationPart properties
    @property
    def text(self) -> str:
        try:
            return self.content.text
        except (ValueError, AttributeError) as e:
            # Enrich the error message with the whole Candidate.
            # The Content object does not have full information.
            raise ValueError(
                "Cannot get the Candidate text.\n"
                f"{e}\n"
                "Candidate:\n" + _dict_to_pretty_string(self.to_dict())
            ) from e

    @property
    def function_calls(self) -> Sequence["FunctionCall"]:
        if not self.content or not self.content.parts:
            return []
        return [
            part.function_call
            for part in self.content.parts
            if part._raw_part._pb.WhichOneof("data") == "function_call"
        ]


class Content:
    r"""The multi-part content of a message.

    Usage:

        ```
        response = model.generate_content(contents=[
            Content(role="user", parts=[Part.from_text("Why is sky blue?")])
        ])
        ```
    """

    def __init__(
        self,
        *,
        parts: List["Part"] = None,
        role: Optional[str] = None,
    ):
        raw_parts = [part._raw_part for part in parts or []]
        self._raw_content = gapic_content_types.Content(parts=raw_parts, role=role)

    @classmethod
    def _from_gapic(cls, raw_content: gapic_content_types.Content) -> "Content":
        content = cls()
        content._raw_content = raw_content
        return content

    @classmethod
    def from_dict(cls, content_dict: Dict[str, Any]) -> "Content":
        raw_content = _dict_to_proto(gapic_content_types.Content, content_dict)
        return cls._from_gapic(raw_content=raw_content)

    def to_dict(self) -> Dict[str, Any]:
        return _proto_to_dict(self._raw_content)

    def __repr__(self) -> str:
        return self._raw_content.__repr__()

    @property
    def parts(self) -> List["Part"]:
        return [
            Part._from_gapic(raw_part=raw_part) for raw_part in self._raw_content.parts
        ]

    @property
    def role(self) -> str:
        return self._raw_content.role

    @role.setter
    def role(self, role: str) -> None:
        self._raw_content.role = role

    # GenerationPart properties
    @property
    def text(self) -> str:
        if len(self.parts) > 1:
            raise ValueError("Multiple content parts are not supported.")
        if not self.parts:
            raise ValueError(
                "Response candidate content has no parts (and thus no text)."
                " The candidate is likely blocked by the safety filters.\n"
                "Content:\n" + _dict_to_pretty_string(self.to_dict())
            )
        return self.parts[0].text


class Part:
    r"""A part of a multi-part Content message.

    Usage:

        ```
        text_part = Part.from_text("Why is sky blue?")
        image_part = Part.from_image(Image.load_from_file("image.jpg"))
        video_part = Part.from_uri(uri="gs://.../video.mp4", mime_type="video/mp4")
        function_response_part = Part.from_function_response(
            name="get_current_weather",
            response={
                "content": {"weather_there": "super nice"},
            }
        )

        response1 = model.generate_content([text_part, image_part])
        response2 = model.generate_content(video_part)
        response3 = chat.send_message(function_response_part)
        ```
    """

    def __init__(self):
        raw_part = gapic_content_types.Part()
        self._raw_part = raw_part

    @classmethod
    def _from_gapic(cls, raw_part: gapic_content_types.Part) -> "Part":
        part = cls()
        part._raw_part = raw_part
        return part

    @classmethod
    def from_dict(cls, part_dict: Dict[str, Any]) -> "Part":
        raw_part = _dict_to_proto(gapic_content_types.Part, part_dict)
        return cls._from_gapic(raw_part=raw_part)

    def __repr__(self) -> str:
        return self._raw_part.__repr__()

    @staticmethod
    def from_data(data: bytes, mime_type: str) -> "Part":
        return Part._from_gapic(
            raw_part=gapic_content_types.Part(
                inline_data=gapic_content_types.Blob(data=data, mime_type=mime_type)
            )
        )

    @staticmethod
    def from_uri(uri: str, mime_type: str) -> "Part":
        return Part._from_gapic(
            raw_part=gapic_content_types.Part(
                file_data=gapic_content_types.FileData(
                    file_uri=uri, mime_type=mime_type
                )
            )
        )

    @staticmethod
    def from_text(text: str) -> "Part":
        return Part._from_gapic(raw_part=gapic_content_types.Part(text=text))

    @staticmethod
    def from_image(image: "Image") -> "Part":
        return Part.from_data(data=image.data, mime_type=image._mime_type)

    @staticmethod
    def from_function_response(name: str, response: Dict[str, Any]) -> "Part":
        return Part._from_gapic(
            raw_part=gapic_content_types.Part(
                function_response=gapic_tool_types.FunctionResponse(
                    name=name,
                    response=response,
                )
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return _proto_to_dict(self._raw_part)

    @property
    def text(self) -> str:
        if "text" not in self._raw_part:
            raise AttributeError(
                "Response candidate content part has no text.\n"
                "Part:\n" + _dict_to_pretty_string(self.to_dict())
            )
        return self._raw_part.text

    @property
    def mime_type(self) -> Optional[str]:
        part_type = self._raw_part._pb.WhichOneof("data")
        if part_type == "inline_data":
            return self._raw_part.inline_data.mime_type
        if part_type == "file_data":
            return self._raw_part.file_data.mime_type
        raise AttributeError(f"Part has no mime_type.\nPart:\n{self.to_dict()}")

    @property
    def inline_data(self) -> gapic_content_types.Blob:
        return self._raw_part.inline_data

    @property
    def file_data(self) -> gapic_content_types.FileData:
        return self._raw_part.file_data

    @property
    def function_call(self) -> "FunctionCall":
        return (
            FunctionCall._from_gapic(self._raw_part.function_call)
            if self._raw_part.function_call
            else None
        )

    @property
    def function_response(self) -> gapic_tool_types.FunctionResponse:
        return self._raw_part.function_response

    @property
    def _image(self) -> "Image":
        return Image.from_bytes(data=self._raw_part.inline_data.data)


class FunctionCall:
    """Function call."""

    def __init__(
        self,
        *,
        name: str,
        args: Dict[str, Any],
    ):
        self._raw_message = aiplatform_types.FunctionCall(name=name, args=args)

    @classmethod
    def _from_gapic(cls, raw_message: aiplatform_types.FunctionCall) -> "FunctionCall":
        response = cls.__new__(cls)
        response._raw_message = raw_message
        return response

    def to_dict(self) -> Dict[str, Any]:
        function_call_dict = _proto_to_dict(self._raw_message)
        function_call_dict["args"] = self._convert_number_values(
            function_call_dict["args"]
        )
        return function_call_dict

    def __repr__(self) -> str:
        return self._raw_message.__repr__()

    @property
    def name(self) -> str:
        return self._raw_message.name

    @property
    def args(self) -> Dict[str, Any]:
        # We cannot use `type(self.args).to_dict(self.args)`
        # due to: AttributeError: type object 'MapComposite' has no attribute 'to_dict'
        args_dict = self.to_dict().get("args")
        return self._convert_number_values(args_dict)

    def _convert_number_values(self, data: Any) -> Any:
        """Converts float values with no decimal part to integers."""
        if isinstance(data, float) and data.is_integer():
            return int(data)
        elif isinstance(data, dict):
            return {
                key: self._convert_number_values(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._convert_number_values(item) for item in data]
        else:
            return data


class SafetySetting:
    """Parameters for the generation."""

    HarmCategory = gapic_content_types.HarmCategory
    HarmBlockMethod = gapic_content_types.SafetySetting.HarmBlockMethod
    HarmBlockThreshold = gapic_content_types.SafetySetting.HarmBlockThreshold

    def __init__(
        self,
        *,
        category: "SafetySetting.HarmCategory",
        threshold: "SafetySetting.HarmBlockThreshold",
        method: Optional["SafetySetting.HarmBlockMethod"] = None,
    ):
        r"""Safety settings.

        Args:
            category: Harm category.
            threshold: The harm block threshold.
            method: Specify if the threshold is used for probability or severity
                score. If not specified, the threshold is used for probability
                score.
        """
        self._raw_safety_setting = gapic_content_types.SafetySetting(
            category=category,
            threshold=threshold,
            method=method,
        )

    @classmethod
    def _from_gapic(
        cls,
        raw_safety_setting: gapic_content_types.SafetySetting,
    ) -> "SafetySetting":
        response = cls(
            category=raw_safety_setting.category,
            threshold=raw_safety_setting.threshold,
        )
        response._raw_safety_setting = raw_safety_setting
        return response

    @classmethod
    def from_dict(cls, safety_setting_dict: Dict[str, Any]) -> "SafetySetting":
        raw_safety_setting = _dict_to_proto(
            aiplatform_types.SafetySetting, safety_setting_dict
        )
        return cls._from_gapic(raw_safety_setting=raw_safety_setting)

    def to_dict(self) -> Dict[str, Any]:
        return _proto_to_dict(self._raw_safety_setting)

    def __repr__(self):
        return self._raw_safety_setting.__repr__()


class grounding:  # pylint: disable=invalid-name
    """Grounding namespace."""

    __module__ = "vertexai.generative_models"

    def __init__(self):
        raise RuntimeError("This class must not be instantiated.")

    class DynamicRetrievalConfig:
        """Config for dynamic retrieval."""

        Mode = gapic_tool_types.DynamicRetrievalConfig.Mode

        def __init__(
            self,
            mode: Mode = Mode.MODE_UNSPECIFIED,
            dynamic_threshold: Optional[float] = None,
        ):
            """Initializes a DynamicRetrievalConfig."""
            self._raw_dynamic_retrieval_config = (
                gapic_tool_types.DynamicRetrievalConfig(
                    mode=mode,
                    dynamic_threshold=dynamic_threshold,
                )
            )

    class GoogleSearchRetrieval:
        r"""Tool to retrieve public web data for grounding, powered by
        Google Search.
        """

        def __init__(
            self,
            dynamic_retrieval_config: Optional[
                "grounding.DynamicRetrievalConfig"
            ] = None,
        ):
            """Initializes a Google Search Retrieval tool."""
            self._raw_google_search_retrieval = gapic_tool_types.GoogleSearchRetrieval(
                dynamic_retrieval_config=(
                    dynamic_retrieval_config._raw_dynamic_retrieval_config
                    if dynamic_retrieval_config
                    else None
                )
            )

    class Retrieval:
        """Defines a retrieval tool that model can call to access external knowledge."""

        def __init__(
            self,
            source: Union["grounding.VertexAISearch"],
            disable_attribution: Optional[bool] = None,
        ):
            """Initializes a Retrieval tool.

            Args:
                source (VertexAISearch):
                    Set to use data source powered by Vertex AI Search.
                disable_attribution (bool):
                    Deprecated. Disable using the result from this
                    tool in detecting grounding attribution. This
                    does not affect how the result is given to the
                    model for generation.
            """
            self._raw_retrieval = gapic_tool_types.Retrieval(
                vertex_ai_search=source._raw_vertex_ai_search,
                disable_attribution=disable_attribution,
            )

    class VertexAISearch:
        r"""Retrieve from Vertex AI Search data store for grounding.
        See https://cloud.google.com/products/agent-builder
        """

        def __init__(
            self,
            datastore: str,
            *,
            project: Optional[str] = None,
            location: Optional[str] = None,
        ):
            """Initializes a Vertex AI Search tool.

            Args:
                datastore (str):
                    Required. Vertex AI Search data store resource name. Format:
                    ``projects/{project}/locations/{location}/collections/default_collection/dataStores/{data_store}``
                    or ``{data_store}``.
                project (str):
                    Optional. Project ID of the data store. Must provide either the full data store resource name or data store id, project ID, and location.
                location (str):
                    Optional. Location of the data store. Must provide either the full data store resource name or data store id, project ID, and location.
            """
            if not re.fullmatch(
                r"^projects/[a-z0-9-]*/locations/[a-z0-9][a-z0-9-]*/collections/[a-z0-9][a-z0-9-_]*/dataStores/[a-z0-9][a-z0-9-_]*$",
                datastore,
            ):
                if not project or not location:
                    raise ValueError(
                        "Must provide either the full data store resource name or data store id, project ID, and location."
                    )
                datastore = f"projects/{project}/locations/{location}/collections/default_collection/dataStores/{datastore}"
            self._raw_vertex_ai_search = gapic_tool_types.VertexAISearch(
                datastore=datastore,
            )


class preview_grounding(grounding):  # pylint: disable=invalid-name
    """Grounding namespace (preview)."""

    __name__ = "grounding"
    __module__ = "vertexai.preview.generative_models"

    def __init__(self):
        raise RuntimeError("This class must not be instantiated.")


def _to_content(
    value: Union[
        gapic_content_types.Content,
        Content,
        str,
        "Image",
        Part,
        gapic_content_types.Part,
        List[Union[str, "Image", Part, gapic_content_types.Part]],
    ],
    role: str = _GenerativeModel._USER_ROLE,
) -> gapic_content_types.Content:
    """Converts different kinds of values to gapic_content_types.Content object."""
    if not value:
        raise TypeError("value must not be empty")
    if isinstance(value, gapic_content_types.Content):
        return value
    if isinstance(value, Content):
        return value._raw_content
    if isinstance(value, (str, Image, Part, gapic_content_types.Part)):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise TypeError(f"Unexpected value type: {value}")
    parts: List[gapic_content_types.Part] = []
    for item in items:
        if isinstance(item, gapic_content_types.Part):
            part = item
        elif isinstance(item, Part):
            part = item._raw_part
        elif isinstance(item, str):
            part = gapic_content_types.Part(text=item)
        elif isinstance(item, Image):
            part = gapic_content_types.Part(
                inline_data=gapic_content_types.Blob(
                    data=item.data, mime_type=item._mime_type
                )
            )
        elif isinstance(item, gapic_content_types.Content):
            raise TypeError(f"A list of Content objects is not supported here: {items}")
        else:
            raise TypeError(
                f"Unexpected item type: {item}."
                "Only types that represent a single Content or a single Part are supported here."
            )
        parts.append(part)
    return gapic_content_types.Content(parts=parts, role=role)


def _append_response(
    base_response: GenerationResponse,
    new_response: GenerationResponse,
):
    _append_gapic_response(
        base_response=base_response._raw_response,
        new_response=new_response._raw_response,
    )


def _append_gapic_response(
    base_response: gapic_prediction_service_types.GenerateContentResponse,
    new_response: gapic_prediction_service_types.GenerateContentResponse,
):
    for idx, candidate in enumerate(new_response.candidates):
        if candidate.index != idx:
            raise ValueError(
                f"Incorrect new candidate ordering: {base_response.candidates}"
            )
        if idx < len(base_response.candidates):
            if base_response.candidates[idx].index != idx:
                raise ValueError(
                    f"Incorrect base candidate ordering: {base_response.candidates}"
                )
            _append_gapic_candidate(base_response.candidates[idx], candidate)
        else:
            if idx != len(base_response.candidates):
                raise ValueError(
                    f"Incorrect base candidate ordering: {base_response.candidates}"
                )
            base_response.candidates.append(candidate)
    # prompt_feedback is always taken from the base_response
    # usage_metadata is always taken from the new_response
    if new_response.usage_metadata:
        base_response.usage_metadata = new_response.usage_metadata


def _append_gapic_candidate(
    base_candidate: gapic_content_types.Candidate,
    new_candidate: gapic_content_types.Candidate,
):
    if base_candidate.index != new_candidate.index:
        raise ValueError(
            f"Incorrect candidate indexes: {base_candidate.index} != {new_candidate.index}"
        )

    # Only merge content if it exists.
    if "content" in new_candidate:
        _append_gapic_content(base_candidate.content, new_candidate.content)

    # For these attributes, the last value wins
    if new_candidate.finish_reason:
        base_candidate.finish_reason = new_candidate.finish_reason
    if new_candidate.safety_ratings:
        base_candidate.safety_ratings = new_candidate.safety_ratings
    if new_candidate.finish_message:
        base_candidate.finish_message = new_candidate.finish_message
    if new_candidate.citation_metadata:
        base_candidate.citation_metadata = new_candidate.citation_metadata


def _append_gapic_content(
    base_content: gapic_content_types.Content,
    new_content: gapic_content_types.Content,
):
    # Handling empty role is a workaround for a case when service returns
    # some chunks with missing role field (e.g. when response is blocked).
    if new_content.role and base_content.role != new_content.role:
        raise ValueError(
            f"Content roles do not match: {base_content.role} != {new_content.role}"
        )

    for part_idx, part in enumerate(new_content.parts):
        if part_idx < len(base_content.parts):
            _append_gapic_part(base_content.parts[part_idx], part)
        else:
            base_content.parts.append(part)


def _append_gapic_part(
    base_part: gapic_content_types.Part,
    new_part: gapic_content_types.Part,
):
    # Text is appended. For other cases, new wins.
    if new_part.text:
        base_part.text += new_part.text
    else:
        base_part._pb = copy.deepcopy(new_part._pb)


def _proto_to_dict(message) -> Dict[str, Any]:
    """Converts a proto-plus protobuf message to a dictionary."""
    # The best way to convert proto to dict is not trivial.
    # Ideally, we want original keys in snake_case.
    # The preserving_proto_field_name flag controls key names, but states have issues:
    # `False` leads to keys using camelCase instead of snake_case.
    # `True` leads to keys using snake_case, but has renamed names like `type_`.
    # We needs to fix this issue using _fix_renamed_proto_dict_keys_in_place.
    result = type(message).to_dict(
        message,
        including_default_value_fields=False,
        use_integers_for_enums=False,
    )
    _fix_renamed_proto_dict_keys_in_place(result)
    return result


def _dict_to_proto(message_type: Type[T], message_dict: Dict[str, Any]) -> T:
    """Converts a dictionary to a proto-plus protobuf message."""
    # We cannot just use `message = message_type(message_dict)` because
    # it fails for classes where GAPIC has renamed proto fields.
    message = message_type()
    json_format.ParseDict(message_dict, message._pb)
    return message


def _dict_to_pretty_string(d: dict) -> str:
    """Format dict as a pretty-printed JSON string."""
    return json.dumps(d, indent=2)


def _fix_renamed_proto_dict_keys_in_place(d: Mapping[str, Any]):
    """Fixes proto dict keys in place."""
    for key, value in list(d.items()):
        if key.endswith("_"):
            new_key = key.rstrip("_")
            del d[key]
            d[new_key] = value
        if isinstance(value, Mapping):
            _fix_renamed_proto_dict_keys_in_place(value)
        if isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                if isinstance(item, Mapping):
                    _fix_renamed_proto_dict_keys_in_place(item)


_FORMAT_TO_MIME_TYPE = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
}


class Image:
    """The image that can be sent to a generative model."""

    _image_bytes: bytes
    _loaded_image: Optional["PIL_Image.Image"] = None

    @staticmethod
    def load_from_file(location: str) -> "Image":
        """Loads image from file.

        Args:
            location: Local path from where to load the image.

        Returns:
            Loaded image as an `Image` object.
        """
        image_bytes = pathlib.Path(location).read_bytes()
        image = Image()
        image._image_bytes = image_bytes
        return image

    @staticmethod
    def from_bytes(data: bytes) -> "Image":
        """Loads image from image bytes.

        Args:
            data: Image bytes.

        Returns:
            Loaded image as an `Image` object.
        """
        image = Image()
        image._image_bytes = data
        return image

    @property
    def _pil_image(self) -> "PIL_Image.Image":
        if self._loaded_image is None:
            if not PIL_Image:
                raise RuntimeError(
                    "The PIL module is not available. Please install the Pillow package."
                )
            self._loaded_image = PIL_Image.open(io.BytesIO(self._image_bytes))
        return self._loaded_image

    @property
    def _mime_type(self) -> str:
        """Returns the MIME type of the image."""
        if PIL_Image:
            return _FORMAT_TO_MIME_TYPE[self._pil_image.format.lower()]
        else:
            # Fall back to jpeg
            return "image/jpeg"

    @property
    def data(self) -> bytes:
        """Returns the image data."""
        return self._image_bytes

    def _repr_png_(self):
        return self._pil_image._repr_png_()


class AutomaticFunctionCallingResponder:
    """Responder that automatically responds to model's function calls."""

    def __init__(self, max_automatic_function_calls: int = 1):
        """Initializes the responder.

        Args:
            max_automatic_function_calls: Maximum number of automatic function calls.
        """
        if not (max_automatic_function_calls > 0):
            raise ValueError("max_automatic_function_calls must be positive.")
        self._max_automatic_function_calls = max_automatic_function_calls

    def _create_responder_for_message(self, tools: List[Tool]) -> "_MessageResponder":
        return AutomaticFunctionCallingResponder._MessageResponder(
            tools=tools,
            max_automatic_function_calls=self._max_automatic_function_calls,
        )

    class _MessageResponder:
        """Automatic Function Calling responder for a single user message."""

        def __init__(
            self,
            *,
            tools: List[Tool],
            max_automatic_function_calls: int = 1,
            **_,
        ):
            self._tools = tools
            self._max_automatic_function_calls = max_automatic_function_calls
            self._remaining_function_calls = max_automatic_function_calls

        def respond_to_model_response(
            self,
            *,
            response: "GenerationResponse",
            **_,
        ) -> Optional["Content"]:
            """Responds to model's response.

            Args:
                response: Model's response that can be auto-responded.

            Returns:
                Optional response to model's response.
            """
            chosen_candidate = response.candidates[0]
            function_calls = chosen_candidate.function_calls
            if not function_calls:
                return None
            tools = self._tools or self._model._tools
            function_response_parts = []
            for function_call in function_calls:
                if self._remaining_function_calls > 0:
                    self._remaining_function_calls -= 1
                else:
                    raise RuntimeError(
                        f"Exceeded the maximum number of automatic function calls ({self._max_automatic_function_calls})."
                        " If more automatic function calls are needed, set `max_automatic_function_calls` to a higher number."
                        f" The last function calls: {function_calls}"
                    )
                callable_function = None
                for tool in tools:
                    new_callable_function = tool._callable_functions.get(
                        function_call.name
                    )
                    if new_callable_function and callable_function:
                        raise ValueError(
                            "Multiple functions with the same name are not supported."
                            f" Found {callable_function} and {new_callable_function}."
                        )
                    callable_function = new_callable_function
                if not callable_function:
                    raise RuntimeError(
                        f"""Model has asked to call function "{function_call.name}" which was not found."""
                    )

                try:
                    function_call_result = callable_function._function(
                        **function_call.args
                    )
                    if not isinstance(function_call_result, Mapping):
                        # If the function returns a single value, wrap it in the
                        # format that Part.from_function_response can accept.
                        function_call_result = {"result": function_call_result}
                except Exception as ex:
                    raise RuntimeError(
                        f"""Error raised when calling function "{function_call.name}" as requested by the model."""
                    ) from ex
                function_response_part = Part.from_function_response(
                    name=function_call.name,
                    response=function_call_result,
                )
                function_response_parts.append(function_response_part)
            function_response_content = Content(
                parts=function_response_parts,
            )
            return function_response_content


class GenerativeModel(_GenerativeModel):
    __module__ = "vertexai.generative_models"

    @functools.cached_property
    def _prediction_client(self) -> prediction_service_v1.PredictionServiceClient:
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        if api_key:
            raise ValueError(
                "Api keys are only supported with the preview namespace. "
                "Import the preview namespace instead:\n"
                "from vertexai.preview import generative_models"
            )
        return aiplatform_initializer.global_config.create_client(
            client_class=prediction_service_v1.PredictionServiceClient,
            location_override=self._location if not api_key else None,
            prediction_client=True,
            api_key=api_key,
        )

    @functools.cached_property
    def _prediction_async_client(
        self,
    ) -> prediction_service_v1.PredictionServiceAsyncClient:
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        if api_key:
            raise ValueError(
                "Api keys are only supported with the preview namespace. "
                "Import the preview namespace instead:\n"
                "from vertexai.preview import generative_models"
            )
        return aiplatform_initializer.global_config.create_client(
            client_class=prediction_service_v1.PredictionServiceAsyncClient,
            location_override=self._location if not api_key else None,
            prediction_client=True,
            api_key=api_key,
        )

    @functools.cached_property
    def _llm_utility_client(self) -> llm_utility_service_v1.LlmUtilityServiceClient:
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        if api_key:
            raise ValueError(
                "Api keys are only supported with the preview namespace. "
                "Import the preview namespace instead:\n"
                "from vertexai.preview import generative_models"
            )
        return aiplatform_initializer.global_config.create_client(
            client_class=llm_utility_service_v1.LlmUtilityServiceClient,
            location_override=self._location if not api_key else None,
            api_key=api_key,
        )

    @functools.cached_property
    def _llm_utility_async_client(
        self,
    ) -> llm_utility_service_v1.LlmUtilityServiceAsyncClient:
        api_key = aiplatform_initializer.global_config.api_key
        if api_key and aiplatform_initializer.global_config.project:
            api_key = None
        if api_key:
            raise ValueError(
                "Api keys are only supported with the preview namespace. "
                "Import the preview namespace instead:\n"
                "from vertexai.preview import generative_models"
            )
        return aiplatform_initializer.global_config.create_client(
            client_class=llm_utility_service_v1.LlmUtilityServiceAsyncClient,
            location_override=self._location if not api_key else None,
            api_key=api_key,
        )

    def _prepare_request(
        self,
        contents: ContentsType,
        *,
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        tools: Optional[List["Tool"]] = None,
        tool_config: Optional["ToolConfig"] = None,
        system_instruction: Optional[PartsType] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> types_v1.GenerateContentRequest:
        """Prepares a GAPIC GenerateContentRequest."""
        request_v1beta1 = super()._prepare_request(
            contents=contents,
            model=model,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
            labels=labels,
        )
        serialized_message_v1beta1 = type(request_v1beta1).serialize(request_v1beta1)
        try:
            response_v1 = types_v1.GenerateContentRequest.deserialize(
                serialized_message_v1beta1
            )
        except Exception as ex:
            raise ValueError(
                "Failed to convert GenerateContentRequest from v1beta1 to v1:\n"
                f"{serialized_message_v1beta1}"
            ) from ex
        return response_v1

    def _parse_response(
        self,
        response: types_v1.GenerateContentResponse,
    ) -> "GenerationResponse":
        response_v1beta1 = aiplatform_types.GenerateContentResponse.deserialize(
            type(response).serialize(response)
        )
        return super()._parse_response(
            response=response_v1beta1,
        )

    # The count_tokens methods need to be overridden since in v1, the
    # `count_tokens` method is implemented by the `LLMUtilityService` API
    # not the `PredictionService` API.
    def _gapic_count_tokens(
        self,
        prediction_resource_name: str,
        contents: List[types_v1.Content],
        system_instruction: Optional[types_v1.Content] = None,
        tools: Optional[List[types_v1.Tool]] = None,
    ) -> types_v1.CountTokensResponse:
        request = types_v1.CountTokensRequest(
            endpoint=prediction_resource_name,
            model=prediction_resource_name,
            contents=contents,
            system_instruction=system_instruction,
            tools=tools,
        )
        return self._llm_utility_client.count_tokens(request=request)

    async def _gapic_count_tokens_async(
        self,
        prediction_resource_name: str,
        contents: List[types_v1.Content],
        system_instruction: Optional[types_v1.Content] = None,
        tools: Optional[List[types_v1.Tool]] = None,
    ) -> types_v1.CountTokensResponse:
        request = types_v1.CountTokensRequest(
            endpoint=prediction_resource_name,
            model=prediction_resource_name,
            contents=contents,
            system_instruction=system_instruction,
            tools=tools,
        )
        return await self._llm_utility_async_client.count_tokens(request=request)

    # The compute_tokens methods need to be overridden since the request types differ.
    def _gapic_compute_tokens(
        self,
        prediction_resource_name: str,
        contents: List[gapic_content_types.Content],
    ) -> gapic_prediction_service_types.CountTokensResponse:
        request = types_v1.ComputeTokensRequest(
            endpoint=prediction_resource_name,
            model=prediction_resource_name,
            contents=contents,
        )
        return self._llm_utility_client.compute_tokens(request=request)

    async def _gapic_compute_tokens_async(
        self,
        prediction_resource_name: str,
        contents: List[gapic_content_types.Content],
    ) -> gapic_prediction_service_types.CountTokensResponse:
        request = types_v1.ComputeTokensRequest(
            endpoint=prediction_resource_name,
            model=prediction_resource_name,
            contents=contents,
        )
        return await self._llm_utility_async_client.compute_tokens(request=request)


class _PreviewGenerativeModel(_GenerativeModel):
    __name__ = "GenerativeModel"
    __module__ = "vertexai.preview.generative_models"

    def start_chat(
        self,
        *,
        history: Optional[List["Content"]] = None,
        response_validation: bool = True,
        # Preview features:
        responder: Optional["AutomaticFunctionCallingResponder"] = None,
    ) -> "ChatSession":
        """Creates a stateful chat session.

        Args:
            history: Previous history to initialize the chat session.
            response_validation: Whether to validate responses before adding
                them to chat history. By default, `send_message` will raise
                error if the request or response is blocked or if the response
                is incomplete due to going over the max token limit.
                If set to `False`, the chat session history will always
                accumulate the request and response messages even if the
                response if blocked or incomplete. This can result in an unusable
                chat session state.
            responder: An responder object that can automatically respond to
                some model messages. Supported responder classes:
                `AutomaticFunctionCallingResponder`.

        Returns:
            A ChatSession object.
        """
        return _PreviewChatSession(
            model=self,
            history=history,
            response_validation=response_validation,
            responder=responder,
        )

    def set_request_response_logging_config(
        self,
        *,
        enabled: bool,
        sampling_rate: float,
        bigquery_destination: str,
        enable_otel_logging: Optional[bool] = None,
    ) -> Union[aiplatform_types.PublisherModelConfig, aiplatform_types.Endpoint]:
        """
        Sets the request/response logging config.

        Args:
            enabled: If logging is enabled or not.
            sampling_rate: Percentage of requests to be logged, expressed as a
                fraction in range(0,1].
        bigquery_destination: BigQuery table for logging. If only given a project,
            a new dataset will be created with name
            ``logging_<endpoint-display-name>_<endpoint-id>`` where will
            be made BigQuery-dataset-name compatible (e.g. most special
            characters will become underscores). If no table name is
            given, a new table will be created with name
            ``request_response_logging``
        enable_otel_logging: This field is used for large models. If true, in
            addition to the original large model logs, logs will be converted in
            OTel schema format, and saved in otel_log column. Default
            value is false.
        Returns:
            The updated PublisherModelConfig or Endpoint.
        """

        logging_config = aiplatform_types.PredictRequestResponseLoggingConfig(
            enabled=enabled,
            sampling_rate=sampling_rate,
            bigquery_destination=aiplatform_types.BigQueryDestination(
                output_uri=bigquery_destination
            ),
            enable_otel_logging=enable_otel_logging,
        )

        if self._endpoint_client.parse_endpoint_path(self._prediction_resource_name):
            return self._endpoint_client.update_endpoint(
                aiplatform_types.UpdateEndpointRequest(
                    endpoint=aiplatform_types.Endpoint(
                        name=self._prediction_resource_name,
                        predict_request_response_logging_config=logging_config,
                    ),
                    update_mask=field_mask_pb2.FieldMask(
                        paths=["predict_request_response_logging_config"]
                    ),
                )
            )

        else:

            operation = self._endpoint_client.set_publisher_model_config(
                aiplatform_types.SetPublisherModelConfigRequest(
                    name=self._prediction_resource_name,
                    publisher_model_config=aiplatform_types.PublisherModelConfig(
                        logging_config=logging_config
                    ),
                )
            )

            return operation.result()
