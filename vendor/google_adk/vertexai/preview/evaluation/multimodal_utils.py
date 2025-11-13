"""Utility functions for multimodal evaluation."""

import logging
import re
from typing import Any, Dict, Union, List, Set

from google.cloud.aiplatform import base
from google.cloud.aiplatform_v1beta1.types import content
from google.cloud.aiplatform_v1beta1.types import (
    evaluation_service as gapic_eval_service_types,
)
from vertexai import generative_models
from vertexai.preview.evaluation import (
    prompt_template as prompt_template_base,
)

from google.protobuf import json_format


ContentMap = gapic_eval_service_types.ContentMap
Content = content.Content
Part = content.Part
_CONTENTS_DETECTOR = "contents {"
_PARTS_DETECTOR = "parts {"

_LOGGER = base.Logger(__name__)


def _string_to_content_list(input_str: str) -> ContentMap.Contents:
    """Converts a string to a list if possible, otherwise returns None."""
    try:
        return json_format.Parse(
            input_str,
            ContentMap.Contents.pb(ContentMap.Contents()),
        )
    except json_format.ParseError as e:
        if _CONTENTS_DETECTOR in input_str and _PARTS_DETECTOR in input_str:
            logging.warning(
                "Failed to parse %s to ContentMap.Contents: %s", input_str, e
            )
        return None


def _is_multimodal_response(response: str) -> bool:
    """Checks if the model response contains multimodal input."""
    content_list = _string_to_content_list(response)
    if content_list is None:
        if _CONTENTS_DETECTOR in response and _PARTS_DETECTOR in response:
            logging.warning(
                "Response contains multimodal input: %s. Please check whether"
                " the response format conforms to ContentMap type.",
                response,
            )
        return False
    else:
        return True


def is_multimodal_instance(
    model_based_metric_instance_input: Dict[str, str],
) -> bool:
    """Checks if the evaluation instance contains multimodal input."""
    for placeholder in model_based_metric_instance_input:
        if _is_multimodal_response(model_based_metric_instance_input[placeholder]):
            return True
    return False


def convert_multimodal_response_to_content_map(
    model_based_metric_instance_input: Dict[str, str],
) -> ContentMap:
    """Converts a multimodal model response to a ContentMap."""
    content_map = ContentMap()
    for placeholder in model_based_metric_instance_input.keys():
        content_list = _string_to_content_list(
            model_based_metric_instance_input[placeholder]
        )
        if content_list is None:
            content_map.values[placeholder] = ContentMap.Contents(
                contents=[
                    Content(
                        parts=[
                            Part(text=model_based_metric_instance_input[placeholder])
                        ]
                    )
                ]
            )
        else:
            content_map.values[placeholder] = content_list

    return content_map


def _split_metric_prompt_template(
    metric_prompt_template: str,
    placeholders: Set[str],
) -> List[str]:
    """Splits the metric prompt template into a list of strings by placeholders."""
    placeholders_with_brackets = [
        re.escape("{" + placeholder + "}") for placeholder in placeholders
    ]
    pattern = "|".join(f"({placeholder})" for placeholder in placeholders_with_brackets)
    split_metric_prompt_template = re.split(pattern, metric_prompt_template)
    return [element for element in split_metric_prompt_template if element]


def _assemble_multi_modal_prompt(
    metric_prompt_template: Union[prompt_template_base.PromptTemplate, str],
    data_row: Dict[str, Any],
    row_index: int,
    placeholders: Set[str],
) -> List[Union[str, generative_models.Part]]:
    """Fills in the split metric prompt template elements with multimodal data to be sent to the model."""
    split_template_elements = _split_metric_prompt_template(
        str(metric_prompt_template), placeholders
    )
    part_inputs = []
    for element in split_template_elements:
        placeholder = element.replace("{", "").replace("}", "")
        if placeholder in data_row.keys():
            content_list = _string_to_content_list(data_row[placeholder])
            if content_list is None:
                part_inputs.append(data_row[placeholder])
            else:
                for content_inp in content_list.contents:
                    for part in content_inp.parts:
                        if part.HasField("text"):
                            part_inputs.append(part.text)
                        elif part.HasField("file_data"):
                            part_inputs.append(
                                generative_models.Part.from_uri(
                                    part.file_data.file_uri,
                                    mime_type=part.file_data.mime_type,
                                )
                            )
                        else:
                            _LOGGER.warning(
                                "The multimodal input you provided "
                                f"at row {row_index} "
                                "contains part types that are not "
                                "yet supported. Currently supported"
                                "part types are text and file_data"
                            )
        else:
            part_inputs.append(element)
    return part_inputs
