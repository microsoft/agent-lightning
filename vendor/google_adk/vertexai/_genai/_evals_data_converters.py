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
"""Dataset converters for evals."""

import copy
import json
import logging
from typing import Any, Optional, Union

from google.genai import _common
from google.genai import types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from . import _evals_utils
from . import _observability_data_converter
from . import types


logger = logging.getLogger("vertexai_genai._evals_data_converters")


class EvalDatasetSchema(_common.CaseInSensitiveEnum):
    """Represents the schema of an evaluation dataset."""

    GEMINI = "gemini"
    FLATTEN = "flatten"
    OPENAI = "openai"
    OBSERVABILITY = "observability"
    UNKNOWN = "unknown"


_PLACEHOLDER_RESPONSE_TEXT = "Error: Missing response for this candidate"


def _create_placeholder_response_candidate(
    text: str = _PLACEHOLDER_RESPONSE_TEXT,
) -> types.ResponseCandidate:
    """Creates a ResponseCandidate with placeholder text."""
    return types.ResponseCandidate(
        response=genai_types.Content(parts=[genai_types.Part(text=text)])
    )


class _GeminiEvalDataConverter(_evals_utils.EvalDataConverter):
    """Converter for dataset in the Gemini format."""

    def _parse_request(self, request_data: dict[str, Any]) -> tuple[
        genai_types.Content,
        genai_types.Content,
        list[types.evals.Message],
        types.ResponseCandidate,
    ]:
        """Parses a request from a Gemini dataset."""
        system_instruction = genai_types.Content()
        prompt = genai_types.Content()
        reference = types.ResponseCandidate()
        conversation_history = []

        if "system_instruction" in request_data:
            system_instruction = genai_types.Content.model_validate(
                request_data["system_instruction"]
            )
        for turn_id, content_dict in enumerate(request_data.get("contents", [])):
            if not isinstance(content_dict, dict):
                raise TypeError(
                    "Expected a dictionary for content at turn %s, but got %s: %s"
                    % (turn_id, type(content_dict).__name__, content_dict)
                )
            if "parts" not in content_dict:
                raise ValueError(
                    "Missing 'parts' key in content structure at turn %s: %s"
                    % (turn_id, content_dict)
                )
            conversation_history.append(
                types.evals.Message(
                    turn_id=str(turn_id),
                    content=genai_types.Content.model_validate(content_dict),
                )
            )
        if conversation_history:
            last_message = conversation_history.pop()
            last_message_role = (
                last_message.content.role if last_message.content else "user"
            )
            if last_message_role in ["user", None]:
                prompt = (
                    last_message.content
                    if last_message.content
                    else genai_types.Content()
                )
            elif last_message_role == "model":
                reference = types.ResponseCandidate(response=last_message.content)
                if conversation_history:
                    second_to_last_message = conversation_history.pop()
                    prompt = (
                        second_to_last_message.content
                        if second_to_last_message.content
                        else genai_types.Content()
                    )
                else:
                    prompt = genai_types.Content()

        return prompt, system_instruction, conversation_history, reference

    @override
    def convert(self, raw_data: list[dict[str, Any]]) -> types.EvaluationDataset:
        """Converts a list of raw data into an EvaluationDataset."""
        eval_cases = []

        for i, item in enumerate(raw_data):
            eval_case_id = "gemini_eval_case_%s" % i
            request_data = item.get("request", {})
            response_data = item.get("response", {})

            (
                prompt,
                system_instruction,
                conversation_history,
                reference,
            ) = self._parse_request(request_data)

            responses = []
            if isinstance(response_data, str):
                responses.append(
                    types.ResponseCandidate(
                        response=genai_types.Content(
                            parts=[genai_types.Part(text=response_data)]
                        )
                    )
                )
            elif isinstance(response_data, dict):
                try:
                    generate_content_response = (
                        genai_types.GenerateContentResponse.model_validate(
                            response_data
                        )
                    )
                    if generate_content_response.candidates:
                        candidate = generate_content_response.candidates[0]
                        if candidate.content:
                            responses.append(
                                types.ResponseCandidate(
                                    response=genai_types.Content.model_validate(
                                        candidate.content
                                    )
                                )
                            )
                    else:
                        responses.append(_create_placeholder_response_candidate())
                except Exception:
                    responses.append(_create_placeholder_response_candidate())
            else:
                responses.append(_create_placeholder_response_candidate())

            eval_case = types.EvalCase(
                eval_case_id=eval_case_id,
                prompt=prompt,
                responses=responses,
                reference=reference,
                system_instruction=system_instruction,
                conversation_history=conversation_history,
            )
            eval_cases.append(eval_case)

        return types.EvaluationDataset(eval_cases=eval_cases)


class _FlattenEvalDataConverter(_evals_utils.EvalDataConverter):
    """Converter for datasets in a structured table format."""

    def convert(self, raw_data: list[dict[str, Any]]) -> types.EvaluationDataset:
        """Converts a list of raw data into an EvaluationDataset."""
        eval_cases = []
        for i, item_dict in enumerate(raw_data):
            if not isinstance(item_dict, dict):
                raise TypeError(
                    "Expected a dictionary for item at index %s, but got %s: %s"
                    % (i, type(item_dict).__name__, item_dict)
                )
            item = copy.deepcopy(item_dict)
            eval_case_id = "eval_case_%s" % i
            prompt_data = item.pop("prompt", None)
            if not prompt_data:
                prompt_data = item.pop("source", None)

            conversation_history_data = item.pop("history", None)
            response_data = item.pop("response", None)
            reference_data = item.pop("reference", None)
            system_instruction_data = item.pop("instruction", None)
            rubric_groups_data = item.pop("rubric_groups", None)
            intermediate_events_data = item.pop("intermediate_events", None)

            if not response_data:
                raise ValueError(
                    "Response is required but missing for %s." % eval_case_id
                )
            if not prompt_data:
                raise ValueError(
                    "Prompt is required but missing for %s." % eval_case_id
                )

            prompt: genai_types.Content
            if isinstance(prompt_data, str):
                prompt = genai_types.Content(parts=[genai_types.Part(text=prompt_data)])
            elif isinstance(prompt_data, dict):
                prompt = genai_types.Content.model_validate(prompt_data)
            elif isinstance(prompt_data, genai_types.Content):
                prompt = prompt_data
            else:
                raise ValueError(
                    "Invalid prompt type for case %s: %s" % (i, type(prompt_data))
                )

            conversation_history: Optional[list[types.evals.Message]] = None
            if isinstance(conversation_history_data, list):
                conversation_history = []
                for turn_id, content in enumerate(conversation_history_data):
                    if isinstance(content, genai_types.Content):
                        conversation_history.append(
                            types.evals.Message(
                                turn_id=str(turn_id),
                                content=content,
                            )
                        )
                    elif isinstance(content, dict):
                        try:
                            validated_content = genai_types.Content.model_validate(
                                content
                            )
                            conversation_history.append(
                                types.evals.Message(
                                    turn_id=str(turn_id),
                                    content=validated_content,
                                )
                            )
                        except ValidationError as e:
                            logger.warning(
                                "Item at index %s in 'history' column for case "
                                " %s is a dict but could not be validated as"
                                " genai_types.Content: %s",
                                turn_id,
                                eval_case_id,
                                e,
                            )
                    else:
                        logger.warning(
                            "Invalid type in 'history' column for case %s at index %s. "
                            "Expected genai_types.Content or dict, but got %s. "
                            "Skipping this history item.",
                            eval_case_id,
                            turn_id,
                            type(content),
                        )

            responses: list[types.ResponseCandidate]
            if isinstance(response_data, dict):
                responses = [
                    types.ResponseCandidate(
                        response=genai_types.Content.model_validate(response_data)
                    )
                ]
            elif isinstance(response_data, str):
                responses = [
                    types.ResponseCandidate(
                        response=genai_types.Content(
                            parts=[genai_types.Part(text=response_data)]
                        )
                    )
                ]
            elif isinstance(response_data, genai_types.Content):
                responses = [types.ResponseCandidate(response=response_data)]
            else:
                raise ValueError(
                    "Invalid response type for case %s: %s" % (i, type(response_data))
                )

            reference: Optional[types.ResponseCandidate] = None
            if reference_data:
                if isinstance(reference_data, dict):
                    reference = types.ResponseCandidate(
                        response=genai_types.Content.model_validate(reference_data)
                    )
                elif isinstance(reference_data, str):
                    reference = types.ResponseCandidate(
                        response=genai_types.Content(
                            parts=[genai_types.Part(text=reference_data)]
                        )
                    )
                elif isinstance(reference_data, genai_types.Content):
                    reference = types.ResponseCandidate(response=reference_data)

            system_instruction: Optional[genai_types.Content] = None
            if system_instruction_data:
                if isinstance(system_instruction_data, dict):
                    system_instruction = genai_types.Content.model_validate(
                        system_instruction_data
                    )
                elif isinstance(system_instruction_data, str):
                    system_instruction = genai_types.Content(
                        parts=[genai_types.Part(text=system_instruction_data)]
                    )
                elif isinstance(system_instruction_data, genai_types.Content):
                    system_instruction = system_instruction_data

            rubric_groups: Optional[dict[str, types.RubricGroup]] = None
            if rubric_groups_data:
                if isinstance(rubric_groups_data, dict):
                    rubric_groups = {}
                    for key, value in rubric_groups_data.items():
                        if isinstance(value, list):
                            try:
                                validated_rubrics = [
                                    (
                                        types.evals.Rubric.model_validate(r)
                                        if isinstance(r, dict)
                                        else r
                                    )
                                    for r in value
                                ]
                                if all(
                                    isinstance(r, types.evals.Rubric)
                                    for r in validated_rubrics
                                ):
                                    rubric_groups[key] = types.RubricGroup(
                                        rubrics=validated_rubrics
                                    )
                                else:
                                    logger.warning(
                                        "Invalid item type in rubric list for group '%s' in case %s.",
                                        key,
                                        i,
                                    )
                            except Exception as e:
                                logger.warning(
                                    "Failed to validate rubrics for group '%s' in case %s: %s",
                                    key,
                                    i,
                                    e,
                                )
                        elif isinstance(value, types.RubricGroup):
                            rubric_groups[key] = value
                        elif isinstance(value, dict):
                            try:
                                rubric_groups[key] = types.RubricGroup.model_validate(
                                    value
                                )
                            except Exception as e:
                                logger.warning(
                                    "Failed to validate RubricGroup dict for group '%s' in case %s: %s",
                                    key,
                                    i,
                                    e,
                                )
                        else:
                            logger.warning(
                                "Invalid type for rubric group '%s' in case %s."
                                " Expected list of rubrics, dict, or RubricGroup.",
                                key,
                                i,
                            )
                else:
                    logger.warning(
                        "Invalid type for rubric_groups in case %s. Expected dict.",
                        i,
                    )

            intermediate_events: Optional[list[types.evals.Event]] = None
            if intermediate_events_data:
                if isinstance(intermediate_events_data, list):
                    intermediate_events = []
                    for event in intermediate_events_data:
                        if isinstance(event, dict):
                            try:
                                validated_event = types.evals.Event.model_validate(
                                    event
                                )
                                intermediate_events.append(validated_event)
                            except Exception as e:
                                logger.warning(
                                    "Failed to validate intermediate event dict for"
                                    " case %s: %s",
                                    i,
                                    e,
                                )
                        elif isinstance(event, types.evals.Event):
                            intermediate_events.append(event)
                        else:
                            logger.warning(
                                "Invalid type for intermediate_event in case"
                                " %s. Expected list of dicts or list of"
                                " types.evals.Event objects.",
                                i,
                            )
                else:
                    logger.warning(
                        "Invalid type for intermediate_events in case %s. Expected"
                        " list of types.evals.Event objects.",
                        i,
                    )

            eval_case = types.EvalCase(
                eval_case_id=eval_case_id,
                prompt=prompt,
                responses=responses,
                reference=reference,
                conversation_history=conversation_history,
                system_instruction=system_instruction,
                rubric_groups=rubric_groups,
                intermediate_events=intermediate_events,
                **item,  # Pass remaining columns as extra fields to EvalCase.
                # They can be used for custom metric prompt templates.
            )
            eval_cases.append(eval_case)

        return types.EvaluationDataset(eval_cases=eval_cases)


class _OpenAIDataConverter(_evals_utils.EvalDataConverter):
    """Converter for dataset in OpenAI's Chat Completion format."""

    def _parse_messages(self, messages: list[dict[str, Any]]) -> tuple[
        Optional[genai_types.Content],
        list[types.evals.Message],
        Optional[genai_types.Content],
        Optional[types.ResponseCandidate],
    ]:
        """Parses a list of messages into instruction, history, prompt, and reference."""
        system_instruction = None
        prompt = None
        reference = None
        conversation_history = []

        if messages and messages[0].get("role") in ["system", "developer"]:
            system_instruction = genai_types.Content(
                parts=[genai_types.Part(text=messages[0].get("content"))]
            )
            messages = messages[1:]

        for turn_id, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation_history.append(
                types.evals.Message(
                    turn_id=str(turn_id),
                    content=genai_types.Content(
                        parts=[genai_types.Part(text=content)], role=role
                    ),
                    author=role,
                )
            )

        if conversation_history:
            last_message = conversation_history.pop()
            if last_message.content and last_message.content.role == "user":
                prompt = last_message.content
            elif last_message.content and last_message.content.role == "assistant":
                reference = types.ResponseCandidate(response=last_message.content)
                if conversation_history:
                    second_to_last_message = conversation_history.pop()
                    prompt = second_to_last_message.content

        return system_instruction, conversation_history, prompt, reference

    @override
    def convert(self, raw_data: list[dict[str, Any]]) -> types.EvaluationDataset:
        """Converts a list of OpenAI ChatCompletion data into an EvaluationDataset."""
        eval_cases = []
        for i, item in enumerate(raw_data):
            eval_case_id = "openai_eval_case_%s" % i

            if "request" not in item or "response" not in item:
                logger.warning(
                    "Skipping case %s due to missing 'request' or 'response' key.", i
                )
                continue

            request_data = item.get("request", {})
            response_data_raw = item.get("response", {})

            response_data = {}
            if isinstance(response_data_raw, str):
                try:
                    loaded_json = json.loads(response_data_raw)
                    if isinstance(loaded_json, dict):
                        response_data = loaded_json
                    else:
                        logger.warning(
                            "Decoded response JSON is not a dictionary for case"
                            " %s. Type: %s",
                            i,
                            type(loaded_json),
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        "Could not decode response JSON string for case %s."
                        " Treating as empty response.",
                        i,
                    )
            elif isinstance(response_data_raw, dict):
                response_data = response_data_raw

            messages = request_data.get("messages", [])
            choices = response_data.get("choices", [])

            (
                system_instruction,
                conversation_history,
                prompt,
                reference,
            ) = self._parse_messages(messages)

            if prompt is None and reference is None:
                logger.warning(
                    "Could not determine a user prompt or reference for case %s."
                    " Skipping.",
                    i,
                )
                continue

            responses = []
            if (
                choices
                and isinstance(choices, list)
                and isinstance(choices[0], dict)
                and choices[0].get("message")
            ):
                response_content = choices[0]["message"].get("content", "")
                responses.append(
                    types.ResponseCandidate(
                        response=genai_types.Content(
                            parts=[genai_types.Part(text=response_content)]
                        )
                    )
                )
            else:
                responses.append(_create_placeholder_response_candidate())

            other_fields = {
                k: v for k, v in item.items() if k not in ["request", "response"]
            }

            eval_case = types.EvalCase(
                eval_case_id=eval_case_id,
                prompt=prompt,
                responses=responses,
                reference=reference,
                system_instruction=system_instruction,
                conversation_history=conversation_history,
                **other_fields,
            )
            eval_cases.append(eval_case)

        return types.EvaluationDataset(eval_cases=eval_cases)


def auto_detect_dataset_schema(
    raw_dataset: list[dict[str, Any]],
) -> Union[EvalDatasetSchema, str]:
    """Detects the schema of a raw dataset."""
    if not raw_dataset:
        return EvalDatasetSchema.UNKNOWN

    first_item = raw_dataset[0]
    keys = set(first_item.keys())

    if "format" in keys:
        format_content = first_item.get("format", "")
        if isinstance(format_content, str) and format_content == "observability":
            return EvalDatasetSchema.OBSERVABILITY

    if "request" in keys and "response" in keys:
        request_content = first_item.get("request", {})
        if isinstance(request_content, dict) and "contents" in request_content:
            contents_list = request_content.get("contents")
            if (
                contents_list
                and isinstance(contents_list, list)
                and isinstance(contents_list[0], dict)
            ):
                if "parts" in contents_list[0]:
                    return EvalDatasetSchema.GEMINI

    if "request" in keys and "response" in keys:
        request_content = first_item.get("request", {})
        if isinstance(request_content, dict) and "messages" in request_content:
            messages_list = request_content.get("messages")
            if (
                messages_list
                and isinstance(messages_list, list)
                and isinstance(messages_list[0], dict)
            ):
                if "role" in messages_list[0] and "content" in messages_list[0]:
                    return EvalDatasetSchema.OPENAI

    if {"prompt", "response"}.issubset(keys) or {
        "response",
        "reference",
    }.issubset(keys):
        return EvalDatasetSchema.FLATTEN
    else:
        return EvalDatasetSchema.UNKNOWN


_CONVERTER_REGISTRY = {
    EvalDatasetSchema.GEMINI: _GeminiEvalDataConverter,
    EvalDatasetSchema.FLATTEN: _FlattenEvalDataConverter,
    EvalDatasetSchema.OPENAI: _OpenAIDataConverter,
    EvalDatasetSchema.OBSERVABILITY: _observability_data_converter.ObservabilityDataConverter,
}


def get_dataset_converter(
    dataset_schema: EvalDatasetSchema,
) -> _evals_utils.EvalDataConverter:
    """Returns the appropriate dataset converter for the given schema."""
    if dataset_schema in _CONVERTER_REGISTRY:
        return _CONVERTER_REGISTRY[dataset_schema]()  # type: ignore[abstract]
    else:
        raise ValueError("Unsupported dataset schema: %s" % dataset_schema)


def _get_first_part_text(content: genai_types.Content) -> str:
    """Safely extracts text from the first part of a content."""
    if (
        content
        and hasattr(content, "parts")
        and isinstance(content.parts, list)
        and content.parts
    ):
        first_part = content.parts[0]
        if hasattr(first_part, "text"):
            return str(first_part.text)
    return ""


def _get_text_from_reference(
    reference: Optional[types.ResponseCandidate],
) -> Optional[str]:
    """Safely extracts text from a reference field."""
    if reference and hasattr(reference, "response") and reference.response:
        return _get_first_part_text(reference.response)
    return None


def _validate_case_consistency(
    base_case: types.EvalCase,
    current_case: types.EvalCase,
    case_idx: int,
    dataset_idx: int,
) -> None:
    """Logs warnings if prompt or reference mismatches occur."""
    if base_case.prompt != current_case.prompt:
        base_prompt_text_preview = _get_first_part_text(base_case.prompt)[:50]
        current_prompt_text_preview = _get_first_part_text(current_case.prompt)[:50]
        logger.warning(
            "Prompt mismatch for case index %d between base dataset (0)"
            " and dataset %d. Using prompt from base. Base prompt"
            " preview: '%s...', Dataset"
            " %d prompt preview: '%s...'",
            case_idx,
            dataset_idx,
            base_prompt_text_preview,
            dataset_idx,
            current_prompt_text_preview,
        )

    base_ref_text = _get_text_from_reference(base_case.reference)
    current_ref_text = _get_text_from_reference(current_case.reference)

    if bool(base_case.reference) != bool(current_case.reference):
        logger.warning(
            "Reference presence mismatch for case index %d between base"
            " dataset (0) and dataset %d. Using reference (or lack"
            " thereof) from base.",
            case_idx,
            dataset_idx,
        )
    elif base_ref_text != current_ref_text:
        logger.warning(
            "Reference text mismatch for case index %d between base"
            " dataset (0) and dataset %d. Using reference from base. "
            " Base ref: '%s...', Current ref:"
            " '%s...'",
            case_idx,
            dataset_idx,
            str(base_ref_text)[:50],
            str(current_ref_text)[:50],
        )


def merge_response_datasets_into_canonical_format(
    raw_datasets: list[list[dict[str, Any]]],
    schemas: list[str],
    agent_info: Optional[types.evals.AgentInfo] = None,
) -> types.EvaluationDataset:
    """Merges multiple raw response datasets into a single EvaluationDataset.

    Assumes that each dataset in raw_datasets has responses corresponding
    to the same set of prompts, in the same order. The prompt, reference,
    system_instruction, and conversation_history are taken from the first dataset.
    """
    if not isinstance(raw_datasets, list):
        raise TypeError(
            "Input 'raw_datasets' must be a list, got %s." % type(raw_datasets)
        )
    if not raw_datasets or not all(isinstance(ds, list) for ds in raw_datasets):
        raise ValueError(
            "Input 'raw_datasets' cannot be empty and must be a list of lists."
        )
    if not schemas or len(schemas) != len(raw_datasets):
        raise ValueError(
            "A list of schemas must be provided, one for each raw dataset. "
            "Got %s schemas for %s datasets." % (len(schemas), len(raw_datasets))
        )

    num_expected_cases = len(raw_datasets[0])
    if num_expected_cases == 0:
        logger.warning(
            "The first dataset has no evaluation cases. Result will be empty."
        )
        return types.EvaluationDataset(eval_cases=[])

    parsed_evaluation_datasets: list[types.EvaluationDataset] = []
    for i, (raw_ds_entry, schema) in enumerate(zip(raw_datasets, schemas)):
        if len(raw_ds_entry) != num_expected_cases:
            raise ValueError(
                "All datasets must have the same number of evaluation cases. "
                "Base dataset (0) has %s, but dataset %s (schema: %s) has %s."
                % (num_expected_cases, i, schema, len(raw_ds_entry))
            )
        converter = get_dataset_converter(schema)
        parsed_evaluation_datasets.append(converter.convert(raw_ds_entry))

    merged_eval_cases: list[types.EvalCase] = []
    base_parsed_dataset = parsed_evaluation_datasets[0]

    for case_idx in range(num_expected_cases):
        base_eval_case: types.EvalCase = (
            base_parsed_dataset.eval_cases[case_idx]
            if base_parsed_dataset.eval_cases
            else types.EvalCase()
        )
        candidate_responses: list[types.ResponseCandidate] = []

        if base_eval_case.responses:
            candidate_responses.append(base_eval_case.responses[0])
        else:
            logger.warning(
                "No response found for base dataset (index 0) in case %s. "
                "Adding placeholder.",
                case_idx,
            )
            candidate_responses.append(
                _create_placeholder_response_candidate(
                    "Missing response from base dataset (0) for case %s" % case_idx
                )
            )

        eval_case_custom_columns = base_eval_case.model_dump(
            exclude={
                "eval_case_id",
                "prompt",
                "responses",
                "reference",
                "system_instruction",
                "conversation_history",
                "intermediate_events",
            },
            exclude_none=True,
        )
        for dataset_idx_offset, current_parsed_ds in enumerate(
            parsed_evaluation_datasets[1:], start=1
        ):
            current_ds_eval_case: types.EvalCase = (
                current_parsed_ds.eval_cases[case_idx]
                if current_parsed_ds.eval_cases
                else types.EvalCase()
            )

            _validate_case_consistency(
                base_eval_case, current_ds_eval_case, case_idx, dataset_idx_offset
            )

            current_ds_extra_attrs = current_ds_eval_case.model_dump(
                exclude={
                    "eval_case_id",
                    "prompt",
                    "responses",
                    "reference",
                    "system_instruction",
                    "conversation_history",
                    "intermediate_events",
                },
                exclude_none=True,
            )
            eval_case_custom_columns.update(current_ds_extra_attrs)

            if current_ds_eval_case.responses:
                candidate_responses.append(current_ds_eval_case.responses[0])
            else:
                logger.warning(
                    "No response found for dataset %s in case %s. Adding placeholder.",
                    dataset_idx_offset,
                    case_idx,
                )
                candidate_responses.append(
                    _create_placeholder_response_candidate(
                        "Missing response from dataset %s for case %s"
                        % (dataset_idx_offset, case_idx)
                    )
                )

        merged_case = types.EvalCase(
            eval_case_id=base_eval_case.eval_case_id
            or "merged_eval_case_%s" % case_idx,
            prompt=base_eval_case.prompt,
            responses=candidate_responses,
            reference=base_eval_case.reference,
            system_instruction=base_eval_case.system_instruction,
            conversation_history=base_eval_case.conversation_history,
            agent_info=agent_info,
            intermediate_events=base_eval_case.intermediate_events,
            **eval_case_custom_columns,
        )
        merged_eval_cases.append(merged_case)

    return types.EvaluationDataset(eval_cases=merged_eval_cases)
