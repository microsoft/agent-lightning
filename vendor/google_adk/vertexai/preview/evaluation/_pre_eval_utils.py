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
"""Utility functions for all pre-evaluation steps."""

from __future__ import annotations

from concurrent import futures
from typing import Callable, Optional, Set, TYPE_CHECKING, Union, List

from google.cloud.aiplatform import base
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)
from vertexai import generative_models
from vertexai.preview.evaluation import _base as evaluation_base
from vertexai.preview.evaluation import constants
from vertexai.preview.evaluation import multimodal_utils
from vertexai.preview.evaluation import (
    prompt_template as prompt_template_base,
)


if TYPE_CHECKING:
    import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        'tqdm is not installed. Please install the SDK using "pip install'
        ' google-cloud-aiplatform[evaluation]"'
    )


_LOGGER = base.Logger(__name__)
_SUCCESSFUL_FINISH_REASONS = [
    gapic_content_types.Candidate.FinishReason.STOP,
    gapic_content_types.Candidate.FinishReason.MAX_TOKENS,
    # Many responses have this finish reason
    gapic_content_types.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED,
]


def _assemble_prompt(
    row: "pd.Series",
    prompt_template: Union[prompt_template_base.PromptTemplate, str],
) -> str:
    """Assembles the prompt template with the given row data."""
    if isinstance(prompt_template, str):
        prompt_template = prompt_template_base.PromptTemplate(prompt_template)
    _check_variable_columns_exist(row, prompt_template.variables)
    return str(
        prompt_template.assemble(
            **row[list(prompt_template.variables)].astype(str).to_dict()
        )
    )


def _generate_content_text_response(
    model: generative_models.GenerativeModel, prompt: str, max_attempts: int = 3
) -> str:
    """Generates a text response from Gemini model from a text prompt with retries .

    Args:
        model: The Gemini model instance.
        prompt: The prompt to send to the model.
        max_attempts: Maximum number of attempts for response generation.

    Returns:
        The text response from the model.

    Raises:
        RuntimeError if the prompt or the response for the prompt is blocked for
        safety reasons.
    """
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(prompt)
            if not response.candidates:
                error_message = (
                    f"The model response was blocked due to"
                    f" {response._raw_response.prompt_feedback.block_reason.name}.\n"
                    f"Blocked reason message:"
                    f" {response._raw_response.prompt_feedback.block_reason_message}.\n"
                    "The input prompt may be blocked for safety reasons.\n"
                    f"Prompt: {prompt}.\n"
                    f"Attempt: {attempt + 1}/{max_attempts}"
                )
                _LOGGER.warning(error_message)
                break
            else:
                candidate = response.candidates[0]
                if candidate.finish_reason not in _SUCCESSFUL_FINISH_REASONS:
                    error_message = (
                        "The model response did not finish"
                        " successfully.\n"
                        f"Finish reason: {candidate.finish_reason}.\n"
                        f"Finish message: {candidate.finish_message}.\n"
                        f"Safety ratings: {candidate.safety_ratings}.\n"
                        "Please adjust the model safety_settings, or"
                        " try a different prompt.\n"
                        f"Attempt: {attempt + 1}/{max_attempts}"
                    )
                    _LOGGER.warning(error_message)
                else:
                    return response.candidates[0].content.parts[0].text
        except Exception as e:
            error_message = (
                f"Failed to generate response candidates from Gemini model"
                f" {model._model_name}.\n"
                f"Error: {e}.\n"
                f"Prompt: {prompt}.\n"
                f"Attempt: {attempt + 1}/{max_attempts}"
            )
            _LOGGER.warning(error_message)
        if attempt < max_attempts - 1:
            _LOGGER.info(
                f"Retrying response generation for prompt: {prompt}, attempt"
                f" {attempt + 1}/{max_attempts}..."
            )

    final_error_message = (
        f"Failed to generate response from Gemini model {model._model_name}.\n"
        f"Prompt: {prompt}."
    )
    _LOGGER.error(final_error_message)
    return constants.RESPONSE_ERROR


def _generate_responses_from_gemini_model(
    model: generative_models.GenerativeModel,
    df: "pd.DataFrame",
    rubric_generation_prompt_template: Optional[str] = None,
) -> List[str]:
    """Generates responses from Gemini model for the given evaluation dataset.

    Args:
        model: The Gemini model instance.
        df: Evaluation Dataset.
    Returns:
        The list of model responses.
    """
    _LOGGER.info(
        f"Generating a total of {df.shape[0]} "
        f"responses from Gemini model {model._model_name.split('/')[-1]}."
    )
    tasks = []
    with tqdm(total=len(df)) as pbar:
        with futures.ThreadPoolExecutor(max_workers=constants.MAX_WORKERS) as executor:
            for idx, row in df.iterrows():
                if rubric_generation_prompt_template:
                    input_columns = prompt_template_base.PromptTemplate(
                        rubric_generation_prompt_template
                    ).variables
                    if multimodal_utils.is_multimodal_instance(
                        row[list(input_columns)].to_dict()
                    ):
                        prompt = multimodal_utils._assemble_multi_modal_prompt(
                            rubric_generation_prompt_template, row, idx, input_columns
                        )
                    else:
                        prompt = _assemble_prompt(
                            row, rubric_generation_prompt_template
                        )
                else:
                    prompt = row[constants.Dataset.PROMPT_COLUMN]
                task = executor.submit(
                    _generate_content_text_response,
                    prompt=prompt,
                    model=model,
                )
                task.add_done_callback(lambda _: pbar.update(1))
                tasks.append(task)
        responses = [future.result() for future in tasks]
    return responses


def _generate_response_from_custom_model_fn(
    model_fn: Callable[[str], str], eval_dataset: "pd.DataFrame"
) -> List[str]:
    """Generates responses from a custom model function.

    Args:
        model_fn: The custom model function.
        eval_dataset: Evaluation Dataset.
    Returns:
        The list of model responses.
    """
    max_workers = 5

    _LOGGER.info(
        f"Generating a total of {eval_dataset.shape[0]} "
        "responses from the custom model function."
    )
    tasks = []
    try:
        with tqdm(total=len(eval_dataset)) as pbar:
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _, row in eval_dataset.iterrows():
                    task = executor.submit(
                        model_fn, row[constants.Dataset.PROMPT_COLUMN]
                    )
                    task.add_done_callback(lambda _: pbar.update(1))
                    tasks.append(task)
    except (ValueError, IndexError) as e:
        _LOGGER.warning(f"Failed to generate response from model function: {e}")

    responses = [task.result() for task in tasks]
    return responses


def populate_eval_dataset_with_model_responses(
    responses: List[str],
    evaluation_run_config: evaluation_base.EvaluationRunConfig,
    is_baseline_model: bool = False,
) -> None:
    """Populates the evaluation dataset with model responses.

    Args:
        responses: The list of model responses.
        evaluation_run_config: Evaluation Run Configurations.
        is_baseline_model: Whether the model is a baseline model for
          PairwiseMetric.
    """
    df = evaluation_run_config.dataset.copy()
    if is_baseline_model:
        evaluation_run_config.dataset = df.assign(baseline_model_response=responses)
    else:
        evaluation_run_config.dataset = df.assign(response=responses)

    _LOGGER.info(
        f"All {evaluation_run_config.dataset.shape[0]} responses are successfully"
        f" generated from model."
    )


def _check_variable_columns_exist(
    dataset_row: "pd.Series", variable_names_set: Set[str]
) -> None:
    """Checks if all variable names exist in the dataset columns.

    Args:
        dataset: The dataset to evaluate.
        variable_names_set: A set of variable names.

    Raises:
        ValueError: If any variable names do not exist in the dataset columns
        or the prompt template is invalid.
    """
    actual_column_names_set = set(dataset_row.to_dict().keys())
    if not variable_names_set.issubset(actual_column_names_set):
        missing_columns = variable_names_set - actual_column_names_set
        raise ValueError(
            "Failed to assemble prompt template: The following column(s) are"
            f" missing: {', '.join(missing_columns)}. "
            f"Please verify prompt_template variables {variable_names_set} and "
            f"evaluation dataset column names {actual_column_names_set}."
        )
