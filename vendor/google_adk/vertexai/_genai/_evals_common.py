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
"""Common utilities for evals."""
import asyncio
import base64
import collections
import concurrent.futures
import datetime
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Literal, Optional, Union

from google.api_core import exceptions as api_exceptions
import vertexai
from google.genai import types as genai_types
from google.genai._api_client import BaseApiClient
from google.genai.models import Models
import pandas as pd
from tqdm import tqdm

from . import _evals_constant
from . import _evals_data_converters
from . import _evals_metric_handlers
from . import _evals_metric_loaders
from . import _gcs_utils

from . import evals
from . import types

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_thread_local_data = threading.local()

MAX_WORKERS = 100
AGENT_MAX_WORKERS = 10


def _get_agent_engine_instance(
    agent_name: str, api_client: BaseApiClient
) -> types.AgentEngine:
    """Gets or creates an agent engine instance for the current thread."""
    if not hasattr(_thread_local_data, "agent_engine_instances"):
        _thread_local_data.agent_engine_instances = {}
    if agent_name not in _thread_local_data.agent_engine_instances:
        client = vertexai.Client(
            project=api_client.project,
            location=api_client.location,
        )
        _thread_local_data.agent_engine_instances[agent_name] = (
            client.agent_engines.get(name=agent_name)
        )
    return _thread_local_data.agent_engine_instances[agent_name]


def _generate_content_with_retry(
    api_client: BaseApiClient,
    model: str,
    contents: Union[genai_types.ContentListUnion, genai_types.ContentListUnionDict],
    config: Optional[genai_types.GenerateContentConfig] = None,
    max_retries: int = 3,
) -> Union[genai_types.GenerateContentResponse, dict[str, Any]]:
    """Generates content using the model's generate_content with retries."""
    models_module = Models(api_client_=api_client)

    for attempt in range(max_retries):
        try:
            response = models_module.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            if not response.candidates:
                logger.warning(
                    "Prompt blocked. Attempt %d/%d. Feedback: %s. Prompt: %s.",
                    attempt + 1,
                    max_retries,
                    response.prompt_feedback,
                    contents,
                )
                if attempt == max_retries - 1:
                    feedback_dict = {}
                    if response.prompt_feedback:
                        feedback_dict = response.prompt_feedback.model_dump(
                            mode="json", exclude_none=True
                        )
                    return {
                        "error": "Prompt blocked after retries",
                        "prompt_feedback": feedback_dict,
                    }
            else:
                candidate = response.candidates[0]
                if candidate.finish_reason not in (
                    genai_types.FinishReason.STOP,
                    genai_types.FinishReason.MAX_TOKENS,
                    genai_types.FinishReason.FINISH_REASON_UNSPECIFIED,
                ):
                    logger.warning(
                        "Generate content did not finish successfully."
                        "Finish reason: %s. Finish message: %s."
                        "Retry attempt: %d/%d",
                        candidate.finish_reason,
                        candidate.finish_message,
                        attempt + 1,
                        max_retries,
                    )
                    if attempt == max_retries - 1:
                        return {
                            "error": (
                                "Generate content unsuccessful after retries:"
                                f" {candidate.finish_reason}"
                            ),
                            "finish_reason": str(candidate.finish_reason),
                            "finish_message": candidate.finish_message or "",
                        }
                else:
                    return response
        except api_exceptions.ResourceExhausted as e:
            logger.warning(
                "Resource Exhausted error on attempt %d/%d: %s. Retrying in %s"
                " seconds...",
                attempt + 1,
                max_retries,
                e,
                2**attempt,
            )
            if attempt == max_retries - 1:
                return {"error": f"Resource exhausted after retries: {e}"}
            time.sleep(2**attempt)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Unexpected error during generate_content on attempt %d/%d: %s",
                attempt + 1,
                max_retries,
                e,
            )

            if attempt == max_retries - 1:
                return {"error": f"Failed after retries: {e}"}
            time.sleep(1)
    return {"error": f"Failed to generate content after {max_retries} retries"}


def _build_generate_content_config(
    request_dict: dict[str, Any],
    global_config: Optional[genai_types.GenerateContentConfig] = None,
) -> genai_types.GenerateContentConfig:
    """Builds a GenerateContentConfig from the request dictionary or provided config."""
    if global_config:
        # If a global config is provided, apply it as a base config. Parts of
        # the global config can be overridden by providing configs in the
        # request.
        merged_config_dict = global_config.model_dump(exclude_none=True)
    else:
        merged_config_dict = {}

    for key in [
        "system_instruction",
        "tools",
        "tools_config",
        "safety_settings",
        "labels",
    ]:
        if key in request_dict:
            merged_config_dict[key] = request_dict[key]
    if "generation_config" in request_dict and isinstance(
        request_dict["generation_config"], dict
    ):
        merged_config_dict.update(request_dict["generation_config"])
    if "labels" in request_dict:
        merged_config_dict["labels"] = request_dict["labels"]

    return genai_types.GenerateContentConfig(**merged_config_dict)


def _extract_contents_for_inference(
    request_dict_or_raw_text: Any,
) -> Any:
    """Extracts contents from a request dictionary or returns the raw text."""
    if not request_dict_or_raw_text:
        raise ValueError("Prompt cannot be empty.")
    if isinstance(request_dict_or_raw_text, dict):
        contents_for_fn = request_dict_or_raw_text.get("contents", None)
        if not contents_for_fn:
            raise ValueError("Contents in the request cannot be empty.")
        return contents_for_fn
    else:
        return request_dict_or_raw_text


def _execute_inference_concurrently(
    api_client: BaseApiClient,
    prompt_dataset: pd.DataFrame,
    progress_desc: str,
    model_or_fn: Optional[Union[str, Callable[[Any], Any]]] = None,
    gemini_config: Optional[genai_types.GenerateContentConfig] = None,
    inference_fn: Optional[Callable[..., Any]] = None,
    agent_engine: Optional[Union[str, types.AgentEngine]] = None,
) -> list[
    Union[genai_types.GenerateContentResponse, dict[str, Any], list[dict[str, Any]]]
]:
    """Internal helper to run inference with concurrency."""
    logger.info(
        "Generating responses for %d prompts using model or function: %s",
        len(prompt_dataset),
        model_or_fn,
    )
    responses: list[
        Union[
            genai_types.GenerateContentResponse,
            dict[str, Any],
            list[dict[str, Any]],
            None,
        ]
    ] = [None] * len(prompt_dataset)
    tasks = []

    primary_prompt_column = (
        "request" if "request" in prompt_dataset.columns else "prompt"
    )
    if primary_prompt_column not in prompt_dataset.columns:
        raise ValueError(
            "Dataset must contain either 'prompt' or 'request'."
            f" Found: {prompt_dataset.columns.tolist()}"
        )

    max_workers = AGENT_MAX_WORKERS if agent_engine else MAX_WORKERS
    with tqdm(total=len(prompt_dataset), desc=progress_desc) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row in prompt_dataset.iterrows():
                request_dict_or_raw_text = row[primary_prompt_column]
                try:
                    contents = _extract_contents_for_inference(request_dict_or_raw_text)
                except ValueError as e:
                    error_message = (
                        f"Failed to extract contents for prompt at index {index}: {e}. "
                        "Skipping prompt."
                    )
                    logger.error(error_message)
                    responses[index] = {"error": error_message}
                    pbar.update(1)
                    continue

                if agent_engine:

                    def agent_run_wrapper(
                        row_arg,
                        contents_arg,
                        agent_engine,
                        inference_fn_arg,
                        api_client_arg,
                    ):
                        if isinstance(agent_engine, str):
                            agent_engine_instance = _get_agent_engine_instance(
                                agent_engine, api_client_arg
                            )
                        elif (
                            hasattr(agent_engine, "api_client")
                            and type(agent_engine).__name__ == "AgentEngine"
                        ):
                            agent_engine_instance = agent_engine
                        return inference_fn_arg(
                            row=row_arg,
                            contents=contents_arg,
                            agent_engine=agent_engine_instance,
                        )

                    future = executor.submit(
                        agent_run_wrapper,
                        row,
                        contents,
                        agent_engine,
                        inference_fn,
                        api_client,
                    )
                elif isinstance(model_or_fn, str):
                    generation_content_config = _build_generate_content_config(
                        request_dict_or_raw_text,
                        gemini_config,
                    )
                    future = executor.submit(
                        inference_fn,
                        api_client=api_client,
                        model=model_or_fn,
                        contents=contents,
                        config=generation_content_config,
                    )
                else:
                    future = executor.submit(model_or_fn, contents)
                future.add_done_callback(lambda _: pbar.update(1))
                tasks.append((future, index))

            for future, index in tasks:
                try:
                    result = future.result()
                    responses[index] = result
                except Exception as e:
                    logger.error(
                        "Error processing prompt at index %d: %s",
                        index,
                        e,
                    )
                    responses[index] = {"error": f"Inference task failed: {e}"}
    return responses  # type: ignore[return-value]


def _run_gemini_inference(
    api_client: BaseApiClient,
    model: str,
    prompt_dataset: pd.DataFrame,
    config: Optional[genai_types.GenerateContentConfig] = None,
) -> list[Union[genai_types.GenerateContentResponse, dict[str, Any]]]:
    """Internal helper to run inference using Gemini model with concurrency."""
    return _execute_inference_concurrently(
        api_client=api_client,
        model_or_fn=model,
        prompt_dataset=prompt_dataset,
        progress_desc="Gemini Inference",
        gemini_config=config,
        inference_fn=_generate_content_with_retry,
    )


def _run_custom_inference(
    model_fn: Callable[[Any], Any],
    prompt_dataset: pd.DataFrame,
) -> list[Any]:
    """Internal helper to run inference using a custom function with concurrency."""
    return _execute_inference_concurrently(
        api_client=None,
        model_or_fn=model_fn,
        prompt_dataset=prompt_dataset,
        progress_desc="Custom Inference",
    )


def _convert_prompt_row_to_litellm_messages(row: pd.Series) -> list[dict[str, Any]]:
    """Converts a DataFrame row into LiteLLM's messages format by detecting the input schema."""
    messages: list[dict[str, Any]] = []
    row_dict = row.to_dict()

    # Case 1: The row is an OpenAI request body itself.
    if "messages" in row_dict and isinstance(row_dict.get("messages"), list):
        return row_dict["messages"]  # type: ignore[no-any-return]

    # Case 2: The row contains a 'request' key with an OpenAI request body.
    elif "request" in row_dict and isinstance(row_dict.get("request"), dict):
        request_body = row_dict["request"]
        if "messages" in request_body and isinstance(
            request_body.get("messages"), list
        ):
            return request_body["messages"]  # type: ignore[no-any-return]

        # Case 3: The 'request' key is in Gemini 'contents' format.
        elif "contents" in request_body and isinstance(
            request_body.get("contents"), list
        ):
            for content in request_body["contents"]:
                role = content.get("role", "user")
                text_parts = [part.get("text", "") for part in content.get("parts", [])]
                messages.append({"role": role, "content": " ".join(text_parts)})
            return messages

    # Case 4: Fallback to a simple 'prompt' key with a raw string.
    elif "prompt" in row_dict and isinstance(row_dict.get("prompt"), str):
        return [{"role": "user", "content": row_dict["prompt"]}]

    raise ValueError(
        "Could not determine prompt/messages format from input row. "
        "Expected OpenAI request body with a 'messages' key, or a 'request' key"
        " with OpenAI request body, or Gemini request body with a 'contents'"
        f" key, or a 'prompt' key with a raw string. Found keys: {list(row_dict.keys())}"
    )


def _call_litellm_completion(
    model: str, messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Wrapper for a single litellm.completion call."""
    try:
        response = litellm.completion(model=model, messages=messages)
        return response.model_dump()  # type: ignore[no-any-return]
    except Exception as e:
        logger.error("LiteLLM completion failed for model %s: %s", model, e)
        return {"error": str(e)}


def _run_litellm_inference(
    model: str, prompt_dataset: pd.DataFrame
) -> list[Optional[dict[str, Any]]]:
    """Runs inference using LiteLLM with concurrency."""
    logger.info(
        "Generating responses for %d prompts using LiteLLM for third party model: %s",
        len(prompt_dataset),
        model,
    )
    responses: list[Optional[dict[str, Any]]] = [None] * len(prompt_dataset)
    tasks = []

    with tqdm(total=len(prompt_dataset), desc=f"LiteLLM Inference ({model})") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for index, row in prompt_dataset.iterrows():
                messages = _convert_prompt_row_to_litellm_messages(row)
                future = executor.submit(
                    _call_litellm_completion, model=model, messages=messages
                )
                future.add_done_callback(lambda _: pbar.update(1))
                tasks.append((future, index))

            for future, index in tasks:
                try:
                    result = future.result()
                    responses[index] = result
                except Exception as e:
                    logger.error("Error processing prompt at index %d: %s", index, e)
                    responses[index] = {"error": f"LiteLLM task failed: {e}"}

    return responses


def _is_litellm_vertex_maas_model(model: str) -> bool:
    """Checks if the model is a Vertex MAAS model to be handled by LiteLLM."""
    return any(
        model.startswith(prefix)
        for prefix in _evals_constant.SUPPORTED_VERTEX_MAAS_MODEL_PREFIXES
    )


def _is_litellm_model(model: str) -> bool:
    """Checks if the model name corresponds to a valid LiteLLM model name."""
    return model in litellm.utils.get_valid_models(model)


def _is_gemini_model(model: str) -> bool:
    """Checks if the model name corresponds to a Gemini/Vertex AI model."""
    return (
        model.startswith("gemini-")
        or model.startswith("projects/")
        or model.startswith("models/")
        or model.startswith("publishers/")
        or model.startswith("tunedModels/")
    )


def _run_inference_internal(
    api_client: BaseApiClient,
    model: Union[Callable[[Any], Any], str],
    prompt_dataset: pd.DataFrame,
    config: Optional[genai_types.GenerateContentConfig] = None,
) -> pd.DataFrame:
    """Runs inference on a given dataset using the specified model or function."""

    if isinstance(model, str) and _is_gemini_model(model):
        if (
            "prompt" not in prompt_dataset.columns
            and "request" not in prompt_dataset.columns
        ):
            raise ValueError(
                "Prompt dataset for Gemini model must contain either 'prompt' or"
                " 'request' column for inference. "
                f"Found columns: {prompt_dataset.columns.tolist()}"
            )

        logger.info("Running inference with Gemini model name: %s", model)
        raw_responses = _run_gemini_inference(
            api_client=api_client,
            model=model,
            prompt_dataset=prompt_dataset,
            config=config,
        )
        processed_responses = []
        for resp_item in raw_responses:
            if isinstance(resp_item, genai_types.GenerateContentResponse):
                text_response = resp_item.text
                processed_responses.append(
                    text_response
                    if text_response is not None
                    else json.dumps({"error": "Empty response text"})
                )
            elif isinstance(resp_item, dict) and "error" in resp_item:
                processed_responses.append(json.dumps(resp_item))
            else:
                error_payload = {
                    "error": "Unexpected response type from Gemini inference",
                    "response_type": str(type(resp_item)),
                    "details": str(resp_item),
                }
                processed_responses.append(json.dumps(error_payload))
        responses = processed_responses
    elif callable(model):
        logger.info("Running inference with custom callable function.")
        custom_responses_raw = _run_custom_inference(
            model_fn=model, prompt_dataset=prompt_dataset
        )
        processed_custom_responses = []
        for resp_item in custom_responses_raw:
            if isinstance(resp_item, str):
                processed_custom_responses.append(resp_item)
            elif isinstance(resp_item, dict) and "error" in resp_item:
                processed_custom_responses.append(json.dumps(resp_item))
            else:
                try:
                    processed_custom_responses.append(json.dumps(resp_item))
                except TypeError:
                    processed_custom_responses.append(str(resp_item))
        responses = processed_custom_responses
    elif isinstance(model, str):
        if litellm is None:
            raise ImportError(
                "The 'litellm' library is required to use this model."
                " Please install it using 'pip install"
                " google-cloud-aiplatform[evaluation]'."
            )

        processed_model_id = model
        if model.startswith("vertex_ai/"):
            # Already correctly prefixed for LiteLLM's Vertex AI provider
            pass
        elif _is_litellm_vertex_maas_model(model):
            processed_model_id = f"vertex_ai/{model}"
            logger.info(
                "Detected Vertex AI Model Garden managed MaaS model. "
                "Using LiteLLM ID: %s",
                processed_model_id,
            )
        elif _is_litellm_model(model):
            # Other LiteLLM supported model
            logger.info("Running inference with LiteLLM for model: %s", model)
        else:
            # Unsupported model string
            raise TypeError(
                f"Unsupported string model name: {model}. Expecting a Gemini model"
                " name (e.g., 'gemini-1.5-pro', 'projects/.../models/...') or a"
                " LiteLLM supported model name (e.g., 'openai/gpt-4o')."
                " If using a third-party model via LiteLLM, ensure the"
                " necessary environment variables are set (e.g., for OpenAI:"
                " `os.environ['OPENAI_API_KEY'] = 'Your API Key'`). See"
                " LiteLLM documentation for details:"
                " https://docs.litellm.ai/docs/set_keys#environment-variables"
            )

        logger.info("Running inference via LiteLLM for model: %s", processed_model_id)
        raw_responses = _run_litellm_inference(
            model=processed_model_id, prompt_dataset=prompt_dataset
        )
        processed_llm_responses = []
        for response_dict in raw_responses:
            if not isinstance(response_dict, dict):
                processed_llm_responses.append(
                    json.dumps(
                        {
                            "error": "Invalid LiteLLM response format",
                            "details": str(response_dict),
                        }
                    )
                )
                continue

            if "error" in response_dict:
                processed_llm_responses.append(json.dumps(response_dict))
                continue

            if (
                "choices" in response_dict
                and isinstance(response_dict["choices"], list)
                and len(response_dict["choices"]) > 0
            ):
                first_choice = response_dict["choices"][0]
                if "message" in first_choice and isinstance(
                    first_choice["message"], dict
                ):
                    message = first_choice["message"]
                    if "content" in message and isinstance(message["content"], str):
                        processed_llm_responses.append(message["content"])
                    else:
                        processed_llm_responses.append(
                            json.dumps(
                                {
                                    "error": "LiteLLM response missing 'content' in message",
                                    "details": response_dict,
                                }
                            )
                        )
                else:
                    processed_llm_responses.append(
                        json.dumps(
                            {
                                "error": "LiteLLM response missing 'message' in first choice",
                                "details": response_dict,
                            }
                        )
                    )
            else:
                processed_llm_responses.append(
                    json.dumps(
                        {
                            "error": "LiteLLM response missing 'choices'",
                            "details": response_dict,
                        }
                    )
                )
        responses = processed_llm_responses
    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. Expecting string (model"
            " name) or Callable."
        )

    if len(responses) != len(prompt_dataset):
        raise RuntimeError(
            "Critical prompt/response count mismatch: %d prompts vs %d"
            " responses. This indicates an issue in response collection."
            % (len(prompt_dataset), len(responses))
        )

    results_df_responses_only = pd.DataFrame(
        {
            "response": responses,
        }
    )

    prompt_dataset_indexed = prompt_dataset.reset_index(drop=True)
    results_df_responses_only_indexed = results_df_responses_only.reset_index(drop=True)

    results_df = pd.concat(
        [prompt_dataset_indexed, results_df_responses_only_indexed], axis=1
    )

    return results_df


def _apply_prompt_template(
    df: pd.DataFrame, prompt_template: types.PromptTemplate
) -> None:
    """Applies a prompt template to a DataFrame.

    The DataFrame is expected to have columns corresponding to the variables
    in the prompt_template_str. The result will be in a new 'request' column.

    Args:
        df: The input DataFrame to modify.
        prompt_template: The prompt template to apply.

    Returns:
        None. The DataFrame is modified in place.
    """
    missing_vars = [var for var in prompt_template.variables if var not in df.columns]
    if missing_vars:
        raise ValueError(
            "Missing columns in DataFrame for prompt template variables:"
            f" {', '.join(missing_vars)}. Available columns:"
            f" {', '.join(df.columns.tolist())}"
        )

    if "prompt" in df.columns:
        logger.info(
            "Templated prompts stored in 'request' and will be used for"
            " inference.Original 'prompt' column is kept but not used for"
            " inference."
        )
    elif "prompt" not in df.columns and "request" in df.columns:
        logger.info("The 'request' column will be replaced with templated prompts.")

    templated_prompts = []
    for _, row in df.iterrows():
        templated_prompts.append(prompt_template.assemble(**row.to_dict()))

    df["request"] = templated_prompts


def _load_dataframe(
    api_client: BaseApiClient, src: Union[str, pd.DataFrame]
) -> pd.DataFrame:
    """Loads and prepares the prompt dataset for inference."""
    logger.info("Loading prompt dataset from: %s", src)
    try:
        loader = _evals_metric_loaders.EvalDatasetLoader(api_client=api_client)
        dataset_list_of_dicts = loader.load(src)
        if not dataset_list_of_dicts:
            raise ValueError("Prompt dataset 'prompt_dataset' must not be empty.")
        return pd.DataFrame(dataset_list_of_dicts)
    except Exception as e:
        logger.error("Failed to load prompt dataset from source: %s. Error: %s", src, e)
        raise e


def _execute_inference(
    *,
    api_client: BaseApiClient,
    src: Union[str, pd.DataFrame],
    model: Optional[Union[Callable[[Any], Any], str]] = None,
    agent_engine: Optional[Union[str, types.AgentEngine]] = None,
    dest: Optional[str] = None,
    config: Optional[genai_types.GenerateContentConfig] = None,
    prompt_template: Optional[Union[str, types.PromptTemplateOrDict]] = None,
) -> pd.DataFrame:
    """Executes inference on a given dataset using the specified model.

    Args:
        api_client: The API client.
        src: The source of the dataset. Can be a string (path to a local file,
          a GCS path, or a BigQuery table) or a Pandas DataFrame.
        model: The model to use for inference. Can be a callable function or a
          string representing a model.
        agent_engine: The agent engine to use for inference. Can be a resource
          name string or an `AgentEngine` instance.
        dest: The destination to save the inference results. Can be a string
          representing a file path or a GCS URI.
        config: The generation configuration for the model.
        prompt_template: The prompt template to use for inference.

    Returns:
        A pandas DataFrame containing the inference results.
    """
    if not api_client:
        raise ValueError("'api_client' instance must be provided.")
    prompt_dataset = _load_dataframe(api_client, src)
    if prompt_template:
        logger.info("Applying prompt template...")
        if isinstance(prompt_template, str):
            prompt_template = types.PromptTemplate(text=prompt_template)
        elif isinstance(prompt_template, dict):
            prompt_template = types.PromptTemplate.model_validate(prompt_template)

        _apply_prompt_template(prompt_dataset, prompt_template)

    if model:
        start_time = time.time()
        logger.debug("Starting inference process ...")
        results_df = _run_inference_internal(
            api_client=api_client,
            model=model,
            prompt_dataset=prompt_dataset,
            config=config,
        )
        end_time = time.time()
        logger.info("Inference completed in %.2f seconds.", end_time - start_time)

        candidate_name = None
        if isinstance(model, str):
            candidate_name = model
        elif callable(model):
            candidate_name = getattr(model, "__name__", None)

        evaluation_dataset = types.EvaluationDataset(
            eval_dataset_df=results_df,
            candidate_name=candidate_name,
        )
    elif agent_engine:
        if not isinstance(agent_engine, str) and not (
            hasattr(agent_engine, "api_client")
            and type(agent_engine).__name__ == "AgentEngine"
        ):
            raise TypeError(
                f"Unsupported agent_engine type: {type(agent_engine)}. Expecting a"
                " string (agent engine resource name in"
                " 'projects/{project_id}/locations/{location_id}/reasoningEngines/{reasoning_engine_id}' format)"
                " or a types.AgentEngine instance."
            )
        if (
            _evals_constant.INTERMEDIATE_EVENTS in prompt_dataset.columns
            or _evals_constant.RESPONSE in prompt_dataset.columns
        ):
            raise ValueError(
                "The eval dataset provided for agent run should not contain"
                f" '{_evals_constant.INTERMEDIATE_EVENTS}' or"
                f" '{_evals_constant.RESPONSE}' columns, as these columns will be"
                " generated by the agent run."
            )
        start_time = time.time()
        logger.debug("Starting Agent Run process ...")
        results_df = _run_agent_internal(
            api_client=api_client,
            agent_engine=agent_engine,
            prompt_dataset=prompt_dataset,
        )
        end_time = time.time()
        logger.info("Agent Run completed in %.2f seconds.", end_time - start_time)

        evaluation_dataset = types.EvaluationDataset(
            eval_dataset_df=results_df,
        )
    else:
        raise ValueError("Either model or agent_engine must be provided.")

    if dest:
        file_name = "inference_results.jsonl" if model else "agent_run_results.jsonl"
        is_gcs_path = dest.startswith(_gcs_utils.GCS_PREFIX)

        if is_gcs_path:
            full_dest_path = os.path.join(dest, file_name)
        else:
            os.makedirs(dest, exist_ok=True)
            full_dest_path = os.path.join(dest, file_name)

        logger.info("Saving inference / agent run results to: %s", full_dest_path)
        try:
            if is_gcs_path:
                _gcs_utils.GcsUtils(api_client=api_client).upload_dataframe(
                    df=results_df,
                    gcs_destination_blob_path=full_dest_path,
                    file_type="jsonl",
                )
                logger.info("Results saved to GCS: %s", full_dest_path)
                evaluation_dataset.gcs_source = types.GcsSource(uris=[full_dest_path])
            else:
                results_df.to_json(full_dest_path, orient="records", lines=True)
                logger.info("Results saved locally to: %s", full_dest_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to save results to %s. Error: %s", full_dest_path, e)

    return evaluation_dataset


def _get_dataset_source(
    ds_item: types.EvaluationDataset,
) -> Union[str, pd.DataFrame]:
    """Returns the source of the dataset, either a DataFrame, GCS URI, or BigQuery URI."""
    if ds_item.eval_dataset_df is not None:
        return ds_item.eval_dataset_df
    elif ds_item.gcs_source is not None and ds_item.gcs_source.uris:
        if len(ds_item.gcs_source.uris) > 1:
            logger.warning(
                "Multiple GCS URIs in GcsSource. Using the first one: %s",
                ds_item.gcs_source.uris[0],
            )
        return ds_item.gcs_source.uris[0]
    elif ds_item.bigquery_source is not None and ds_item.bigquery_source.input_uri:
        return ds_item.bigquery_source.input_uri
    else:
        raise ValueError(
            "EvaluationDataset item has no valid source"
            " (eval_dataset_df, gcs_source with uris, or bigquery_source with"
            " input_uri)."
        )


def _resolve_dataset_inputs(
    dataset: list[types.EvaluationDataset],
    dataset_schema: Optional[Literal["GEMINI", "FLATTEN", "OPENAI"]],
    loader: "_evals_metric_loaders.EvalDatasetLoader",
    agent_info: Optional[types.evals.AgentInfo] = None,
) -> tuple[types.EvaluationDataset, int]:
    """Loads and processes single or multiple datasets for evaluation.

    Args:
      dataset: The dataset(s) to process. Can be a single EvaluationDataset or a
        list of them.
      dataset_schema: The schema to use for the dataset(s). If None, it will be
        auto-detected.
      loader: An instance of EvalDatasetLoader to load data.
      agent_info: The agent info of the agent under evaluation.

    Returns:
      A tuple containing:
        - processed_eval_dataset: The processed EvaluationDataset containing
        evaluation cases.
        - num_response_candidates: The number of response candidates.
    """
    if not dataset:
        raise ValueError("Input dataset list cannot be empty.")

    num_response_candidates = len(dataset)
    datasets_to_process = dataset
    logger.info("Processing %s dataset(s).", num_response_candidates)

    loaded_raw_datasets: list[list[dict[str, Any]]] = []
    schemas_for_merge: list[str] = []

    for i, ds_item in enumerate(datasets_to_process):
        if not isinstance(ds_item, types.EvaluationDataset):
            logger.error(
                "Unexpected item type in dataset list at index %d: %s. Expected"
                " types.EvaluationDataset.",
                i,
                type(ds_item),
            )
            raise TypeError(
                f"Item at index {i} is not an EvaluationDataset: {type(ds_item)}"
            )

        ds_source_for_loader = _get_dataset_source(ds_item)
        current_loaded_data = loader.load(ds_source_for_loader)
        loaded_raw_datasets.append(current_loaded_data)

        if dataset_schema:
            current_schema = _evals_data_converters.EvalDatasetSchema(dataset_schema)
        else:
            current_schema = _evals_data_converters.auto_detect_dataset_schema(  # type: ignore[assignment]
                current_loaded_data
            )
        schemas_for_merge.append(current_schema)

        logger.info(
            "Dataset %d: Schema: %s. Using %s converter.",
            i,
            current_schema,
            _evals_data_converters.get_dataset_converter(
                current_schema
            ).__class__.__name__,
        )

    processed_eval_dataset = (
        _evals_data_converters.merge_response_datasets_into_canonical_format(
            raw_datasets=loaded_raw_datasets,
            schemas=schemas_for_merge,
            agent_info=agent_info,
        )
    )

    if not processed_eval_dataset.eval_cases:
        raise ValueError("No evaluation cases found in the dataset.")
    return processed_eval_dataset, num_response_candidates


def _resolve_evaluation_run_metrics(
    metrics: list[types.EvaluationRunMetric], api_client: Any
) -> list[types.EvaluationRunMetric]:
    """Resolves a list of evaluation run metric instances, loading RubricMetric if necessary."""
    if not metrics:
        return []
    resolved_metrics_list = []
    for metric_instance in metrics:
        if isinstance(metric_instance, types.EvaluationRunMetric):
            resolved_metrics_list.append(metric_instance)
        elif isinstance(
            metric_instance, _evals_metric_loaders.LazyLoadedPrebuiltMetric
        ):
            try:
                resolved_metric = metric_instance.resolve(api_client=api_client)
                if resolved_metric.name:
                    resolved_metrics_list.append(
                        types.EvaluationRunMetric(
                            metric=resolved_metric.name,
                            metric_config=types.UnifiedMetric(
                                predefined_metric_spec=types.PredefinedMetricSpec(
                                    metric_spec_name=resolved_metric.name,
                                )
                            ),
                        )
                    )
            except Exception as e:
                logger.error(
                    "Failed to resolve RubricMetric %s@%s: %s",
                    metric_instance.name,
                    metric_instance.version,
                    e,
                )
                raise
        else:
            try:
                metric_name_str = str(metric_instance)
                lazy_metric_instance = getattr(
                    _evals_metric_loaders.RubricMetric, metric_name_str.upper()
                )
                if isinstance(
                    lazy_metric_instance, _evals_metric_loaders.LazyLoadedPrebuiltMetric
                ):
                    resolved_metric = lazy_metric_instance.resolve(
                        api_client=api_client
                    )
                    if resolved_metric.name:
                        resolved_metrics_list.append(
                            types.EvaluationRunMetric(
                                metric=resolved_metric.name,
                                metric_config=types.UnifiedMetric(
                                    predefined_metric_spec=types.PredefinedMetricSpec(
                                        metric_spec_name=resolved_metric.name,
                                    )
                                ),
                            )
                        )
                else:
                    raise TypeError(
                        f"RubricMetric.{metric_name_str.upper()} cannot be resolved."
                    )
            except AttributeError as exc:
                raise TypeError(
                    "Unsupported metric type or invalid RubricMetric name:"
                    f" {metric_instance}"
                ) from exc
    return resolved_metrics_list


def _resolve_metrics(
    metrics: list[types.Metric], api_client: Any
) -> list[types.Metric]:
    """Resolves a list of metric instances, loading RubricMetric if necessary."""
    resolved_metrics_list = []
    for metric_instance in metrics:
        if isinstance(metric_instance, _evals_metric_loaders.LazyLoadedPrebuiltMetric):
            try:
                resolved_metrics_list.append(
                    metric_instance.resolve(api_client=api_client)
                )
            except Exception as e:
                logger.error(
                    "Failed to resolve RubricMetric %s@%s: %s",
                    metric_instance.name,
                    metric_instance.version,
                    e,
                )
                raise
        elif isinstance(metric_instance, types.Metric):
            resolved_metrics_list.append(metric_instance)
        else:
            try:
                metric_name_str = str(metric_instance)
                lazy_metric_instance = getattr(
                    _evals_metric_loaders.RubricMetric, metric_name_str.upper()
                )
                if isinstance(
                    lazy_metric_instance, _evals_metric_loaders.LazyLoadedPrebuiltMetric
                ):
                    resolved_metrics_list.append(
                        lazy_metric_instance.resolve(api_client=api_client)
                    )
                else:
                    raise TypeError(
                        f"RubricMetric.{metric_name_str.upper()} cannot be resolved."
                    )
            except AttributeError as exc:
                raise TypeError(
                    "Unsupported metric type or invalid RubricMetric name:"
                    f" {metric_instance}"
                ) from exc
    return resolved_metrics_list


def _execute_evaluation(
    *,
    api_client: Any,
    dataset: Union[types.EvaluationDataset, list[types.EvaluationDataset]],
    metrics: list[types.Metric],
    dataset_schema: Optional[Literal["GEMINI", "FLATTEN", "OPENAI"]] = None,
    dest: Optional[str] = None,
    **kwargs,
) -> types.EvaluationResult:
    """Evaluates a dataset using the provided metrics.

    Args:
        api_client: The API client.
        dataset: The dataset to evaluate.
        metrics: The metrics to evaluate the dataset against.
        dataset_schema: The schema of the dataset.
        dest: The destination to save the evaluation results.
        **kwargs: Extra arguments to pass to evaluation, such as `agent_info`.

    Returns:
        The evaluation result.
    """

    logger.info("Preparing dataset(s) and metrics...")
    if isinstance(dataset, types.EvaluationDataset):
        dataset_list = [dataset]
    elif isinstance(dataset, list):
        for item in dataset:
            if not isinstance(item, types.EvaluationDataset):
                raise TypeError(
                    f"Unsupported dataset type: {type(item)}. "
                    "Must be EvaluationDataset."
                )
        dataset_list = dataset
    else:
        raise TypeError(
            f"Unsupported dataset type: {type(dataset)}. Must be an"
            " EvaluationDataset or a list of EvaluationDataset."
        )
    original_candidate_names = [
        ds.candidate_name or f"candidate_{i + 1}" for i, ds in enumerate(dataset_list)
    ]
    name_counts = collections.Counter(original_candidate_names)
    deduped_candidate_names = []
    current_name_counts: collections.defaultdict[Any, int] = collections.defaultdict(
        int
    )

    for name in original_candidate_names:
        if name_counts[name] > 1:
            current_name_counts[name] += 1
            deduped_candidate_names.append(f"{name} #{current_name_counts[name]}")
        else:
            deduped_candidate_names.append(name)

    loader = _evals_metric_loaders.EvalDatasetLoader(api_client=api_client)

    agent_info = kwargs.get("agent_info", None)
    validated_agent_info = None
    if agent_info:
        if isinstance(agent_info, dict):
            validated_agent_info = types.evals.AgentInfo.model_validate(agent_info)
        elif isinstance(agent_info, types.evals.AgentInfo):
            validated_agent_info = agent_info
        else:
            raise TypeError(
                f"agent_info values must be of type types.evals.AgentInfo or dict, but got {type(agent_info)}'"
            )

    processed_eval_dataset, num_response_candidates = _resolve_dataset_inputs(
        dataset=dataset_list,
        dataset_schema=dataset_schema,
        loader=loader,
        agent_info=validated_agent_info,
    )

    resolved_metrics = _resolve_metrics(metrics, api_client)

    evaluation_run_config = _evals_metric_handlers.EvaluationRunConfig(
        evals_module=evals.Evals(api_client_=api_client),
        dataset=processed_eval_dataset,
        metrics=resolved_metrics,
        num_response_candidates=num_response_candidates,
    )

    logger.info("Running Metric Computation...")
    t1 = time.perf_counter()
    evaluation_result = _evals_metric_handlers.compute_metrics_and_aggregate(
        evaluation_run_config
    )
    t2 = time.perf_counter()
    logger.info("Evaluation took: %f seconds", t2 - t1)

    evaluation_result.evaluation_dataset = dataset_list
    evaluation_result.agent_info = validated_agent_info

    if not evaluation_result.metadata:
        evaluation_result.metadata = types.EvaluationRunMetadata()

    evaluation_result.metadata.creation_timestamp = datetime.datetime.now(
        datetime.timezone.utc
    )

    if deduped_candidate_names:
        evaluation_result.metadata.candidate_names = deduped_candidate_names

    logger.info("Evaluation run completed.")

    if dest:
        uploaded_path = _gcs_utils.GcsUtils(
            api_client=api_client
        ).upload_json_to_prefix(
            data=evaluation_result.model_dump(
                mode="json",
                exclude_none=True,
                exclude={"evaluation_dataset"},
            ),
            gcs_dest_prefix=dest,
            filename_prefix="evaluation_result",
        )
        logger.info(
            "Evaluation results uploaded successfully to GCS: %s", uploaded_path
        )
    return evaluation_result


def _run_agent_internal(
    api_client: BaseApiClient,
    agent_engine: Union[str, types.AgentEngine],
    prompt_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Runs an agent."""
    raw_responses = _run_agent(
        api_client=api_client, agent_engine=agent_engine, prompt_dataset=prompt_dataset
    )
    processed_intermediate_events = []
    processed_responses = []
    for resp_item in raw_responses:
        intermediate_events_row = []
        response_row = None
        if isinstance(resp_item, list):
            try:
                response_row = resp_item[-1]["content"]["parts"][0]["text"]
                for intermediate_event in resp_item[:-1]:
                    intermediate_events_row.append(
                        {
                            "event_id": intermediate_event["id"],
                            "content": intermediate_event["content"],
                            "creation_timestamp": intermediate_event["timestamp"],
                            "author": intermediate_event["author"],
                        }
                    )
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_payload = {
                    "error": (
                        f"Failed to parse agent run response {str(resp_item)} to "
                        f"intermediate events and final response: {e}"
                    ),
                }
                response_row = json.dumps(error_payload)
        else:
            error_payload = {
                "error": "Unexpected response type from agent run",
                "response_type": str(type(resp_item)),
                "details": str(resp_item),
            }
            response_row = json.dumps(error_payload)

        processed_intermediate_events.append(intermediate_events_row)
        processed_responses.append(response_row)

    if len(processed_responses) != len(prompt_dataset) or len(
        processed_responses
    ) != len(processed_intermediate_events):
        raise RuntimeError(
            "Critical prompt/response/intermediate_events count mismatch: %d prompts vs %d vs %d"
            " responses. This indicates an issue in response collection."
            % (
                len(prompt_dataset),
                len(processed_responses),
                len(processed_intermediate_events),
            )
        )

    results_df_responses_only = pd.DataFrame(
        {
            _evals_constant.INTERMEDIATE_EVENTS: processed_intermediate_events,
            _evals_constant.RESPONSE: processed_responses,
        }
    )

    prompt_dataset_indexed = prompt_dataset.reset_index(drop=True)
    results_df_responses_only_indexed = results_df_responses_only.reset_index(drop=True)

    results_df = pd.concat(
        [prompt_dataset_indexed, results_df_responses_only_indexed], axis=1
    )
    return results_df


def _run_agent(
    api_client: BaseApiClient,
    agent_engine: Union[str, types.AgentEngine],
    prompt_dataset: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Internal helper to run inference using Gemini model with concurrency."""
    return _execute_inference_concurrently(
        api_client=api_client,
        agent_engine=agent_engine,
        prompt_dataset=prompt_dataset,
        progress_desc="Agent Run",
        gemini_config=None,
        inference_fn=_execute_agent_run_with_retry,
    )


def _execute_agent_run_with_retry(
    row: pd.Series,
    contents: Union[genai_types.ContentListUnion, genai_types.ContentListUnionDict],
    agent_engine: types.AgentEngine,
    max_retries: int = 3,
) -> Union[list[dict[str, Any]], dict[str, Any]]:
    """Executes agent run for a single prompt."""
    try:
        if isinstance(row["session_inputs"], str):
            session_inputs = types.evals.SessionInput.model_validate(
                json.loads(row["session_inputs"])
            )
        elif isinstance(row["session_inputs"], dict):
            session_inputs = types.evals.SessionInput.model_validate(
                row["session_inputs"]
            )
        elif isinstance(row["session_inputs"], types.evals.SessionInput):
            session_inputs = row["session_inputs"]
        else:
            raise TypeError(
                f"Unsupported session_inputs type: {type(row['session_inputs'])}. "
                "Expecting string or dict in types.evals.SessionInput format."
            )
        user_id = session_inputs.user_id
        session_state = session_inputs.state
        session = agent_engine.create_session(
            user_id=user_id,
            state=session_state,
        )
    except KeyError as e:
        return {"error": f"Failed to get all required agent engine inputs: {e}"}
    except Exception as e:
        return {"error": f"Failed to create a new session : {e}"}
    for attempt in range(max_retries):
        try:
            responses = []
            for event in agent_engine.stream_query(
                user_id=user_id,
                session_id=session["id"],
                message=contents,
            ):
                if event and "content" in event and "parts" in event["content"]:
                    responses.append(event)
            return responses
        except api_exceptions.ResourceExhausted as e:
            logger.warning(
                "Resource Exhausted error on attempt %d/%d: %s. Retrying in %s"
                " seconds...",
                attempt + 1,
                max_retries,
                e,
                2**attempt,
            )
            if attempt == max_retries - 1:
                return {"error": f"Resource exhausted after retries: {e}"}
            time.sleep(2**attempt)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Unexpected error during generate_content on attempt %d/%d: %s",
                attempt + 1,
                max_retries,
                e,
            )

            if attempt == max_retries - 1:
                return {"error": f"Failed after retries: {e}"}
            time.sleep(1)
    return {"error": f"Failed to get agent run results after {max_retries} retries"}


def _convert_gcs_to_evaluation_item_result(
    api_client: BaseApiClient,
    gcs_uri: str,
) -> types.EvaluationItemResult:
    """Converts a json file to an EvaluationItemResult."""
    logger.info("Loading evaluation item result from GCS: %s", gcs_uri)
    gcs_utils = _gcs_utils.GcsUtils(api_client=api_client)
    try:
        eval_item_data = json.loads(gcs_utils.read_file_contents(gcs_uri))
        return types.EvaluationItemResult(**eval_item_data)
    except Exception as e:
        logger.error(
            "Failed to load evaluation result from GCS: %s. Error: %s", gcs_uri, e
        )
    return types.EvaluationItemResult()


def _convert_gcs_to_evaluation_item_request(
    api_client: BaseApiClient,
    gcs_uri: str,
) -> types.EvaluationItemRequest:
    """Converts a json file to an EvaluationItemRequest."""
    logger.info("Loading evaluation item request from GCS: %s", gcs_uri)
    gcs_utils = _gcs_utils.GcsUtils(api_client=api_client)
    try:
        eval_item_data = json.loads(gcs_utils.read_file_contents(gcs_uri))
        return types.EvaluationItemRequest(**eval_item_data)
    except Exception as e:
        logger.error(
            "Failed to load evaluation request from GCS: %s. Error: %s", gcs_uri, e
        )
    return types.EvaluationItemRequest()


def _get_aggregated_metrics(
    results: types.EvaluationRunResults,
) -> list[types.AggregatedMetricResult]:
    """Retrieves an EvaluationResult from the resource name."""
    if (
        not results
        or not results.summary_metrics
        or not results.summary_metrics.metrics
    ):
        return []

    aggregated_metrics_dict = {}
    for name, value in results.summary_metrics.metrics.items():
        result = name.rsplit("/", 1)
        full_metric_name = result[0]
        aggregated_metric_name = result[1]
        if full_metric_name not in aggregated_metrics_dict:
            aggregated_metrics_dict[full_metric_name] = {}
            aggregated_metrics_dict[full_metric_name]["sub_metric_name"] = (
                full_metric_name.split("/")[-1]
            )
        aggregated_metrics_dict[full_metric_name][aggregated_metric_name] = value

    items_sorted = sorted(
        aggregated_metrics_dict.items(),
        key=lambda item: (item[1]["sub_metric_name"], item[0]),
    )

    return [
        types.AggregatedMetricResult(
            metric_name=name.split("/")[-1],
            mean_score=values.get("AVERAGE"),
            stdev_score=values.get("STANDARD_DEVIATION"),
        )
        for name, values in items_sorted
    ]


def _get_eval_case_result_from_eval_item(
    index: int,
    eval_item: types.EvaluationItem,
) -> types.EvalCaseResult:
    """Transforms EvaluationItem to EvalCaseResult."""
    metric_results = {}
    if eval_item.evaluation_response.candidate_results:
        for candidate_result in eval_item.evaluation_response.candidate_results:
            metric_results[candidate_result.metric] = types.EvalCaseMetricResult(
                metric_name=candidate_result.metric,
                score=candidate_result.score,
                explanation=candidate_result.explanation,
                rubric_verdicts=candidate_result.rubric_verdicts,
                error_message=(eval_item.error.message if eval_item.error else None),
            )
    return types.EvalCaseResult(
        eval_case_index=index,
        response_candidate_results=[
            types.ResponseCandidateResult(
                response_index=0,
                metric_results=metric_results,
            )
        ],
    )


def _convert_request_to_dataset_row(
    request: types.EvaluationItemRequest,
) -> dict[str, Any]:
    """Converts an EvaluationItemRequest to a dictionary."""
    dict_row = {}
    dict_row[_evals_constant.PROMPT] = (
        request.prompt.text if request.prompt.text else None
    )
    dict_row[_evals_constant.REFERENCE] = request.golden_response
    intermediate_events = []
    if request.candidate_responses:
        for candidate in request.candidate_responses:
            dict_row[candidate.candidate] = candidate.text if candidate.text else None
            if candidate.events:
                for event in candidate.events:
                    content_dict = {"parts": event.parts, "role": event.role}
                    int_events_dict = {
                        "event_id": candidate.candidate,
                        "content": content_dict,
                    }
                    intermediate_events.append(int_events_dict)
    dict_row[_evals_constant.INTERMEDIATE_EVENTS] = intermediate_events
    return dict_row


def _transform_dataframe(rows: list[dict[str, Any]]) -> list[types.EvaluationDataset]:
    """Transforms rows to a list of EvaluationDatasets.

    Args:
      rows: A list of rows, each row is a dictionary of candidate name to response
        text.
    Returns:
      A list of EvaluationDatasets, one for each candidate.
    """
    df = pd.DataFrame(rows)
    candidates = [
        col for col in df.columns if col not in _evals_constant.COMMON_DATASET_COLUMNS
    ]

    eval_dfs = [
        types.EvaluationDataset(
            candidate_name=candidate,
            eval_dataset_df=df.rename(columns={candidate: _evals_constant.RESPONSE}),
        )
        for candidate in candidates
    ]
    return eval_dfs


def _get_eval_cases_eval_dfs_from_eval_items(
    eval_items: list[types.EvaluationItem],
) -> tuple[list[types.EvalCaseResult], list[types.EvaluationDataset]]:
    """Converts an EvaluationSet to a list of EvaluationCaseResults and EvaluationDatasets.

    Args:
      api_client: The API client.
      evaluation_set_name: The name of the evaluation set.
    Returns:
      A tuple of two lists:
        - eval_case_results: A list of EvalCaseResults, one for each evaluation
          item.
        - eval_dfs: A list of EvaluationDatasets, one for each candidate.
    """
    dataset_rows = []
    eval_case_results = []
    for index, eval_item in enumerate(eval_items):
        if (
            eval_item
            and eval_item.evaluation_response
            and eval_item.evaluation_response.request
        ):
            eval_case_results.append(
                _get_eval_case_result_from_eval_item(index, eval_item)
            )
            dataset_rows.append(
                _convert_request_to_dataset_row(eval_item.evaluation_response.request)
            )
    eval_dfs = _transform_dataframe(dataset_rows)
    return eval_case_results, eval_dfs


def _get_agent_info_from_inference_configs(
    candidate_names: list[str],
    inference_configs: Optional[dict[str, types.EvaluationRunInferenceConfig]] = None,
) -> Optional[types.evals.AgentInfo]:
    """Retrieves an AgentInfo from the inference configs."""
    # TODO(lakeyk): Support multiple agents.
    if not (
        inference_configs
        and candidate_names
        and candidate_names[0] in inference_configs
        and inference_configs[candidate_names[0]].agent_config
    ):
        return None
    if len(inference_configs.keys()) > 1:
        logger.warning(
            "Multiple agents are not supported yet. Displaying the first agent."
        )
    agent_config = inference_configs[candidate_names[0]].agent_config
    di = agent_config.developer_instruction
    instruction = di.parts[0].text if di and di.parts and di.parts[0].text else None
    return types.evals.AgentInfo(
        name=candidate_names[0],
        instruction=instruction,
        tool_declarations=agent_config.tools,
    )


def _get_eval_result_from_eval_items(
    results: types.EvaluationRunResults,
    eval_items: list[types.EvaluationItem],
    inference_configs: Optional[dict[str, types.EvaluationRunInferenceConfig]] = None,
) -> types.EvaluationResult:
    """Retrieves an EvaluationResult from the EvaluationRunResults.

    This function is used to convert an EvaluationRunResults object used by the
    Evaluation Management API to an EvaluationResult object. It is used to display
    the evaluation results in the UI.

    Args:
      results: The EvaluationRunResults object.
      eval_items: The list of EvaluationItems.
    Returns:
      An EvaluationResult object.
    """
    aggregated_metrics = _get_aggregated_metrics(results)
    eval_case_results, eval_dfs = _get_eval_cases_eval_dfs_from_eval_items(eval_items)
    candidate_names = [eval_df.candidate_name for eval_df in eval_dfs]
    eval_result = types.EvaluationResult(
        summary_metrics=aggregated_metrics,
        eval_case_results=eval_case_results,
        evaluation_dataset=eval_dfs,
        metadata=types.EvaluationRunMetadata(
            candidate_names=candidate_names,
        ),
        agent_info=_get_agent_info_from_inference_configs(
            candidate_names, inference_configs
        ),
    )
    return eval_result


def _convert_evaluation_run_results(
    api_client: BaseApiClient,
    evaluation_run_results: types.EvaluationRunResults,
    inference_configs: Optional[dict[str, types.EvaluationRunInferenceConfig]] = None,
) -> list[types.EvaluationItem]:
    """Retrieves an EvaluationItem from the EvaluationRunResults."""
    if not evaluation_run_results or not evaluation_run_results.evaluation_set:
        return []

    evals_module = evals.Evals(api_client_=api_client)
    eval_set = evals_module.get_evaluation_set(
        name=evaluation_run_results.evaluation_set
    )

    eval_items = []
    if eval_set and eval_set.evaluation_items:
        eval_items = [
            evals_module.get_evaluation_item(name=item_name)
            for item_name in eval_set.evaluation_items
        ]
    return _get_eval_result_from_eval_items(
        evaluation_run_results, eval_items, inference_configs
    )


async def _convert_evaluation_run_results_async(
    api_client: BaseApiClient,
    evaluation_run_results: types.EvaluationRunResults,
    inference_configs: Optional[dict[str, types.EvaluationRunInferenceConfig]] = None,
) -> list[types.EvaluationItem]:
    """Retrieves an EvaluationItem from the EvaluationRunResults."""
    if not evaluation_run_results or not evaluation_run_results.evaluation_set:
        return []

    evals_module = evals.AsyncEvals(api_client_=api_client)
    eval_set = await evals_module.get_evaluation_set(
        name=evaluation_run_results.evaluation_set
    )

    eval_items = []
    if eval_set and eval_set.evaluation_items:
        tasks = [
            evals_module.get_evaluation_item(name=eval_item)
            for eval_item in eval_set.evaluation_items
        ]
        eval_items = await asyncio.gather(*tasks)
    return _get_eval_result_from_eval_items(
        evaluation_run_results, eval_items, inference_configs
    )


def _object_to_dict(obj) -> dict[str, Any]:
    """Converts an object to a dictionary."""
    if not hasattr(obj, "__dict__"):
        return obj  # Not an object with attributes, return as is (e.g., int, str)

    result: dict[str, Any] = {}
    for key, value in obj.__dict__.items():
        if value is None:
            continue
        if isinstance(value, (int, float, str, bool)):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = [_object_to_dict(item) for item in value]
        elif isinstance(value, bytes):
            result[key] = base64.b64encode(value).decode("utf-8")
        elif hasattr(value, "__dict__"):  # Nested object
            result[key] = _object_to_dict(value)
        else:
            result[key] = value  # Handle other types like sets, etc.
    return result


def _create_evaluation_set_from_dataframe(
    api_client: BaseApiClient,
    gcs_dest_prefix: str,
    eval_df: pd.DataFrame,
    candidate_name: Optional[str] = None,
) -> types.EvaluationSet:
    """Converts a dataframe to an EvaluationSet."""
    eval_item_requests = []
    for _, row in eval_df.iterrows():
        intermediate_events = []
        if (
            _evals_constant.INTERMEDIATE_EVENTS in row
            and isinstance(row[_evals_constant.INTERMEDIATE_EVENTS], list)
            and len(row[_evals_constant.INTERMEDIATE_EVENTS]) > 0
        ):
            for event in row[_evals_constant.INTERMEDIATE_EVENTS]:
                if "content" in event:
                    intermediate_events.append(event["content"])
        eval_item_requests.append(
            types.EvaluationItemRequest(
                prompt=(
                    types.EvaluationPrompt(text=row[_evals_constant.PROMPT])
                    if _evals_constant.PROMPT in row
                    else None
                ),
                golden_response=(
                    types.CandidateResponse(text=row[_evals_constant.REFERENCE])
                    if _evals_constant.REFERENCE in row
                    else None
                ),
                candidate_responses=[
                    types.CandidateResponse(
                        candidate=candidate_name or "Candidate 1",
                        text=row.get(_evals_constant.RESPONSE, None),
                        events=(
                            intermediate_events
                            if len(intermediate_events) > 0
                            else None
                        ),
                    )
                ],
            )
        )
    logger.info("Writing evaluation item requests to GCS.")
    gcs_utils = _gcs_utils.GcsUtils(api_client=api_client)
    evals_module = evals.Evals(api_client_=api_client)
    eval_items = []
    for eval_item_request in eval_item_requests:
        gcs_uri = gcs_utils.upload_json_to_prefix(
            data=_object_to_dict(eval_item_request),
            gcs_dest_prefix=gcs_dest_prefix,
            filename_prefix="request",
        )
        eval_item = evals_module.create_evaluation_item(
            evaluation_item_type=types.EvaluationItemType.REQUEST,
            gcs_uri=gcs_uri,
            display_name="sdk-generated-eval-item",
        )
        eval_items.append(eval_item.name)
    logger.info("Creating evaluation set from GCS URIs")
    evaluation_set = evals_module.create_evaluation_set(
        evaluation_items=eval_items,
    )

    return evaluation_set
