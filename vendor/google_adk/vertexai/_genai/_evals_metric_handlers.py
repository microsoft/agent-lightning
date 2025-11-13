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
"""Handlers for computing evaluation metrics."""

import abc
import collections
from concurrent import futures
import json
import logging
import statistics
import time
from typing import Any, Callable, Optional, TypeVar, Union

from google.genai import errors as genai_errors
from google.genai import _common
from google.genai import types as genai_types
from tqdm import tqdm
from typing_extensions import override

from . import _evals_common
from . import _evals_constant
from . import evals
from . import types


logger = logging.getLogger(__name__)
_MAX_RETRIES = 3


def _has_tool_call(intermediate_events: Optional[list[types.evals.Event]]) -> bool:
    """Checks if any event in intermediate_events has a function call."""
    if not intermediate_events:
        return False
    for event in intermediate_events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    return True
    return False


def _extract_text_from_content(
    content: Optional[genai_types.Content], warn_property: str = "text"
) -> Optional[str]:
    """Extracts and concatenates all text parts from a Content object."""
    if not content or not content.parts:
        return None

    text_accumulator = ""
    any_text_part_found = False
    non_text_part_names = []

    for part_obj in content.parts:
        part_dump = part_obj.model_dump(exclude={"text", "thought"})
        for field_name, field_value in part_dump.items():
            if field_value is not None:
                if field_name not in non_text_part_names:
                    non_text_part_names.append(field_name)

        if isinstance(part_obj.text, str):
            if (
                hasattr(part_obj, "thought")
                and isinstance(part_obj.thought, bool)
                and part_obj.thought
            ):
                continue
            any_text_part_found = True
            text_accumulator += part_obj.text

    if non_text_part_names and any_text_part_found:
        logger.warning(
            "Warning: content contains non-text parts: %s. Returning"
            " concatenated %s result from text parts. Inspect individual parts"
            " for full content.",
            non_text_part_names,
            warn_property,
        )
    return text_accumulator if any_text_part_found else None


def _default_aggregate_scores(
    metric_name: str,
    eval_case_metric_results: list[types.EvalCaseMetricResult],
) -> types.AggregatedMetricResult:
    """Default aggregation logic using mean and standard deviation."""
    scores = []
    num_error = 0
    num_valid = 0

    for result in eval_case_metric_results:
        if result.error_message is None and result.score is not None:
            try:
                scores.append(float(result.score))
                num_valid += 1
            except (ValueError, TypeError):
                logger.warning(
                    "Could not convert score '%s' to float for metric '%s' during"
                    " default aggregation. Counting as error.",
                    result.score,
                    metric_name,
                )
                num_error += 1
        else:
            num_error += 1

    mean_score = None
    stdev_score = None
    if num_valid > 0:
        try:
            mean_score = statistics.mean(scores)
        except statistics.StatisticsError as e:
            logger.warning("Could not calculate mean for %s: %s", metric_name, e)
    if num_valid > 1:
        try:
            stdev_score = statistics.stdev(scores)
        except statistics.StatisticsError as e:
            logger.warning("Could not calculate stdev for %s: %s", metric_name, e)

    return types.AggregatedMetricResult(
        metric_name=metric_name,
        num_cases_total=len(eval_case_metric_results),
        num_cases_valid=num_valid,
        num_cases_error=num_error,
        mean_score=mean_score,
        stdev_score=stdev_score,
    )


class MetricHandler(abc.ABC):
    """Abstract base class for metric handlers."""

    def __init__(self, module: "evals.Evals", metric: types.Metric):
        self.module = module
        self.metric = metric

    @abc.abstractmethod
    def get_metric_result(
        self, eval_case: types.EvalCase, response_index: int
    ) -> types.EvalCaseMetricResult:
        """Processes a single evaluation case for a specific metric."""
        raise NotImplementedError()

    @abc.abstractmethod
    def aggregate(
        self, eval_case_metric_results: list[types.EvalCaseMetricResult]
    ) -> types.AggregatedMetricResult:
        """Aggregates the metric results for a specific metric."""
        raise NotImplementedError()


class ComputationMetricHandler(MetricHandler):
    """Metric handler for computation metrics."""

    SUPPORTED_COMPUTATION_METRICS = frozenset(
        {
            "exact_match",
            "bleu",
            "rouge_1",
            "rouge_l_sum",
            "tool_call_valid",
            "tool_name_match",
            "tool_parameter_key_match",
            "tool_parameter_kv_match",
            # TODO b/423934249 - Add trajectory metrics once they are supported.
        }
    )

    def __init__(self, module: "evals.Evals", metric: types.Metric):
        super().__init__(module=module, metric=metric)
        if self.metric.name not in self.SUPPORTED_COMPUTATION_METRICS:
            raise ValueError(
                f"Metric '{self.metric.name}' is not supported for computation."
            )

    def _build_request_payload(
        self, eval_case: types.EvalCase, response_index: int
    ) -> dict[str, Any]:
        """Builds the request parameters for evaluate instances."""
        request_payload = {}
        if response_index >= len(eval_case.responses):
            raise IndexError(
                f"response_index {response_index} out of bounds for eval_case with"
                f" {len(eval_case.responses)} responses."
            )
        if eval_case.responses is None:
            raise ValueError(
                f"No responses found for eval_case with ID {eval_case.eval_case_id}."
            )
        current_response_candidate = eval_case.responses[response_index]
        if _extract_text_from_content(current_response_candidate.response) is None:
            raise ValueError(
                f"Response text missing for candidate {response_index} in eval_case"
                f" {eval_case.eval_case_id or 'Unknown ID'}."
            )

        if (
            eval_case.reference is None
            or _extract_text_from_content(eval_case.reference.response) is None
        ):
            raise ValueError(
                "Reference text missing for eval_case"
                f" {eval_case.eval_case_id or 'Unknown ID'}."
            )
        logger.debug("eval_case: %s", eval_case)

        if self.metric.name and self.metric.name.startswith("rouge"):
            request_payload["rouge_input"] = {
                "metric_spec": {
                    "rouge_type": (
                        "rougeLsum" if self.metric.name == "rouge_l_sum" else "rouge1"
                    ),
                },
                "instances": [
                    {
                        "prediction": _extract_text_from_content(
                            current_response_candidate.response
                        ),
                        "reference": _extract_text_from_content(
                            eval_case.reference.response
                        ),
                    }
                ],
            }
        else:
            request_payload[f"{self.metric.name}_input"] = {
                "metric_spec": {},
                "instances": [
                    {
                        "prediction": _extract_text_from_content(
                            current_response_candidate.response
                        ),
                        "reference": _extract_text_from_content(
                            eval_case.reference.response
                        ),
                    }
                ],
            }
        logger.debug("request_payload: %s", request_payload)
        return request_payload

    @override
    def get_metric_result(
        self, eval_case: types.EvalCase, response_index: int
    ) -> types.EvalCaseMetricResult:
        """Processes a single evaluation case for a specific computation metric."""

        metric_name = self.metric.name
        logger.debug(
            "ComputationMetricHandler: Processing '%s' for case: %s",
            metric_name,
            eval_case.model_dump(exclude_none=True),
        )
        response = self.module.evaluate_instances(
            metric_config=self._build_request_payload(eval_case, response_index)
        ).model_dump(exclude_none=True)
        logger.debug("response: %s", response)
        score = None
        for _, result_value in response.items():
            if isinstance(result_value, dict) and result_value:
                for _, metric_value in result_value.items():
                    if isinstance(metric_value, list) and metric_value:
                        score = metric_value[0]["score"]
                        break
        logger.debug("Metric result: %s", score)
        return types.EvalCaseMetricResult(
            metric_name=metric_name,
            score=score,
        )

    @override
    def aggregate(
        self, eval_case_metric_results: list[types.EvalCaseMetricResult]
    ) -> types.AggregatedMetricResult:
        """Aggregates the metric results for a computation metric."""
        logger.debug("Aggregating results for computation metric: %s", self.metric.name)
        return _default_aggregate_scores(self.metric.name, eval_case_metric_results)


class TranslationMetricHandler(MetricHandler):
    """Metric handler for translation metrics."""

    SUPPORTED_TRANSLATION_METRICS = frozenset({"comet", "metricx"})

    def __init__(self, module: "evals.Evals", metric: types.Metric):
        super().__init__(module=module, metric=metric)

        if self.metric.name not in self.SUPPORTED_TRANSLATION_METRICS:
            raise ValueError(
                f"Metric '{self.metric.name}' is not supported for translation."
            )

    def _build_request_payload(
        self, eval_case: types.EvalCase, response_index: int
    ) -> dict[str, Any]:
        """Builds the request parameters for evaluate instances."""
        request_payload = {}
        metric_input_name = f"{self.metric.name}_input"
        version = None
        if hasattr(self.metric, "version"):
            version = self.metric.version
        elif self.metric.name == "comet":
            version = "COMET_22_SRC_REF"
        elif self.metric.name == "metricx":
            version = "METRICX_24_SRC_REF"

        source_language = None
        target_language = None
        if hasattr(self.metric, "source_language"):
            source_language = self.metric.source_language
        if hasattr(self.metric, "target_language"):
            target_language = self.metric.target_language

        if response_index >= len(eval_case.responses):
            raise IndexError(
                f"response_index {response_index} out of bounds for eval_case with"
                f" {len(eval_case.responses)} responses."
            )

        if eval_case.responses is None:
            raise ValueError(
                f"No responses found for eval_case with ID {eval_case.eval_case_id}."
            )
        current_response_candidate = eval_case.responses[response_index]
        if _extract_text_from_content(current_response_candidate.response) is None:
            raise ValueError(
                f"Response text missing for candidate {response_index} in eval_case"
                f" {eval_case.eval_case_id or 'Unknown ID'}."
            )

        if (
            eval_case.reference is None
            or _extract_text_from_content(eval_case.reference.response) is None
        ):
            raise ValueError(
                "Reference text missing for eval_case"
                f" {eval_case.eval_case_id or 'Unknown ID'}."
            )
        if _extract_text_from_content(eval_case.prompt) is None:
            raise ValueError(
                "Prompt text (source for translation) missing for eval_case"
                f" {eval_case.eval_case_id or 'Unknown ID'}."
            )

        request_payload[metric_input_name] = {
            "metric_spec": {
                "version": version,
                "source_language": source_language,
                "target_language": target_language,
            },
            "instance": {
                "prediction": _extract_text_from_content(
                    current_response_candidate.response
                ),
                "reference": _extract_text_from_content(eval_case.reference.response),
                "source": _extract_text_from_content(eval_case.prompt),
            },
        }
        return request_payload

    @override
    def get_metric_result(
        self, eval_case: types.EvalCase, response_index: int
    ) -> types.EvalCaseMetricResult:
        """Processes a single evaluation case for a specific translation metric."""
        metric_name = self.metric.name
        logger.debug(
            "TranslationMetricHandler: Processing '%s' for case: %s",
            metric_name,
            eval_case,
        )
        api_response = self.module.evaluate_instances(
            metric_config=self._build_request_payload(eval_case, response_index)
        )
        logger.debug("API Response: %s", api_response)

        score = None
        error_message = None

        try:
            if metric_name == "comet":
                if api_response and api_response.comet_result:
                    score = api_response.comet_result.score
                else:
                    logger.warning(
                        "Comet result missing in API response for metric '%s'."
                        " API response: %s",
                        metric_name,
                        (
                            api_response.model_dump_json(exclude_none=True)
                            if api_response
                            else "None"
                        ),
                    )
            elif metric_name == "metricx":
                if api_response and api_response.metricx_result:
                    score = api_response.metricx_result.score
                else:
                    logger.warning(
                        "MetricX result missing in API response for metric '%s'."
                        " API response: %s",
                        metric_name,
                        (
                            api_response.model_dump_json(exclude_none=True)
                            if api_response
                            else "None"
                        ),
                    )
            if score is None and not error_message:
                logger.warning(
                    "Score could not be extracted for translation metric '%s'."
                    " API response: %s",
                    metric_name,
                    (
                        api_response.model_dump_json(exclude_none=True)
                        if api_response
                        else "None"
                    ),
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error processing/extracting score for translation metric '%s': %s."
                " API response: %s",
                metric_name,
                e,
                (
                    api_response.model_dump_json(exclude_none=True)
                    if api_response
                    else "None"
                ),
                exc_info=True,
            )
            error_message = f"Error extracting score: {e}"

        return types.EvalCaseMetricResult(
            metric_name=metric_name,
            score=score,
            error_message=error_message,
        )

    @override
    def aggregate(
        self, eval_case_metric_results: list[types.EvalCaseMetricResult]
    ) -> types.AggregatedMetricResult:
        """Aggregates the metric results for a translation metric."""
        logger.debug("Aggregating results for translation metric: %s", self.metric.name)
        return _default_aggregate_scores(self.metric.name, eval_case_metric_results)


class LLMMetricHandler(MetricHandler):
    """Metric handler for LLM metrics."""

    def __init__(self, module: "evals.Evals", metric: types.LLMMetric):
        super().__init__(module=module, metric=metric)

    def _build_rubric_based_input(
        self, eval_case: types.EvalCase, response_content: genai_types.Content
    ) -> dict[str, Any]:
        """Builds the payload for a rubric-based LLM metric."""
        eval_case_dict = eval_case.model_dump(exclude={"responses"})
        rubric_groups_data = eval_case_dict.get("rubric_groups")

        if not isinstance(rubric_groups_data, dict):
            raise ValueError(
                f"Dataset column 'rubric_groups' for case {eval_case.eval_case_id} "
                "must be a dictionary."
            )

        rubric_group_from_data = rubric_groups_data.get(
            self.metric.rubric_group_name, {}
        )
        if isinstance(rubric_group_from_data, dict):
            rubrics_list = rubric_group_from_data.get("rubrics", [])
        else:
            rubrics_list = []

        if not isinstance(rubrics_list, list):
            logger.warning(
                "Rubrics for group '%s' in case %s is not a list: %s. "
                "Skipping rubrics for this case.",
                self.metric.rubric_group_name,
                eval_case.eval_case_id,
                rubrics_list,
            )
            rubrics_list = []

        parsed_rubrics = [types.evals.Rubric(**r) for r in rubrics_list]
        rubric_enhanced_contents = {
            "prompt": (
                [eval_case.prompt.model_dump(mode="json", exclude_none=True)]
                if eval_case.prompt
                else None
            ),
            "response": [response_content.model_dump(mode="json", exclude_none=True)],
            "rubric_groups": {
                self.metric.rubric_group_name: {
                    "rubrics": [
                        r.model_dump(mode="json", exclude_none=True)
                        for r in parsed_rubrics
                    ]
                }
            },
        }

        metric_spec_payload = {
            "metric_prompt_template": self.metric.prompt_template,
            "rubric_group_key": self.metric.rubric_group_name,
        }

        return {
            "rubric_based_metric_input": {
                "metric_spec": metric_spec_payload,
                "instance": {"rubric_enhanced_contents": rubric_enhanced_contents},
            }
        }

    def _build_pointwise_input(
        self, eval_case: types.EvalCase, response_content: genai_types.Content
    ) -> dict[str, Any]:
        """Builds the payload for a standard pointwise LLM metric."""
        instance_data = {
            "prompt": eval_case.prompt,
            "response": response_content,
        }
        template_obj = types.PromptTemplate(text=self.metric.prompt_template)
        required_vars = template_obj.variables - set(instance_data.keys())
        for var_name in required_vars:
            if hasattr(eval_case, var_name):
                instance_data[var_name] = getattr(eval_case, var_name)

        content_map_values = {}
        for key, value in instance_data.items():
            content_list_to_serialize = []
            if isinstance(value, genai_types.Content):
                content_list_to_serialize = [value]
            elif isinstance(value, types.ResponseCandidate):
                if value.response:  # pytype: disable=attribute-error
                    content_list_to_serialize = [value.response]
            elif isinstance(value, list) and value:
                if isinstance(value[0], genai_types.Content):
                    content_list_to_serialize = value
                elif isinstance(value[0], types.evals.Message):
                    history_texts = []
                    for msg_obj in value:
                        msg_text = _extract_text_from_content(msg_obj.content)
                        if msg_text:
                            role = msg_obj.content.role or msg_obj.author or "user"
                            history_texts.append(f"{role}: {msg_text}")
                    content_list_to_serialize = [
                        genai_types.Content(
                            parts=[genai_types.Part(text="\n".join(history_texts))]
                        )
                    ]
                else:
                    content_list_to_serialize = [
                        genai_types.Content(
                            parts=[genai_types.Part(text=json.dumps(value))]
                        )
                    ]
            elif isinstance(value, dict):
                content_list_to_serialize = [
                    genai_types.Content(
                        parts=[genai_types.Part(text=json.dumps(value))]
                    )
                ]
            else:
                content_list_to_serialize = [
                    genai_types.Content(parts=[genai_types.Part(text=str(value))])
                ]

            content_map_values[key] = types.ContentMapContents(
                contents=content_list_to_serialize
            )

        instance_payload = types.PointwiseMetricInstance(
            content_map_instance=types.ContentMap(values=content_map_values)
        )

        metric_spec_payload = {"metric_prompt_template": self.metric.prompt_template}
        if self.metric.return_raw_output is not None:
            metric_spec_payload["custom_output_format_config"] = {  # type: ignore[assignment]
                "return_raw_output": self.metric.return_raw_output
            }
        if self.metric.judge_model_system_instruction:
            metric_spec_payload["system_instruction"] = (
                self.metric.judge_model_system_instruction
            )

        return {
            "pointwise_metric_input": {
                "metric_spec": metric_spec_payload,
                "instance": instance_payload.model_dump(mode="json", exclude_none=True),
            }
        }

    def _add_autorater_config(self, payload: dict[str, Any]) -> None:
        """Adds autorater config to the request payload if specified."""
        autorater_config = {}
        if self.metric.judge_model:
            autorater_config["autorater_model"] = self.metric.judge_model
        if self.metric.judge_model_sampling_count:
            autorater_config["sampling_count"] = self.metric.judge_model_sampling_count  # type: ignore[assignment]

        if not autorater_config:
            return

        if "rubric_based_metric_input" in payload:
            spec = payload["rubric_based_metric_input"]["metric_spec"]
            if "judge_autorater_config" not in spec:
                spec["judge_autorater_config"] = {}
            spec["judge_autorater_config"].update(autorater_config)
        else:
            payload["autorater_config"] = autorater_config

    def _build_request_payload(
        self, eval_case: types.EvalCase, response_index: int
    ) -> dict[str, Any]:
        """Builds the request parameters for evaluate instances request."""
        if not eval_case.responses or response_index >= len(eval_case.responses):
            raise IndexError(f"response_index {response_index} is out of bounds.")

        response_content = eval_case.responses[response_index].response
        if not response_content:
            raise ValueError(
                f"Response content missing for candidate {response_index}."
            )

        if self.metric.rubric_group_name:
            payload = self._build_rubric_based_input(eval_case, response_content)
        else:
            payload = self._build_pointwise_input(eval_case, response_content)

        self._add_autorater_config(payload)
        return payload

    @override
    def get_metric_result(
        self, eval_case: types.EvalCase, response_index: int
    ) -> types.EvalCaseMetricResult:
        """Processes a single evaluation case for a specific LLM metric."""
        metric_name = self.metric.name
        try:
            payload = self._build_request_payload(eval_case, response_index)
            response = self.module.evaluate_instances(metric_config=payload)

            if self.metric.rubric_group_name:
                result_data = response.rubric_based_metric_result
                return types.EvalCaseMetricResult(
                    metric_name=metric_name,
                    score=result_data.score if result_data else None,
                    rubric_verdicts=result_data.rubric_verdicts if result_data else [],
                )
            else:
                result_data = response.pointwise_metric_result  # type: ignore[assignment]
                return types.EvalCaseMetricResult(
                    metric_name=metric_name,
                    score=result_data.score if result_data else None,
                    explanation=result_data.explanation if result_data else None,  # type: ignore[attr-defined]
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error processing metric %s for case %s: %s",
                metric_name,
                eval_case.eval_case_id,
                e,
                exc_info=True,
            )
            return types.EvalCaseMetricResult(
                metric_name=metric_name, error_message=str(e)
            )

    @override
    def aggregate(
        self, eval_case_metric_results: list[types.EvalCaseMetricResult]
    ) -> types.AggregatedMetricResult:
        """Aggregates the metric results for a LLM metric."""
        if self.metric.aggregate_summary_fn and callable(
            self.metric.aggregate_summary_fn
        ):
            logger.info(
                "Using custom aggregate_summary_fn for metric '%s'", self.metric.name
            )
            try:
                custom_summary_dict = self.metric.aggregate_summary_fn(
                    eval_case_metric_results
                )
                if not isinstance(custom_summary_dict, dict):
                    raise TypeError("aggregate_summary_fn must return a dictionary.")

                num_cases_total = len(eval_case_metric_results)
                num_cases_error = len(
                    [
                        result
                        for result in eval_case_metric_results
                        if result.error_message is not None
                    ]
                )
                num_cases_valid = num_cases_total - num_cases_error
                required_fields = {
                    "num_cases_total": num_cases_total,
                    "num_cases_error": num_cases_error,
                    "num_cases_valid": num_cases_valid,
                }
                final_summary_dict = {**required_fields, **custom_summary_dict}

                return types.AggregatedMetricResult(
                    metric_name=self.metric.name,
                    **final_summary_dict,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "Error executing custom aggregate_summary_fn for metric '%s': %s."
                    " Falling back to default aggregation.",
                    self.metric.name,
                    e,
                    exc_info=True,
                )
                return _default_aggregate_scores(
                    self.metric.name, eval_case_metric_results
                )
        else:
            logger.debug(
                "Using default aggregation for LLM metric '%s'", self.metric.name
            )
            return _default_aggregate_scores(self.metric.name, eval_case_metric_results)


class CustomMetricHandler(MetricHandler):
    """Metric handler for custom metrics."""

    def __init__(self, module: "evals.Evals", metric: types.Metric):
        super().__init__(module=module, metric=metric)

        if not self.metric.custom_function:
            raise ValueError(
                f"CustomMetricHandler for '{self.metric.name}' needs "
                " Metric.custom_function to be set."
            )
        if not isinstance(self.metric.custom_function, Callable):
            raise ValueError(
                f"CustomMetricHandler for '{self.metric.name}' needs "
                " Metric.custom_function to be a callable function."
            )

    @override
    def get_metric_result(
        self, eval_case: types.EvalCase, response_index: int
    ) -> types.EvalCaseMetricResult:
        """Processes a single evaluation case for a custom metric."""
        metric_name = self.metric.name
        logger.debug(
            "CustomMetricHandler: Processing '%s' for case: %s",
            metric_name,
            eval_case.model_dump(exclude_none=True),
        )

        if response_index >= len(eval_case.responses):
            return types.EvalCaseMetricResult(
                metric_name=self.metric.name,
                error_message=(
                    f"response_index {response_index} out of bounds for EvalCase"
                    f" {eval_case.eval_case_id or 'Unknown ID'}."
                ),
            )

        if not eval_case.responses:
            raise ValueError(f"EvalCase {eval_case.eval_case_id} has no responses.")

        current_response_candidate = eval_case.responses[response_index]

        instance_for_custom_fn = eval_case.model_dump(
            exclude={"responses"}, mode="json", exclude_none=True
        )
        instance_for_custom_fn["response"] = current_response_candidate.model_dump(
            mode="json", exclude_none=True
        ).get("response")

        error_msg = None
        score = None
        explanation = None
        try:
            if self.metric.custom_function:
                custom_function_result = self.metric.custom_function(
                    instance_for_custom_fn
                )

                if isinstance(custom_function_result, types.EvalCaseMetricResult):
                    return custom_function_result
                elif (
                    isinstance(custom_function_result, dict)
                    and "score" in custom_function_result
                ):
                    score = custom_function_result["score"]
                    explanation = custom_function_result.get("explanation", None)
                elif isinstance(custom_function_result, (float, int)):
                    score = custom_function_result
                    explanation = None
                else:
                    error_msg = (
                        f"CustomFunctionError({self.metric.custom_function}): Returned"
                        f" unexpected type {type(custom_function_result)}"
                    )

        except Exception as e:  # pylint: disable=broad-exception-caught
            if self.metric.custom_function and hasattr(
                self.metric.custom_function, "__name__"
            ):
                custom_function_name = self.metric.custom_function.__name__
            else:
                custom_function_name = "unknown_custom_function"
            error_msg = f"CustomFunctionError({custom_function_name}): {e}"
            score = None
            explanation = None

        return types.EvalCaseMetricResult(
            metric_name=self.metric.name,
            score=score,
            explanation=explanation,
            error_message=error_msg,
        )

    @override
    def aggregate(
        self, eval_case_metric_results: list[types.EvalCaseMetricResult]
    ) -> types.AggregatedMetricResult:
        """Aggregates the metric results for a custom metric."""
        logger.debug("Aggregating results for custom metric: %s", self.metric.name)
        return _default_aggregate_scores(self.metric.name, eval_case_metric_results)


class PredefinedMetricHandler(MetricHandler):
    """Metric handler for predefined metrics."""

    def __init__(self, module: "evals.Evals", metric: types.Metric):
        super().__init__(module=module, metric=metric)
        if self.metric.name not in _evals_constant.SUPPORTED_PREDEFINED_METRICS:
            raise ValueError(
                f"Metric '{self.metric.name}' is not a supported predefined metric."
            )

    @staticmethod
    def _content_to_instance_data(
        content: Optional[genai_types.Content],
    ) -> Optional[types.evals.InstanceData]:
        """Converts a genai_types.Content object to a types.InstanceData object."""
        if not content:
            return None
        return types.evals.InstanceData(
            contents=types.evals.InstanceDataContents(contents=[content])
        )

    @staticmethod
    def _eval_case_to_agent_data(
        eval_case: types.EvalCase,
    ) -> Optional[types.evals.AgentData]:
        """Converts an EvalCase object to an AgentData object."""
        if not eval_case.agent_info and not eval_case.intermediate_events:
            return None
        tools = None
        developer_instruction = None
        agent_config = None
        tool_declarations = []
        event_contents = []

        if eval_case.agent_info:
            agent_info = eval_case.agent_info
            if agent_info.instruction:
                developer_instruction = types.evals.InstanceData(
                    text=agent_info.instruction
                )
            if agent_info.tool_declarations:
                tool_declarations = agent_info.tool_declarations
            tools = types.evals.Tools(tool=tool_declarations)

            if tools or developer_instruction:
                agent_config = types.evals.AgentConfig(
                    tools=tools,
                    developer_instruction=developer_instruction,
                )

        if eval_case.intermediate_events:
            event_contents = [
                event.content
                for event in eval_case.intermediate_events
                if event.content
            ]
        events = types.evals.Events(event=event_contents)

        return types.evals.AgentData(
            agent_config=agent_config,
            events=events,
        )

    def _build_request_payload(
        self, eval_case: types.EvalCase, response_index: int
    ) -> dict[str, Any]:
        """Builds the request parameters for evaluate instances request."""
        if not eval_case.responses or response_index >= len(eval_case.responses):
            raise IndexError(f"response_index {response_index} is out of bounds.")

        response_content = eval_case.responses[response_index].response
        if not response_content:
            raise ValueError(
                f"Response content missing for candidate {response_index}."
            )

        if self.metric.name == "tool_use_quality_v1":
            if not _has_tool_call(eval_case.intermediate_events):
                logger.warning(
                    "Metric 'tool_use_quality_v1' requires tool usage in "
                    "'intermediate_events', but no tool usage was found for case %s.",
                    eval_case.eval_case_id,
                )

        reference_instance_data = None
        if eval_case.reference:
            reference_instance_data = PredefinedMetricHandler._content_to_instance_data(
                eval_case.reference.response
            )

        prompt_instance_data = None
        if self.metric.name.startswith("multi_turn"):
            prompt_contents = []
            if eval_case.conversation_history:
                for message in eval_case.conversation_history:
                    prompt_contents.append(message.content)
            if eval_case.prompt:
                prompt_contents.append(eval_case.prompt)

            prompt_instance_data = types.evals.InstanceData(
                contents=types.evals.InstanceDataContents(contents=prompt_contents)
            )
        else:
            prompt_instance_data = PredefinedMetricHandler._content_to_instance_data(
                eval_case.prompt
            )

        other_data_map = {}
        if hasattr(eval_case, "context") and eval_case.context:
            if isinstance(eval_case.context, str):
                other_data_map["context"] = types.evals.InstanceData(
                    text=eval_case.context
                )
            elif isinstance(eval_case.context, genai_types.Content):
                other_data_map["context"] = (
                    PredefinedMetricHandler._content_to_instance_data(eval_case.context)
                )
            else:
                logger.warning(
                    f"Unsupported type for context: {type(eval_case.context)}"
                )
        instance_payload = types.EvaluationInstance(
            prompt=prompt_instance_data,
            response=PredefinedMetricHandler._content_to_instance_data(
                response_content
            ),
            reference=reference_instance_data,
            rubric_groups=eval_case.rubric_groups,
            other_data=(
                types.MapInstance(map_instance=other_data_map)
                if other_data_map
                else None
            ),
            agent_data=PredefinedMetricHandler._eval_case_to_agent_data(eval_case),
        )

        return {
            "instance": instance_payload,
        }

    @override
    def get_metric_result(
        self, eval_case: types.EvalCase, response_index: int
    ) -> types.EvalCaseMetricResult:
        """Processes a single evaluation case for a specific predefined metric."""
        metric_name = self.metric.name
        try:
            payload = self._build_request_payload(eval_case, response_index)
            for attempt in range(_MAX_RETRIES):
                try:
                    api_response = self.module._evaluate_instances(
                        metrics=[self.metric], instance=payload.get("instance")
                    )
                    break
                except genai_errors.ClientError as e:
                    if e.code == 429:
                        logger.warning(
                            "Resource Exhausted error on attempt %d/%d: %s. Retrying in %s"
                            " seconds...",
                            attempt + 1,
                            _MAX_RETRIES,
                            e,
                            2**attempt,
                        )
                        if attempt == _MAX_RETRIES - 1:
                            return types.EvalCaseMetricResult(
                                metric_name=metric_name,
                                error_message=f"Judge model resource exhausted after {_MAX_RETRIES} retries: {e}",
                            )
                        time.sleep(2**attempt)
                    else:
                        raise e

            if (
                api_response
                and hasattr(api_response, "metric_results")
                and api_response.metric_results
            ):
                result_data = api_response.metric_results[0]

                error_message = None
                if result_data.error and getattr(result_data.error, "code"):
                    error_message = f"Error in metric result: {result_data.error}"
                return types.EvalCaseMetricResult(
                    metric_name=metric_name,
                    score=result_data.score,
                    explanation=result_data.explanation,
                    rubric_verdicts=result_data.rubric_verdicts,
                    error_message=error_message,
                )
            else:
                logger.error(
                    "Metric results missing in API response for predefined metric '%s'."
                    " API response: %s",
                    metric_name,
                    (
                        api_response.model_dump_json(exclude_none=True)
                        if api_response
                        else "None"
                    ),
                )
                return types.EvalCaseMetricResult(
                    metric_name=metric_name,
                    error_message="Metric results missing in API response.",
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error processing metric %s for case %s: %s",
                metric_name,
                eval_case.eval_case_id,
                e,
                exc_info=True,
            )
            return types.EvalCaseMetricResult(
                metric_name=metric_name, error_message=str(e)
            )

    @override
    def aggregate(
        self, eval_case_metric_results: list[types.EvalCaseMetricResult]
    ) -> types.AggregatedMetricResult:
        """Aggregates the metric results for a predefined metric."""
        logger.debug("Aggregating results for predefined metric: %s", self.metric.name)
        return _default_aggregate_scores(self.metric.name, eval_case_metric_results)


_METRIC_HANDLER_MAPPING = [
    (
        lambda m: m.custom_function and isinstance(m.custom_function, Callable),
        CustomMetricHandler,
    ),
    (
        lambda m: m.name in ComputationMetricHandler.SUPPORTED_COMPUTATION_METRICS,
        ComputationMetricHandler,
    ),
    (
        lambda m: m.name in TranslationMetricHandler.SUPPORTED_TRANSLATION_METRICS,
        TranslationMetricHandler,
    ),
    (
        lambda m: m.name in _evals_constant.SUPPORTED_PREDEFINED_METRICS,
        PredefinedMetricHandler,
    ),
    (lambda m: isinstance(m, types.LLMMetric), LLMMetricHandler),
]

MetricHandlerType = TypeVar(
    "MetricHandlerType",
    ComputationMetricHandler,
    TranslationMetricHandler,
    LLMMetricHandler,
    CustomMetricHandler,
    PredefinedMetricHandler,
)


def get_handler_for_metric(
    module: "evals.Evals", metric: types.Metric
) -> Union[MetricHandlerType, Any]:
    """Returns a metric handler for the given metric."""
    for condition, handler_class in _METRIC_HANDLER_MAPPING:
        if condition(metric):  # type: ignore[no-untyped-call]
            return handler_class(module=module, metric=metric)
    raise ValueError(f"Unsupported metric: {metric.name}")


def calculate_win_rates(eval_result: types.EvaluationResult) -> dict[str, Any]:
    """Calculates win/tie rates for comparison results."""
    if not eval_result.eval_case_results:
        return {}
    max_models = max(
        (
            len(case.response_candidate_results)
            for case in eval_result.eval_case_results
            if case.response_candidate_results
        ),
        default=0,
    )
    if max_models == 0:
        return {}
    stats: collections.defaultdict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"wins": [0] * max_models, "ties": 0, "valid_comparisons": 0}
    )
    for case in eval_result.eval_case_results:
        if not case.response_candidate_results:
            continue
        scores_by_metric = collections.defaultdict(list)
        for idx, candidate in enumerate(case.response_candidate_results):
            for name, res in (
                candidate.metric_results.items() if candidate.metric_results else {}
            ):
                if res.score is not None:
                    scores_by_metric[name].append({"score": res.score, "cand_idx": idx})
        for name, scores in scores_by_metric.items():
            if not scores:
                continue
            stats[name]["valid_comparisons"] += 1
            max_score = max(s["score"] for s in scores)
            winners = [s["cand_idx"] for s in scores if s["score"] == max_score]
            if len(winners) == 1:
                stats[name]["wins"][winners[0]] += 1
            else:
                stats[name]["ties"] += 1
    win_rates = {}
    for name, metric_stats in stats.items():
        if metric_stats["valid_comparisons"] > 0:
            win_rates[name] = {
                "win_rates": [
                    w / metric_stats["valid_comparisons"] for w in metric_stats["wins"]
                ],
                "tie_rate": metric_stats["ties"] / metric_stats["valid_comparisons"],
            }
    return win_rates


def _aggregate_metric_results(
    metric_handlers: list[MetricHandler],
    eval_case_results: list[types.EvalCaseResult],
) -> list[types.AggregatedMetricResult]:
    """Aggregates results by calling the aggregate method of each handler."""
    aggregated_metric_results = []
    logger.info("Aggregating results per metric...")
    for handler in metric_handlers:
        metric_name = handler.metric.name
        results_for_this_metric: list[types.EvalCaseMetricResult] = []
        for case_result in eval_case_results:
            if case_result.response_candidate_results:
                for response_candidate_res in case_result.response_candidate_results:
                    if (
                        response_candidate_res.metric_results
                        and metric_name in response_candidate_res.metric_results
                        and isinstance(metric_name, str)
                    ):
                        results_for_this_metric.append(
                            response_candidate_res.metric_results[metric_name]
                        )
        if not results_for_this_metric:
            logger.warning(
                "No results found for metric '%s' to aggregate.", metric_name
            )
            continue

        try:
            summary = handler.aggregate(results_for_this_metric)
            aggregated_metric_results.append(summary)
        except NotImplementedError:
            logger.warning(
                "Aggregation not implemented for metric handler: %s (metric: '%s')."
                " Skipping summary.",
                type(handler).__name__,
                metric_name,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error during aggregation for metric '%s' using handler %s: %s",
                metric_name,
                type(handler).__name__,
                e,
                exc_info=True,
            )
            aggregated_metric_results.append(
                types.AggregatedMetricResult(
                    metric_name=metric_name,
                    num_cases_total=len(results_for_this_metric),
                    num_cases_valid=0,
                    num_cases_error=len(results_for_this_metric),
                    mean_score=None,
                    stdev_score=None,
                )
            )
    logger.debug("Finished aggregation, returning: %s", aggregated_metric_results)
    return aggregated_metric_results


class EvaluationRunConfig(_common.BaseModel):
    """Configuration for an evaluation run."""

    evals_module: Any
    """The module to be used for the evaluation run."""
    dataset: types.EvaluationDataset
    """The dataset to be used for the evaluation run."""
    metrics: list[types.Metric]
    """The list of metrics to be used for the evaluation run."""
    num_response_candidates: int
    """The number of response candidates for the evaluation run."""


def compute_metrics_and_aggregate(
    evaluation_run_config: EvaluationRunConfig,
) -> types.EvaluationResult:
    """Computes metrics and aggregates them for a given evaluation run config."""
    metric_handlers = []
    all_futures = []
    results_by_case_response_metric: collections.defaultdict[
        Any, collections.defaultdict[Any, dict[Any, Any]]
    ] = collections.defaultdict(lambda: collections.defaultdict(dict))
    submission_errors = []
    execution_errors = []
    case_indices_with_errors = set()

    for eval_metric in evaluation_run_config.metrics:
        metric_handlers.append(
            get_handler_for_metric(evaluation_run_config.evals_module, eval_metric)
        )

    eval_case_count = len(evaluation_run_config.dataset.eval_cases)
    logger.info("Total number of evaluation cases: %d", eval_case_count)
    logger.info(
        "Number of response candidates: %d",
        evaluation_run_config.num_response_candidates,
    )
    total_metric_computations = (
        eval_case_count
        * len(metric_handlers)
        * evaluation_run_config.num_response_candidates
    )
    logger.info("Total number of metric computations: %d", total_metric_computations)

    with tqdm(
        total=total_metric_computations,
        desc="Computing Metrics for Evaluation Dataset",
    ) as pbar:
        with futures.ThreadPoolExecutor(
            max_workers=_evals_common.MAX_WORKERS
        ) as executor:
            for metric_handler_instance in metric_handlers:
                for eval_case_index, eval_case in enumerate(
                    evaluation_run_config.dataset.eval_cases
                ):
                    actual_num_candidates_for_case = min(
                        evaluation_run_config.num_response_candidates,
                        len(eval_case.responses),
                    )
                    for response_index in range(actual_num_candidates_for_case):
                        try:
                            future = executor.submit(
                                metric_handler_instance.get_metric_result,
                                eval_case,
                                response_index,
                            )
                            future.add_done_callback(lambda _: pbar.update(1))
                            logger.debug(
                                "Submitting metric computation for case %d, "
                                "response %d for metric %s.",
                                eval_case_index,
                                response_index,
                                metric_handler_instance.metric.name,
                            )
                            all_futures.append(
                                (
                                    future,
                                    metric_handler_instance.metric.name,
                                    eval_case_index,
                                    response_index,
                                )
                            )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            logger.error(
                                "Error submitting metric computation for case %d, "
                                "response %d for metric %s: %s",
                                eval_case_index,
                                response_index,
                                metric_handler_instance.metric.name,
                                e,
                                exc_info=True,
                            )
                            submission_errors.append(
                                (
                                    metric_handler_instance.metric.name,
                                    eval_case_index,
                                    response_index,
                                    f"Error: {e}",
                                )
                            )
                            error_result = types.EvalCaseMetricResult(
                                metric_name=metric_handler_instance.metric.name,
                                error_message=f"Submission Error: {e}",
                            )
                            results_by_case_response_metric[eval_case_index][
                                response_index
                            ][metric_handler_instance.metric.name] = error_result
                            case_indices_with_errors.add(eval_case_index)
                            pbar.update(1)

        for future, metric_name, eval_case_index, response_index in all_futures:
            try:
                eval_case_metric_result = future.result()
                logger.debug(
                    "Successfully obtained result for metric '%s', case %d, response"
                    " %d: %s.",
                    metric_name,
                    eval_case_index,
                    response_index,
                    eval_case_metric_result,
                )
                results_by_case_response_metric[eval_case_index][response_index][
                    metric_name
                ] = eval_case_metric_result
                logger.debug(
                    "Stored result for metric '%s', case %d, response %d.",
                    metric_name,
                    eval_case_index,
                    response_index,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "Error executing metric '%s' for case %s, response %s: %s",
                    metric_name,
                    eval_case_index,
                    response_index,
                    e,
                    exc_info=True,
                )
                error_msg = (
                    f"Error executing metric '{metric_name}' for case"
                    f" {eval_case_index}, response {response_index}: {e}"
                )
                execution_errors.append(
                    (
                        metric_name,
                        eval_case_index,
                        response_index,
                        error_msg,
                    )
                )
                case_indices_with_errors.add(eval_case_index)
                error_result = types.EvalCaseMetricResult(
                    metric_name=metric_name,
                    error_message=error_msg,
                )
                results_by_case_response_metric[eval_case_index][response_index][
                    metric_name
                ] = error_result

    final_eval_case_results = []
    sorted_eval_case_indices = sorted(results_by_case_response_metric.keys())
    for eval_case_index in sorted_eval_case_indices:
        per_response_results_for_this_case = results_by_case_response_metric[
            eval_case_index
        ]

        current_response_candidate_results_list = []
        sorted_response_indices = sorted(per_response_results_for_this_case.keys())

        for response_index in sorted_response_indices:
            metric_results_for_this_response = per_response_results_for_this_case[
                response_index
            ]

            response_candidate_result_obj = types.ResponseCandidateResult(
                response_index=response_index,
                metric_results=metric_results_for_this_response,
            )
            current_response_candidate_results_list.append(
                response_candidate_result_obj
            )

        if current_response_candidate_results_list:
            eval_case_result = types.EvalCaseResult(
                eval_case_index=eval_case_index,
                response_candidate_results=current_response_candidate_results_list,
            )
            final_eval_case_results.append(eval_case_result)
        elif eval_case_index in case_indices_with_errors or any(
            err_case_idx == eval_case_index
            for _, err_case_idx, _, _ in submission_errors
        ):
            logger.warning(
                "EvalCase %d had errors but no metric results were"
                " processed into the structure.",
                eval_case_index,
            )
            eval_case_result = types.EvalCaseResult(
                eval_case_index=eval_case_index,
                response_candidate_results=[],
            )
            final_eval_case_results.append(eval_case_result)

    if submission_errors:
        logger.warning("Encountered %d submission errors.", len(submission_errors))
        logger.warning("Submission errors: %s", submission_errors)
    if execution_errors:
        logger.warning("Encountered %d execution errors.", len(execution_errors))
        logger.warning("Execution errors: %s", execution_errors)

    aggregated_metric_results = _aggregate_metric_results(
        metric_handlers, final_eval_case_results
    )
    eval_result = types.EvaluationResult(
        eval_case_results=final_eval_case_results,
        summary_metrics=aggregated_metric_results,
    )
    if evaluation_run_config.num_response_candidates > 1:
        try:
            eval_result.win_rates = calculate_win_rates(eval_result)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error calculating win rates: %s", e, exc_info=True)
    return eval_result
