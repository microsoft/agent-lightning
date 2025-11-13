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
"""Library for metrics computation with Gen AI Evaluation Service."""

import json
from typing import Any, Dict, List, Union

from google import api_core
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform_v1beta1.services import (
    evaluation_service as gapic_evaluation_services,
)
from google.cloud.aiplatform_v1beta1.types import (
    evaluation_service as gapic_eval_service_types,
)
from vertexai.preview.evaluation import _base as eval_base
from vertexai.preview.evaluation import constants
from vertexai.preview.evaluation import multimodal_utils
from vertexai.preview.evaluation import (
    prompt_template as prompt_template_base,
)
from vertexai.preview.evaluation import utils
from vertexai.preview.evaluation.metrics import (
    _base as metrics_base,
)
from vertexai.preview.evaluation.metrics import (
    _default_templates,
)
from vertexai.preview.evaluation.metrics import _rouge
from vertexai.preview.evaluation.metrics import (
    _trajectory_single_tool_use,
)
from vertexai.preview.evaluation.metrics import (
    custom_output_config as custom_output_config_class,
)
from vertexai.preview.evaluation.metrics import pairwise_metric
from vertexai.preview.evaluation.metrics import pointwise_metric
from google.protobuf import json_format


_LOGGER = base.Logger(__name__)
_METRIC_NAME_TO_METRIC_SPEC = {
    # Automatic Metrics.
    constants.Metric.EXACT_MATCH: (gapic_eval_service_types.ExactMatchSpec()),
    constants.Metric.BLEU: gapic_eval_service_types.BleuSpec(),
    constants.Metric.ROUGE: gapic_eval_service_types.RougeSpec(),
    constants.Metric.ROUGE_1: gapic_eval_service_types.RougeSpec(rouge_type="rouge1"),
    constants.Metric.ROUGE_2: gapic_eval_service_types.RougeSpec(rouge_type="rouge2"),
    constants.Metric.ROUGE_L: gapic_eval_service_types.RougeSpec(rouge_type="rougeL"),
    constants.Metric.ROUGE_L_SUM: gapic_eval_service_types.RougeSpec(
        rouge_type="rougeLsum"
    ),
    constants.Metric.TOOL_CALL_VALID: (gapic_eval_service_types.ToolCallValidSpec()),
    constants.Metric.TOOL_NAME_MATCH: (gapic_eval_service_types.ToolNameMatchSpec()),
    constants.Metric.TOOL_PARAMETER_KV_MATCH: (
        gapic_eval_service_types.ToolParameterKVMatchSpec()
    ),
    constants.Metric.TOOL_PARAMETER_KEY_MATCH: (
        gapic_eval_service_types.ToolParameterKeyMatchSpec()
    ),
    # Pointwise Metrics.
    constants.Metric.POINTWISE_METRIC: (gapic_eval_service_types.PointwiseMetricSpec()),
    # Pairwise Metrics.
    constants.Metric.PAIRWISE_METRIC: (gapic_eval_service_types.PairwiseMetricSpec()),
    constants.Metric.RUBRIC_BASED_INSTRUCTION_FOLLOWING: (
        gapic_eval_service_types.RubricBasedInstructionFollowingSpec()
    ),
    constants.Metric.TRAJECTORY_EXACT_MATCH: (
        gapic_eval_service_types.TrajectoryExactMatchSpec()
    ),
    constants.Metric.TRAJECTORY_IN_ORDER_MATCH: (
        gapic_eval_service_types.TrajectoryInOrderMatchSpec()
    ),
    constants.Metric.TRAJECTORY_ANY_ORDER_MATCH: (
        gapic_eval_service_types.TrajectoryAnyOrderMatchSpec()
    ),
    constants.Metric.TRAJECTORY_PRECISION: (
        gapic_eval_service_types.TrajectoryPrecisionSpec()
    ),
    constants.Metric.TRAJECTORY_RECALL: (
        gapic_eval_service_types.TrajectoryRecallSpec()
    ),
    constants.Metric.TRAJECTORY_SINGLE_TOOL_USE: (
        gapic_eval_service_types.TrajectorySingleToolUseSpec()
    ),
}
_QUESTION_TEMPLATE = """<question>{question}"""


def _format_rubrics(questions: List[str]) -> str:
    """Formats the list of rubrics into a question block."""
    question_block = "\n".join(
        _QUESTION_TEMPLATE.format(question=q.strip()) for q in questions
    )
    return question_block


def build_custom_output_format_config(
    custom_output_config: custom_output_config_class.CustomOutputConfig,
) -> Union[gapic_eval_service_types.CustomOutputFormatConfig, None]:
    """Builds a CustomOutputFormatConfig from user input."""
    custom_output_cfg = gapic_eval_service_types.CustomOutputFormatConfig()
    if custom_output_config.return_raw_output:
        custom_output_cfg.return_raw_output = True
        return custom_output_cfg
    else:
        return None


def build_trajectory(
    trajectory: Union[str, List[Dict[str, Any]]],
) -> gapic_eval_service_types.Trajectory:
    """Builds a trajectory from user input."""
    if not trajectory:
        return

    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)

    if isinstance(trajectory, List):
        try:
            tool_calls = []
            for tool_call_dict in trajectory:
                tool_input_str = json.dumps(tool_call_dict["tool_input"])
                tool_calls.append(
                    gapic_eval_service_types.ToolCall(
                        tool_name=tool_call_dict["tool_name"], tool_input=tool_input_str
                    )
                )
            return gapic_eval_service_types.Trajectory(tool_calls=tool_calls)
        except KeyError as e:
            _LOGGER.error(f"Failed to parse trajectory: {e}")
    else:
        _LOGGER.error(
            f"Unsupported trajectory type: {type(trajectory)}, expected list or"
            " a JSON array."
        )


def build_request(
    metric: Union[str, metrics_base._Metric],
    row_dict: Dict[str, Any],
    evaluation_run_config: eval_base.EvaluationRunConfig,
) -> gapic_eval_service_types.EvaluateInstancesRequest:
    """Builds a metric instance and form the request for the evaluation service.

    Args:
        metric: The name of the metric to evaluate.
        row_dict: An evaluation dataset instance as a dictionary.
        evaluation_run_config: Evaluation run configurations.

    Returns:
        A single EvaluateInstancesRequest.

    Raises:
        ValueError: If required request fields are not provided.
    """
    project = initializer.global_config.project
    location = initializer.global_config.location
    if not project or not location:
        raise ValueError(
            "No project or location specified. Please run `vertexai.init()` to"
            " provide these parameters."
        )
    location_path = (
        gapic_evaluation_services.EvaluationServiceClient.common_location_path(
            project, location
        )
    )

    if isinstance(metric, pointwise_metric.PointwiseMetric):
        metric_name = constants.Metric.POINTWISE_METRIC
    elif isinstance(metric, pairwise_metric.PairwiseMetric):
        metric_name = constants.Metric.PAIRWISE_METRIC
    else:
        metric_name = str(metric)

    try:
        metric_spec = _METRIC_NAME_TO_METRIC_SPEC[metric_name]
    except KeyError as e:
        raise ValueError(f"Metric name: {metric_name} is not supported.") from e

    model_based_metric_instance_input = {}
    metric_column_mapping = evaluation_run_config.metric_column_mapping
    if isinstance(
        metric, metrics_base._ModelBasedMetric  # pylint: disable=protected-access
    ):
        metric_spec.metric_prompt_template = metric.metric_prompt_template
        metric_spec.system_instruction = metric.system_instruction
        if metric.custom_output_config:
            metric_spec.custom_output_format_config = build_custom_output_format_config(
                metric.custom_output_config
            )
        for variable in prompt_template_base.PromptTemplate(
            metric.metric_prompt_template
        ).variables:
            model_based_metric_instance_input[variable] = row_dict.get(
                metric_column_mapping.get(variable),
                "",
            )
        if isinstance(metric, pairwise_metric.PairwiseMetric):
            metric_column_mapping = evaluation_run_config.metric_column_mapping
            metric_spec.candidate_response_field_name = metric_column_mapping.get(
                constants.Dataset.MODEL_RESPONSE_COLUMN,
                constants.Dataset.MODEL_RESPONSE_COLUMN,
            )
            metric_spec.baseline_response_field_name = metric_column_mapping.get(
                constants.Dataset.BASELINE_MODEL_RESPONSE_COLUMN,
                constants.Dataset.BASELINE_MODEL_RESPONSE_COLUMN,
            )
    elif isinstance(metric, _rouge.Rouge):
        metric_spec.rouge_type = metric.rouge_type
        metric_spec.use_stemmer = metric.use_stemmer
        metric_spec.split_summaries = metric.split_summaries
    elif isinstance(metric, _trajectory_single_tool_use.TrajectorySingleToolUse):
        metric_spec.tool_name = metric.tool_name

    response = row_dict.get(
        metric_column_mapping.get(constants.Dataset.MODEL_RESPONSE_COLUMN), ""
    )
    reference = row_dict.get(
        metric_column_mapping.get(constants.Dataset.REFERENCE_COLUMN), ""
    )
    predicted_trajectory = build_trajectory(
        row_dict.get(
            metric_column_mapping.get(constants.Dataset.PREDICTED_TRAJECTORY_COLUMN),
            "",
        )
    )
    reference_trajectory = build_trajectory(
        row_dict.get(
            metric_column_mapping.get(constants.Dataset.REFERENCE_TRAJECTORY_COLUMN),
            "",
        )
    )
    if isinstance(metric, metrics_base._ModelBasedMetric):
        if metric_spec.metric_prompt_template in (
            _default_templates.INSTRUCTION_FOLLOWING_RUBRIC_CRITIQUE_TEMPLATE,
            _default_templates.MULTIMODAL_UNDERSTANDING_RUBRIC_CRITIQUE_TEMPLATE,
            _default_templates.TEXT_QUALITY_RUBRIC_CRITIQUE_TEMPLATE,
            _default_templates.PAIRWISE_INSTRUCTION_FOLLOWING_RUBRIC_CRITIQUE_TEMPLATE,
            _default_templates.PAIRWISE_MULTIMODAL_UNDERSTANDING_RUBRIC_CRITIQUE_TEMPLATE,
            _default_templates.PAIRWISE_TEXT_QUALITY_RUBRIC_CRITIQUE_TEMPLATE,
        ):
            model_based_metric_instance_input[constants.Dataset.RUBRICS_COLUMN] = (
                _format_rubrics(
                    model_based_metric_instance_input[constants.Dataset.RUBRICS_COLUMN]
                )
            )
        if (
            constants.Dataset.RUBRICS_COLUMN in model_based_metric_instance_input
            and isinstance(
                model_based_metric_instance_input[constants.Dataset.RUBRICS_COLUMN],
                List,
            )
        ):
            model_based_metric_instance_input[constants.Dataset.RUBRICS_COLUMN] = (
                "\n".join(
                    model_based_metric_instance_input[constants.Dataset.RUBRICS_COLUMN]
                )
            )

    if metric_name == constants.Metric.EXACT_MATCH:
        instance = gapic_eval_service_types.ExactMatchInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.ExactMatchInstance(
                    prediction=response,
                    reference=reference,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            exact_match_input=instance,
        )
    elif metric_name == constants.Metric.BLEU:
        instance = gapic_eval_service_types.BleuInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.BleuInstance(
                    prediction=response,
                    reference=reference,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            bleu_input=instance,
        )
    elif metric_name in (
        constants.Metric.ROUGE,
        constants.Metric.ROUGE_1,
        constants.Metric.ROUGE_2,
        constants.Metric.ROUGE_L,
        constants.Metric.ROUGE_L_SUM,
    ):
        instance = gapic_eval_service_types.RougeInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.RougeInstance(
                    prediction=response,
                    reference=reference,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            rouge_input=instance,
        )
    elif metric_name == constants.Metric.TOOL_CALL_VALID:
        instance = gapic_eval_service_types.ToolCallValidInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.ToolCallValidInstance(
                    prediction=response,
                    reference=reference,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            tool_call_valid_input=instance,
        )
    elif metric_name == constants.Metric.TOOL_NAME_MATCH:
        instance = gapic_eval_service_types.ToolNameMatchInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.ToolNameMatchInstance(
                    prediction=response,
                    reference=reference,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            tool_name_match_input=instance,
        )
    elif metric_name == constants.Metric.TOOL_PARAMETER_KEY_MATCH:
        instance = gapic_eval_service_types.ToolParameterKeyMatchInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.ToolParameterKeyMatchInstance(
                    prediction=response,
                    reference=reference,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            tool_parameter_key_match_input=instance,
        )
    elif metric_name == constants.Metric.TOOL_PARAMETER_KV_MATCH:
        instance = gapic_eval_service_types.ToolParameterKVMatchInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.ToolParameterKVMatchInstance(
                    prediction=response,
                    reference=reference,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            tool_parameter_kv_match_input=instance,
        )
    elif metric_name == constants.Metric.POINTWISE_METRIC:
        if multimodal_utils.is_multimodal_instance(model_based_metric_instance_input):
            instance = gapic_eval_service_types.PointwiseMetricInput(
                metric_spec=metric_spec,
                instance=gapic_eval_service_types.PointwiseMetricInstance(
                    content_map_instance=multimodal_utils.convert_multimodal_response_to_content_map(
                        model_based_metric_instance_input
                    ),
                ),
            )
        else:
            instance = gapic_eval_service_types.PointwiseMetricInput(
                metric_spec=metric_spec,
                instance=gapic_eval_service_types.PointwiseMetricInstance(
                    json_instance=json.dumps(model_based_metric_instance_input),
                ),
            )
        autorater_config = evaluation_run_config.autorater_config
        if (
            isinstance(metric, metrics_base._ModelBasedMetric)
            and metric.autorater_config
        ):
            autorater_config = metric.autorater_config
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            pointwise_metric_input=instance,
            autorater_config=autorater_config,
        )
    elif metric_name == constants.Metric.PAIRWISE_METRIC:
        if multimodal_utils.is_multimodal_instance(model_based_metric_instance_input):
            instance = gapic_eval_service_types.PairwiseMetricInput(
                metric_spec=metric_spec,
                instance=gapic_eval_service_types.PairwiseMetricInstance(
                    content_map_instance=multimodal_utils.convert_multimodal_response_to_content_map(
                        model_based_metric_instance_input
                    ),
                ),
            )
        else:
            instance = gapic_eval_service_types.PairwiseMetricInput(
                metric_spec=metric_spec,
                instance=gapic_eval_service_types.PairwiseMetricInstance(
                    json_instance=json.dumps(model_based_metric_instance_input),
                ),
            )
        autorater_config = evaluation_run_config.autorater_config
        if (
            isinstance(metric, metrics_base._ModelBasedMetric)
            and metric.autorater_config
        ):
            autorater_config = metric.autorater_config
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            pairwise_metric_input=instance,
            autorater_config=autorater_config,
        )
    elif metric_name == constants.Metric.RUBRIC_BASED_INSTRUCTION_FOLLOWING:
        required_rbif_fields = [
            constants.Dataset.MODEL_RESPONSE_COLUMN,
            constants.Dataset.PROMPT_COLUMN,
        ]
        for field in required_rbif_fields:
            column_name = metric_column_mapping.get(field)
            value = row_dict.get(column_name)
            if value is None and field in required_rbif_fields:
                raise ValueError(
                    f"Missing required field: `{field}` for "
                    f"{constants.Metric.RUBRIC_BASED_INSTRUCTION_FOLLOWING}."
                )
            else:
                model_based_metric_instance_input[field] = value
        instance = gapic_eval_service_types.RubricBasedInstructionFollowingInput(
            metric_spec=metric_spec,
            instance=gapic_eval_service_types.RubricBasedInstructionFollowingInstance(
                json_instance=json.dumps(model_based_metric_instance_input),
            ),
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            rubric_based_instruction_following_input=instance,
        )
    elif metric_name == constants.Metric.TRAJECTORY_EXACT_MATCH:
        instance = gapic_eval_service_types.TrajectoryExactMatchInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.TrajectoryExactMatchInstance(
                    predicted_trajectory=predicted_trajectory,
                    reference_trajectory=reference_trajectory,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            trajectory_exact_match_input=instance,
        )
    elif metric_name == constants.Metric.TRAJECTORY_IN_ORDER_MATCH:
        instance = gapic_eval_service_types.TrajectoryInOrderMatchInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.TrajectoryInOrderMatchInstance(
                    predicted_trajectory=predicted_trajectory,
                    reference_trajectory=reference_trajectory,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            trajectory_in_order_match_input=instance,
        )
    elif metric_name == constants.Metric.TRAJECTORY_ANY_ORDER_MATCH:
        instance = gapic_eval_service_types.TrajectoryAnyOrderMatchInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.TrajectoryAnyOrderMatchInstance(
                    predicted_trajectory=predicted_trajectory,
                    reference_trajectory=reference_trajectory,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            trajectory_any_order_match_input=instance,
        )
    elif metric_name == constants.Metric.TRAJECTORY_PRECISION:
        instance = gapic_eval_service_types.TrajectoryPrecisionInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.TrajectoryPrecisionInstance(
                    predicted_trajectory=predicted_trajectory,
                    reference_trajectory=reference_trajectory,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            trajectory_precision_input=instance,
        )
    elif metric_name == constants.Metric.TRAJECTORY_RECALL:
        instance = gapic_eval_service_types.TrajectoryRecallInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.TrajectoryRecallInstance(
                    predicted_trajectory=predicted_trajectory,
                    reference_trajectory=reference_trajectory,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            trajectory_recall_input=instance,
        )
    elif metric_name == constants.Metric.TRAJECTORY_SINGLE_TOOL_USE:
        instance = gapic_eval_service_types.TrajectorySingleToolUseInput(
            metric_spec=metric_spec,
            instances=[
                gapic_eval_service_types.TrajectorySingleToolUseInstance(
                    predicted_trajectory=predicted_trajectory,
                )
            ],
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            trajectory_single_tool_use_input=instance,
        )
    else:
        raise ValueError(f"Unknown metric type: {metric_name}")


def _parse_autometric_results(
    metric_result_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Parses the automatic metric results from the evaluation results.

    Args:
        metric_result_dict: The metric results dictionary.

    Returns:
        A dictionary containing metric score of the metric.
    """
    for value in metric_result_dict.values():
        return {
            constants.MetricResult.SCORE_KEY: value[0].get(
                constants.MetricResult.SCORE_KEY
            )
        }


def _parse_pointwise_results(
    metric_result_dict: Dict[str, Any],
    metric: Union[str, metrics_base._Metric],
) -> Dict[str, Any]:
    """Parses the model-based pointwise metric results from the evaluation results.

    Args:
        metric_result_dict: The metric results dictionary.
        metric: The metric to evaluate.

    Returns:
        One of the following:
        1. A dictionary containing raw outputs from the judge model if
        return_raw_output is set to True in custom_output_config.
        2. A dictionary containing metric score and explanation of the
        metric if custom_output_config is not set.
    """
    if (
        isinstance(metric, pointwise_metric.PointwiseMetric)
        and getattr(metric, "custom_output_config", None)
        and getattr(metric.custom_output_config, "return_raw_output", False)
    ):
        raw_outputs = (
            metric_result_dict.get(constants.MetricResult.CUSTOM_OUTPUT_KEY)
            .get(constants.MetricResult.RAW_OUTPUTS_KEY)
            .get(constants.MetricResult.RAW_OUTPUT_KEY)
        )
        if (
            isinstance(metric, pointwise_metric.PointwiseMetric)
            and getattr(metric, "custom_output_config", None)
            and getattr(metric.custom_output_config, "parsing_fn", None)
        ):
            parsing_fn = metric.custom_output_config.parsing_fn
            return parsing_fn(raw_outputs)
        return {constants.MetricResult.RAW_OUTPUT_KEY: raw_outputs}
    else:
        return {
            constants.MetricResult.SCORE_KEY: metric_result_dict.get(
                constants.MetricResult.SCORE_KEY
            ),
            constants.MetricResult.EXPLANATION_KEY: metric_result_dict.get(
                constants.MetricResult.EXPLANATION_KEY
            ),
        }


def _parse_pairwise_results(
    metric_result_dict: Dict[str, Any],
    metric: Union[str, metrics_base._Metric],
) -> Dict[str, Any]:
    """Parses the pairwise metric results from the evaluation results.

    Args:
        metric_result_dict: The metric results dictionary.
        metric: The metric to evaluate.

    Returns:
        One of the following:
        1. A dictionary containing raw outputs from the judge model if
        return_raw_output is set to True in custom_output_config.
        2. A dictionary containing metric score and explanation of the
        metric if custom_output_config is not set.
    """
    if (
        isinstance(metric, pairwise_metric.PairwiseMetric)
        and getattr(metric, "custom_output_config", None)
        and getattr(metric.custom_output_config, "return_raw_output", False)
    ):
        raw_outputs = (
            metric_result_dict.get(constants.MetricResult.CUSTOM_OUTPUT_KEY)
            .get(constants.MetricResult.RAW_OUTPUTS_KEY)
            .get(constants.MetricResult.RAW_OUTPUT_KEY)
        )
        if (
            isinstance(metric, pairwise_metric.PairwiseMetric)
            and getattr(metric, "custom_output_config", None)
            and getattr(metric.custom_output_config, "parsing_fn", None)
        ):
            parsing_fn = metric.custom_output_config.parsing_fn
            return parsing_fn(raw_outputs)
        return {constants.MetricResult.RAW_OUTPUT_KEY: raw_outputs}
    else:
        return {
            constants.MetricResult.PAIRWISE_CHOICE_KEY: metric_result_dict.get(
                constants.MetricResult.PAIRWISE_CHOICE_KEY,
            ),
            constants.MetricResult.EXPLANATION_KEY: metric_result_dict.get(
                constants.MetricResult.EXPLANATION_KEY
            ),
        }


def _parse_rubric_based_instruction_following_results(
    metric_result_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Parses the rubric-based instruction following metric results from the evaluation results.

    Args:
        metric_result_dict: The metric results dictionary.

    Returns:
        A dictionary containing a list of rubrics and corresponding verdicts and
        an overall instruction following score.
    """
    rubric_critique_results = []
    for rc_result in metric_result_dict["rubric_critique_results"]:
        if "verdict" not in rc_result:
            rc_result["verdict"] = False  # proto3 shows False bool as unset
        rubric_critique_results.append(
            {
                "rubric": rc_result["rubric"],
                "verdict": rc_result["verdict"],
            }
        )
    return {
        constants.MetricResult.RUBRIC_LEVEL_INSTRUCTION_FOLLOWING_KEY: (
            rubric_critique_results
        ),
        constants.MetricResult.SCORE_KEY: (
            metric_result_dict.get(constants.MetricResult.SCORE_KEY)
        ),
    }


def handle_response(
    response: Union[str, gapic_eval_service_types.EvaluateInstancesResponse],
    metric: Union[str, metrics_base._Metric],
) -> Union[str, Dict[str, Any]]:
    """Handles the response from the evaluation service.

    Args:
        response: The response from the evaluation service.
        metric: The metric to evaluate to check the output type.

    Returns:
        A parsed metric result dictionary, or an error message string.
    """
    if isinstance(response, str):
        return response

    metric_type = response._pb.WhichOneof(  # pylint: disable=protected-access
        "evaluation_results"
    )

    if metric_type == constants.MetricResult.EXACT_MATCH_RESULTS:
        metric_result = response.exact_match_results
    elif metric_type == constants.MetricResult.BLEU_RESULTS:
        metric_result = response.bleu_results
    elif metric_type == constants.MetricResult.ROUGE_RESULTS:
        metric_result = response.rouge_results
    elif metric_type == constants.MetricResult.TOOL_CALL_VALID_RESULTS:
        metric_result = response.tool_call_valid_results
    elif metric_type == constants.MetricResult.TOOL_NAME_MATCH_RESULTS:
        metric_result = response.tool_name_match_results
    elif metric_type == constants.MetricResult.TOOL_PARAMETER_KEY_MATCH_RESULTS:
        metric_result = response.tool_parameter_key_match_results
    elif metric_type == constants.MetricResult.TOOL_PARAMETER_KV_MATCH_RESULTS:
        metric_result = response.tool_parameter_kv_match_results
    elif metric_type == constants.MetricResult.POINTWISE_METRIC_RESULT:
        metric_result = response.pointwise_metric_result
    elif metric_type == constants.MetricResult.PAIRWISE_METRIC_RESULT:
        metric_result = response.pairwise_metric_result
    elif metric_type == constants.MetricResult.TRAJECTORY_EXACT_MATCH_RESULTS:
        metric_result = response.trajectory_exact_match_results
    elif metric_type == constants.MetricResult.TRAJECTORY_IN_ORDER_MATCH_RESULTS:
        metric_result = response.trajectory_in_order_match_results
    elif metric_type == constants.MetricResult.TRAJECTORY_ANY_ORDER_MATCH_RESULTS:
        metric_result = response.trajectory_any_order_match_results
    elif metric_type == constants.MetricResult.TRAJECTORY_PRECISION_RESULTS:
        metric_result = response.trajectory_precision_results
    elif metric_type == constants.MetricResult.TRAJECTORY_RECALL_RESULTS:
        metric_result = response.trajectory_recall_results
    elif metric_type == constants.MetricResult.TRAJECTORY_SINGLE_TOOL_USE_RESULTS:
        metric_result = response.trajectory_single_tool_use_results
    elif (
        metric_type == constants.MetricResult.RUBRIC_BASED_INSTRUCTION_FOLLOWING_RESULT
    ):
        metric_result = response.rubric_based_instruction_following_result
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    metric_result_dict = json_format.MessageToDict(
        metric_result._pb,  # pylint: disable=protected-access
        preserving_proto_field_name=True,
    )
    if metric_type in (constants.MetricResult.AUTOMATIC_METRIC_RESULTS_LIST):
        result = _parse_autometric_results(metric_result_dict)
    elif metric_type == constants.MetricResult.POINTWISE_METRIC_RESULT:
        result = _parse_pointwise_results(metric_result_dict, metric)
    elif metric_type == constants.MetricResult.PAIRWISE_METRIC_RESULT:
        result = _parse_pairwise_results(metric_result_dict, metric)
    elif (
        metric_type == constants.MetricResult.RUBRIC_BASED_INSTRUCTION_FOLLOWING_RESULT
    ):
        result = _parse_rubric_based_instruction_following_results(metric_result_dict)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
    return result


def evaluate_instances(
    client: gapic_evaluation_services.EvaluationServiceClient,
    request: gapic_eval_service_types.EvaluateInstancesRequest,
    rate_limiter: utils.RateLimiter,
    retry_timeout: float,
) -> gapic_eval_service_types.EvaluateInstancesResponse:
    """Evaluates an instance using Vertex Gen AI Evaluation Service.

    Args:
        client: The Vertex Gen AI evaluation service client for evaluation.
        request: An EvaluateInstancesRequest.
        rate_limiter: The rate limiter for evaluation service requests.
        retry_timeout: How long to keep retrying the evaluation requests, in seconds.

    Returns:
        An EvaluateInstancesResponse from Vertex Gen AI Evaluation Service.
    """
    rate_limiter.sleep_and_advance()
    return client.evaluate_instances(
        request=request,
        retry=api_core.retry.Retry(
            initial=0.250,
            maximum=90.0,
            multiplier=1.45,
            timeout=retry_timeout,
            predicate=api_core.retry.if_exception_type(
                api_core.exceptions.Aborted,
                api_core.exceptions.DeadlineExceeded,
                api_core.exceptions.ResourceExhausted,
                api_core.exceptions.ServiceUnavailable,
                api_core.exceptions.Cancelled,
            ),
        ),
    )
