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
from typing import Any, Dict, Union

from google import api_core
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform_v1.services import (
    evaluation_service as gapic_evaluation_services,
)
from google.cloud.aiplatform_v1.types import (
    evaluation_service as gapic_eval_service_types,
)
from vertexai.evaluation import _base as eval_base
from vertexai.evaluation import constants
from vertexai.evaluation import (
    prompt_template as prompt_template_base,
)
from vertexai.evaluation import utils
from vertexai.evaluation.metrics import (
    _base as metrics_base,
)
from vertexai.evaluation.metrics import _rouge
from vertexai.evaluation.metrics import pairwise_metric
from vertexai.evaluation.metrics import pointwise_metric
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
    # Model-based Translation Metrics.
    constants.Metric.COMET: gapic_eval_service_types.CometSpec(),
    constants.Metric.METRICX: gapic_eval_service_types.MetricxSpec(),
}


def build_request(
    metric: Union[str, metrics_base._Metric],
    row_dict: Dict[str, Any],
    evaluation_run_config: eval_base.EvaluationRunConfig,
) -> gapic_eval_service_types.EvaluateInstancesRequest:
    """Builds an EvaluateInstancesRequest for Vertex Gen AI Evaluation Service.

    Args:
        metric: The metric to be evaluated.
        row_dict: An evaluation dataset instance as a dictionary.
        evaluation_run_config: Evaluation run configurations.

    Returns:
        An EvaluateInstancesRequest for Vertex Gen AI Evaluation Service.
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
        metric, metrics_base._ModelBasedMetric
    ):  # pylint: disable=protected-access
        metric_spec.metric_prompt_template = metric.metric_prompt_template
        for variable in prompt_template_base.PromptTemplate(
            metric.metric_prompt_template
        ).variables:
            model_based_metric_instance_input[variable] = row_dict.get(
                metric_column_mapping.get(variable),
                "",
            )
    elif isinstance(metric, _rouge.Rouge):
        metric_spec.rouge_type = metric.rouge_type
        metric_spec.use_stemmer = metric.use_stemmer
        metric_spec.split_summaries = metric.split_summaries
    elif isinstance(
        metric, metrics_base._TranslationMetric  # pylint: disable=protected-access
    ):
        metric_spec.version = metric.version
        metric_spec.source_language = metric.source_language
        metric_spec.target_language = metric.target_language

    response = row_dict.get(
        metric_column_mapping.get(constants.Dataset.MODEL_RESPONSE_COLUMN), ""
    )
    reference = row_dict.get(
        metric_column_mapping.get(constants.Dataset.REFERENCE_COLUMN), ""
    )
    source = row_dict.get(
        metric_column_mapping.get(constants.Dataset.SOURCE_COLUMN), ""
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
        instance = gapic_eval_service_types.PointwiseMetricInput(
            metric_spec=metric_spec,
            instance=gapic_eval_service_types.PointwiseMetricInstance(
                json_instance=json.dumps(model_based_metric_instance_input),
            ),
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            pointwise_metric_input=instance,
        )
    elif metric_name == constants.Metric.PAIRWISE_METRIC:
        instance = gapic_eval_service_types.PairwiseMetricInput(
            metric_spec=metric_spec,
            instance=gapic_eval_service_types.PairwiseMetricInstance(
                json_instance=json.dumps(model_based_metric_instance_input),
            ),
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path, pairwise_metric_input=instance
        )
    elif metric_name == constants.Metric.COMET:
        instance = gapic_eval_service_types.CometInput(
            metric_spec=metric_spec,
            instance=gapic_eval_service_types.CometInstance(
                prediction=response,
                reference=reference,
                source=source,
            ),
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            comet_input=instance,
        )
    elif metric_name == constants.Metric.METRICX:
        instance = gapic_eval_service_types.MetricxInput(
            metric_spec=metric_spec,
            instance=gapic_eval_service_types.MetricxInstance(
                prediction=response,
                reference=reference,
                source=source,
            ),
        )
        return gapic_eval_service_types.EvaluateInstancesRequest(
            location=location_path,
            metricx_input=instance,
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
) -> Dict[str, Any]:
    """Parses the model-based pointwise metric result.

    Args:
        metric_result_dict: The metric result dictionary.

    Returns:
        A dictionary containing metric score, explanation of the pointwise
        metric result.
    """
    return {
        constants.MetricResult.SCORE_KEY: metric_result_dict.get(
            constants.MetricResult.SCORE_KEY
        ),
        constants.MetricResult.EXPLANATION_KEY: metric_result_dict.get(
            constants.MetricResult.EXPLANATION_KEY
        ),
    }


def _parse_model_based_translation_results(
    metric_result_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Parses the model-based pointwise translation metric result.

    Args:
        metric_result_dict: The metric result dictionary.

    Returns:
        A dictionary containing metric score.
    """
    return {
        constants.MetricResult.SCORE_KEY: metric_result_dict.get(
            constants.MetricResult.SCORE_KEY
        ),
    }


def _parse_pairwise_results(
    metric_result_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Parses the pairwise metric result.

    Args:
        metric_result_dict: The metric result dictionary.

    Returns:
        A dictionary containing metric score, explanation of the pairwise metric
        result.
    """
    return {
        constants.MetricResult.PAIRWISE_CHOICE_KEY: metric_result_dict.get(
            constants.MetricResult.PAIRWISE_CHOICE_KEY,
        ),
        constants.MetricResult.EXPLANATION_KEY: metric_result_dict.get(
            constants.MetricResult.EXPLANATION_KEY
        ),
    }


def handle_response(
    response: Union[str, gapic_eval_service_types.EvaluateInstancesResponse],
) -> Union[str, Dict[str, Any]]:
    """Handles the response from the evaluation service.

    Args:
        response: The response from the evaluation service.

    Returns:
        A parsed metric result dictionary, or an error message string.
    """
    if isinstance(response, str):
        return response

    metric_type = response._pb.WhichOneof("evaluation_results")

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
    elif metric_type == constants.MetricResult.COMET_RESULT:
        metric_result = response.comet_result
    elif metric_type == constants.MetricResult.METRICX_RESULT:
        metric_result = response.metricx_result
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    metric_result_dict = json_format.MessageToDict(
        metric_result._pb, preserving_proto_field_name=True
    )
    if metric_type in constants.MetricResult.AUTOMATIC_METRIC_RESULTS_LIST:
        result = _parse_autometric_results(metric_result_dict)
    elif metric_type == constants.MetricResult.POINTWISE_METRIC_RESULT:
        result = _parse_pointwise_results(metric_result_dict)
    elif metric_type == constants.MetricResult.PAIRWISE_METRIC_RESULT:
        result = _parse_pairwise_results(metric_result_dict)
    elif metric_type in (
        constants.MetricResult.COMET_RESULT,
        constants.MetricResult.METRICX_RESULT,
    ):
        result = _parse_model_based_translation_results(metric_result_dict)
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
