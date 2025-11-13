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

"""Transformers module for Vertex addons."""
from typing import Any  # type: ignore[attr-defined]

from google.genai._common import get_value_by_path as getv

from . import _evals_constant
from . import types


def t_metrics(
    metrics: list["types.MetricSubclass"],
    set_default_aggregation_metrics: bool = False,
) -> list[dict[str, Any]]:
    """Prepares the metric payload for the evaluation request.

    Args:
        metrics: A list of metrics used for evaluation.
        set_default_aggregation_metrics: Whether to set default aggregation metrics.
    Returns:
        A list of resolved metric payloads for the evaluation request.
    """
    metrics_payload = []

    for metric in metrics:
        metric_payload_item: dict[str, Any] = {}

        metric_name = getv(metric, ["name"]).lower()

        if set_default_aggregation_metrics:
            metric_payload_item["aggregation_metrics"] = [
                "AVERAGE",
                "STANDARD_DEVIATION",
            ]

        if metric_name == "exact_match":
            metric_payload_item["exact_match_spec"] = {}
        elif metric_name == "bleu":
            metric_payload_item["bleu_spec"] = {}
        elif metric_name.startswith("rouge"):
            rouge_type = metric_name.replace("_", "")
            metric_payload_item["rouge_spec"] = {"rouge_type": rouge_type}
        # API Pre-defined metrics
        elif metric_name in _evals_constant.SUPPORTED_PREDEFINED_METRICS:
            metric_payload_item["predefined_metric_spec"] = {
                "metric_spec_name": metric_name,
                "metric_spec_parameters": metric.metric_spec_parameters,
            }
        # Pointwise metrics
        elif hasattr(metric, "prompt_template") and metric.prompt_template:
            pointwise_spec = {"metric_prompt_template": metric.prompt_template}
            system_instruction = getv(metric, ["judge_model_system_instruction"])
            if system_instruction:
                pointwise_spec["system_instruction"] = system_instruction
            return_raw_output = getv(metric, ["return_raw_output"])
            if return_raw_output:
                pointwise_spec["custom_output_format_config"] = {  # type: ignore[assignment]
                    "return_raw_output": return_raw_output
                }
            metric_payload_item["pointwise_metric_spec"] = pointwise_spec
        else:
            raise ValueError(
                f"Unsupported metric type or invalid metric name: {metric_name}"
            )
        metrics_payload.append(metric_payload_item)
    return metrics_payload
