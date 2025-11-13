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
"""Evaluation Metrics Module."""

from vertexai.preview.evaluation.metrics import _base
from vertexai.preview.evaluation.metrics import _rouge
from vertexai.preview.evaluation.metrics import (
    _trajectory_single_tool_use,
)
from vertexai.preview.evaluation.metrics import (
    custom_output_config,
)
from vertexai.preview.evaluation.metrics import (
    metric_prompt_template,
)
from vertexai.preview.evaluation.metrics import (
    metric_prompt_template_examples,
)
from vertexai.preview.evaluation.metrics import pairwise_metric
from vertexai.preview.evaluation.metrics import pointwise_metric
from vertexai.preview.evaluation.metrics import (
    predefined_rubric_metrics,
)
from vertexai.preview.evaluation.metrics import (
    rubric_based_metric,
)


PairwiseMetric = pairwise_metric.PairwiseMetric
PointwiseMetric = pointwise_metric.PointwiseMetric
CustomMetric = _base.CustomMetric
PairwiseMetricPromptTemplate = metric_prompt_template.PairwiseMetricPromptTemplate
PointwiseMetricPromptTemplate = metric_prompt_template.PointwiseMetricPromptTemplate
MetricPromptTemplateExamples = (
    metric_prompt_template_examples.MetricPromptTemplateExamples
)
Rouge = _rouge.Rouge
TrajectorySingleToolUse = _trajectory_single_tool_use.TrajectorySingleToolUse
CustomOutputConfig = custom_output_config.CustomOutputConfig
RubricBasedMetric = rubric_based_metric.RubricBasedMetric
RubricGenerationConfig = _base.RubricGenerationConfig
PredefinedRubricMetrics = predefined_rubric_metrics.PredefinedRubricMetrics


__all__ = [
    "CustomMetric",
    "PairwiseMetric",
    "PointwiseMetric",
    "PairwiseMetricPromptTemplate",
    "PointwiseMetricPromptTemplate",
    "MetricPromptTemplateExamples",
    "Rouge",
    "TrajectorySingleToolUse",
    "CustomOutputConfig",
    "RubricBasedMetric",
    "RubricGenerationConfig",
    "PredefinedRubricMetrics",
]
