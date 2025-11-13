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

from vertexai.evaluation.metrics import _base
from vertexai.evaluation.metrics import _rouge
from vertexai.evaluation.metrics import (
    metric_prompt_template,
)
from vertexai.evaluation.metrics import (
    metric_prompt_template_examples,
)
from vertexai.evaluation.metrics import pairwise_metric
from vertexai.evaluation.metrics import pointwise_metric


PairwiseMetric = pairwise_metric.PairwiseMetric
PointwiseMetric = pointwise_metric.PointwiseMetric
CustomMetric = _base.CustomMetric
PairwiseMetricPromptTemplate = metric_prompt_template.PairwiseMetricPromptTemplate
PointwiseMetricPromptTemplate = metric_prompt_template.PointwiseMetricPromptTemplate
MetricPromptTemplateExamples = (
    metric_prompt_template_examples.MetricPromptTemplateExamples
)
Rouge = _rouge.Rouge


__all__ = [
    "CustomMetric",
    "PairwiseMetric",
    "PointwiseMetric",
    "PairwiseMetricPromptTemplate",
    "PointwiseMetricPromptTemplate",
    "MetricPromptTemplateExamples",
    "Rouge",
]
