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
"""Vertex Gen AI Evaluation Service Module."""

from vertexai.preview.evaluation import _base
from vertexai.preview.evaluation import autorater_utils
from vertexai.preview.evaluation import eval_task
from vertexai.preview.evaluation import metrics
from vertexai.preview.evaluation import prompt_template


EvalResult = _base.EvalResult
EvalTask = eval_task.EvalTask
PairwiseMetric = metrics.PairwiseMetric
PointwiseMetric = metrics.PointwiseMetric
CustomMetric = metrics.CustomMetric
PromptTemplate = prompt_template.PromptTemplate
PairwiseMetricPromptTemplate = metrics.PairwiseMetricPromptTemplate
PointwiseMetricPromptTemplate = metrics.PointwiseMetricPromptTemplate
MetricPromptTemplateExamples = metrics.MetricPromptTemplateExamples
AutoraterConfig = autorater_utils.AutoraterConfig
CustomOutputConfig = metrics.CustomOutputConfig
RubricBasedMetric = metrics.RubricBasedMetric
RubricGenerationConfig = metrics.RubricGenerationConfig
PredefinedRubricMetrics = metrics.PredefinedRubricMetrics

__all__ = [
    "EvalTask",
    "EvalResult",
    "PairwiseMetric",
    "PointwiseMetric",
    "CustomMetric",
    "PromptTemplate",
    "PairwiseMetricPromptTemplate",
    "PointwiseMetricPromptTemplate",
    "MetricPromptTemplateExamples",
    "AutoraterConfig",
    "CustomOutputConfig",
    "RubricBasedMetric",
    "RubricGenerationConfig",
    "PredefinedRubricMetrics",
]
