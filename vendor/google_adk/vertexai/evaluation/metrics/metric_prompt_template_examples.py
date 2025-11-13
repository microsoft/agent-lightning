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
"""Example metric prompt templates for model-based evaluation."""

from typing import List

from google.cloud.aiplatform.utils import _ipython_utils
from vertexai.evaluation import constants
from vertexai.evaluation.metrics import (
    _default_templates,
)
from vertexai.evaluation.metrics import pairwise_metric
from vertexai.evaluation.metrics import pointwise_metric


class MetricPromptTemplateExamples:
    """Examples of metric prompt templates for model-based evaluation."""

    _PROMPT_TEMPLATE_MAP = {
        constants.Metric.COHERENCE: _default_templates.COHERENCE_PROMPT_TEMPLATE,
        constants.Metric.FLUENCY: _default_templates.FLUENCY_PROMPT_TEMPLATE,
        constants.Metric.SAFETY: _default_templates.SAFETY_PROMPT_TEMPLATE,
        constants.Metric.GROUNDEDNESS: (
            _default_templates.GROUNDEDNESS_PROMPT_TEMPLATE
        ),
        constants.Metric.INSTRUCTION_FOLLOWING: (
            _default_templates.INSTRUCTION_FOLLOWING_PROMPT_TEMPLATE
        ),
        constants.Metric.VERBOSITY: _default_templates.VERBOSITY_PROMPT_TEMPLATE,
        constants.Metric.TEXT_QUALITY: (
            _default_templates.TEXT_QUALITY_PROMPT_TEMPLATE
        ),
        constants.Metric.SUMMARIZATION_QUALITY: (
            _default_templates.SUMMARIZATION_QUALITY_PROMPT_TEMPLATE
        ),
        constants.Metric.QUESTION_ANSWERING_QUALITY: (
            _default_templates.QUESTION_ANSWERING_QUALITY_PROMPT_TEMPLATE
        ),
        constants.Metric.MULTI_TURN_CHAT_QUALITY: (
            _default_templates.MULTI_TURN_CHAT_QUALITY_PROMPT_TEMPLATE
        ),
        constants.Metric.MULTI_TURN_SAFETY: (
            _default_templates.MULTI_TURN_SAFETY_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_COHERENCE: (
            _default_templates.PAIRWISE_COHERENCE_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_FLUENCY: (
            _default_templates.PAIRWISE_FLUENCY_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_SAFETY: (
            _default_templates.PAIRWISE_SAFETY_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_GROUNDEDNESS: (
            _default_templates.PAIRWISE_GROUNDEDNESS_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_INSTRUCTION_FOLLOWING: (
            _default_templates.PAIRWISE_INSTRUCTION_FOLLOWING_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_VERBOSITY: (
            _default_templates.PAIRWISE_VERBOSITY_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_TEXT_QUALITY: (
            _default_templates.PAIRWISE_TEXT_QUALITY_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_SUMMARIZATION_QUALITY: (
            _default_templates.PAIRWISE_SUMMARIZATION_QUALITY_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_QUESTION_ANSWERING_QUALITY: (
            _default_templates.PAIRWISE_QUESTION_ANSWERING_QUALITY_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_MULTI_TURN_CHAT_QUALITY: (
            _default_templates.PAIRWISE_MULTI_TURN_CHAT_QUALITY_PROMPT_TEMPLATE
        ),
        constants.Metric.PAIRWISE_MULTI_TURN_SAFETY: (
            _default_templates.PAIRWISE_MULTI_TURN_SAFETY_PROMPT_TEMPLATE
        ),
    }

    @classmethod
    def get_prompt_template(cls, metric_name: str) -> str:
        """Returns the prompt template for the given metric name."""
        return cls._PROMPT_TEMPLATE_MAP[metric_name]

    @classmethod
    def list_example_metric_names(cls) -> List[str]:
        """Returns a list of all metric prompt templates."""
        _ipython_utils.display_browse_prebuilt_metrics_button()
        return list(cls._PROMPT_TEMPLATE_MAP.keys())

    class Pointwise:
        """Example PointwiseMetric instances."""

        FLUENCY = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.FLUENCY,
            metric_prompt_template=_default_templates.FLUENCY_PROMPT_TEMPLATE,
        )
        COHERENCE = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.COHERENCE,
            metric_prompt_template=_default_templates.COHERENCE_PROMPT_TEMPLATE,
        )
        SAFETY = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.SAFETY,
            metric_prompt_template=_default_templates.SAFETY_PROMPT_TEMPLATE,
        )
        GROUNDEDNESS = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.GROUNDEDNESS,
            metric_prompt_template=_default_templates.GROUNDEDNESS_PROMPT_TEMPLATE,
        )
        INSTRUCTION_FOLLOWING = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.INSTRUCTION_FOLLOWING,
            metric_prompt_template=_default_templates.INSTRUCTION_FOLLOWING_PROMPT_TEMPLATE,
        )
        VERBOSITY = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.VERBOSITY,
            metric_prompt_template=_default_templates.VERBOSITY_PROMPT_TEMPLATE,
        )
        TEXT_QUALITY = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.TEXT_QUALITY,
            metric_prompt_template=_default_templates.TEXT_QUALITY_PROMPT_TEMPLATE,
        )
        SUMMARIZATION_QUALITY = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.SUMMARIZATION_QUALITY,
            metric_prompt_template=_default_templates.SUMMARIZATION_QUALITY_PROMPT_TEMPLATE,
        )
        QUESTION_ANSWERING_QUALITY = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.QUESTION_ANSWERING_QUALITY,
            metric_prompt_template=_default_templates.QUESTION_ANSWERING_QUALITY_PROMPT_TEMPLATE,
        )
        MULTI_TURN_CHAT_QUALITY = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.MULTI_TURN_CHAT_QUALITY,
            metric_prompt_template=_default_templates.MULTI_TURN_CHAT_QUALITY_PROMPT_TEMPLATE,
        )
        MULTI_TURN_SAFETY = pointwise_metric.PointwiseMetric(
            metric=constants.Metric.MULTI_TURN_SAFETY,
            metric_prompt_template=_default_templates.MULTI_TURN_SAFETY_PROMPT_TEMPLATE,
        )

    class Pairwise:
        """Example PairwiseMetric instances."""

        FLUENCY = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_FLUENCY,
            metric_prompt_template=_default_templates.PAIRWISE_FLUENCY_PROMPT_TEMPLATE,
        )
        COHERENCE = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_COHERENCE,
            metric_prompt_template=_default_templates.PAIRWISE_COHERENCE_PROMPT_TEMPLATE,
        )
        SAFETY = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_SAFETY,
            metric_prompt_template=_default_templates.PAIRWISE_SAFETY_PROMPT_TEMPLATE,
        )
        GROUNDEDNESS = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_GROUNDEDNESS,
            metric_prompt_template=_default_templates.PAIRWISE_GROUNDEDNESS_PROMPT_TEMPLATE,
        )
        INSTRUCTION_FOLLOWING = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_INSTRUCTION_FOLLOWING,
            metric_prompt_template=_default_templates.PAIRWISE_INSTRUCTION_FOLLOWING_PROMPT_TEMPLATE,
        )
        VERBOSITY = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_VERBOSITY,
            metric_prompt_template=_default_templates.PAIRWISE_VERBOSITY_PROMPT_TEMPLATE,
        )
        TEXT_QUALITY = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_TEXT_QUALITY,
            metric_prompt_template=_default_templates.PAIRWISE_TEXT_QUALITY_PROMPT_TEMPLATE,
        )
        SUMMARIZATION_QUALITY = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_SUMMARIZATION_QUALITY,
            metric_prompt_template=_default_templates.PAIRWISE_SUMMARIZATION_QUALITY_PROMPT_TEMPLATE,
        )
        QUESTION_ANSWERING_QUALITY = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_QUESTION_ANSWERING_QUALITY,
            metric_prompt_template=_default_templates.PAIRWISE_QUESTION_ANSWERING_QUALITY_PROMPT_TEMPLATE,
        )
        MULTI_TURN_CHAT_QUALITY = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_MULTI_TURN_CHAT_QUALITY,
            metric_prompt_template=_default_templates.PAIRWISE_MULTI_TURN_CHAT_QUALITY_PROMPT_TEMPLATE,
        )
        MULTI_TURN_SAFETY = pairwise_metric.PairwiseMetric(
            metric=constants.Metric.PAIRWISE_MULTI_TURN_SAFETY,
            metric_prompt_template=_default_templates.PAIRWISE_MULTI_TURN_SAFETY_PROMPT_TEMPLATE,
        )
