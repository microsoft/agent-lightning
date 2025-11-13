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
"""Model-based Pointwise Metric."""

from typing import Union

from vertexai.evaluation import constants
from vertexai.evaluation.metrics import _base
from vertexai.evaluation.metrics import (
    metric_prompt_template as metric_prompt_template_base,
)


class PointwiseMetric(_base._ModelBasedMetric):  # pylint: disable=protected-access
    """A Model-based Pointwise Metric.

    A model-based evaluation metric that evaluate a single generative model's
    response.

    For more details on when to use model-based pointwise metrics, see
    [Evaluation methods and metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval).

    Usage Examples:

        ```
        candidate_model = GenerativeModel("gemini-1.5-pro")
        eval_dataset = pd.DataFrame({
            "prompt"  : [...],
        })
        fluency_metric = PointwiseMetric(
            metric="fluency",
            metric_prompt_template=MetricPromptTemplateExamples.get_prompt_template('fluency'),
        )
        pointwise_eval_task = EvalTask(
            dataset=eval_dataset,
            metrics=[
                fluency_metric,
                MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
            ],
        )
        pointwise_result = pointwise_eval_task.evaluate(
            model=candidate_model,
        )
        ```
    """

    def __init__(
        self,
        *,
        metric: str,
        metric_prompt_template: Union[
            metric_prompt_template_base.PointwiseMetricPromptTemplate, str
        ],
    ):
        """Initializes a pointwise evaluation metric.

        Args:
          metric: The pointwise evaluation metric name.
          metric_prompt_template: Pointwise metric prompt template for performing
            the model-based evaluation. A freeform string is also accepted.
        """
        super().__init__(
            metric_prompt_template=metric_prompt_template,
            metric=metric,
        )


class Comet(_base._TranslationMetric):  # pylint: disable=protected-access
    """A COMET metric.

    Evaluates a score for the given instance using
    https://huggingface.co/Unbabel/wmt22-comet-da
    """

    _metric_name = constants.Metric.COMET

    def __init__(
        self,
        *,
        version: str = "COMET_22_SRC_REF",
        source_language: str = None,
        target_language: str = None,
    ):
        """Initializes the COMET metric.

        Args:
          version: The COMET version to use for evaluation eg.
            "COMET_22_SRC_REF".
          source_language: Optional. The source language of the translation.
          target_language: Optional. The target language of the translation.
        """

        super().__init__(
            name=Comet._metric_name,
            version=version,
            source_language=source_language,
            target_language=target_language,
        )


class MetricX(_base._TranslationMetric):  # pylint: disable=protected-access
    """A MetricX metric.

    Evaluates a score for the given instance using
    https://github.com/google-research/metricx
    """

    _metric_name = constants.Metric.METRICX

    def __init__(
        self,
        *,
        version: str = "METRICX_24_SRC_REF",
        source_language: str = None,
        target_language: str = None,
    ):
        """Initializes the MetricX metric.

        Args:
          version: The MetricX version to use for evaluation. Can be one of
            "METRICX_24_SRC_REF", "METRICX_24_SRC", or "METRICX_24_REF".
          source_language: Optional. The source language of the translation.
          target_language: Optional. The target language of the translation.
        """

        super().__init__(
            name=MetricX._metric_name,
            version=version,
            source_language=source_language,
            target_language=target_language,
        )
