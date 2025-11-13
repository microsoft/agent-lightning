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
"""Base classes for evaluation metrics."""

import abc
from typing import Any, Callable, Dict, Literal, Optional, Union, List

from google.cloud.aiplatform_v1beta1.types import (
    evaluation_service as gapic_eval_service_types,
)
from vertexai import generative_models
from vertexai.preview.evaluation import constants
from vertexai.preview.evaluation.metrics import (
    custom_output_config as custom_output_config_class,
)
from vertexai.preview.evaluation.metrics import (
    metric_prompt_template as metric_prompt_template_base,
)


_ModelType = Union[generative_models.GenerativeModel, Callable[[str], str]]


class _Metric(abc.ABC):
    """The abstract class for evaluation metric."""

    def __init__(self, metric: str):
        self._metric = metric

    def __str__(self):
        return self.metric_name

    @property
    def metric_name(self) -> str:
        return self._metric


class _ModelBasedMetric(_Metric):
    """A Model-based Metric.

    An evaluation metric that evaluates generative AI model responses with
    another generative model as a judge. This metric can be used to evaluate a
    single model, or two models side-by-side.

    For more details on when to use model-based metrics, see
    [Evaluation methods and metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval).
    """

    def __init__(
        self,
        *,
        metric: str,
        metric_prompt_template: Union[
            metric_prompt_template_base.PointwiseMetricPromptTemplate,
            metric_prompt_template_base.PairwiseMetricPromptTemplate,
            str,
        ],
        system_instruction: Optional[str] = None,
        autorater_config: Optional[gapic_eval_service_types.AutoraterConfig] = None,
        custom_output_config: Optional[
            custom_output_config_class.CustomOutputConfig
        ] = None,
    ):
        """Initializes the model-based evaluation metric.

        Args:
          metric: Generic model based metric name.
          metric_prompt_template: A metric prompt template for performing
            the model-based evaluation. A freeform string is also accepted.
          system_instruction: The system instruction to be used in the metric
            prompt.
          autorater_config: The config for judge model.
          custom_output_config: Config for custom output from the judge model.
        """
        super().__init__(metric=metric)
        self.metric_prompt_template = str(metric_prompt_template)
        self.system_instruction = system_instruction
        self.autorater_config = autorater_config
        self.custom_output_config = custom_output_config


class CustomMetric(_Metric):
    """The custom evaluation metric.

    A fully-customized CustomMetric that can be used to evaluate a single model
    by defining a metric function for a computation-based metric. The
    CustomMetric is computed on the client-side using the user-defined metric
    function in SDK only, not by the Vertex Gen AI Evaluation Service.

      Attributes:
        name: The name of the metric.
        metric_function: The user-defined evaluation function to compute a metric
          score. Must use the dataset row dictionary as the metric function
          input and return per-instance metric result as a dictionary output.
          The metric score must mapped to the name of the CustomMetric as key.
    """

    def __init__(
        self,
        name: str,
        metric_function: Callable[
            [Dict[str, Any]],
            Dict[str, Any],
        ],
    ):
        """Initializes the evaluation metric."""
        super().__init__(name)
        self.name = name
        self.metric_function = metric_function


class _AutomaticMetric(_Metric):
    """An automatic metric that computes deterministic score based on reference.

    An lexicon-based evaluation metric that evaluate a generative model's
    response on the given evaluation task with reference ground truth answers.
    It is a type of pointwise evaluation metric.

    For more details on when to use automatic metrics, see
    [Evaluation methods and
    metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval).
    """

    def __init__(
        self,
        metric: Literal[constants.Metric.ROUGE],
    ):
        """Initializes the automatic evaluation metric.

        Args:
          metric: The automatic evaluation metric name.
        """
        super().__init__(metric=metric)


class RubricGenerationConfig:
    """The rubric generation config."""

    def __init__(
        self,
        prompt_template: str,
        model: Optional[_ModelType] = None,
        parsing_fn: Optional[Callable[[str], List[str]]] = None,
    ):
        """Initializes the rubric generation config.

        Args:
          prompt_template: The prompt template for rubric generation.
          model: The model to use for rubric generation.
          parsing_fn: The function to parse the rubric generation response.
        """
        self.prompt_template = prompt_template
        self.model = model
        self.parsing_fn = parsing_fn
