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
"""Model-based Pairwise Metric."""

from typing import Callable, Optional, Union

from google.cloud.aiplatform_v1beta1.types import (
    evaluation_service as gapic_eval_service_types,
)
from vertexai.preview import generative_models
from vertexai.preview.evaluation.metrics import _base
from vertexai.preview.evaluation.metrics import (
    custom_output_config as custom_output_config_class,
)
from vertexai.preview.evaluation.metrics import (
    metric_prompt_template as metric_prompt_template_base,
)


class PairwiseMetric(_base._ModelBasedMetric):  # pylint: disable=protected-access
    """A Model-based Pairwise Metric.

    A model-based evaluation metric that compares two generative models' responses
    side-by-side, and allows users to A/B test their generative models to
    determine which model is performing better.

    For more details on when to use pairwise metrics, see
    [Evaluation methods and
    metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval#pointwise_versus_pairwise).

    Result Details:

        * In `EvalResult.summary_metrics`, win rates for both the baseline and
        candidate model are computed. The win rate is computed as proportion of
        wins of one model's responses to total attempts as a decimal value
        between 0 and 1.

        * In `EvalResult.metrics_table`, a pairwise metric produces two
        evaluation results per dataset row:
            * `pairwise_choice`: The choice shows whether the candidate model or
              the baseline model performs better, or if they are equally good.
            * `explanation`: The rationale behind each verdict using
              chain-of-thought reasoning. The explanation helps users scrutinize
              the judgment and builds appropriate trust in the decisions.

        See [documentation
        page](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval#understand-results)
        for more details on understanding the metric results.

    Usage Examples:

        ```
        baseline_model = GenerativeModel("gemini-1.0-pro")
        candidate_model = GenerativeModel("gemini-1.5-pro")

        pairwise_groundedness = PairwiseMetric(
            metric_prompt_template=MetricPromptTemplateExamples.get_prompt_template(
                "pairwise_groundedness"
            ),
            baseline_model=baseline_model,
        )
        eval_dataset = pd.DataFrame({
              "prompt"  : [...],
        })
        pairwise_task = EvalTask(
            dataset=eval_dataset,
            metrics=[pairwise_groundedness],
            experiment="my-pairwise-experiment",
        )
        pairwise_result = pairwise_task.evaluate(
            model=candidate_model,
            experiment_run_name="gemini-pairwise-eval-run",
        )
        ```
    """

    def __init__(
        self,
        *,
        metric: str,
        metric_prompt_template: Union[
            metric_prompt_template_base.PairwiseMetricPromptTemplate, str
        ],
        baseline_model: Optional[
            Union[generative_models.GenerativeModel, Callable[[str], str]]
        ] = None,
        system_instruction: Optional[str] = None,
        autorater_config: Optional[gapic_eval_service_types.AutoraterConfig] = None,
        custom_output_config: Optional[
            custom_output_config_class.CustomOutputConfig
        ] = None,
    ):
        """Initializes a pairwise evaluation metric.

        Args:
          metric: The pairwise evaluation metric name.
          metric_prompt_template: Pairwise metric prompt template for performing
            the pairwise model-based evaluation. A freeform string is also accepted.
          baseline_model: The baseline model for side-by-side comparison. If not
            specified, `baseline_model_response` column is required in the dataset
            to perform bring-your-own-response(BYOR) evaluation.
          system_instruction: The system instruction for the evaluation.
          autorater_config: The config for judge model.
          custom_output_config: Config for custom output from the judge model.
        """
        super().__init__(
            metric_prompt_template=metric_prompt_template,
            metric=metric,
            system_instruction=system_instruction,
            autorater_config=autorater_config,
            custom_output_config=custom_output_config,
        )
        self._baseline_model = baseline_model

    @property
    def baseline_model(
        self,
    ) -> Union[generative_models.GenerativeModel, Callable[[str], str]]:
        return self._baseline_model
