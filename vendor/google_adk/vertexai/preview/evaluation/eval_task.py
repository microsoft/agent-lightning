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
"""Evaluation Task class."""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Union
import uuid
import warnings

from google.api_core import exceptions
import vertexai
from google.cloud.aiplatform import base
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.metadata import metadata
from vertexai import generative_models
from vertexai.preview import reasoning_engines
from vertexai.preview.evaluation import _base as eval_base
from vertexai.preview.evaluation import _evaluation
from vertexai.preview.evaluation import constants
from vertexai.preview.evaluation import utils as eval_utils
from vertexai.preview.evaluation.metrics import (
    _base as metrics_base,
)
from vertexai.preview.evaluation.metrics import pairwise_metric
from vertexai.preview.evaluation.metrics import pointwise_metric
from vertexai.preview.evaluation.metrics import (
    rubric_based_metric,
)
import numpy as np


if TYPE_CHECKING:
    import pandas as pd
    from google.colab import sheets


# pylint: disable=g-import-not-at-top
try:
    from IPython import display as IPython_display
except ImportError:
    IPython_display = None

_LOGGER = base.Logger(__name__)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

AutoraterConfig = eval_base.AutoraterConfig
EvalResult = eval_base.EvalResult
GenerativeModel = generative_models.GenerativeModel

_RunnableType = Union[reasoning_engines.Queryable, Callable[[str], Dict[str, str]]]
_ModelType = Union[generative_models.GenerativeModel, Callable[[str], str]]


class EvalTask:
    """A class representing an EvalTask.

    An evaluation task assesses the ability of a Gen AI model, agent or
    application to perform a specific task in response to prompts.
    Each evaluation task includes an evaluation dataset, which can be a set of
    test cases and a set of metrics for assessment. These tasks provide the
    framework for running evaluations in a standardized and repeatable way,
    allowing for comparative assessment with varying run-specific parameters.


    Dataset Details:

        Default dataset column names:
            * prompt_column_name: "prompt"
            * reference_column_name: "reference"
            * response_column_name: "response"
            * baseline_model_response_column_name: "baseline_model_response"
            * rubrics_column_name: "rubrics"


        Requirement for different use cases:
          * Bring-your-own-response (BYOR): You already have the data that you
              want to evaluate stored in the dataset. Response column name can be
              customized by providing `response_column_name` parameter, or in the
              `metric_column_mapping`. For BYOR pairwise evaluation, the baseline
              model response column name can be customized by providing
              `baseline_model_response_column_name` parameter, or
              in the `metric_column_mapping`. If the `response` column or
              `baseline_model_response` column is present while the
              corresponding model is specified, an error will be raised.

          * Perform model/agent inference without a prompt template: You have a dataset
              containing the input prompts to the model/agent and want to perform
              inference before evaluation. A column named `prompt` is required
              in the evaluation dataset and is used directly as input to the model/agent.

          * Perform model/agent inference with a prompt template: You have a dataset
              containing the input variables to the prompt template and want to
              assemble the prompts for inference. Evaluation dataset
              must contain column names corresponding to the variable names in
              the prompt template. For example, if prompt template is
              "Instruction: {instruction}, context: {context}", the dataset must
              contain `instruction` and `context` columns.

    Metrics Details:

        The supported metrics descriptions, rating rubrics, and the required
        input variables can be found on the Vertex AI public documentation page.
        [Evaluation methods and metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval).

    Usage Examples:

        1. To perform bring-your-own-response(BYOR) evaluation, provide the model
        responses in the `response` column in the dataset. If a pairwise metric is
        used for BYOR evaluation, provide the baseline model responses in the
        `baseline_model_response` column.

          ```
          eval_dataset = pd.DataFrame({
                  "prompt"  : [...],
                  "reference": [...],
                  "response" : [...],
                  "baseline_model_response": [...],
          })
          eval_task = EvalTask(
            dataset=eval_dataset,
            metrics=[
                    "bleu",
                    "rouge_l_sum",
                    MetricPromptTemplateExamples.Pointwise.FLUENCY,
                    MetricPromptTemplateExamples.Pairwise.SAFETY
            ],
            experiment="my-experiment",
          )
          eval_result = eval_task.evaluate(experiment_run_name="eval-experiment-run")
          ```

        2. To perform evaluation with Gemini model inference, specify the `model`
        parameter with a `GenerativeModel` instance.  The input column name to the
        model is `prompt` and must be present in the dataset.

          ```
          eval_dataset = pd.DataFrame({
                "reference": [...],
                "prompt"  : [...],
          })
          result = EvalTask(
              dataset=eval_dataset,
              metrics=["exact_match", "bleu", "rouge_1", "rouge_l_sum"],
              experiment="my-experiment",
          ).evaluate(
              model=GenerativeModel("gemini-1.5-pro"),
              experiment_run_name="gemini-eval-run"
          )
          ```

        3. If a `prompt_template` is specified, the `prompt` column is not required.
        Prompts can be assembled from the evaluation dataset, and all prompt
        template variable names must be present in the dataset columns.
          ```
          eval_dataset = pd.DataFrame({
              "context"    : [...],
              "instruction": [...],
          })
          result = EvalTask(
              dataset=eval_dataset,
              metrics=[MetricPromptTemplateExamples.Pointwise.SUMMARIZATION_QUALITY],
          ).evaluate(
              model=GenerativeModel("gemini-1.5-pro"),
              prompt_template="{instruction}. Article: {context}. Summary:",
          )
          ```

        4. To perform evaluation with custom model inference, specify the `model`
        parameter with a custom inference function. The input column name to the
        custom inference function is `prompt` and must be present in the dataset.

          ```
          from openai import OpenAI
          client = OpenAI()
          def custom_model_fn(input: str) -> str:
            response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "user", "content": input}
              ]
            )
            return response.choices[0].message.content

          eval_dataset = pd.DataFrame({
                "prompt"  : [...],
                "reference": [...],
          })
          result = EvalTask(
              dataset=eval_dataset,
              metrics=[MetricPromptTemplateExamples.Pointwise.SAFETY],
              experiment="my-experiment",
          ).evaluate(
              model=custom_model_fn,
              experiment_run_name="gpt-eval-run"
          )
          ```

        5. To perform pairwise metric evaluation with model inference step, specify
        the `baseline_model` input to a `PairwiseMetric` instance and the candidate
        `model` input to the `EvalTask.evaluate()` function. The input column name
        to both models is `prompt` and must be present in the dataset.

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
          result = EvalTask(
              dataset=eval_dataset,
              metrics=[pairwise_groundedness],
              experiment="my-pairwise-experiment",
          ).evaluate(
              model=candidate_model,
              experiment_run_name="gemini-pairwise-eval-run",
          )
          ```
    """

    def __init__(
        self,
        *,
        dataset: Union["pd.DataFrame", str, Dict[str, Any], "sheets.InteractiveSheet"],
        metrics: List[
            Union[
                Literal[
                    "exact_match",
                    "bleu",
                    "rouge_1",
                    "rouge_2",
                    "rouge_l",
                    "rouge_l_sum",
                    "tool_call_valid",
                    "tool_name_match",
                    "tool_parameter_key_match",
                    "tool_parameter_kv_match",
                    "trajectory_exact_match",
                    "trajectory_in_order_match",
                    "trajectory_any_order_match",
                    "trajectory_precision",
                    "trajectory_recall",
                    "rubric_based_instruction_following",
                ],
                metrics_base.CustomMetric,
                metrics_base._AutomaticMetric,
                pointwise_metric.PointwiseMetric,
                pairwise_metric.PairwiseMetric,
                rubric_based_metric.RubricBasedMetric,
            ]
        ],
        experiment: Optional[str] = None,
        metric_column_mapping: Optional[Dict[str, str]] = None,
        output_uri_prefix: Optional[str] = "",
        autorater_config: Optional[AutoraterConfig] = None,
    ):
        """Initializes an EvalTask.

        Args:
            dataset: The dataset to be evaluated.
                Supports the following dataset formats:
                * pandas.DataFrame: Used directly for evaluation.
                * Dict: Converted to a pandas DataFrame before evaluation.
                * str: Interpreted as a file path or URI. Supported formats include:
                    * Local JSONL or CSV files:  Loaded from the local filesystem.
                    * GCS JSONL or CSV files: Loaded from Google Cloud Storage
                        (e.g., 'gs://bucket/data.csv').
                    * BigQuery table URI: Loaded from Google Cloud BigQuery
                        (e.g., 'bq://project-id.dataset.table_name').
            metrics: The list of metric names, or Metric instances to evaluate.
              Prompt template is required for PairwiseMetric.
            experiment: The name of the experiment to log the evaluations to.
            metric_column_mapping: An optional dictionary column mapping that
              overrides the metric prompt template input variable names with
              mapped the evaluation dataset column names, used during evaluation.
              For example, if the input_variables of the metric prompt template
              are ["context", "reference"], the metric_column_mapping can be
                {
                    "context": "news_context",
                    "reference": "ground_truth",
                    "response": "model_1_response"
                }
              if the dataset has columns "news_context", "ground_truth" and
              "model_1_response".
            output_uri_prefix: GCS location to store the metrics_table from
              evaluation results.
            autorater_config: The autorater config for model based evaluation.
              If autorater config is specified on a metric, it will override the
              autorater config specified here.
        """
        self._dataset = eval_utils.load_dataset(dataset)
        self._metrics = metrics
        self._experiment = experiment
        self._metric_column_mapping = eval_utils.initialize_metric_column_mapping(
            metric_column_mapping, self._dataset
        )
        self.output_uri_prefix = output_uri_prefix
        self._autorater_config = autorater_config

    @property
    def dataset(self) -> "pd.DataFrame":
        """Returns evaluation dataset."""
        return self._dataset

    @property
    def metrics(self) -> List[Union[str, metrics_base.CustomMetric]]:
        """Returns metrics."""
        return self._metrics

    @property
    def autorater_config(self) -> Optional[AutoraterConfig]:
        """Returns autorater config."""
        return self._autorater_config

    @property
    def experiment(self) -> Optional[str]:
        """Returns experiment name."""
        return self._experiment

    def _evaluate_with_experiment(
        self,
        model: Optional[_ModelType] = None,
        runnable: Optional[_RunnableType] = None,
        prompt_template: Optional[str] = None,
        experiment_run_name: Optional[str] = None,
        evaluation_service_qps: Optional[float] = None,
        retry_timeout: float = 120.0,
        output_file_name: Optional[str] = None,
    ) -> EvalResult:
        """Runs an evaluation for the EvalTask with an experiment.

        Args:
          model: A GenerativeModel instance or a custom model function to generate
            responses to evaluate. If not provided, the evaluation is computed with
            the `response` column in the `dataset`.
          runnable: The runnable to generate responses to evaluate. If not provided,
            the evaluation is computed with the `response` and/or `predicted_trajectory`
            column in the `dataset`.
          prompt_template: The prompt template to use for the evaluation. If not
            set, the prompt template that was used to create the EvalTask will be
            used.
          experiment_run_name: The name of the experiment run to log the evaluation
            to if an experiment is set for this EvalTask. If not provided, a random
            unique experiment run name is used.
          evaluation_service_qps: The custom QPS limit for the evaluation service.
          retry_timeout: How long to keep retrying the evaluation requests for
            the whole evaluation dataset, in seconds.
          output_path: The file name with csv suffix to store the output
            metrics_table to be tracked in the experiment run.

        Returns:
          The evaluation result.
        """
        self._validate_experiment_run()
        with vertexai.preview.start_run(experiment_run_name):
            self._log_eval_experiment_param(
                model=model,
                runnable=runnable,
                prompt_template=prompt_template,
                output_file_name=output_file_name,
            )
            eval_result = _evaluation.evaluate(
                dataset=self._dataset,
                metrics=self._metrics,
                model=model,
                runnable=runnable,
                prompt_template=prompt_template,
                metric_column_mapping=self._metric_column_mapping,
                evaluation_service_qps=evaluation_service_qps,
                retry_timeout=retry_timeout,
                autorater_config=self._autorater_config,
            )

            eval_result.summary_metrics = {
                k: ("NaN" if isinstance(v, float) and np.isnan(v) else v)
                for k, v in eval_result.summary_metrics.items()
            }
            eval_result.metadata = {
                "experiment": self._experiment,
                "experiment_run": experiment_run_name,
            }
            try:
                vertexai.preview.log_metrics(eval_result.summary_metrics)
            except (TypeError, exceptions.InvalidArgument) as e:
                _LOGGER.warning(f"Experiment metrics logging failed: {str(e)}")
        return eval_result

    def evaluate(
        self,
        *,
        model: Optional[_ModelType] = None,
        runnable: Optional[_RunnableType] = None,
        prompt_template: Optional[str] = None,
        experiment_run_name: Optional[str] = None,
        response_column_name: Optional[str] = None,
        baseline_model_response_column_name: Optional[str] = None,
        evaluation_service_qps: Optional[float] = None,
        retry_timeout: float = 120.0,
        output_file_name: Optional[str] = "",
    ) -> EvalResult:
        """Runs an evaluation for the EvalTask.

        Args:
          model: A GenerativeModel instance or a custom model function to generate
            responses to evaluate. If not provided, the evaluation can be performed
            in the bring-your-own-response (BYOR) mode.
          runnable: The runnable to generate responses to evaluate. If not provided,
            the evaluation is computed with the `response` and/or `predicted_trajectory`
            column in the `dataset`.
          prompt_template: The prompt template to use for the evaluation. If not
            set, the prompt template that was used to create the EvalTask will be
            used.
          experiment_run_name: The name of the experiment run to log the evaluation
            to if an experiment is set for this EvalTask. If not provided, a random
            unique experiment run name is used.
          response_column_name: The column name of model response in the dataset. If
            provided, this will override the `metric_column_mapping` of the `EvalTask`.
          baseline_model_response_column_name: The column name of baseline model
            response in the dataset for pairwise metrics. If provided, this will
            override the `metric_column_mapping` of the `EvalTask`
          evaluation_service_qps: The custom QPS limit for the evaluation service.
          retry_timeout: How long to keep retrying the evaluation requests for
            the whole evaluation dataset, in seconds.
          output_file_name: The file name with csv suffix to store the output
            metrics_table.

        Returns:
          The evaluation result.
        """
        global_experiment_name = (
            metadata._experiment_tracker.experiment_name
        )  # pylint: disable=protected-access
        if experiment_run_name and not self._experiment and not global_experiment_name:
            raise ValueError(
                "Experiment is not set. Please initialize EvalTask with an"
                " experiment, or initialize a global experiment with "
                "`vertexai.init(experiment='experiment_name')`for logging this"
                " evaluation run."
            )

        self._verify_and_set_response_column_name(
            response_column_name=response_column_name,
            metric_column_mapping_key=constants.Dataset.MODEL_RESPONSE_COLUMN,
        )
        self._verify_and_set_response_column_name(
            response_column_name=baseline_model_response_column_name,
            metric_column_mapping_key=constants.Dataset.BASELINE_MODEL_RESPONSE_COLUMN,
        )
        if self.output_uri_prefix and not output_file_name:
            output_file_name = f"eval_results_{utils.timestamped_unique_name()}.csv"
        experiment_run_name = experiment_run_name or f"{uuid.uuid4()}"
        if self._experiment and global_experiment_name:
            metadata._experiment_tracker.set_experiment(  # pylint: disable=protected-access
                experiment=self._experiment, backing_tensorboard=False
            )
            eval_result = self._evaluate_with_experiment(
                model=model,
                runnable=runnable,
                prompt_template=prompt_template,
                experiment_run_name=experiment_run_name,
                evaluation_service_qps=evaluation_service_qps,
                retry_timeout=retry_timeout,
                output_file_name=output_file_name,
            )
            metadata._experiment_tracker.set_experiment(  # pylint: disable=protected-access
                experiment=global_experiment_name, backing_tensorboard=False
            )
        elif self._experiment and not global_experiment_name:
            metadata._experiment_tracker.set_experiment(  # pylint: disable=protected-access
                experiment=self._experiment, backing_tensorboard=False
            )
            eval_result = self._evaluate_with_experiment(
                model=model,
                runnable=runnable,
                prompt_template=prompt_template,
                experiment_run_name=experiment_run_name,
                evaluation_service_qps=evaluation_service_qps,
                retry_timeout=retry_timeout,
                output_file_name=output_file_name,
            )
            metadata._experiment_tracker.reset()  # pylint: disable=protected-access
        elif not self._experiment and global_experiment_name:
            eval_result = self._evaluate_with_experiment(
                model=model,
                runnable=runnable,
                prompt_template=prompt_template,
                experiment_run_name=experiment_run_name,
                evaluation_service_qps=evaluation_service_qps,
                retry_timeout=retry_timeout,
                output_file_name=output_file_name,
            )
        else:
            eval_result = _evaluation.evaluate(
                dataset=self._dataset,
                metrics=self._metrics,
                model=model,
                runnable=runnable,
                prompt_template=prompt_template,
                metric_column_mapping=self._metric_column_mapping,
                evaluation_service_qps=evaluation_service_qps,
                retry_timeout=retry_timeout,
                autorater_config=self._autorater_config,
            )
        eval_utils.upload_evaluation_results(
            eval_result, self.output_uri_prefix, output_file_name
        )
        return eval_result

    def _validate_experiment_run(self) -> None:
        """Checks if an experiment run already exists."""
        if (
            metadata._experiment_tracker.experiment_run
        ):  # pylint: disable=protected-access
            raise ValueError(
                "Experiment run already exists. Please specify the name of the"
                " experiment run to assign current session within this evaluation."
            )

    def _log_eval_experiment_param(
        self,
        model: _ModelType = None,
        runnable: _RunnableType = None,
        prompt_template: Optional[str] = None,
        output_file_name: Optional[str] = None,
    ) -> None:
        """Logs variable input parameters of an evaluation to an experiment run."""
        eval_metadata = {}
        if prompt_template is not None:
            eval_metadata.update({"prompt_template": prompt_template})

        if model:
            if isinstance(model, GenerativeModel):
                eval_metadata.update(
                    {
                        "model_name": model._model_name,  # pylint: disable=protected-access
                    }
                )

                if (
                    model._generation_config
                    and isinstance(  # pylint: disable=protected-access
                        model._generation_config,
                        dict,  # pylint: disable=protected-access
                    )
                ):
                    eval_metadata.update(
                        **model._generation_config
                    )  # pylint: disable=protected-access

                if model._safety_settings and isinstance(
                    model._safety_settings, dict
                ):  # pylint: disable=protected-access
                    safety_settings = (
                        model._safety_settings
                    )  # pylint: disable=protected-access
                    safety_settings_as_str = {
                        category.name: threshold.name
                        for category, threshold in safety_settings.items()
                    }
                    eval_metadata.update(safety_settings_as_str)

        if runnable:
            if isinstance(runnable, reasoning_engines.LangchainAgent):
                eval_metadata.update(
                    {
                        "model_name": runnable._model_name,
                        "tools": runnable._tools,
                    }  # pylint: disable=protected-access
                )

        if self.output_uri_prefix and output_file_name:
            eval_metadata.update(
                {"output_file": self.output_uri_prefix + "/" + output_file_name}
            )

        if eval_metadata:
            _LOGGER.info(
                f"Logging Eval experiment evaluation metadata: {eval_metadata}"
            )
            try:
                vertexai.preview.log_params(eval_metadata)
            except (ValueError, TypeError) as e:
                _LOGGER.warning(
                    f"Experiment evaluation metadata logging failed: {str(e)}"
                )

    def _verify_and_set_response_column_name(
        self, response_column_name: str, metric_column_mapping_key: str
    ) -> None:
        """Verifies and sets the model response column names."""
        if response_column_name:
            if response_column_name in self._dataset.columns:
                self._metric_column_mapping[metric_column_mapping_key] = (
                    response_column_name
                )
            else:
                raise ValueError(
                    f"(Baseline) Model response column {response_column_name} is not"
                    " found in the dataset."
                )

    def display_runs(self):
        """Displays experiment runs associated with this EvalTask."""
        if not self._experiment:
            raise ValueError("Experiment is not set.")
        elif IPython_display:
            IPython_display.display(
                vertexai.preview.get_experiment_df(self._experiment)
            )
