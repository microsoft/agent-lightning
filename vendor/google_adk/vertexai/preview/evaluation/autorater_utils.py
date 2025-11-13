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
"""Autorater Utils Class and Functions."""

import logging
import time
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Union

from vertexai import generative_models
from vertexai.preview.evaluation import _base as evaluation_base
from vertexai.preview.evaluation import eval_task
from vertexai.preview.evaluation.metrics import pairwise_metric
from vertexai.preview.evaluation.metrics import pointwise_metric
from vertexai.preview.tuning import sft
from sklearn import metrics


if TYPE_CHECKING:
    import pandas as pd

AutoraterConfig = evaluation_base.AutoraterConfig
AutoraterEvalResult = evaluation_base.AutoraterEvalResult
EvalTask = eval_task.EvalTask
PointwiseMetric = pointwise_metric.PointwiseMetric
PairwiseMetric = pairwise_metric.PairwiseMetric
_SCORE = "score"
_METRIC = "metric"
_PAIRWISE_CHOICE = "pairwise_choice"
_HUMAN_RATING = "human_rating"
_HUMAN_PAIRWISE_CHOICE = "human_pairwise_choice"
_ACCURACY_BALANCED = "accuracy_balanced"
_F1_SCORE_BALANCED = "f1_score_balanced"
_CONFUSION_MATRIX = "confusion_matrix"
_CONFUSION_MATRIX_LABELS = "confusion_matrix_labels"
_METRICS_CATEGORY_LIMIT = 10
_NAN = "nan"
_ERROR = "error"


def tune_autorater(
    *,
    base_model: Union[str, generative_models.GenerativeModel],
    train_dataset: str,
    validation_dataset: Optional[str] = None,
    tuned_model_display_name: Optional[str] = None,
    epochs: Optional[int] = None,
    learning_rate_multiplier: Optional[float] = None,
    adapter_size: Optional[Literal[1, 4, 8, 16]] = None,
    labels: Optional[Dict[str, str]] = None,
    time_out_hours: int = 10,
) -> AutoraterConfig:
    """Lora Tune an autorater model.

    Args:
        base_model: Model name for tuning, e.g., "gemini-1.0-pro-002".
        train_dataset: Cloud Storage path to file containing training dataset for
          tuning. The dataset should be in JSONL format.
        validation_dataset: Cloud Storage path to file containing validation
          dataset for tuning. The dataset should be in JSONL format.
        tuned_model_display_name: The display name of the
          [TunedModel][google.cloud.aiplatform.v1.Model]. The name can be up to
          128 characters long and can consist of any UTF-8 characters.
        epochs: Number of training epoches for this tuning job.
        learning_rate_multiplier: Learning rate multiplier for tuning.
        adapter_size: Adapter size for tuning.
        labels: User-defined metadata to be associated with trained models
        time_out_hours: Timeout in hours for tuning job. Default value is 10
          hours.

    Returns:
        A `AutoraterConfig` object with tuned model endpoint.
    """
    tune_job = sft.train(
        source_model=base_model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        tuned_model_display_name=tuned_model_display_name,
        epochs=epochs,
        learning_rate_multiplier=learning_rate_multiplier,
        adapter_size=adapter_size,
        labels=labels,
    )
    time_out_seconds = time_out_hours * 60 * 60
    while not tune_job.refresh().has_ended and time_out_seconds > 0:
        time.sleep(60)
        time_out_seconds -= 60

    if tune_job.has_succeeded:
        return AutoraterConfig(autorater_model=tune_job.tuned_model_endpoint_name)
    else:
        raise ValueError(
            "Failed to tune autorater model. Please check the logs for more details."
        )


def _get_evaluation_result(
    metric: Union[PointwiseMetric, PairwiseMetric],
    autorater_eval_results: List[str],
    human_eval_results: List[str],
) -> Dict[str, Any]:
    """Get evaluation result for autorater."""
    filtered_autorater_eval_results = []
    filtered_human_eval_results = []
    for autorater_eval_result, human_eval_result in zip(
        autorater_eval_results, human_eval_results
    ):
        # Filter failed pointwise evaluation results.
        if autorater_eval_result.lower() == _NAN or human_eval_result.lower() == _NAN:
            continue

        # Filter failed pairwise evaluation results.
        if (
            autorater_eval_result.lower() == _ERROR
            or human_eval_result.lower() == _ERROR
        ):
            continue

        filtered_autorater_eval_results.append(autorater_eval_result)
        filtered_human_eval_results.append(human_eval_result)

    labels = list(
        sorted(set(filtered_autorater_eval_results) | set(filtered_human_eval_results))
    )
    eval_result = {_METRIC: metric.metric_name}
    eval_result[_ACCURACY_BALANCED] = metrics.balanced_accuracy_score(
        filtered_human_eval_results, filtered_autorater_eval_results
    )
    eval_result[_F1_SCORE_BALANCED] = metrics.f1_score(
        filtered_human_eval_results,
        filtered_autorater_eval_results,
        average="weighted",
    )
    if len(labels) > _METRICS_CATEGORY_LIMIT:
        logging.warning(
            "Confusion matrix is not provided as the number of"
            " rating rubric values %d is greater than the limit %d.",
            len(labels),
            _METRICS_CATEGORY_LIMIT,
        )
    else:
        eval_result[_CONFUSION_MATRIX] = metrics.confusion_matrix(
            filtered_human_eval_results,
            filtered_autorater_eval_results,
            labels=labels,
        )
        eval_result[_CONFUSION_MATRIX_LABELS] = labels
    return eval_result


def evaluate_autorater(
    *,
    evaluate_autorater_input: "pd.DataFrame",
    eval_metrics: List[Union[PointwiseMetric, PairwiseMetric]],
    autorater_config: Optional[AutoraterConfig] = None,
    eval_dataset_metadata: Dict[str, Any] = None,
    **kwargs,
) -> AutoraterEvalResult:
    """Evaluates the autorater model using human evaluation results.

    Args:
        evaluate_autorater_input: Autorater evaluation input, including
          evaluation results from human evaluation and autorater model.
        eval_metrics: List of model based metrics.
        autorater_config: Autorater configuration.
        eval_dataset_metadata: Evaluation dataset metadata.
        **kwargs: Additional arguments added to AutoraterEvalResult.

    Returns:
        Autorater evalaution result .
    """
    eval_result = []
    for metric in eval_metrics:
        if isinstance(metric, PointwiseMetric):
            autorater_score = list(
                map(
                    lambda x: str(float(x)),
                    list(evaluate_autorater_input[metric.metric_name + "/" + _SCORE]),
                )
            )
            human_score = list(
                map(
                    lambda x: str(float(x)),
                    list(
                        evaluate_autorater_input[
                            metric.metric_name + "/" + _HUMAN_RATING
                        ]
                    ),
                )
            )
            eval_result.append(
                _get_evaluation_result(metric, autorater_score, human_score)
            )
        elif isinstance(metric, PairwiseMetric):
            autorater_choice = list(
                map(
                    str,
                    list(
                        evaluate_autorater_input[
                            metric.metric_name + "/" + _PAIRWISE_CHOICE
                        ]
                    ),
                )
            )
            human_choice = list(
                map(
                    str,
                    list(
                        evaluate_autorater_input[
                            metric.metric_name + "/" + _HUMAN_PAIRWISE_CHOICE
                        ]
                    ),
                )
            )
            eval_result.append(
                _get_evaluation_result(metric, autorater_choice, human_choice)
            )
        else:
            continue
    return AutoraterEvalResult(
        eval_result=eval_result,
        eval_dataset_metadata=eval_dataset_metadata,
        autorater_config=autorater_config,
        **kwargs,
    )
