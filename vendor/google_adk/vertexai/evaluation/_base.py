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
"""Base classes for evaluation."""


import dataclasses
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from google.cloud.aiplatform_v1.services import (
    evaluation_service as gapic_evaluation_services,
)
from vertexai.evaluation.metrics import (
    _base as metrics_base,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclasses.dataclass
class EvaluationRunConfig:
    """Evaluation Run Configurations.

    Attributes:
      dataset: The dataset to evaluate.
      metrics: The list of metric names, or Metric instances to evaluate.
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
      client: The evaluation service client.
      evaluation_service_qps: The custom QPS limit for the evaluation service.
      retry_timeout: How long to keep retrying the evaluation requests, in seconds.
    """

    dataset: "pd.DataFrame"
    metrics: List[Union[str, metrics_base._Metric]]
    metric_column_mapping: Dict[str, str]
    client: gapic_evaluation_services.EvaluationServiceClient
    evaluation_service_qps: float
    retry_timeout: float

    def validate_dataset_column(self, column_name: str) -> None:
        """Validates that the column names in the column map are in the dataset.

        Args:
          column_name: The column name to validate.

        Raises:
          KeyError: If any of the column names are not in the dataset.
        """
        if (
            self.metric_column_mapping.get(column_name, column_name)
            not in self.dataset.columns
        ):
            raise KeyError(
                "Required column"
                f" `{self.metric_column_mapping.get(column_name, column_name)}`"
                " not found in the evaluation dataset. The columns in the"
                f" evaluation dataset are {list(self.dataset.columns)}."
            )


@dataclasses.dataclass
class EvalResult:
    """Evaluation result.

    Attributes:
      summary_metrics: A dictionary of summary evaluation metrics for an evaluation run.
      metrics_table: A pandas.DataFrame table containing evaluation dataset inputs,
        predictions, explanations, and metric results per row.
      metadata: The metadata for the evaluation run.
    """

    summary_metrics: Dict[str, float]
    metrics_table: Optional["pd.DataFrame"] = None
    metadata: Optional[Dict[str, str]] = None
