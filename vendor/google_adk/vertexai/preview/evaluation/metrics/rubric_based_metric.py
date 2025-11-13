# Copyright 2025 Google LLC
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

import collections
from typing import Union, TYPE_CHECKING

from google.cloud.aiplatform import base
from vertexai import generative_models
from vertexai.preview.evaluation import _pre_eval_utils
from vertexai.preview.evaluation import constants
from vertexai.preview.evaluation import utils
from vertexai.preview.evaluation.metrics import (
    _base as metrics_base,
)
from vertexai.preview.evaluation.metrics import pairwise_metric
from vertexai.preview.evaluation.metrics import pointwise_metric

if TYPE_CHECKING:
    import pandas as pd

_DEFAULT_MODEL_NAME = "gemini-2.0-flash-001"
_LOGGER = base.Logger(__name__)


class RubricBasedMetric(metrics_base._Metric):
    """Config for Rubric-Based Eval."""

    def __init__(
        self,
        *,
        generation_config: metrics_base.RubricGenerationConfig,
        critique_metric: Union[
            pointwise_metric.PointwiseMetric, pairwise_metric.PairwiseMetric
        ]
    ):
        """Initializes RubricBasedMetric.

        Args:
          generation_config: Config for rubric generation.
          critique_metric: Pointwise/pairwise metric for rubric critique.
        """
        super().__init__(metric=critique_metric._metric)

        self.generation_config = generation_config
        self.critique_metric = critique_metric

    def generate_rubrics(
        self,
        eval_dataset: "pd.Dataframe",
    ) -> "pd.DataFrame":
        """Generates rubrics for given eval dataset."""
        if not self.generation_config.model:
            model = generative_models.GenerativeModel(model_name=_DEFAULT_MODEL_NAME)
        else:
            model = self.generation_config.model

        if constants.Dataset.RUBRICS_COLUMN in eval_dataset.columns:
            _LOGGER.warning(
                "Rubrics column already exists in the dataset. Skipping rubric"
                " generation."
            )
            return eval_dataset

        responses = _pre_eval_utils._generate_responses_from_gemini_model(
            model,
            eval_dataset,
            self.generation_config.prompt_template,
        )
        if self.generation_config.parsing_fn:
            parsing_fn = self.generation_config.parsing_fn
        else:
            parsing_fn = utils.parse_rubrics
        dataset_with_rubrics = eval_dataset.copy()
        aggregated = collections.defaultdict(list)
        for idx, response in enumerate(responses):
            result = parsing_fn(response)
            if isinstance(result, dict):
                questions = result.pop("questions", None)
                if questions is not None:
                    aggregated[constants.Dataset.RUBRICS_COLUMN].append(
                        (idx, questions)
                    )
                for key, value in result.items():
                    aggregated[key].append((idx, value))
            else:
                aggregated[constants.Dataset.RUBRICS_COLUMN].append((idx, result))
        for key, values in aggregated.items():
            dataset_with_rubrics[key] = None
            dataset_with_rubrics[key] = dataset_with_rubrics[key].astype(object)
            for idx, value in values:
                dataset_with_rubrics.at[idx, key] = value
        return dataset_with_rubrics
