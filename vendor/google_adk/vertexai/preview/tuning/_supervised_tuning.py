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

from typing import Dict, Literal, Optional, Union

from google.cloud.aiplatform.preview import datasets
from google.cloud.aiplatform.utils import _ipython_utils
from google.cloud.aiplatform_v1beta1.types import (
    tuning_job as gca_tuning_job_types,
)
from vertexai import generative_models
from vertexai.preview.tuning import _tuning
from vertexai.preview.tuning._tuning import SourceModel


def train(
    *,
    source_model: Union[
        str,
        generative_models.GenerativeModel,
        SourceModel,
    ],
    train_dataset: Union[str, datasets.MultimodalDataset],
    validation_dataset: Optional[Union[str, datasets.MultimodalDataset]] = None,
    tuned_model_display_name: Optional[str] = None,
    tuning_mode: Optional[Literal["FULL", "PEFT_ADAPTER"]] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    learning_rate_multiplier: Optional[float] = None,
    adapter_size: Optional[Literal[1, 4, 8, 16, 32]] = None,
    labels: Optional[Dict[str, str]] = None,
    output_uri: Optional[str] = None,
) -> "SupervisedTuningJob":
    """Tunes a model using supervised training.

    Args:
        source_model (str): Model name for tuning, e.g., "gemini-1.0-pro-002".
        train_dataset: Training dataset used for tuning. The dataset can be a JSONL file on Google Cloud Storage (specified as its GCS URI) or a Vertex Multimodal Dataset (either as the dataset object itself or as its resource name).
        validation_dataset: Validation dataset used for tuning. The dataset can be a JSONL file on Google Cloud Storage (specified as its GCS URI) or a Vertex Multimodal Dataset (either as the dataset object itself or as the resource name).
        tuned_model_display_name: The display name of the
          [TunedModel][google.cloud.aiplatform.v1.Model]. The name can be up to
          128 characters long and can consist of any UTF-8 characters.
        epochs: Number of training epoches for this tuning job.
        tuning_mode: Tuning mode for this tuning job. Can only be used with OSS
          models.
        learning_rate: Learning rate for tuning. Can only be used with OSS
          models. Mutually exclusive with `learning_rate_multiplier`.
        learning_rate_multiplier: Learning rate multiplier for tuning. Mutually
          exclusive with `learning_rate`.
        adapter_size: Adapter size for tuning.
        labels: User-defined metadata to be associated with trained models

    Returns:
        A `TuningJob` object.
    """
    if tuning_mode is None:
        tuning_mode_value = None
    elif tuning_mode == "FULL":
        tuning_mode_value = (
            gca_tuning_job_types.SupervisedTuningSpec.TuningMode.TUNING_MODE_FULL
        )
    elif tuning_mode == "PEFT_ADAPTER":
        tuning_mode_value = (
            gca_tuning_job_types.SupervisedTuningSpec.TuningMode.TUNING_MODE_PEFT_ADAPTER
        )
    else:
        raise ValueError(
            f"Unsupported tuning mode: {tuning_mode}. The supported tuning modes are [FULL, PEFT_ADAPTER]"
        )

    if learning_rate and learning_rate_multiplier:
        raise ValueError(
            "Only one of `learning_rate` and `learning_rate_multiplier` can be set."
        )

    if adapter_size is None:
        adapter_size_value = None
    elif adapter_size == 1:
        adapter_size_value = (
            gca_tuning_job_types.SupervisedHyperParameters.AdapterSize.ADAPTER_SIZE_ONE
        )
    elif adapter_size == 4:
        adapter_size_value = (
            gca_tuning_job_types.SupervisedHyperParameters.AdapterSize.ADAPTER_SIZE_FOUR
        )
    elif adapter_size == 8:
        adapter_size_value = (
            gca_tuning_job_types.SupervisedHyperParameters.AdapterSize.ADAPTER_SIZE_EIGHT
        )
    elif adapter_size == 16:
        adapter_size_value = (
            gca_tuning_job_types.SupervisedHyperParameters.AdapterSize.ADAPTER_SIZE_SIXTEEN
        )
    elif adapter_size == 32:
        adapter_size_value = (
            gca_tuning_job_types.SupervisedHyperParameters.AdapterSize.ADAPTER_SIZE_THIRTY_TWO
        )
    else:
        raise ValueError(
            f"Unsupported adapter size: {adapter_size}. The supported sizes are [1, 4, 8, 16, 32]"
        )
    if isinstance(train_dataset, datasets.MultimodalDataset):
        train_dataset = train_dataset.resource_name
    if isinstance(validation_dataset, datasets.MultimodalDataset):
        validation_dataset = validation_dataset.resource_name
    supervised_tuning_spec = gca_tuning_job_types.SupervisedTuningSpec(
        tuning_mode=tuning_mode_value,
        training_dataset_uri=train_dataset,
        validation_dataset_uri=validation_dataset,
        hyper_parameters=gca_tuning_job_types.SupervisedHyperParameters(
            epoch_count=epochs,
            learning_rate=learning_rate,
            learning_rate_multiplier=learning_rate_multiplier,
            adapter_size=adapter_size_value,
        ),
    )

    if isinstance(source_model, generative_models.GenerativeModel):
        source_model = source_model._prediction_resource_name.rpartition("/")[-1]

    if labels is None:
        labels = {}
    if "mg-source" not in labels:
        labels["mg-source"] = "sdk"

    supervised_tuning_job = (
        SupervisedTuningJob._create(  # pylint: disable=protected-access
            base_model=source_model,
            tuning_spec=supervised_tuning_spec,
            tuned_model_display_name=tuned_model_display_name,
            labels=labels,
            output_uri=output_uri,
        )
    )
    _ipython_utils.display_model_tuning_button(supervised_tuning_job)

    return supervised_tuning_job


class SupervisedTuningJob(_tuning.TuningJob):
    def __init__(self, tuning_job_name: str):
        super().__init__(tuning_job_name=tuning_job_name)
        _ipython_utils.display_model_tuning_button(self)
