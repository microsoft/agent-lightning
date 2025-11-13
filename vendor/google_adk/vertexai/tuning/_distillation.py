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
# pylint: disable=protected-access
"""Classes for model tuning based on distillation."""

from typing import Optional

from google.cloud.aiplatform.utils import gcs_utils
from google.cloud.aiplatform_v1beta1.types import (
    tuning_job as gca_tuning_job_types,
)

from vertexai import generative_models
from vertexai.tuning import _tuning


def distill_model(
    *,
    student_model: str,
    teacher_model: str,
    training_dataset: str,
    validation_dataset: Optional[str] = None,
    epoch_count: Optional[int] = None,
    learning_rate_multiplier: Optional[float] = None,
    tuned_model_display_name: Optional[str] = None,
) -> "DistillationJob":
    """Tunes a model using distillation.

    Args:
        student_model:
            Student model name for distillation, e.g., "gemma-1.1-2b-it".
        teacher_model:
            Teacher model name for distillation, e.g., "gemini-1.5-flash-001".
        training_dataset: Cloud Storage path to file containing training dataset for distillation.
            The dataset should be in JSONL format.
        validation_dataset: Cloud Storage path to file containing validation dataset for distillation.
            The dataset should be in JSONL format.
        epoch_count: Number of training epoches for this tuning job.
        learning_rate_multiplier: Learning rate multiplier for tuning.
        tuned_model_display_name: The display name of the
            [TunedModel][google.cloud.aiplatform.v1.Model]. The name can
            be up to 128 characters long and can consist of any UTF-8 characters.

    Returns:
        A `TuningJob` object.
    """

    if isinstance(student_model, generative_models.GenerativeModel):
        student_model = student_model._prediction_resource_name

    student_model = student_model.rpartition("/")[-1]
    teacher_model = teacher_model.rpartition("/")[-1]

    pipeline_root = (
        gcs_utils.create_gcs_bucket_for_pipeline_artifacts_if_it_does_not_exist()
    )

    distillation_spec = gca_tuning_job_types.DistillationSpec(
        student_model=student_model,
        base_teacher_model=teacher_model,
        training_dataset_uri=training_dataset,
        validation_dataset_uri=validation_dataset,
        hyper_parameters=gca_tuning_job_types.DistillationHyperParameters(
            epoch_count=epoch_count,
            learning_rate_multiplier=learning_rate_multiplier,
        ),
        pipeline_root_directory=pipeline_root,
    )

    return DistillationJob._create(  # pylint: disable=protected-access
        base_model=None,
        tuning_spec=distillation_spec,
        tuned_model_display_name=tuned_model_display_name,
    )


class DistillationJob(_tuning.TuningJob):
    pass
