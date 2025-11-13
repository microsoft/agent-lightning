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
"""Class to support Batch Prediction with GenAI models."""
# pylint: disable=protected-access

import logging
import re
from typing import List, Optional, Union

from google.cloud.aiplatform import base as aiplatform_base
from google.cloud.aiplatform import initializer as aiplatform_initializer
from google.cloud.aiplatform import jobs
from google.cloud.aiplatform import models
from google.cloud.aiplatform import utils as aiplatform_utils
from google.cloud.aiplatform_v1 import types as gca_types
from vertexai import generative_models

from google.rpc import status_pb2


_LOGGER = aiplatform_base.Logger(__name__)

_GEMINI_MODEL_PATTERN = r"publishers/google/models/gemini"
_LLAMA_MODEL_PATTERN = r"publishers/meta/models/llama"
_CLAUDE_MODEL_PATTERN = r"publishers/anthropic/models/claude"
_GPT_MODEL_PATTERN = r"publishers/openai/models/gpt"
_QWEN_MODEL_PATTERN = r"publishers/qwen/models/qwen"
_DEEPSEEK_MODEL_PATTERN = r"publishers/deepseek-ai/models/deepseek"
_E5_MODEL_PATTERN = r"publishers/intfloat/models/multilingual"
_GEMINI_TUNED_MODEL_PATTERN = r"^projects/[0-9]+?/locations/[0-9a-z-]+?/models/[0-9]+?$"


class BatchPredictionJob(aiplatform_base._VertexAiResourceNounPlus):
    """Represents a BatchPredictionJob that runs with GenAI models."""

    _resource_noun = "batchPredictionJobs"
    _getter_method = "get_batch_prediction_job"
    _list_method = "list_batch_prediction_jobs"
    _delete_method = "delete_batch_prediction_job"
    _job_type = "batch-predictions"
    _parse_resource_name_method = "parse_batch_prediction_job_path"
    _format_resource_name_method = "batch_prediction_job_path"

    client_class = aiplatform_utils.JobClientWithOverride

    def __init__(self, batch_prediction_job_name: str):
        """Retrieves a BatchPredictionJob resource that runs with a GenAI model.

        Args:
            batch_prediction_job_name (str):
                Required. A fully-qualified BatchPredictionJob resource name or
                ID. Example: "projects/.../locations/.../batchPredictionJobs/456"
                or "456" when project and location are initialized.

        Raises:
            ValueError: If batch_prediction_job_name represents a BatchPredictionJob
            resource that runs with another type of model.
        """
        super().__init__(resource_name=batch_prediction_job_name)
        self._gca_resource = self._get_gca_resource(
            resource_name=batch_prediction_job_name
        )
        if not self._is_genai_model(self.model_name):
            raise ValueError(
                f"BatchPredictionJob '{batch_prediction_job_name}' "
                f"runs with the model '{self.model_name}', "
                "which is not a GenAI model."
            )

    @property
    def model_name(self) -> str:
        """Returns the model name used for this batch prediction job."""
        return self._gca_resource.model

    @property
    def state(self) -> gca_types.JobState:
        """Returns the state of this batch prediction job."""
        return self._gca_resource.state

    @property
    def has_ended(self) -> bool:
        """Returns true if this batch prediction job has ended."""
        return self.state in jobs._JOB_COMPLETE_STATES

    @property
    def has_succeeded(self) -> bool:
        """Returns true if this batch prediction job has succeeded."""
        return self.state == gca_types.JobState.JOB_STATE_SUCCEEDED

    @property
    def error(self) -> Optional[status_pb2.Status]:
        """Returns detailed error info for this Job resource."""
        return self._gca_resource.error

    @property
    def output_location(self) -> str:
        """Returns the output location of this batch prediction job."""
        return (
            self._gca_resource.output_info.gcs_output_directory
            or self._gca_resource.output_info.bigquery_output_table
        )

    @classmethod
    def submit(
        cls,
        source_model: Union[str, generative_models.GenerativeModel],
        input_dataset: Union[str, List[str]],
        *,
        output_uri_prefix: Optional[str] = None,
        job_display_name: Optional[str] = None,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        starting_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
    ) -> "BatchPredictionJob":
        """Submits a batch prediction job for a GenAI model.

        Args:
            source_model (Union[str, generative_models.GenerativeModel]):
                A GenAI model name or a tuned model name or a GenerativeModel instance
                for batch prediction.
                Supported formats for model name: "gemini-1.0-pro",
                "models/gemini-1.0-pro", and "publishers/google/models/gemini-1.0-pro"
                Supported formats for tuned model name: "789" and
                "projects/123/locations/456/models/789"
            input_dataset (Union[str,List[str]]):
                GCS URI(-s) or BigQuery URI to your input data to run batch
                prediction on. Example: "gs://path/to/input/data.jsonl" or
                "bq://projectId.bqDatasetId.bqTableId"
            output_uri_prefix (str):
                GCS or BigQuery URI prefix for the output predictions. Example:
                "gs://path/to/output/data" or "bq://projectId.bqDatasetId"
                If not specified, f"{STAGING_BUCKET}/gen-ai-batch-prediction" will
                be used for GCS source and
                f"bq://projectId.gen_ai_batch_prediction.predictions_{TIMESTAMP}"
                will be used for BigQuery source.
            job_display_name (str):
                The user-defined name of the BatchPredictionJob.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            machine_type (str):
                The type of machine for running batch prediction job.
            accelerator_type (str):
                The type of accelerator for running batch prediction job.
            accelerator_count (int):
                The number of accelerators for running batch prediction job.
            starting_replica_count (int):
                The starting number of replica for running batch prediction job.
            max_replica_count (int):
                The maximum number of replica for running batch prediction job.

        Returns:
            Instantiated BatchPredictionJob.

        Raises:
            ValueError: If source_model is not a GenAI model.
            Or if input_dataset or output_uri_prefix are not in supported formats.
            Or if output_uri_prefix is not specified and staging_bucket is not
            set in vertexai.init().
        """
        # Handle model name
        model_name = cls._reconcile_model_name(
            source_model._model_name
            if isinstance(source_model, generative_models.GenerativeModel)
            else source_model
        )
        if not cls._is_genai_model(model_name):
            raise ValueError(f"Model '{model_name}' is not a Generative AI model.")

        # Handle input URI
        gcs_source = None
        bigquery_source = None
        first_input_uri = (
            input_dataset if isinstance(input_dataset, str) else input_dataset[0]
        )
        if first_input_uri.startswith("gs://"):
            gcs_source = input_dataset
        elif first_input_uri.startswith("bq://"):
            if not isinstance(input_dataset, str):
                raise ValueError("Multiple BigQuery input datasets are not supported.")
            bigquery_source = input_dataset
        else:
            raise ValueError(
                f"Unsupported input URI: {input_dataset}. "
                "Supported formats: 'gs://path/to/input/data.jsonl' and "
                "'bq://projectId.bqDatasetId.bqTableId'"
            )

        # Handle output URI
        gcs_destination_prefix = None
        bigquery_destination_prefix = None
        if output_uri_prefix:
            if output_uri_prefix.startswith("gs://"):
                gcs_destination_prefix = output_uri_prefix
            elif output_uri_prefix.startswith("bq://"):
                # Temporarily handle this in SDK, will remove once b/338423462 is fixed.
                bigquery_destination_prefix = cls._complete_bq_uri(output_uri_prefix)
            else:
                raise ValueError(
                    f"Unsupported output URI: {output_uri_prefix}. "
                    "Supported formats: 'gs://path/to/output/data' and "
                    "'bq://projectId.bqDatasetId'"
                )
        else:
            if first_input_uri.startswith("gs://"):
                if not aiplatform_initializer.global_config.staging_bucket:
                    raise ValueError(
                        "Please either specify output_uri_prefix or "
                        "set staging_bucket in vertexai.init()."
                    )
                gcs_destination_prefix = (
                    aiplatform_initializer.global_config.staging_bucket.rstrip("/")
                    + "/gen-ai-batch-prediction"
                )
            else:
                bigquery_destination_prefix = cls._complete_bq_uri()

        # Reuse aiplatform class to submit the job (override _LOGGER)
        logging.getLogger("google.cloud.aiplatform.jobs").disabled = True
        try:
            aiplatform_job = jobs.BatchPredictionJob.submit(
                model_name=model_name,
                job_display_name=job_display_name,
                gcs_source=gcs_source,
                bigquery_source=bigquery_source,
                gcs_destination_prefix=gcs_destination_prefix,
                bigquery_destination_prefix=bigquery_destination_prefix,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                starting_replica_count=starting_replica_count,
                max_replica_count=max_replica_count,
            )
            job = cls._empty_constructor()
            job._gca_resource = aiplatform_job._gca_resource

            _LOGGER.log_create_complete(
                cls, job._gca_resource, "job", module_name="batch_prediction"
            )
            _LOGGER.info("View Batch Prediction Job:\n%s" % job._dashboard_uri())

            return job
        finally:
            logging.getLogger("google.cloud.aiplatform.jobs").disabled = False

    def refresh(self) -> "BatchPredictionJob":
        """Refreshes the batch prediction job from the service."""
        self._sync_gca_resource()
        return self

    def cancel(self):
        """Cancels this BatchPredictionJob.

        Success of cancellation is not guaranteed. Use `job.refresh()` and
        `job.state` to verify if cancellation was successful.
        """
        _LOGGER.log_action_start_against_resource("Cancelling", "run", self)
        self.api_client.cancel_batch_prediction_job(name=self.resource_name)

    def delete(self):
        """Deletes this BatchPredictionJob resource.

        WARNING: This deletion is permanent.
        """
        self._delete()

    @classmethod
    def list(cls, filter=None) -> List["BatchPredictionJob"]:
        """Lists all BatchPredictionJob instances that run with GenAI models."""
        return cls._list(
            cls_filter=lambda gca_resource: cls._is_genai_model(gca_resource.model),
            filter=filter,
        )

    def _dashboard_uri(self) -> Optional[str]:
        """Returns the Google Cloud console URL where job can be viewed."""
        fields = self._parse_resource_name(self.resource_name)
        location = fields.pop("location")
        project = fields.pop("project")
        job = list(fields.values())[0]
        return (
            "https://console.cloud.google.com/ai/platform/locations/"
            f"{location}/{self._job_type}/{job}?project={project}"
        )

    @classmethod
    def _reconcile_model_name(cls, model_name: str) -> str:
        """Reconciles model name to a publisher model resource name or a tuned model resource name."""
        if not model_name:
            raise ValueError("model_name must not be empty")

        if "/" not in model_name:
            # model name (e.g., gemini-1.0-pro)
            if model_name.startswith("gemini"):
                return "publishers/google/models/" + model_name
            else:
                raise ValueError(
                    "Abbreviated model names are only supported for Gemini models. "
                    "Please provide the full publisher model name."
                )
        elif model_name.startswith("models/"):
            # publisher model name (e.g., models/gemini-1.0-pro)
            return "publishers/google/" + model_name
        elif (
            re.match(
                r"^publishers/(?P<publisher>[^/]+)/models/(?P<model>[^@]+)@(?P<version>[^@]+)$",
                model_name,
            )
            or model_name.startswith("publishers/google/models/")
            or model_name.startswith("publishers/meta/models/")
            or model_name.startswith("publishers/anthropic/models/")
            or model_name.startswith("publishers/openai/models/")
            or model_name.startswith("publishers/qwen/models/")
            or model_name.startswith("publishers/deepseek-ai/models/")
            or model_name.startswith("publishers/intfloat/models/")
            or re.search(_GEMINI_TUNED_MODEL_PATTERN, model_name)
        ):
            return model_name
        else:
            raise ValueError(f"Invalid format for model name: {model_name}.")

    @classmethod
    def _is_genai_model(cls, model_name: str) -> bool:
        """Validates if a given model_name represents a GenAI model."""
        if re.search(_GEMINI_MODEL_PATTERN, model_name):
            # Model is a Gemini model.
            return True

        if re.search(_GEMINI_TUNED_MODEL_PATTERN, model_name):
            model = models.Model(model_name)
            if (
                model.gca_resource.model_source_info.source_type
                == gca_types.model.ModelSourceInfo.ModelSourceType.GENIE
            ):
                # Model is a tuned Gemini model.
                return True

        if re.search(_LLAMA_MODEL_PATTERN, model_name):
            # Model is a Llama3 model.
            return True

        if re.search(_CLAUDE_MODEL_PATTERN, model_name):
            # Model is a claude model.
            return True

        if re.search(_GPT_MODEL_PATTERN, model_name):
            # Model is a GPT model.
            return True

        if re.search(_QWEN_MODEL_PATTERN, model_name):
            # Model is a Qwen model.
            return True

        if re.search(_DEEPSEEK_MODEL_PATTERN, model_name):
            # Model is a DeepSeek model.
            return True

        if re.search(_E5_MODEL_PATTERN, model_name):
            # Model is an E5 model.
            return True

        if re.match(
            r"^publishers/(?P<publisher>[^/]+)/models/(?P<model>[^@]+)@(?P<version>[^@]+)$",
            model_name,
        ):
            # Model is a self-hosted model.
            return True

        return False

    @classmethod
    def num_pending_jobs(cls) -> int:
        """Returns the number of pending batch prediction jobs.

        The pending jobs are those defined in _JOB_PENDING_STATES from
        google/cloud/aiplatform/jobs.py
        e.g. JOB_STATE_QUEUED, JOB_STATE_PENDING, JOB_STATE_RUNNING,
        JOB_STATE_CANCELLING, JOB_STATE_UPDATING.
        It will be used to manage the number of concurrent batch that is limited
        according to
        https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#concurrent-batch-requests
        """
        return len(
            cls._list(
                cls_filter=lambda gca_resource: cls._is_genai_model(gca_resource.model),
                filter=" OR ".join(
                    f'state="{pending_state.name}"'
                    for pending_state in jobs._JOB_PENDING_STATES
                ),
            )
        )

    @classmethod
    def _complete_bq_uri(cls, uri: Optional[str] = None):
        """Completes a BigQuery uri to a BigQuery table uri."""
        uri_parts = uri.split(".") if uri else []
        uri_len = len(uri_parts)
        if len(uri_parts) > 3:
            raise ValueError(
                f"Invalid URI: {uri}. "
                "Supported formats: 'bq://projectId.bqDatasetId.bqTableId'"
            )

        schema_and_project = (
            uri_parts[0]
            if uri_len >= 1
            else f"bq://{aiplatform_initializer.global_config.project}"
        )
        if not schema_and_project.startswith("bq://"):
            raise ValueError("URI must start with 'bq://'")

        dataset = uri_parts[1] if uri_len >= 2 else "gen_ai_batch_prediction"

        table = (
            uri_parts[2]
            if uri_len >= 3
            else f"predictions_{aiplatform_utils.timestamped_unique_name()}"
        )

        return f"{schema_and_project}.{dataset}.{table}"
