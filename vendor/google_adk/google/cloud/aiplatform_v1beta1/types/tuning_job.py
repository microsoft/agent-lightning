# -*- coding: utf-8 -*-
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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.aiplatform_v1beta1.types import content
from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1beta1.types import evaluation_service
from google.cloud.aiplatform_v1beta1.types import job_state
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "TuningJob",
        "TunedModel",
        "SupervisedTuningDatasetDistribution",
        "SupervisedTuningDataStats",
        "DatasetDistribution",
        "DatasetStats",
        "DistillationDataStats",
        "TuningDataStats",
        "SupervisedHyperParameters",
        "SupervisedTuningSpec",
        "DistillationSpec",
        "DistillationHyperParameters",
        "PartnerModelTuningSpec",
        "TunedModelRef",
        "VeoHyperParameters",
        "VeoTuningSpec",
        "EvaluationConfig",
        "EvaluateDatasetRun",
        "TunedModelCheckpoint",
        "PreTunedModel",
    },
)


class TuningJob(proto.Message):
    r"""Represents a TuningJob that runs with Google owned models.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        base_model (str):
            The base model that is being tuned. See `Supported
            models <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning#supported_models>`__.

            This field is a member of `oneof`_ ``source_model``.
        pre_tuned_model (google.cloud.aiplatform_v1beta1.types.PreTunedModel):
            The pre-tuned model for continuous tuning.

            This field is a member of `oneof`_ ``source_model``.
        supervised_tuning_spec (google.cloud.aiplatform_v1beta1.types.SupervisedTuningSpec):
            Tuning Spec for Supervised Fine Tuning.

            This field is a member of `oneof`_ ``tuning_spec``.
        distillation_spec (google.cloud.aiplatform_v1beta1.types.DistillationSpec):
            Tuning Spec for Distillation.

            This field is a member of `oneof`_ ``tuning_spec``.
        partner_model_tuning_spec (google.cloud.aiplatform_v1beta1.types.PartnerModelTuningSpec):
            Tuning Spec for open sourced and third party
            Partner models.

            This field is a member of `oneof`_ ``tuning_spec``.
        veo_tuning_spec (google.cloud.aiplatform_v1beta1.types.VeoTuningSpec):
            Tuning Spec for Veo Tuning.

            This field is a member of `oneof`_ ``tuning_spec``.
        name (str):
            Output only. Identifier. Resource name of a TuningJob.
            Format:
            ``projects/{project}/locations/{location}/tuningJobs/{tuning_job}``
        tuned_model_display_name (str):
            Optional. The display name of the
            [TunedModel][google.cloud.aiplatform.v1.Model]. The name can
            be up to 128 characters long and can consist of any UTF-8
            characters.
        description (str):
            Optional. The description of the
            [TuningJob][google.cloud.aiplatform.v1.TuningJob].
        custom_base_model (str):
            Optional. The user-provided path to custom model weights.
            Set this field to tune a custom model. The path must be a
            Cloud Storage directory that contains the model weights in
            .safetensors format along with associated model metadata
            files. If this field is set, the base_model field must still
            be set to indicate which base model the custom model is
            derived from. This feature is only available for open source
            models.
        state (google.cloud.aiplatform_v1beta1.types.JobState):
            Output only. The detailed state of the job.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the
            [TuningJob][google.cloud.aiplatform.v1.TuningJob] was
            created.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the
            [TuningJob][google.cloud.aiplatform.v1.TuningJob] for the
            first time entered the ``JOB_STATE_RUNNING`` state.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the TuningJob entered any of the
            following [JobStates][google.cloud.aiplatform.v1.JobState]:
            ``JOB_STATE_SUCCEEDED``, ``JOB_STATE_FAILED``,
            ``JOB_STATE_CANCELLED``, ``JOB_STATE_EXPIRED``.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the
            [TuningJob][google.cloud.aiplatform.v1.TuningJob] was most
            recently updated.
        error (google.rpc.status_pb2.Status):
            Output only. Only populated when job's state is
            ``JOB_STATE_FAILED`` or ``JOB_STATE_CANCELLED``.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined metadata to organize
            [TuningJob][google.cloud.aiplatform.v1.TuningJob] and
            generated resources such as
            [Model][google.cloud.aiplatform.v1.Model] and
            [Endpoint][google.cloud.aiplatform.v1.Endpoint].

            Label keys and values can be no longer than 64 characters
            (Unicode codepoints), can only contain lowercase letters,
            numeric characters, underscores and dashes. International
            characters are allowed.

            See https://goo.gl/xmQnxf for more information and examples
            of labels.
        experiment (str):
            Output only. The Experiment associated with this
            [TuningJob][google.cloud.aiplatform.v1.TuningJob].
        tuned_model (google.cloud.aiplatform_v1beta1.types.TunedModel):
            Output only. The tuned model resources associated with this
            [TuningJob][google.cloud.aiplatform.v1.TuningJob].
        tuning_data_stats (google.cloud.aiplatform_v1beta1.types.TuningDataStats):
            Output only. The tuning data statistics associated with this
            [TuningJob][google.cloud.aiplatform.v1.TuningJob].
        pipeline_job (str):
            Output only. The resource name of the PipelineJob associated
            with the [TuningJob][google.cloud.aiplatform.v1.TuningJob].
            Format:
            ``projects/{project}/locations/{location}/pipelineJobs/{pipeline_job}``.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Customer-managed encryption key options for a
            TuningJob. If this is set, then all resources
            created by the TuningJob will be encrypted with
            the provided encryption key.
        service_account (str):
            The service account that the tuningJob workload runs as. If
            not specified, the Vertex AI Secure Fine-Tuned Service Agent
            in the project will be used. See
            https://cloud.google.com/iam/docs/service-agents#vertex-ai-secure-fine-tuning-service-agent

            Users starting the pipeline must have the
            ``iam.serviceAccounts.actAs`` permission on this service
            account.
        output_uri (str):
            Optional. Cloud Storage path to the directory
            where tuning job outputs are written to. This
            field is only available and required for open
            source models.
        evaluate_dataset_runs (MutableSequence[google.cloud.aiplatform_v1beta1.types.EvaluateDatasetRun]):
            Output only. Evaluation runs for the Tuning
            Job.
    """

    base_model: str = proto.Field(
        proto.STRING,
        number=4,
        oneof="source_model",
    )
    pre_tuned_model: "PreTunedModel" = proto.Field(
        proto.MESSAGE,
        number=31,
        oneof="source_model",
        message="PreTunedModel",
    )
    supervised_tuning_spec: "SupervisedTuningSpec" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="tuning_spec",
        message="SupervisedTuningSpec",
    )
    distillation_spec: "DistillationSpec" = proto.Field(
        proto.MESSAGE,
        number=17,
        oneof="tuning_spec",
        message="DistillationSpec",
    )
    partner_model_tuning_spec: "PartnerModelTuningSpec" = proto.Field(
        proto.MESSAGE,
        number=21,
        oneof="tuning_spec",
        message="PartnerModelTuningSpec",
    )
    veo_tuning_spec: "VeoTuningSpec" = proto.Field(
        proto.MESSAGE,
        number=33,
        oneof="tuning_spec",
        message="VeoTuningSpec",
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tuned_model_display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    custom_base_model: str = proto.Field(
        proto.STRING,
        number=26,
    )
    state: job_state.JobState = proto.Field(
        proto.ENUM,
        number=6,
        enum=job_state.JobState,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=10,
        message=timestamp_pb2.Timestamp,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=11,
        message=status_pb2.Status,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=12,
    )
    experiment: str = proto.Field(
        proto.STRING,
        number=13,
    )
    tuned_model: "TunedModel" = proto.Field(
        proto.MESSAGE,
        number=14,
        message="TunedModel",
    )
    tuning_data_stats: "TuningDataStats" = proto.Field(
        proto.MESSAGE,
        number=15,
        message="TuningDataStats",
    )
    pipeline_job: str = proto.Field(
        proto.STRING,
        number=18,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=16,
        message=gca_encryption_spec.EncryptionSpec,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=22,
    )
    output_uri: str = proto.Field(
        proto.STRING,
        number=25,
    )
    evaluate_dataset_runs: MutableSequence["EvaluateDatasetRun"] = proto.RepeatedField(
        proto.MESSAGE,
        number=32,
        message="EvaluateDatasetRun",
    )


class TunedModel(proto.Message):
    r"""The Model Registry Model and Online Prediction Endpoint associated
    with this [TuningJob][google.cloud.aiplatform.v1.TuningJob].

    Attributes:
        model (str):
            Output only. The resource name of the TunedModel. Format:

            ``projects/{project}/locations/{location}/models/{model}@{version_id}``

            When tuning from a base model, the version_id will be 1.

            For continuous tuning, the version id will be incremented by
            1 from the last version id in the parent model. E.g.,
            ``projects/{project}/locations/{location}/models/{model}@{last_version_id + 1}``
        endpoint (str):
            Output only. A resource name of an Endpoint. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``.
        checkpoints (MutableSequence[google.cloud.aiplatform_v1beta1.types.TunedModelCheckpoint]):
            Output only. The checkpoints associated with
            this TunedModel. This field is only populated
            for tuning jobs that enable intermediate
            checkpoints.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    endpoint: str = proto.Field(
        proto.STRING,
        number=2,
    )
    checkpoints: MutableSequence["TunedModelCheckpoint"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="TunedModelCheckpoint",
    )


class SupervisedTuningDatasetDistribution(proto.Message):
    r"""Dataset distribution for Supervised Tuning.

    Attributes:
        sum (int):
            Output only. Sum of a given population of
            values.
        billable_sum (int):
            Output only. Sum of a given population of
            values that are billable.
        min_ (float):
            Output only. The minimum of the population
            values.
        max_ (float):
            Output only. The maximum of the population
            values.
        mean (float):
            Output only. The arithmetic mean of the
            values in the population.
        median (float):
            Output only. The median of the values in the
            population.
        p5 (float):
            Output only. The 5th percentile of the values
            in the population.
        p95 (float):
            Output only. The 95th percentile of the
            values in the population.
        buckets (MutableSequence[google.cloud.aiplatform_v1beta1.types.SupervisedTuningDatasetDistribution.DatasetBucket]):
            Output only. Defines the histogram bucket.
    """

    class DatasetBucket(proto.Message):
        r"""Dataset bucket used to create a histogram for the
        distribution given a population of values.

        Attributes:
            count (float):
                Output only. Number of values in the bucket.
            left (float):
                Output only. Left bound of the bucket.
            right (float):
                Output only. Right bound of the bucket.
        """

        count: float = proto.Field(
            proto.DOUBLE,
            number=1,
        )
        left: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )
        right: float = proto.Field(
            proto.DOUBLE,
            number=3,
        )

    sum: int = proto.Field(
        proto.INT64,
        number=1,
    )
    billable_sum: int = proto.Field(
        proto.INT64,
        number=9,
    )
    min_: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    max_: float = proto.Field(
        proto.DOUBLE,
        number=3,
    )
    mean: float = proto.Field(
        proto.DOUBLE,
        number=4,
    )
    median: float = proto.Field(
        proto.DOUBLE,
        number=5,
    )
    p5: float = proto.Field(
        proto.DOUBLE,
        number=6,
    )
    p95: float = proto.Field(
        proto.DOUBLE,
        number=7,
    )
    buckets: MutableSequence[DatasetBucket] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message=DatasetBucket,
    )


class SupervisedTuningDataStats(proto.Message):
    r"""Tuning data statistics for Supervised Tuning.

    Attributes:
        tuning_dataset_example_count (int):
            Output only. Number of examples in the tuning
            dataset.
        total_tuning_character_count (int):
            Output only. Number of tuning characters in
            the tuning dataset.
        total_billable_character_count (int):
            Output only. Number of billable characters in
            the tuning dataset.
        total_billable_token_count (int):
            Output only. Number of billable tokens in the
            tuning dataset.
        tuning_step_count (int):
            Output only. Number of tuning steps for this
            Tuning Job.
        user_input_token_distribution (google.cloud.aiplatform_v1beta1.types.SupervisedTuningDatasetDistribution):
            Output only. Dataset distributions for the
            user input tokens.
        user_output_token_distribution (google.cloud.aiplatform_v1beta1.types.SupervisedTuningDatasetDistribution):
            Output only. Dataset distributions for the
            user output tokens.
        user_message_per_example_distribution (google.cloud.aiplatform_v1beta1.types.SupervisedTuningDatasetDistribution):
            Output only. Dataset distributions for the
            messages per example.
        user_dataset_examples (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
            Output only. Sample user messages in the
            training dataset uri.
        total_truncated_example_count (int):
            Output only. The number of examples in the
            dataset that have been dropped. An example can
            be dropped for reasons including: too many
            tokens, contains an invalid image, contains too
            many images, etc.
        truncated_example_indices (MutableSequence[int]):
            Output only. A partial sample of the indices
            (starting from 1) of the dropped examples.
        dropped_example_reasons (MutableSequence[str]):
            Output only. For each index in
            ``truncated_example_indices``, the user-facing reason why
            the example was dropped.
    """

    tuning_dataset_example_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    total_tuning_character_count: int = proto.Field(
        proto.INT64,
        number=2,
    )
    total_billable_character_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    total_billable_token_count: int = proto.Field(
        proto.INT64,
        number=9,
    )
    tuning_step_count: int = proto.Field(
        proto.INT64,
        number=4,
    )
    user_input_token_distribution: "SupervisedTuningDatasetDistribution" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="SupervisedTuningDatasetDistribution",
    )
    user_output_token_distribution: "SupervisedTuningDatasetDistribution" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="SupervisedTuningDatasetDistribution",
    )
    user_message_per_example_distribution: "SupervisedTuningDatasetDistribution" = (
        proto.Field(
            proto.MESSAGE,
            number=7,
            message="SupervisedTuningDatasetDistribution",
        )
    )
    user_dataset_examples: MutableSequence[content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message=content.Content,
    )
    total_truncated_example_count: int = proto.Field(
        proto.INT64,
        number=10,
    )
    truncated_example_indices: MutableSequence[int] = proto.RepeatedField(
        proto.INT64,
        number=11,
    )
    dropped_example_reasons: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=12,
    )


class DatasetDistribution(proto.Message):
    r"""Distribution computed over a tuning dataset.

    Attributes:
        sum (float):
            Output only. Sum of a given population of
            values.
        min_ (float):
            Output only. The minimum of the population
            values.
        max_ (float):
            Output only. The maximum of the population
            values.
        mean (float):
            Output only. The arithmetic mean of the
            values in the population.
        median (float):
            Output only. The median of the values in the
            population.
        p5 (float):
            Output only. The 5th percentile of the values
            in the population.
        p95 (float):
            Output only. The 95th percentile of the
            values in the population.
        buckets (MutableSequence[google.cloud.aiplatform_v1beta1.types.DatasetDistribution.DistributionBucket]):
            Output only. Defines the histogram bucket.
    """

    class DistributionBucket(proto.Message):
        r"""Dataset bucket used to create a histogram for the
        distribution given a population of values.

        Attributes:
            count (int):
                Output only. Number of values in the bucket.
            left (float):
                Output only. Left bound of the bucket.
            right (float):
                Output only. Right bound of the bucket.
        """

        count: int = proto.Field(
            proto.INT64,
            number=1,
        )
        left: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )
        right: float = proto.Field(
            proto.DOUBLE,
            number=3,
        )

    sum: float = proto.Field(
        proto.DOUBLE,
        number=1,
    )
    min_: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    max_: float = proto.Field(
        proto.DOUBLE,
        number=3,
    )
    mean: float = proto.Field(
        proto.DOUBLE,
        number=4,
    )
    median: float = proto.Field(
        proto.DOUBLE,
        number=5,
    )
    p5: float = proto.Field(
        proto.DOUBLE,
        number=6,
    )
    p95: float = proto.Field(
        proto.DOUBLE,
        number=7,
    )
    buckets: MutableSequence[DistributionBucket] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message=DistributionBucket,
    )


class DatasetStats(proto.Message):
    r"""Statistics computed over a tuning dataset.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tuning_dataset_example_count (int):
            Output only. Number of examples in the tuning
            dataset.
        total_tuning_character_count (int):
            Output only. Number of tuning characters in
            the tuning dataset.
        total_billable_character_count (int):
            Output only. Number of billable characters in
            the tuning dataset.
        tuning_step_count (int):
            Output only. Number of tuning steps for this
            Tuning Job.
        user_input_token_distribution (google.cloud.aiplatform_v1beta1.types.DatasetDistribution):
            Output only. Dataset distributions for the
            user input tokens.
        user_output_token_distribution (google.cloud.aiplatform_v1beta1.types.DatasetDistribution):
            Output only. Dataset distributions for the
            user output tokens.

            This field is a member of `oneof`_ ``_user_output_token_distribution``.
        user_message_per_example_distribution (google.cloud.aiplatform_v1beta1.types.DatasetDistribution):
            Output only. Dataset distributions for the
            messages per example.
        user_dataset_examples (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
            Output only. Sample user messages in the
            training dataset uri.
    """

    tuning_dataset_example_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    total_tuning_character_count: int = proto.Field(
        proto.INT64,
        number=2,
    )
    total_billable_character_count: int = proto.Field(
        proto.INT64,
        number=3,
    )
    tuning_step_count: int = proto.Field(
        proto.INT64,
        number=4,
    )
    user_input_token_distribution: "DatasetDistribution" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="DatasetDistribution",
    )
    user_output_token_distribution: "DatasetDistribution" = proto.Field(
        proto.MESSAGE,
        number=6,
        optional=True,
        message="DatasetDistribution",
    )
    user_message_per_example_distribution: "DatasetDistribution" = proto.Field(
        proto.MESSAGE,
        number=7,
        message="DatasetDistribution",
    )
    user_dataset_examples: MutableSequence[content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message=content.Content,
    )


class DistillationDataStats(proto.Message):
    r"""Statistics computed for datasets used for distillation.

    Attributes:
        training_dataset_stats (google.cloud.aiplatform_v1beta1.types.DatasetStats):
            Output only. Statistics computed for the
            training dataset.
    """

    training_dataset_stats: "DatasetStats" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="DatasetStats",
    )


class TuningDataStats(proto.Message):
    r"""The tuning data statistic values for
    [TuningJob][google.cloud.aiplatform.v1.TuningJob].

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        supervised_tuning_data_stats (google.cloud.aiplatform_v1beta1.types.SupervisedTuningDataStats):
            The SFT Tuning data stats.

            This field is a member of `oneof`_ ``tuning_data_stats``.
        distillation_data_stats (google.cloud.aiplatform_v1beta1.types.DistillationDataStats):
            Output only. Statistics for distillation.

            This field is a member of `oneof`_ ``tuning_data_stats``.
    """

    supervised_tuning_data_stats: "SupervisedTuningDataStats" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="tuning_data_stats",
        message="SupervisedTuningDataStats",
    )
    distillation_data_stats: "DistillationDataStats" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="tuning_data_stats",
        message="DistillationDataStats",
    )


class SupervisedHyperParameters(proto.Message):
    r"""Hyperparameters for SFT.

    Attributes:
        epoch_count (int):
            Optional. Number of complete passes the model
            makes over the entire training dataset during
            training.
        learning_rate_multiplier (float):
            Optional. Multiplier for adjusting the default learning
            rate. Mutually exclusive with ``learning_rate``.
        learning_rate (float):
            Optional. Learning rate for tuning. Mutually exclusive with
            ``learning_rate_multiplier``. This feature is only available
            for open source models.
        adapter_size (google.cloud.aiplatform_v1beta1.types.SupervisedHyperParameters.AdapterSize):
            Optional. Adapter size for tuning.
        batch_size (int):
            Optional. Batch size for tuning.
            This feature is only available for open source
            models.
    """

    class AdapterSize(proto.Enum):
        r"""Supported adapter sizes for tuning.

        Values:
            ADAPTER_SIZE_UNSPECIFIED (0):
                Adapter size is unspecified.
            ADAPTER_SIZE_ONE (1):
                Adapter size 1.
            ADAPTER_SIZE_TWO (6):
                Adapter size 2.
            ADAPTER_SIZE_FOUR (2):
                Adapter size 4.
            ADAPTER_SIZE_EIGHT (3):
                Adapter size 8.
            ADAPTER_SIZE_SIXTEEN (4):
                Adapter size 16.
            ADAPTER_SIZE_THIRTY_TWO (5):
                Adapter size 32.
        """

        ADAPTER_SIZE_UNSPECIFIED = 0
        ADAPTER_SIZE_ONE = 1
        ADAPTER_SIZE_TWO = 6
        ADAPTER_SIZE_FOUR = 2
        ADAPTER_SIZE_EIGHT = 3
        ADAPTER_SIZE_SIXTEEN = 4
        ADAPTER_SIZE_THIRTY_TWO = 5

    epoch_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    learning_rate_multiplier: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    learning_rate: float = proto.Field(
        proto.DOUBLE,
        number=6,
    )
    adapter_size: AdapterSize = proto.Field(
        proto.ENUM,
        number=3,
        enum=AdapterSize,
    )
    batch_size: int = proto.Field(
        proto.INT64,
        number=5,
    )


class SupervisedTuningSpec(proto.Message):
    r"""Tuning Spec for Supervised Tuning for first party models.

    Attributes:
        training_dataset_uri (str):
            Required. Training dataset used for tuning.
            The dataset can be specified as either a Cloud
            Storage path to a JSONL file or as the resource
            name of a Vertex Multimodal Dataset.
        validation_dataset_uri (str):
            Optional. Validation dataset used for tuning.
            The dataset can be specified as either a Cloud
            Storage path to a JSONL file or as the resource
            name of a Vertex Multimodal Dataset.
        hyper_parameters (google.cloud.aiplatform_v1beta1.types.SupervisedHyperParameters):
            Optional. Hyperparameters for SFT.
        export_last_checkpoint_only (bool):
            Optional. If set to true, disable
            intermediate checkpoints for SFT and only the
            last checkpoint will be exported. Otherwise,
            enable intermediate checkpoints for SFT. Default
            is false.
        evaluation_config (google.cloud.aiplatform_v1beta1.types.EvaluationConfig):
            Optional. Evaluation Config for Tuning Job.
        tuning_mode (google.cloud.aiplatform_v1beta1.types.SupervisedTuningSpec.TuningMode):
            Tuning mode.
    """

    class TuningMode(proto.Enum):
        r"""Supported tuning modes.

        Values:
            TUNING_MODE_UNSPECIFIED (0):
                Tuning mode is unspecified.
            TUNING_MODE_FULL (1):
                Full fine-tuning mode.
            TUNING_MODE_PEFT_ADAPTER (2):
                PEFT adapter tuning mode.
        """

        TUNING_MODE_UNSPECIFIED = 0
        TUNING_MODE_FULL = 1
        TUNING_MODE_PEFT_ADAPTER = 2

    training_dataset_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    validation_dataset_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    hyper_parameters: "SupervisedHyperParameters" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="SupervisedHyperParameters",
    )
    export_last_checkpoint_only: bool = proto.Field(
        proto.BOOL,
        number=6,
    )
    evaluation_config: "EvaluationConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="EvaluationConfig",
    )
    tuning_mode: TuningMode = proto.Field(
        proto.ENUM,
        number=7,
        enum=TuningMode,
    )


class DistillationSpec(proto.Message):
    r"""Tuning Spec for Distillation.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        base_teacher_model (str):
            The base teacher model that is being distilled. See
            `Supported
            models <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning#supported_models>`__.

            This field is a member of `oneof`_ ``teacher_model``.
        tuned_teacher_model_source (str):
            The resource name of the Tuned teacher model. Format:
            ``projects/{project}/locations/{location}/models/{model}``.

            This field is a member of `oneof`_ ``teacher_model``.
        training_dataset_uri (str):
            Required. Cloud Storage path to file
            containing training dataset for tuning. The
            dataset must be formatted as a JSONL file.
        validation_dataset_uri (str):
            Optional. Cloud Storage path to file
            containing validation dataset for tuning. The
            dataset must be formatted as a JSONL file.

            This field is a member of `oneof`_ ``_validation_dataset_uri``.
        hyper_parameters (google.cloud.aiplatform_v1beta1.types.DistillationHyperParameters):
            Optional. Hyperparameters for Distillation.
        student_model (str):
            The student model that is being tuned, e.g.,
            "google/gemma-2b-1.1-it".
        pipeline_root_directory (str):
            Required. A path in a Cloud Storage bucket,
            which will be treated as the root output
            directory of the distillation pipeline. It is
            used by the system to generate the paths of
            output artifacts.
    """

    base_teacher_model: str = proto.Field(
        proto.STRING,
        number=5,
        oneof="teacher_model",
    )
    tuned_teacher_model_source: str = proto.Field(
        proto.STRING,
        number=6,
        oneof="teacher_model",
    )
    training_dataset_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    validation_dataset_uri: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    hyper_parameters: "DistillationHyperParameters" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="DistillationHyperParameters",
    )
    student_model: str = proto.Field(
        proto.STRING,
        number=4,
    )
    pipeline_root_directory: str = proto.Field(
        proto.STRING,
        number=7,
    )


class DistillationHyperParameters(proto.Message):
    r"""Hyperparameters for Distillation.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        epoch_count (int):
            Optional. Number of complete passes the model
            makes over the entire training dataset during
            training.

            This field is a member of `oneof`_ ``_epoch_count``.
        learning_rate_multiplier (float):
            Optional. Multiplier for adjusting the
            default learning rate.

            This field is a member of `oneof`_ ``_learning_rate_multiplier``.
        adapter_size (google.cloud.aiplatform_v1beta1.types.SupervisedHyperParameters.AdapterSize):
            Optional. Adapter size for distillation.
    """

    epoch_count: int = proto.Field(
        proto.INT64,
        number=1,
        optional=True,
    )
    learning_rate_multiplier: float = proto.Field(
        proto.DOUBLE,
        number=2,
        optional=True,
    )
    adapter_size: "SupervisedHyperParameters.AdapterSize" = proto.Field(
        proto.ENUM,
        number=3,
        enum="SupervisedHyperParameters.AdapterSize",
    )


class PartnerModelTuningSpec(proto.Message):
    r"""Tuning spec for Partner models.

    Attributes:
        training_dataset_uri (str):
            Required. Cloud Storage path to file
            containing training dataset for tuning. The
            dataset must be formatted as a JSONL file.
        validation_dataset_uri (str):
            Optional. Cloud Storage path to file
            containing validation dataset for tuning. The
            dataset must be formatted as a JSONL file.
        hyper_parameters (MutableMapping[str, google.protobuf.struct_pb2.Value]):
            Hyperparameters for tuning. The accepted hyper_parameters
            and their valid range of values will differ depending on the
            base model.
    """

    training_dataset_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    validation_dataset_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    hyper_parameters: MutableMapping[str, struct_pb2.Value] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Value,
    )


class TunedModelRef(proto.Message):
    r"""TunedModel Reference for legacy model migration.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tuned_model (str):
            Support migration from model registry.

            This field is a member of `oneof`_ ``tuned_model_ref``.
        tuning_job (str):
            Support migration from tuning job list page,
            from gemini-1.0-pro-002 to 1.5 and above.

            This field is a member of `oneof`_ ``tuned_model_ref``.
        pipeline_job (str):
            Support migration from tuning job list page,
            from bison model to gemini model.

            This field is a member of `oneof`_ ``tuned_model_ref``.
    """

    tuned_model: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="tuned_model_ref",
    )
    tuning_job: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="tuned_model_ref",
    )
    pipeline_job: str = proto.Field(
        proto.STRING,
        number=3,
        oneof="tuned_model_ref",
    )


class VeoHyperParameters(proto.Message):
    r"""Hyperparameters for Veo.

    Attributes:
        epoch_count (int):
            Optional. Number of complete passes the model
            makes over the entire training dataset during
            training.
        learning_rate_multiplier (float):
            Optional. Multiplier for adjusting the
            default learning rate.
        tuning_task (google.cloud.aiplatform_v1beta1.types.VeoHyperParameters.TuningTask):
            Optional. The tuning task. Either I2V or T2V.
    """

    class TuningTask(proto.Enum):
        r"""An enum defining the tuning task used for Veo.

        Values:
            TUNING_TASK_UNSPECIFIED (0):
                Default value. This value is unused.
            TUNING_TASK_I2V (1):
                Tuning task for image to video.
            TUNING_TASK_T2V (2):
                Tuning task for text to video.
        """

        TUNING_TASK_UNSPECIFIED = 0
        TUNING_TASK_I2V = 1
        TUNING_TASK_T2V = 2

    epoch_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    learning_rate_multiplier: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    tuning_task: TuningTask = proto.Field(
        proto.ENUM,
        number=3,
        enum=TuningTask,
    )


class VeoTuningSpec(proto.Message):
    r"""Tuning Spec for Veo Model Tuning.

    Attributes:
        training_dataset_uri (str):
            Required. Training dataset used for tuning.
            The dataset can be specified as either a Cloud
            Storage path to a JSONL file or as the resource
            name of a Vertex Multimodal Dataset.
        validation_dataset_uri (str):
            Optional. Validation dataset used for tuning.
            The dataset can be specified as either a Cloud
            Storage path to a JSONL file or as the resource
            name of a Vertex Multimodal Dataset.
        hyper_parameters (google.cloud.aiplatform_v1beta1.types.VeoHyperParameters):
            Optional. Hyperparameters for Veo.
    """

    training_dataset_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    validation_dataset_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    hyper_parameters: "VeoHyperParameters" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="VeoHyperParameters",
    )


class EvaluationConfig(proto.Message):
    r"""Evaluation Config for Tuning Job.

    Attributes:
        metrics (MutableSequence[google.cloud.aiplatform_v1beta1.types.Metric]):
            Required. The metrics used for evaluation.
        output_config (google.cloud.aiplatform_v1beta1.types.OutputConfig):
            Required. Config for evaluation output.
        autorater_config (google.cloud.aiplatform_v1beta1.types.AutoraterConfig):
            Optional. Autorater config for evaluation.
    """

    metrics: MutableSequence[evaluation_service.Metric] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=evaluation_service.Metric,
    )
    output_config: evaluation_service.OutputConfig = proto.Field(
        proto.MESSAGE,
        number=2,
        message=evaluation_service.OutputConfig,
    )
    autorater_config: evaluation_service.AutoraterConfig = proto.Field(
        proto.MESSAGE,
        number=3,
        message=evaluation_service.AutoraterConfig,
    )


class EvaluateDatasetRun(proto.Message):
    r"""Evaluate Dataset Run Result for Tuning Job.

    Attributes:
        operation_name (str):
            Output only. The operation ID of the evaluation run. Format:
            ``projects/{project}/locations/{location}/operations/{operation_id}``.
        checkpoint_id (str):
            Output only. The checkpoint id used in the
            evaluation run. Only populated when evaluating
            checkpoints.
        evaluate_dataset_response (google.cloud.aiplatform_v1beta1.types.EvaluateDatasetResponse):
            Output only. Results for
            EvaluationService.EvaluateDataset.
        error (google.rpc.status_pb2.Status):
            Output only. The error of the evaluation run
            if any.
    """

    operation_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    checkpoint_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    evaluate_dataset_response: evaluation_service.EvaluateDatasetResponse = proto.Field(
        proto.MESSAGE,
        number=3,
        message=evaluation_service.EvaluateDatasetResponse,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=4,
        message=status_pb2.Status,
    )


class TunedModelCheckpoint(proto.Message):
    r"""TunedModelCheckpoint for the Tuned Model of a Tuning Job.

    Attributes:
        checkpoint_id (str):
            The ID of the checkpoint.
        epoch (int):
            The epoch of the checkpoint.
        step (int):
            The step of the checkpoint.
        endpoint (str):
            The Endpoint resource name that the checkpoint is deployed
            to. Format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``.
    """

    checkpoint_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    epoch: int = proto.Field(
        proto.INT64,
        number=2,
    )
    step: int = proto.Field(
        proto.INT64,
        number=3,
    )
    endpoint: str = proto.Field(
        proto.STRING,
        number=4,
    )


class PreTunedModel(proto.Message):
    r"""A pre-tuned model for continuous tuning.

    Attributes:
        tuned_model_name (str):
            The resource name of the Model. E.g., a model resource name
            with a specified version id or alias:

            ``projects/{project}/locations/{location}/models/{model}@{version_id}``

            ``projects/{project}/locations/{location}/models/{model}@{alias}``

            Or, omit the version id to use the default version:

            ``projects/{project}/locations/{location}/models/{model}``
        checkpoint_id (str):
            Optional. The source checkpoint id. If not
            specified, the default checkpoint will be used.
        base_model (str):
            Output only. The name of the base model this
            [PreTunedModel][google.cloud.aiplatform.v1beta1.PreTunedModel]
            was tuned from.
    """

    tuned_model_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    checkpoint_id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    base_model: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
