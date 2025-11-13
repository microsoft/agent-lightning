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
from google.cloud.aiplatform_v1beta1.types import io
from google.cloud.aiplatform_v1beta1.types import operation


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "PairwiseChoice",
        "EvaluateDatasetOperationMetadata",
        "EvaluateDatasetResponse",
        "OutputInfo",
        "AggregationOutput",
        "AggregationResult",
        "EvaluateDatasetRequest",
        "OutputConfig",
        "Metric",
        "EvaluationDataset",
        "AutoraterConfig",
        "EvaluateInstancesRequest",
        "EvaluateInstancesResponse",
        "ExactMatchInput",
        "ExactMatchInstance",
        "ExactMatchSpec",
        "ExactMatchResults",
        "ExactMatchMetricValue",
        "BleuInput",
        "BleuInstance",
        "BleuSpec",
        "BleuResults",
        "BleuMetricValue",
        "RougeInput",
        "RougeInstance",
        "RougeSpec",
        "RougeResults",
        "RougeMetricValue",
        "CoherenceInput",
        "CoherenceInstance",
        "CoherenceSpec",
        "CoherenceResult",
        "FluencyInput",
        "FluencyInstance",
        "FluencySpec",
        "FluencyResult",
        "SafetyInput",
        "SafetyInstance",
        "SafetySpec",
        "SafetyResult",
        "GroundednessInput",
        "GroundednessInstance",
        "GroundednessSpec",
        "GroundednessResult",
        "FulfillmentInput",
        "FulfillmentInstance",
        "FulfillmentSpec",
        "FulfillmentResult",
        "SummarizationQualityInput",
        "SummarizationQualityInstance",
        "SummarizationQualitySpec",
        "SummarizationQualityResult",
        "PairwiseSummarizationQualityInput",
        "PairwiseSummarizationQualityInstance",
        "PairwiseSummarizationQualitySpec",
        "PairwiseSummarizationQualityResult",
        "SummarizationHelpfulnessInput",
        "SummarizationHelpfulnessInstance",
        "SummarizationHelpfulnessSpec",
        "SummarizationHelpfulnessResult",
        "SummarizationVerbosityInput",
        "SummarizationVerbosityInstance",
        "SummarizationVerbositySpec",
        "SummarizationVerbosityResult",
        "QuestionAnsweringQualityInput",
        "QuestionAnsweringQualityInstance",
        "QuestionAnsweringQualitySpec",
        "QuestionAnsweringQualityResult",
        "PairwiseQuestionAnsweringQualityInput",
        "PairwiseQuestionAnsweringQualityInstance",
        "PairwiseQuestionAnsweringQualitySpec",
        "PairwiseQuestionAnsweringQualityResult",
        "QuestionAnsweringRelevanceInput",
        "QuestionAnsweringRelevanceInstance",
        "QuestionAnsweringRelevanceSpec",
        "QuestionAnsweringRelevanceResult",
        "QuestionAnsweringHelpfulnessInput",
        "QuestionAnsweringHelpfulnessInstance",
        "QuestionAnsweringHelpfulnessSpec",
        "QuestionAnsweringHelpfulnessResult",
        "QuestionAnsweringCorrectnessInput",
        "QuestionAnsweringCorrectnessInstance",
        "QuestionAnsweringCorrectnessSpec",
        "QuestionAnsweringCorrectnessResult",
        "PointwiseMetricInput",
        "PointwiseMetricInstance",
        "PointwiseMetricSpec",
        "CustomOutputFormatConfig",
        "PointwiseMetricResult",
        "CustomOutput",
        "RawOutput",
        "PairwiseMetricInput",
        "PairwiseMetricInstance",
        "PairwiseMetricSpec",
        "PairwiseMetricResult",
        "ToolCallValidInput",
        "ToolCallValidSpec",
        "ToolCallValidInstance",
        "ToolCallValidResults",
        "ToolCallValidMetricValue",
        "ToolNameMatchInput",
        "ToolNameMatchSpec",
        "ToolNameMatchInstance",
        "ToolNameMatchResults",
        "ToolNameMatchMetricValue",
        "ToolParameterKeyMatchInput",
        "ToolParameterKeyMatchSpec",
        "ToolParameterKeyMatchInstance",
        "ToolParameterKeyMatchResults",
        "ToolParameterKeyMatchMetricValue",
        "ToolParameterKVMatchInput",
        "ToolParameterKVMatchSpec",
        "ToolParameterKVMatchInstance",
        "ToolParameterKVMatchResults",
        "ToolParameterKVMatchMetricValue",
        "CometInput",
        "CometSpec",
        "CometInstance",
        "CometResult",
        "MetricxInput",
        "MetricxSpec",
        "MetricxInstance",
        "MetricxResult",
        "RubricBasedInstructionFollowingInput",
        "RubricBasedInstructionFollowingInstance",
        "RubricBasedInstructionFollowingSpec",
        "RubricBasedInstructionFollowingResult",
        "RubricCritiqueResult",
        "TrajectoryExactMatchInput",
        "TrajectoryExactMatchSpec",
        "TrajectoryExactMatchInstance",
        "TrajectoryExactMatchResults",
        "TrajectoryExactMatchMetricValue",
        "TrajectoryInOrderMatchInput",
        "TrajectoryInOrderMatchSpec",
        "TrajectoryInOrderMatchInstance",
        "TrajectoryInOrderMatchResults",
        "TrajectoryInOrderMatchMetricValue",
        "TrajectoryAnyOrderMatchInput",
        "TrajectoryAnyOrderMatchSpec",
        "TrajectoryAnyOrderMatchInstance",
        "TrajectoryAnyOrderMatchResults",
        "TrajectoryAnyOrderMatchMetricValue",
        "TrajectoryPrecisionInput",
        "TrajectoryPrecisionSpec",
        "TrajectoryPrecisionInstance",
        "TrajectoryPrecisionResults",
        "TrajectoryPrecisionMetricValue",
        "TrajectoryRecallInput",
        "TrajectoryRecallSpec",
        "TrajectoryRecallInstance",
        "TrajectoryRecallResults",
        "TrajectoryRecallMetricValue",
        "TrajectorySingleToolUseInput",
        "TrajectorySingleToolUseSpec",
        "TrajectorySingleToolUseInstance",
        "TrajectorySingleToolUseResults",
        "TrajectorySingleToolUseMetricValue",
        "Trajectory",
        "ToolCall",
        "ContentMap",
    },
)


class PairwiseChoice(proto.Enum):
    r"""Pairwise prediction autorater preference.

    Values:
        PAIRWISE_CHOICE_UNSPECIFIED (0):
            Unspecified prediction choice.
        BASELINE (1):
            Baseline prediction wins
        CANDIDATE (2):
            Candidate prediction wins
        TIE (3):
            Winner cannot be determined
    """

    PAIRWISE_CHOICE_UNSPECIFIED = 0
    BASELINE = 1
    CANDIDATE = 2
    TIE = 3


class EvaluateDatasetOperationMetadata(proto.Message):
    r"""Operation metadata for Dataset Evaluation.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            Generic operation metadata.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


class EvaluateDatasetResponse(proto.Message):
    r"""Response in LRO for EvaluationService.EvaluateDataset.

    Attributes:
        aggregation_output (google.cloud.aiplatform_v1beta1.types.AggregationOutput):
            Output only. Aggregation statistics derived
            from results of
            EvaluationService.EvaluateDataset.
        output_info (google.cloud.aiplatform_v1beta1.types.OutputInfo):
            Output only. Output info for
            EvaluationService.EvaluateDataset.
    """

    aggregation_output: "AggregationOutput" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="AggregationOutput",
    )
    output_info: "OutputInfo" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="OutputInfo",
    )


class OutputInfo(proto.Message):
    r"""Describes the info for output of
    EvaluationService.EvaluateDataset.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_output_directory (str):
            Output only. The full path of the Cloud
            Storage directory created, into which the
            evaluation results and aggregation results are
            written.

            This field is a member of `oneof`_ ``output_location``.
    """

    gcs_output_directory: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="output_location",
    )


class AggregationOutput(proto.Message):
    r"""The aggregation result for the entire dataset and all
    metrics.

    Attributes:
        dataset (google.cloud.aiplatform_v1beta1.types.EvaluationDataset):
            The dataset used for evaluation &
            aggregation.
        aggregation_results (MutableSequence[google.cloud.aiplatform_v1beta1.types.AggregationResult]):
            One AggregationResult per metric.
    """

    dataset: "EvaluationDataset" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="EvaluationDataset",
    )
    aggregation_results: MutableSequence["AggregationResult"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="AggregationResult",
    )


class AggregationResult(proto.Message):
    r"""The aggregation result for a single metric.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        pointwise_metric_result (google.cloud.aiplatform_v1beta1.types.PointwiseMetricResult):
            Result for pointwise metric.

            This field is a member of `oneof`_ ``aggregation_result``.
        pairwise_metric_result (google.cloud.aiplatform_v1beta1.types.PairwiseMetricResult):
            Result for pairwise metric.

            This field is a member of `oneof`_ ``aggregation_result``.
        exact_match_metric_value (google.cloud.aiplatform_v1beta1.types.ExactMatchMetricValue):
            Results for exact match metric.

            This field is a member of `oneof`_ ``aggregation_result``.
        bleu_metric_value (google.cloud.aiplatform_v1beta1.types.BleuMetricValue):
            Results for bleu metric.

            This field is a member of `oneof`_ ``aggregation_result``.
        rouge_metric_value (google.cloud.aiplatform_v1beta1.types.RougeMetricValue):
            Results for rouge metric.

            This field is a member of `oneof`_ ``aggregation_result``.
        aggregation_metric (google.cloud.aiplatform_v1beta1.types.Metric.AggregationMetric):
            Aggregation metric.
    """

    pointwise_metric_result: "PointwiseMetricResult" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="aggregation_result",
        message="PointwiseMetricResult",
    )
    pairwise_metric_result: "PairwiseMetricResult" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="aggregation_result",
        message="PairwiseMetricResult",
    )
    exact_match_metric_value: "ExactMatchMetricValue" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="aggregation_result",
        message="ExactMatchMetricValue",
    )
    bleu_metric_value: "BleuMetricValue" = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="aggregation_result",
        message="BleuMetricValue",
    )
    rouge_metric_value: "RougeMetricValue" = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="aggregation_result",
        message="RougeMetricValue",
    )
    aggregation_metric: "Metric.AggregationMetric" = proto.Field(
        proto.ENUM,
        number=4,
        enum="Metric.AggregationMetric",
    )


class EvaluateDatasetRequest(proto.Message):
    r"""Request message for EvaluationService.EvaluateDataset.

    Attributes:
        location (str):
            Required. The resource name of the Location to evaluate the
            dataset. Format: ``projects/{project}/locations/{location}``
        dataset (google.cloud.aiplatform_v1beta1.types.EvaluationDataset):
            Required. The dataset used for evaluation.
        metrics (MutableSequence[google.cloud.aiplatform_v1beta1.types.Metric]):
            Required. The metrics used for evaluation.
        output_config (google.cloud.aiplatform_v1beta1.types.OutputConfig):
            Required. Config for evaluation output.
        autorater_config (google.cloud.aiplatform_v1beta1.types.AutoraterConfig):
            Optional. Autorater config used for evaluation. Currently
            only publisher Gemini models are supported. Format:
            ``projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{MODEL}.``
    """

    location: str = proto.Field(
        proto.STRING,
        number=1,
    )
    dataset: "EvaluationDataset" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="EvaluationDataset",
    )
    metrics: MutableSequence["Metric"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="Metric",
    )
    output_config: "OutputConfig" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="OutputConfig",
    )
    autorater_config: "AutoraterConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="AutoraterConfig",
    )


class OutputConfig(proto.Message):
    r"""Config for evaluation output.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_destination (google.cloud.aiplatform_v1beta1.types.GcsDestination):
            Cloud storage destination for evaluation
            output.

            This field is a member of `oneof`_ ``destination``.
    """

    gcs_destination: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="destination",
        message=io.GcsDestination,
    )


class Metric(proto.Message):
    r"""The metric used for dataset level evaluation.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        pointwise_metric_spec (google.cloud.aiplatform_v1beta1.types.PointwiseMetricSpec):
            Spec for pointwise metric.

            This field is a member of `oneof`_ ``metric_spec``.
        pairwise_metric_spec (google.cloud.aiplatform_v1beta1.types.PairwiseMetricSpec):
            Spec for pairwise metric.

            This field is a member of `oneof`_ ``metric_spec``.
        exact_match_spec (google.cloud.aiplatform_v1beta1.types.ExactMatchSpec):
            Spec for exact match metric.

            This field is a member of `oneof`_ ``metric_spec``.
        bleu_spec (google.cloud.aiplatform_v1beta1.types.BleuSpec):
            Spec for bleu metric.

            This field is a member of `oneof`_ ``metric_spec``.
        rouge_spec (google.cloud.aiplatform_v1beta1.types.RougeSpec):
            Spec for rouge metric.

            This field is a member of `oneof`_ ``metric_spec``.
        aggregation_metrics (MutableSequence[google.cloud.aiplatform_v1beta1.types.Metric.AggregationMetric]):
            Optional. The aggregation metrics to use.
    """

    class AggregationMetric(proto.Enum):
        r"""The aggregation metrics supported by
        EvaluationService.EvaluateDataset.

        Values:
            AGGREGATION_METRIC_UNSPECIFIED (0):
                Unspecified aggregation metric.
            AVERAGE (1):
                Average aggregation metric. Not supported for
                Pairwise metric.
            MODE (2):
                Mode aggregation metric.
            STANDARD_DEVIATION (3):
                Standard deviation aggregation metric. Not
                supported for pairwise metric.
            VARIANCE (4):
                Variance aggregation metric. Not supported
                for pairwise metric.
            MINIMUM (5):
                Minimum aggregation metric. Not supported for
                pairwise metric.
            MAXIMUM (6):
                Maximum aggregation metric. Not supported for
                pairwise metric.
            MEDIAN (7):
                Median aggregation metric. Not supported for
                pairwise metric.
            PERCENTILE_P90 (8):
                90th percentile aggregation metric. Not
                supported for pairwise metric.
            PERCENTILE_P95 (9):
                95th percentile aggregation metric. Not
                supported for pairwise metric.
            PERCENTILE_P99 (10):
                99th percentile aggregation metric. Not
                supported for pairwise metric.
        """

        AGGREGATION_METRIC_UNSPECIFIED = 0
        AVERAGE = 1
        MODE = 2
        STANDARD_DEVIATION = 3
        VARIANCE = 4
        MINIMUM = 5
        MAXIMUM = 6
        MEDIAN = 7
        PERCENTILE_P90 = 8
        PERCENTILE_P95 = 9
        PERCENTILE_P99 = 10

    pointwise_metric_spec: "PointwiseMetricSpec" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="metric_spec",
        message="PointwiseMetricSpec",
    )
    pairwise_metric_spec: "PairwiseMetricSpec" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="metric_spec",
        message="PairwiseMetricSpec",
    )
    exact_match_spec: "ExactMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="metric_spec",
        message="ExactMatchSpec",
    )
    bleu_spec: "BleuSpec" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="metric_spec",
        message="BleuSpec",
    )
    rouge_spec: "RougeSpec" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="metric_spec",
        message="RougeSpec",
    )
    aggregation_metrics: MutableSequence[AggregationMetric] = proto.RepeatedField(
        proto.ENUM,
        number=1,
        enum=AggregationMetric,
    )


class EvaluationDataset(proto.Message):
    r"""The dataset used for evaluation.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_source (google.cloud.aiplatform_v1beta1.types.GcsSource):
            Cloud storage source holds the dataset.
            Currently only one Cloud Storage file path is
            supported.

            This field is a member of `oneof`_ ``source``.
        bigquery_source (google.cloud.aiplatform_v1beta1.types.BigQuerySource):
            BigQuery source holds the dataset.

            This field is a member of `oneof`_ ``source``.
    """

    gcs_source: io.GcsSource = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="source",
        message=io.GcsSource,
    )
    bigquery_source: io.BigQuerySource = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=io.BigQuerySource,
    )


class AutoraterConfig(proto.Message):
    r"""The configs for autorater. This is applicable to both
    EvaluateInstances and EvaluateDataset.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        sampling_count (int):
            Optional. Number of samples for each instance
            in the dataset. If not specified, the default is
            4. Minimum value is 1, maximum value is 32.

            This field is a member of `oneof`_ ``_sampling_count``.
        flip_enabled (bool):
            Optional. Default is true. Whether to flip the candidate and
            baseline responses. This is only applicable to the pairwise
            metric. If enabled, also provide
            PairwiseMetricSpec.candidate_response_field_name and
            PairwiseMetricSpec.baseline_response_field_name. When
            rendering PairwiseMetricSpec.metric_prompt_template, the
            candidate and baseline fields will be flipped for half of
            the samples to reduce bias.

            This field is a member of `oneof`_ ``_flip_enabled``.
        autorater_model (str):
            Optional. The fully qualified name of the publisher model or
            tuned autorater endpoint to use.

            Publisher model format:
            ``projects/{project}/locations/{location}/publishers/*/models/*``

            Tuned model endpoint format:
            ``projects/{project}/locations/{location}/endpoints/{endpoint}``
    """

    sampling_count: int = proto.Field(
        proto.INT32,
        number=1,
        optional=True,
    )
    flip_enabled: bool = proto.Field(
        proto.BOOL,
        number=2,
        optional=True,
    )
    autorater_model: str = proto.Field(
        proto.STRING,
        number=3,
    )


class EvaluateInstancesRequest(proto.Message):
    r"""Request message for EvaluationService.EvaluateInstances.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        exact_match_input (google.cloud.aiplatform_v1beta1.types.ExactMatchInput):
            Auto metric instances.
            Instances and metric spec for exact match
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        bleu_input (google.cloud.aiplatform_v1beta1.types.BleuInput):
            Instances and metric spec for bleu metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        rouge_input (google.cloud.aiplatform_v1beta1.types.RougeInput):
            Instances and metric spec for rouge metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        fluency_input (google.cloud.aiplatform_v1beta1.types.FluencyInput):
            LLM-based metric instance.
            General text generation metrics, applicable to
            other categories. Input for fluency metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        coherence_input (google.cloud.aiplatform_v1beta1.types.CoherenceInput):
            Input for coherence metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        safety_input (google.cloud.aiplatform_v1beta1.types.SafetyInput):
            Input for safety metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        groundedness_input (google.cloud.aiplatform_v1beta1.types.GroundednessInput):
            Input for groundedness metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        fulfillment_input (google.cloud.aiplatform_v1beta1.types.FulfillmentInput):
            Input for fulfillment metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        summarization_quality_input (google.cloud.aiplatform_v1beta1.types.SummarizationQualityInput):
            Input for summarization quality metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        pairwise_summarization_quality_input (google.cloud.aiplatform_v1beta1.types.PairwiseSummarizationQualityInput):
            Input for pairwise summarization quality
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        summarization_helpfulness_input (google.cloud.aiplatform_v1beta1.types.SummarizationHelpfulnessInput):
            Input for summarization helpfulness metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        summarization_verbosity_input (google.cloud.aiplatform_v1beta1.types.SummarizationVerbosityInput):
            Input for summarization verbosity metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        question_answering_quality_input (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringQualityInput):
            Input for question answering quality metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        pairwise_question_answering_quality_input (google.cloud.aiplatform_v1beta1.types.PairwiseQuestionAnsweringQualityInput):
            Input for pairwise question answering quality
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        question_answering_relevance_input (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringRelevanceInput):
            Input for question answering relevance
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        question_answering_helpfulness_input (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringHelpfulnessInput):
            Input for question answering helpfulness
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        question_answering_correctness_input (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringCorrectnessInput):
            Input for question answering correctness
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        pointwise_metric_input (google.cloud.aiplatform_v1beta1.types.PointwiseMetricInput):
            Input for pointwise metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        pairwise_metric_input (google.cloud.aiplatform_v1beta1.types.PairwiseMetricInput):
            Input for pairwise metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        tool_call_valid_input (google.cloud.aiplatform_v1beta1.types.ToolCallValidInput):
            Tool call metric instances.
            Input for tool call valid metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        tool_name_match_input (google.cloud.aiplatform_v1beta1.types.ToolNameMatchInput):
            Input for tool name match metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        tool_parameter_key_match_input (google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchInput):
            Input for tool parameter key match metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        tool_parameter_kv_match_input (google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchInput):
            Input for tool parameter key value match
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        comet_input (google.cloud.aiplatform_v1beta1.types.CometInput):
            Translation metrics.
            Input for Comet metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        metricx_input (google.cloud.aiplatform_v1beta1.types.MetricxInput):
            Input for Metricx metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        trajectory_exact_match_input (google.cloud.aiplatform_v1beta1.types.TrajectoryExactMatchInput):
            Input for trajectory exact match metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        trajectory_in_order_match_input (google.cloud.aiplatform_v1beta1.types.TrajectoryInOrderMatchInput):
            Input for trajectory in order match metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        trajectory_any_order_match_input (google.cloud.aiplatform_v1beta1.types.TrajectoryAnyOrderMatchInput):
            Input for trajectory match any order metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        trajectory_precision_input (google.cloud.aiplatform_v1beta1.types.TrajectoryPrecisionInput):
            Input for trajectory precision metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        trajectory_recall_input (google.cloud.aiplatform_v1beta1.types.TrajectoryRecallInput):
            Input for trajectory recall metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        trajectory_single_tool_use_input (google.cloud.aiplatform_v1beta1.types.TrajectorySingleToolUseInput):
            Input for trajectory single tool use metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        rubric_based_instruction_following_input (google.cloud.aiplatform_v1beta1.types.RubricBasedInstructionFollowingInput):
            Rubric Based Instruction Following metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        location (str):
            Required. The resource name of the Location to evaluate the
            instances. Format:
            ``projects/{project}/locations/{location}``
        autorater_config (google.cloud.aiplatform_v1beta1.types.AutoraterConfig):
            Optional. Autorater config used for
            evaluation.
    """

    exact_match_input: "ExactMatchInput" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="metric_inputs",
        message="ExactMatchInput",
    )
    bleu_input: "BleuInput" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="metric_inputs",
        message="BleuInput",
    )
    rouge_input: "RougeInput" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="metric_inputs",
        message="RougeInput",
    )
    fluency_input: "FluencyInput" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="metric_inputs",
        message="FluencyInput",
    )
    coherence_input: "CoherenceInput" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="metric_inputs",
        message="CoherenceInput",
    )
    safety_input: "SafetyInput" = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="metric_inputs",
        message="SafetyInput",
    )
    groundedness_input: "GroundednessInput" = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="metric_inputs",
        message="GroundednessInput",
    )
    fulfillment_input: "FulfillmentInput" = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="metric_inputs",
        message="FulfillmentInput",
    )
    summarization_quality_input: "SummarizationQualityInput" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="metric_inputs",
        message="SummarizationQualityInput",
    )
    pairwise_summarization_quality_input: "PairwiseSummarizationQualityInput" = (
        proto.Field(
            proto.MESSAGE,
            number=23,
            oneof="metric_inputs",
            message="PairwiseSummarizationQualityInput",
        )
    )
    summarization_helpfulness_input: "SummarizationHelpfulnessInput" = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="metric_inputs",
        message="SummarizationHelpfulnessInput",
    )
    summarization_verbosity_input: "SummarizationVerbosityInput" = proto.Field(
        proto.MESSAGE,
        number=15,
        oneof="metric_inputs",
        message="SummarizationVerbosityInput",
    )
    question_answering_quality_input: "QuestionAnsweringQualityInput" = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="metric_inputs",
        message="QuestionAnsweringQualityInput",
    )
    pairwise_question_answering_quality_input: (
        "PairwiseQuestionAnsweringQualityInput"
    ) = proto.Field(
        proto.MESSAGE,
        number=24,
        oneof="metric_inputs",
        message="PairwiseQuestionAnsweringQualityInput",
    )
    question_answering_relevance_input: "QuestionAnsweringRelevanceInput" = proto.Field(
        proto.MESSAGE,
        number=16,
        oneof="metric_inputs",
        message="QuestionAnsweringRelevanceInput",
    )
    question_answering_helpfulness_input: "QuestionAnsweringHelpfulnessInput" = (
        proto.Field(
            proto.MESSAGE,
            number=17,
            oneof="metric_inputs",
            message="QuestionAnsweringHelpfulnessInput",
        )
    )
    question_answering_correctness_input: "QuestionAnsweringCorrectnessInput" = (
        proto.Field(
            proto.MESSAGE,
            number=18,
            oneof="metric_inputs",
            message="QuestionAnsweringCorrectnessInput",
        )
    )
    pointwise_metric_input: "PointwiseMetricInput" = proto.Field(
        proto.MESSAGE,
        number=28,
        oneof="metric_inputs",
        message="PointwiseMetricInput",
    )
    pairwise_metric_input: "PairwiseMetricInput" = proto.Field(
        proto.MESSAGE,
        number=29,
        oneof="metric_inputs",
        message="PairwiseMetricInput",
    )
    tool_call_valid_input: "ToolCallValidInput" = proto.Field(
        proto.MESSAGE,
        number=19,
        oneof="metric_inputs",
        message="ToolCallValidInput",
    )
    tool_name_match_input: "ToolNameMatchInput" = proto.Field(
        proto.MESSAGE,
        number=20,
        oneof="metric_inputs",
        message="ToolNameMatchInput",
    )
    tool_parameter_key_match_input: "ToolParameterKeyMatchInput" = proto.Field(
        proto.MESSAGE,
        number=21,
        oneof="metric_inputs",
        message="ToolParameterKeyMatchInput",
    )
    tool_parameter_kv_match_input: "ToolParameterKVMatchInput" = proto.Field(
        proto.MESSAGE,
        number=22,
        oneof="metric_inputs",
        message="ToolParameterKVMatchInput",
    )
    comet_input: "CometInput" = proto.Field(
        proto.MESSAGE,
        number=31,
        oneof="metric_inputs",
        message="CometInput",
    )
    metricx_input: "MetricxInput" = proto.Field(
        proto.MESSAGE,
        number=32,
        oneof="metric_inputs",
        message="MetricxInput",
    )
    trajectory_exact_match_input: "TrajectoryExactMatchInput" = proto.Field(
        proto.MESSAGE,
        number=33,
        oneof="metric_inputs",
        message="TrajectoryExactMatchInput",
    )
    trajectory_in_order_match_input: "TrajectoryInOrderMatchInput" = proto.Field(
        proto.MESSAGE,
        number=34,
        oneof="metric_inputs",
        message="TrajectoryInOrderMatchInput",
    )
    trajectory_any_order_match_input: "TrajectoryAnyOrderMatchInput" = proto.Field(
        proto.MESSAGE,
        number=35,
        oneof="metric_inputs",
        message="TrajectoryAnyOrderMatchInput",
    )
    trajectory_precision_input: "TrajectoryPrecisionInput" = proto.Field(
        proto.MESSAGE,
        number=37,
        oneof="metric_inputs",
        message="TrajectoryPrecisionInput",
    )
    trajectory_recall_input: "TrajectoryRecallInput" = proto.Field(
        proto.MESSAGE,
        number=38,
        oneof="metric_inputs",
        message="TrajectoryRecallInput",
    )
    trajectory_single_tool_use_input: "TrajectorySingleToolUseInput" = proto.Field(
        proto.MESSAGE,
        number=39,
        oneof="metric_inputs",
        message="TrajectorySingleToolUseInput",
    )
    rubric_based_instruction_following_input: "RubricBasedInstructionFollowingInput" = (
        proto.Field(
            proto.MESSAGE,
            number=40,
            oneof="metric_inputs",
            message="RubricBasedInstructionFollowingInput",
        )
    )
    location: str = proto.Field(
        proto.STRING,
        number=1,
    )
    autorater_config: "AutoraterConfig" = proto.Field(
        proto.MESSAGE,
        number=30,
        message="AutoraterConfig",
    )


class EvaluateInstancesResponse(proto.Message):
    r"""Response message for EvaluationService.EvaluateInstances.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        exact_match_results (google.cloud.aiplatform_v1beta1.types.ExactMatchResults):
            Auto metric evaluation results.
            Results for exact match metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        bleu_results (google.cloud.aiplatform_v1beta1.types.BleuResults):
            Results for bleu metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        rouge_results (google.cloud.aiplatform_v1beta1.types.RougeResults):
            Results for rouge metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        fluency_result (google.cloud.aiplatform_v1beta1.types.FluencyResult):
            LLM-based metric evaluation result.
            General text generation metrics, applicable to
            other categories. Result for fluency metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        coherence_result (google.cloud.aiplatform_v1beta1.types.CoherenceResult):
            Result for coherence metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        safety_result (google.cloud.aiplatform_v1beta1.types.SafetyResult):
            Result for safety metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        groundedness_result (google.cloud.aiplatform_v1beta1.types.GroundednessResult):
            Result for groundedness metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        fulfillment_result (google.cloud.aiplatform_v1beta1.types.FulfillmentResult):
            Result for fulfillment metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        summarization_quality_result (google.cloud.aiplatform_v1beta1.types.SummarizationQualityResult):
            Summarization only metrics.
            Result for summarization quality metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        pairwise_summarization_quality_result (google.cloud.aiplatform_v1beta1.types.PairwiseSummarizationQualityResult):
            Result for pairwise summarization quality
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        summarization_helpfulness_result (google.cloud.aiplatform_v1beta1.types.SummarizationHelpfulnessResult):
            Result for summarization helpfulness metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        summarization_verbosity_result (google.cloud.aiplatform_v1beta1.types.SummarizationVerbosityResult):
            Result for summarization verbosity metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        question_answering_quality_result (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringQualityResult):
            Question answering only metrics.
            Result for question answering quality metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        pairwise_question_answering_quality_result (google.cloud.aiplatform_v1beta1.types.PairwiseQuestionAnsweringQualityResult):
            Result for pairwise question answering
            quality metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        question_answering_relevance_result (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringRelevanceResult):
            Result for question answering relevance
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        question_answering_helpfulness_result (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringHelpfulnessResult):
            Result for question answering helpfulness
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        question_answering_correctness_result (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringCorrectnessResult):
            Result for question answering correctness
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        pointwise_metric_result (google.cloud.aiplatform_v1beta1.types.PointwiseMetricResult):
            Generic metrics.
            Result for pointwise metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        pairwise_metric_result (google.cloud.aiplatform_v1beta1.types.PairwiseMetricResult):
            Result for pairwise metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        tool_call_valid_results (google.cloud.aiplatform_v1beta1.types.ToolCallValidResults):
            Tool call metrics.
            Results for tool call valid metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        tool_name_match_results (google.cloud.aiplatform_v1beta1.types.ToolNameMatchResults):
            Results for tool name match metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        tool_parameter_key_match_results (google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchResults):
            Results for tool parameter key match  metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        tool_parameter_kv_match_results (google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchResults):
            Results for tool parameter key value match
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        comet_result (google.cloud.aiplatform_v1beta1.types.CometResult):
            Translation metrics.
            Result for Comet metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        metricx_result (google.cloud.aiplatform_v1beta1.types.MetricxResult):
            Result for Metricx metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        trajectory_exact_match_results (google.cloud.aiplatform_v1beta1.types.TrajectoryExactMatchResults):
            Result for trajectory exact match metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        trajectory_in_order_match_results (google.cloud.aiplatform_v1beta1.types.TrajectoryInOrderMatchResults):
            Result for trajectory in order match metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        trajectory_any_order_match_results (google.cloud.aiplatform_v1beta1.types.TrajectoryAnyOrderMatchResults):
            Result for trajectory any order match metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        trajectory_precision_results (google.cloud.aiplatform_v1beta1.types.TrajectoryPrecisionResults):
            Result for trajectory precision metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        trajectory_recall_results (google.cloud.aiplatform_v1beta1.types.TrajectoryRecallResults):
            Results for trajectory recall metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        trajectory_single_tool_use_results (google.cloud.aiplatform_v1beta1.types.TrajectorySingleToolUseResults):
            Results for trajectory single tool use
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        rubric_based_instruction_following_result (google.cloud.aiplatform_v1beta1.types.RubricBasedInstructionFollowingResult):
            Result for rubric based instruction following
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
    """

    exact_match_results: "ExactMatchResults" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="evaluation_results",
        message="ExactMatchResults",
    )
    bleu_results: "BleuResults" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="evaluation_results",
        message="BleuResults",
    )
    rouge_results: "RougeResults" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="evaluation_results",
        message="RougeResults",
    )
    fluency_result: "FluencyResult" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="evaluation_results",
        message="FluencyResult",
    )
    coherence_result: "CoherenceResult" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="evaluation_results",
        message="CoherenceResult",
    )
    safety_result: "SafetyResult" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="evaluation_results",
        message="SafetyResult",
    )
    groundedness_result: "GroundednessResult" = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="evaluation_results",
        message="GroundednessResult",
    )
    fulfillment_result: "FulfillmentResult" = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="evaluation_results",
        message="FulfillmentResult",
    )
    summarization_quality_result: "SummarizationQualityResult" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="evaluation_results",
        message="SummarizationQualityResult",
    )
    pairwise_summarization_quality_result: "PairwiseSummarizationQualityResult" = (
        proto.Field(
            proto.MESSAGE,
            number=22,
            oneof="evaluation_results",
            message="PairwiseSummarizationQualityResult",
        )
    )
    summarization_helpfulness_result: "SummarizationHelpfulnessResult" = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="evaluation_results",
        message="SummarizationHelpfulnessResult",
    )
    summarization_verbosity_result: "SummarizationVerbosityResult" = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="evaluation_results",
        message="SummarizationVerbosityResult",
    )
    question_answering_quality_result: "QuestionAnsweringQualityResult" = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="evaluation_results",
        message="QuestionAnsweringQualityResult",
    )
    pairwise_question_answering_quality_result: (
        "PairwiseQuestionAnsweringQualityResult"
    ) = proto.Field(
        proto.MESSAGE,
        number=23,
        oneof="evaluation_results",
        message="PairwiseQuestionAnsweringQualityResult",
    )
    question_answering_relevance_result: "QuestionAnsweringRelevanceResult" = (
        proto.Field(
            proto.MESSAGE,
            number=15,
            oneof="evaluation_results",
            message="QuestionAnsweringRelevanceResult",
        )
    )
    question_answering_helpfulness_result: "QuestionAnsweringHelpfulnessResult" = (
        proto.Field(
            proto.MESSAGE,
            number=16,
            oneof="evaluation_results",
            message="QuestionAnsweringHelpfulnessResult",
        )
    )
    question_answering_correctness_result: "QuestionAnsweringCorrectnessResult" = (
        proto.Field(
            proto.MESSAGE,
            number=17,
            oneof="evaluation_results",
            message="QuestionAnsweringCorrectnessResult",
        )
    )
    pointwise_metric_result: "PointwiseMetricResult" = proto.Field(
        proto.MESSAGE,
        number=27,
        oneof="evaluation_results",
        message="PointwiseMetricResult",
    )
    pairwise_metric_result: "PairwiseMetricResult" = proto.Field(
        proto.MESSAGE,
        number=28,
        oneof="evaluation_results",
        message="PairwiseMetricResult",
    )
    tool_call_valid_results: "ToolCallValidResults" = proto.Field(
        proto.MESSAGE,
        number=18,
        oneof="evaluation_results",
        message="ToolCallValidResults",
    )
    tool_name_match_results: "ToolNameMatchResults" = proto.Field(
        proto.MESSAGE,
        number=19,
        oneof="evaluation_results",
        message="ToolNameMatchResults",
    )
    tool_parameter_key_match_results: "ToolParameterKeyMatchResults" = proto.Field(
        proto.MESSAGE,
        number=20,
        oneof="evaluation_results",
        message="ToolParameterKeyMatchResults",
    )
    tool_parameter_kv_match_results: "ToolParameterKVMatchResults" = proto.Field(
        proto.MESSAGE,
        number=21,
        oneof="evaluation_results",
        message="ToolParameterKVMatchResults",
    )
    comet_result: "CometResult" = proto.Field(
        proto.MESSAGE,
        number=29,
        oneof="evaluation_results",
        message="CometResult",
    )
    metricx_result: "MetricxResult" = proto.Field(
        proto.MESSAGE,
        number=30,
        oneof="evaluation_results",
        message="MetricxResult",
    )
    trajectory_exact_match_results: "TrajectoryExactMatchResults" = proto.Field(
        proto.MESSAGE,
        number=31,
        oneof="evaluation_results",
        message="TrajectoryExactMatchResults",
    )
    trajectory_in_order_match_results: "TrajectoryInOrderMatchResults" = proto.Field(
        proto.MESSAGE,
        number=32,
        oneof="evaluation_results",
        message="TrajectoryInOrderMatchResults",
    )
    trajectory_any_order_match_results: "TrajectoryAnyOrderMatchResults" = proto.Field(
        proto.MESSAGE,
        number=33,
        oneof="evaluation_results",
        message="TrajectoryAnyOrderMatchResults",
    )
    trajectory_precision_results: "TrajectoryPrecisionResults" = proto.Field(
        proto.MESSAGE,
        number=35,
        oneof="evaluation_results",
        message="TrajectoryPrecisionResults",
    )
    trajectory_recall_results: "TrajectoryRecallResults" = proto.Field(
        proto.MESSAGE,
        number=36,
        oneof="evaluation_results",
        message="TrajectoryRecallResults",
    )
    trajectory_single_tool_use_results: "TrajectorySingleToolUseResults" = proto.Field(
        proto.MESSAGE,
        number=37,
        oneof="evaluation_results",
        message="TrajectorySingleToolUseResults",
    )
    rubric_based_instruction_following_result: (
        "RubricBasedInstructionFollowingResult"
    ) = proto.Field(
        proto.MESSAGE,
        number=38,
        oneof="evaluation_results",
        message="RubricBasedInstructionFollowingResult",
    )


class ExactMatchInput(proto.Message):
    r"""Input for exact match metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ExactMatchSpec):
            Required. Spec for exact match metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ExactMatchInstance]):
            Required. Repeated exact match instances.
    """

    metric_spec: "ExactMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ExactMatchSpec",
    )
    instances: MutableSequence["ExactMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ExactMatchInstance",
    )


class ExactMatchInstance(proto.Message):
    r"""Spec for exact match instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ExactMatchSpec(proto.Message):
    r"""Spec for exact match metric - returns 1 if prediction and
    reference exactly matches, otherwise 0.

    """


class ExactMatchResults(proto.Message):
    r"""Results for exact match metric.

    Attributes:
        exact_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ExactMatchMetricValue]):
            Output only. Exact match metric values.
    """

    exact_match_metric_values: MutableSequence["ExactMatchMetricValue"] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="ExactMatchMetricValue",
        )
    )


class ExactMatchMetricValue(proto.Message):
    r"""Exact match metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Exact match score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class BleuInput(proto.Message):
    r"""Input for bleu metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.BleuSpec):
            Required. Spec for bleu score metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.BleuInstance]):
            Required. Repeated bleu instances.
    """

    metric_spec: "BleuSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="BleuSpec",
    )
    instances: MutableSequence["BleuInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="BleuInstance",
    )


class BleuInstance(proto.Message):
    r"""Spec for bleu instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class BleuSpec(proto.Message):
    r"""Spec for bleu score metric - calculates the precision of
    n-grams in the prediction as compared to reference - returns a
    score ranging between 0 to 1.

    Attributes:
        use_effective_order (bool):
            Optional. Whether to use_effective_order to compute bleu
            score.
    """

    use_effective_order: bool = proto.Field(
        proto.BOOL,
        number=1,
    )


class BleuResults(proto.Message):
    r"""Results for bleu metric.

    Attributes:
        bleu_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.BleuMetricValue]):
            Output only. Bleu metric values.
    """

    bleu_metric_values: MutableSequence["BleuMetricValue"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="BleuMetricValue",
    )


class BleuMetricValue(proto.Message):
    r"""Bleu metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Bleu score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class RougeInput(proto.Message):
    r"""Input for rouge metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.RougeSpec):
            Required. Spec for rouge score metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.RougeInstance]):
            Required. Repeated rouge instances.
    """

    metric_spec: "RougeSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RougeSpec",
    )
    instances: MutableSequence["RougeInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="RougeInstance",
    )


class RougeInstance(proto.Message):
    r"""Spec for rouge instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class RougeSpec(proto.Message):
    r"""Spec for rouge score metric - calculates the recall of
    n-grams in prediction as compared to reference - returns a score
    ranging between 0 and 1.

    Attributes:
        rouge_type (str):
            Optional. Supported rouge types are rougen[1-9], rougeL, and
            rougeLsum.
        use_stemmer (bool):
            Optional. Whether to use stemmer to compute
            rouge score.
        split_summaries (bool):
            Optional. Whether to split summaries while
            using rougeLsum.
    """

    rouge_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    use_stemmer: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    split_summaries: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class RougeResults(proto.Message):
    r"""Results for rouge metric.

    Attributes:
        rouge_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.RougeMetricValue]):
            Output only. Rouge metric values.
    """

    rouge_metric_values: MutableSequence["RougeMetricValue"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="RougeMetricValue",
    )


class RougeMetricValue(proto.Message):
    r"""Rouge metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Rouge score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class CoherenceInput(proto.Message):
    r"""Input for coherence metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.CoherenceSpec):
            Required. Spec for coherence score metric.
        instance (google.cloud.aiplatform_v1beta1.types.CoherenceInstance):
            Required. Coherence instance.
    """

    metric_spec: "CoherenceSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CoherenceSpec",
    )
    instance: "CoherenceInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="CoherenceInstance",
    )


class CoherenceInstance(proto.Message):
    r"""Spec for coherence instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )


class CoherenceSpec(proto.Message):
    r"""Spec for coherence score metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class CoherenceResult(proto.Message):
    r"""Spec for coherence result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Coherence score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for coherence score.
        confidence (float):
            Output only. Confidence for coherence score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class FluencyInput(proto.Message):
    r"""Input for fluency metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.FluencySpec):
            Required. Spec for fluency score metric.
        instance (google.cloud.aiplatform_v1beta1.types.FluencyInstance):
            Required. Fluency instance.
    """

    metric_spec: "FluencySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="FluencySpec",
    )
    instance: "FluencyInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="FluencyInstance",
    )


class FluencyInstance(proto.Message):
    r"""Spec for fluency instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )


class FluencySpec(proto.Message):
    r"""Spec for fluency score metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class FluencyResult(proto.Message):
    r"""Spec for fluency result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Fluency score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for fluency score.
        confidence (float):
            Output only. Confidence for fluency score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class SafetyInput(proto.Message):
    r"""Input for safety metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.SafetySpec):
            Required. Spec for safety metric.
        instance (google.cloud.aiplatform_v1beta1.types.SafetyInstance):
            Required. Safety instance.
    """

    metric_spec: "SafetySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SafetySpec",
    )
    instance: "SafetyInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SafetyInstance",
    )


class SafetyInstance(proto.Message):
    r"""Spec for safety instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )


class SafetySpec(proto.Message):
    r"""Spec for safety metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class SafetyResult(proto.Message):
    r"""Spec for safety result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Safety score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for safety score.
        confidence (float):
            Output only. Confidence for safety score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class GroundednessInput(proto.Message):
    r"""Input for groundedness metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.GroundednessSpec):
            Required. Spec for groundedness metric.
        instance (google.cloud.aiplatform_v1beta1.types.GroundednessInstance):
            Required. Groundedness instance.
    """

    metric_spec: "GroundednessSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="GroundednessSpec",
    )
    instance: "GroundednessInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="GroundednessInstance",
    )


class GroundednessInstance(proto.Message):
    r"""Spec for groundedness instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        context (str):
            Required. Background information provided in
            context used to compare against the prediction.

            This field is a member of `oneof`_ ``_context``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class GroundednessSpec(proto.Message):
    r"""Spec for groundedness metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class GroundednessResult(proto.Message):
    r"""Spec for groundedness result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Groundedness score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for groundedness
            score.
        confidence (float):
            Output only. Confidence for groundedness
            score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class FulfillmentInput(proto.Message):
    r"""Input for fulfillment metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.FulfillmentSpec):
            Required. Spec for fulfillment score metric.
        instance (google.cloud.aiplatform_v1beta1.types.FulfillmentInstance):
            Required. Fulfillment instance.
    """

    metric_spec: "FulfillmentSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="FulfillmentSpec",
    )
    instance: "FulfillmentInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="FulfillmentInstance",
    )


class FulfillmentInstance(proto.Message):
    r"""Spec for fulfillment instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        instruction (str):
            Required. Inference instruction prompt to
            compare prediction with.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class FulfillmentSpec(proto.Message):
    r"""Spec for fulfillment metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class FulfillmentResult(proto.Message):
    r"""Spec for fulfillment result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Fulfillment score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for fulfillment
            score.
        confidence (float):
            Output only. Confidence for fulfillment
            score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class SummarizationQualityInput(proto.Message):
    r"""Input for summarization quality metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.SummarizationQualitySpec):
            Required. Spec for summarization quality
            score metric.
        instance (google.cloud.aiplatform_v1beta1.types.SummarizationQualityInstance):
            Required. Summarization quality instance.
    """

    metric_spec: "SummarizationQualitySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SummarizationQualitySpec",
    )
    instance: "SummarizationQualityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SummarizationQualityInstance",
    )


class SummarizationQualityInstance(proto.Message):
    r"""Spec for summarization quality instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to be summarized.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. Summarization prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class SummarizationQualitySpec(proto.Message):
    r"""Spec for summarization quality score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute summarization quality.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class SummarizationQualityResult(proto.Message):
    r"""Spec for summarization quality result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Summarization Quality score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for summarization
            quality score.
        confidence (float):
            Output only. Confidence for summarization
            quality score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class PairwiseSummarizationQualityInput(proto.Message):
    r"""Input for pairwise summarization quality metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.PairwiseSummarizationQualitySpec):
            Required. Spec for pairwise summarization
            quality score metric.
        instance (google.cloud.aiplatform_v1beta1.types.PairwiseSummarizationQualityInstance):
            Required. Pairwise summarization quality
            instance.
    """

    metric_spec: "PairwiseSummarizationQualitySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="PairwiseSummarizationQualitySpec",
    )
    instance: "PairwiseSummarizationQualityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="PairwiseSummarizationQualityInstance",
    )


class PairwiseSummarizationQualityInstance(proto.Message):
    r"""Spec for pairwise summarization quality instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the candidate model.

            This field is a member of `oneof`_ ``_prediction``.
        baseline_prediction (str):
            Required. Output of the baseline model.

            This field is a member of `oneof`_ ``_baseline_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to be summarized.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. Summarization prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    baseline_prediction: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=5,
        optional=True,
    )


class PairwiseSummarizationQualitySpec(proto.Message):
    r"""Spec for pairwise summarization quality score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute pairwise summarization quality.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class PairwiseSummarizationQualityResult(proto.Message):
    r"""Spec for pairwise summarization quality result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        pairwise_choice (google.cloud.aiplatform_v1beta1.types.PairwiseChoice):
            Output only. Pairwise summarization
            prediction choice.
        explanation (str):
            Output only. Explanation for summarization
            quality score.
        confidence (float):
            Output only. Confidence for summarization
            quality score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    pairwise_choice: "PairwiseChoice" = proto.Field(
        proto.ENUM,
        number=1,
        enum="PairwiseChoice",
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class SummarizationHelpfulnessInput(proto.Message):
    r"""Input for summarization helpfulness metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.SummarizationHelpfulnessSpec):
            Required. Spec for summarization helpfulness
            score metric.
        instance (google.cloud.aiplatform_v1beta1.types.SummarizationHelpfulnessInstance):
            Required. Summarization helpfulness instance.
    """

    metric_spec: "SummarizationHelpfulnessSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SummarizationHelpfulnessSpec",
    )
    instance: "SummarizationHelpfulnessInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SummarizationHelpfulnessInstance",
    )


class SummarizationHelpfulnessInstance(proto.Message):
    r"""Spec for summarization helpfulness instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to be summarized.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Optional. Summarization prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class SummarizationHelpfulnessSpec(proto.Message):
    r"""Spec for summarization helpfulness score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute summarization helpfulness.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class SummarizationHelpfulnessResult(proto.Message):
    r"""Spec for summarization helpfulness result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Summarization Helpfulness score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for summarization
            helpfulness score.
        confidence (float):
            Output only. Confidence for summarization
            helpfulness score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class SummarizationVerbosityInput(proto.Message):
    r"""Input for summarization verbosity metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.SummarizationVerbositySpec):
            Required. Spec for summarization verbosity
            score metric.
        instance (google.cloud.aiplatform_v1beta1.types.SummarizationVerbosityInstance):
            Required. Summarization verbosity instance.
    """

    metric_spec: "SummarizationVerbositySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SummarizationVerbositySpec",
    )
    instance: "SummarizationVerbosityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SummarizationVerbosityInstance",
    )


class SummarizationVerbosityInstance(proto.Message):
    r"""Spec for summarization verbosity instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to be summarized.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Optional. Summarization prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class SummarizationVerbositySpec(proto.Message):
    r"""Spec for summarization verbosity score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute summarization verbosity.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class SummarizationVerbosityResult(proto.Message):
    r"""Spec for summarization verbosity result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Summarization Verbosity score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for summarization
            verbosity score.
        confidence (float):
            Output only. Confidence for summarization
            verbosity score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class QuestionAnsweringQualityInput(proto.Message):
    r"""Input for question answering quality metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringQualitySpec):
            Required. Spec for question answering quality
            score metric.
        instance (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringQualityInstance):
            Required. Question answering quality
            instance.
    """

    metric_spec: "QuestionAnsweringQualitySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="QuestionAnsweringQualitySpec",
    )
    instance: "QuestionAnsweringQualityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="QuestionAnsweringQualityInstance",
    )


class QuestionAnsweringQualityInstance(proto.Message):
    r"""Spec for question answering quality instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to answer the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. Question Answering prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class QuestionAnsweringQualitySpec(proto.Message):
    r"""Spec for question answering quality score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering quality.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class QuestionAnsweringQualityResult(proto.Message):
    r"""Spec for question answering quality result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Question Answering Quality
            score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for question
            answering quality score.
        confidence (float):
            Output only. Confidence for question
            answering quality score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class PairwiseQuestionAnsweringQualityInput(proto.Message):
    r"""Input for pairwise question answering quality metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.PairwiseQuestionAnsweringQualitySpec):
            Required. Spec for pairwise question
            answering quality score metric.
        instance (google.cloud.aiplatform_v1beta1.types.PairwiseQuestionAnsweringQualityInstance):
            Required. Pairwise question answering quality
            instance.
    """

    metric_spec: "PairwiseQuestionAnsweringQualitySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="PairwiseQuestionAnsweringQualitySpec",
    )
    instance: "PairwiseQuestionAnsweringQualityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="PairwiseQuestionAnsweringQualityInstance",
    )


class PairwiseQuestionAnsweringQualityInstance(proto.Message):
    r"""Spec for pairwise question answering quality instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the candidate model.

            This field is a member of `oneof`_ ``_prediction``.
        baseline_prediction (str):
            Required. Output of the baseline model.

            This field is a member of `oneof`_ ``_baseline_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to answer the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. Question Answering prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    baseline_prediction: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=5,
        optional=True,
    )


class PairwiseQuestionAnsweringQualitySpec(proto.Message):
    r"""Spec for pairwise question answering quality score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering quality.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class PairwiseQuestionAnsweringQualityResult(proto.Message):
    r"""Spec for pairwise question answering quality result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        pairwise_choice (google.cloud.aiplatform_v1beta1.types.PairwiseChoice):
            Output only. Pairwise question answering
            prediction choice.
        explanation (str):
            Output only. Explanation for question
            answering quality score.
        confidence (float):
            Output only. Confidence for question
            answering quality score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    pairwise_choice: "PairwiseChoice" = proto.Field(
        proto.ENUM,
        number=1,
        enum="PairwiseChoice",
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class QuestionAnsweringRelevanceInput(proto.Message):
    r"""Input for question answering relevance metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringRelevanceSpec):
            Required. Spec for question answering
            relevance score metric.
        instance (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringRelevanceInstance):
            Required. Question answering relevance
            instance.
    """

    metric_spec: "QuestionAnsweringRelevanceSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="QuestionAnsweringRelevanceSpec",
    )
    instance: "QuestionAnsweringRelevanceInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="QuestionAnsweringRelevanceInstance",
    )


class QuestionAnsweringRelevanceInstance(proto.Message):
    r"""Spec for question answering relevance instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Optional. Text provided as context to answer
            the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. The question asked and other
            instruction in the inference prompt.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class QuestionAnsweringRelevanceSpec(proto.Message):
    r"""Spec for question answering relevance metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering relevance.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class QuestionAnsweringRelevanceResult(proto.Message):
    r"""Spec for question answering relevance result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Question Answering Relevance
            score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for question
            answering relevance score.
        confidence (float):
            Output only. Confidence for question
            answering relevance score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class QuestionAnsweringHelpfulnessInput(proto.Message):
    r"""Input for question answering helpfulness metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringHelpfulnessSpec):
            Required. Spec for question answering
            helpfulness score metric.
        instance (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringHelpfulnessInstance):
            Required. Question answering helpfulness
            instance.
    """

    metric_spec: "QuestionAnsweringHelpfulnessSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="QuestionAnsweringHelpfulnessSpec",
    )
    instance: "QuestionAnsweringHelpfulnessInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="QuestionAnsweringHelpfulnessInstance",
    )


class QuestionAnsweringHelpfulnessInstance(proto.Message):
    r"""Spec for question answering helpfulness instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Optional. Text provided as context to answer
            the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. The question asked and other
            instruction in the inference prompt.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class QuestionAnsweringHelpfulnessSpec(proto.Message):
    r"""Spec for question answering helpfulness metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering helpfulness.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class QuestionAnsweringHelpfulnessResult(proto.Message):
    r"""Spec for question answering helpfulness result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Question Answering Helpfulness
            score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for question
            answering helpfulness score.
        confidence (float):
            Output only. Confidence for question
            answering helpfulness score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class QuestionAnsweringCorrectnessInput(proto.Message):
    r"""Input for question answering correctness metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringCorrectnessSpec):
            Required. Spec for question answering
            correctness score metric.
        instance (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringCorrectnessInstance):
            Required. Question answering correctness
            instance.
    """

    metric_spec: "QuestionAnsweringCorrectnessSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="QuestionAnsweringCorrectnessSpec",
    )
    instance: "QuestionAnsweringCorrectnessInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="QuestionAnsweringCorrectnessInstance",
    )


class QuestionAnsweringCorrectnessInstance(proto.Message):
    r"""Spec for question answering correctness instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Optional. Text provided as context to answer
            the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. The question asked and other
            instruction in the inference prompt.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class QuestionAnsweringCorrectnessSpec(proto.Message):
    r"""Spec for question answering correctness metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering correctness.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class QuestionAnsweringCorrectnessResult(proto.Message):
    r"""Spec for question answering correctness result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Question Answering Correctness
            score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for question
            answering correctness score.
        confidence (float):
            Output only. Confidence for question
            answering correctness score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class PointwiseMetricInput(proto.Message):
    r"""Input for pointwise metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.PointwiseMetricSpec):
            Required. Spec for pointwise metric.
        instance (google.cloud.aiplatform_v1beta1.types.PointwiseMetricInstance):
            Required. Pointwise metric instance.
    """

    metric_spec: "PointwiseMetricSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="PointwiseMetricSpec",
    )
    instance: "PointwiseMetricInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="PointwiseMetricInstance",
    )


class PointwiseMetricInstance(proto.Message):
    r"""Pointwise metric instance. Usually one instance corresponds
    to one row in an evaluation dataset.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        json_instance (str):
            Instance specified as a json string. String key-value pairs
            are expected in the json_instance to render
            PointwiseMetricSpec.instance_prompt_template.

            This field is a member of `oneof`_ ``instance``.
        content_map_instance (google.cloud.aiplatform_v1beta1.types.ContentMap):
            Key-value contents for the mutlimodality
            input, including text, image, video, audio, and
            pdf, etc. The key is placeholder in metric
            prompt template, and the value is the multimodal
            content.

            This field is a member of `oneof`_ ``instance``.
    """

    json_instance: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="instance",
    )
    content_map_instance: "ContentMap" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="instance",
        message="ContentMap",
    )


class PointwiseMetricSpec(proto.Message):
    r"""Spec for pointwise metric.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        metric_prompt_template (str):
            Required. Metric prompt template for
            pointwise metric.

            This field is a member of `oneof`_ ``_metric_prompt_template``.
        system_instruction (str):
            Optional. System instructions for pointwise
            metric.

            This field is a member of `oneof`_ ``_system_instruction``.
        custom_output_format_config (google.cloud.aiplatform_v1beta1.types.CustomOutputFormatConfig):
            Optional. CustomOutputFormatConfig allows customization of
            metric output. By default, metrics return a score and
            explanation. When this config is set, the default output is
            replaced with either:

            - The raw output string.
            - A parsed output based on a user-defined schema. If a
              custom format is chosen, the ``score`` and ``explanation``
              fields in the corresponding metric result will be empty.
    """

    metric_prompt_template: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    system_instruction: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    custom_output_format_config: "CustomOutputFormatConfig" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="CustomOutputFormatConfig",
    )


class CustomOutputFormatConfig(proto.Message):
    r"""Spec for custom output format configuration.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        return_raw_output (bool):
            Optional. Whether to return raw output.

            This field is a member of `oneof`_ ``custom_output_format_config``.
    """

    return_raw_output: bool = proto.Field(
        proto.BOOL,
        number=1,
        oneof="custom_output_format_config",
    )


class PointwiseMetricResult(proto.Message):
    r"""Spec for pointwise metric result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Pointwise metric score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for pointwise metric
            score.
        custom_output (google.cloud.aiplatform_v1beta1.types.CustomOutput):
            Output only. Spec for custom output.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    custom_output: "CustomOutput" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="CustomOutput",
    )


class CustomOutput(proto.Message):
    r"""Spec for custom output.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        raw_outputs (google.cloud.aiplatform_v1beta1.types.RawOutput):
            Output only. List of raw output strings.

            This field is a member of `oneof`_ ``custom_output``.
    """

    raw_outputs: "RawOutput" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="custom_output",
        message="RawOutput",
    )


class RawOutput(proto.Message):
    r"""Raw output.

    Attributes:
        raw_output (MutableSequence[str]):
            Output only. Raw output string.
    """

    raw_output: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )


class PairwiseMetricInput(proto.Message):
    r"""Input for pairwise metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.PairwiseMetricSpec):
            Required. Spec for pairwise metric.
        instance (google.cloud.aiplatform_v1beta1.types.PairwiseMetricInstance):
            Required. Pairwise metric instance.
    """

    metric_spec: "PairwiseMetricSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="PairwiseMetricSpec",
    )
    instance: "PairwiseMetricInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="PairwiseMetricInstance",
    )


class PairwiseMetricInstance(proto.Message):
    r"""Pairwise metric instance. Usually one instance corresponds to
    one row in an evaluation dataset.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        json_instance (str):
            Instance specified as a json string. String key-value pairs
            are expected in the json_instance to render
            PairwiseMetricSpec.instance_prompt_template.

            This field is a member of `oneof`_ ``instance``.
        content_map_instance (google.cloud.aiplatform_v1beta1.types.ContentMap):
            Key-value contents for the mutlimodality
            input, including text, image, video, audio, and
            pdf, etc. The key is placeholder in metric
            prompt template, and the value is the multimodal
            content.

            This field is a member of `oneof`_ ``instance``.
    """

    json_instance: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="instance",
    )
    content_map_instance: "ContentMap" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="instance",
        message="ContentMap",
    )


class PairwiseMetricSpec(proto.Message):
    r"""Spec for pairwise metric.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        metric_prompt_template (str):
            Required. Metric prompt template for pairwise
            metric.

            This field is a member of `oneof`_ ``_metric_prompt_template``.
        candidate_response_field_name (str):
            Optional. The field name of the candidate
            response.
        baseline_response_field_name (str):
            Optional. The field name of the baseline
            response.
        system_instruction (str):
            Optional. System instructions for pairwise
            metric.

            This field is a member of `oneof`_ ``_system_instruction``.
        custom_output_format_config (google.cloud.aiplatform_v1beta1.types.CustomOutputFormatConfig):
            Optional. CustomOutputFormatConfig allows customization of
            metric output. When this config is set, the default output
            is replaced with the raw output string. If a custom format
            is chosen, the ``pairwise_choice`` and ``explanation``
            fields in the corresponding metric result will be empty.
    """

    metric_prompt_template: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    candidate_response_field_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    baseline_response_field_name: str = proto.Field(
        proto.STRING,
        number=3,
    )
    system_instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )
    custom_output_format_config: "CustomOutputFormatConfig" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="CustomOutputFormatConfig",
    )


class PairwiseMetricResult(proto.Message):
    r"""Spec for pairwise metric result.

    Attributes:
        pairwise_choice (google.cloud.aiplatform_v1beta1.types.PairwiseChoice):
            Output only. Pairwise metric choice.
        explanation (str):
            Output only. Explanation for pairwise metric
            score.
        custom_output (google.cloud.aiplatform_v1beta1.types.CustomOutput):
            Output only. Spec for custom output.
    """

    pairwise_choice: "PairwiseChoice" = proto.Field(
        proto.ENUM,
        number=1,
        enum="PairwiseChoice",
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    custom_output: "CustomOutput" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="CustomOutput",
    )


class ToolCallValidInput(proto.Message):
    r"""Input for tool call valid metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ToolCallValidSpec):
            Required. Spec for tool call valid metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolCallValidInstance]):
            Required. Repeated tool call valid instances.
    """

    metric_spec: "ToolCallValidSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ToolCallValidSpec",
    )
    instances: MutableSequence["ToolCallValidInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ToolCallValidInstance",
    )


class ToolCallValidSpec(proto.Message):
    r"""Spec for tool call valid metric."""


class ToolCallValidInstance(proto.Message):
    r"""Spec for tool call valid instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ToolCallValidResults(proto.Message):
    r"""Results for tool call valid metric.

    Attributes:
        tool_call_valid_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolCallValidMetricValue]):
            Output only. Tool call valid metric values.
    """

    tool_call_valid_metric_values: MutableSequence["ToolCallValidMetricValue"] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="ToolCallValidMetricValue",
        )
    )


class ToolCallValidMetricValue(proto.Message):
    r"""Tool call valid metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Tool call valid score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class ToolNameMatchInput(proto.Message):
    r"""Input for tool name match metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ToolNameMatchSpec):
            Required. Spec for tool name match metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolNameMatchInstance]):
            Required. Repeated tool name match instances.
    """

    metric_spec: "ToolNameMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ToolNameMatchSpec",
    )
    instances: MutableSequence["ToolNameMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ToolNameMatchInstance",
    )


class ToolNameMatchSpec(proto.Message):
    r"""Spec for tool name match metric."""


class ToolNameMatchInstance(proto.Message):
    r"""Spec for tool name match instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ToolNameMatchResults(proto.Message):
    r"""Results for tool name match metric.

    Attributes:
        tool_name_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolNameMatchMetricValue]):
            Output only. Tool name match metric values.
    """

    tool_name_match_metric_values: MutableSequence["ToolNameMatchMetricValue"] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="ToolNameMatchMetricValue",
        )
    )


class ToolNameMatchMetricValue(proto.Message):
    r"""Tool name match metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Tool name match score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class ToolParameterKeyMatchInput(proto.Message):
    r"""Input for tool parameter key match metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchSpec):
            Required. Spec for tool parameter key match
            metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchInstance]):
            Required. Repeated tool parameter key match
            instances.
    """

    metric_spec: "ToolParameterKeyMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ToolParameterKeyMatchSpec",
    )
    instances: MutableSequence["ToolParameterKeyMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ToolParameterKeyMatchInstance",
    )


class ToolParameterKeyMatchSpec(proto.Message):
    r"""Spec for tool parameter key match metric."""


class ToolParameterKeyMatchInstance(proto.Message):
    r"""Spec for tool parameter key match instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ToolParameterKeyMatchResults(proto.Message):
    r"""Results for tool parameter key match metric.

    Attributes:
        tool_parameter_key_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchMetricValue]):
            Output only. Tool parameter key match metric
            values.
    """

    tool_parameter_key_match_metric_values: MutableSequence[
        "ToolParameterKeyMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ToolParameterKeyMatchMetricValue",
    )


class ToolParameterKeyMatchMetricValue(proto.Message):
    r"""Tool parameter key match metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Tool parameter key match score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class ToolParameterKVMatchInput(proto.Message):
    r"""Input for tool parameter key value match metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchSpec):
            Required. Spec for tool parameter key value
            match metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchInstance]):
            Required. Repeated tool parameter key value
            match instances.
    """

    metric_spec: "ToolParameterKVMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ToolParameterKVMatchSpec",
    )
    instances: MutableSequence["ToolParameterKVMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ToolParameterKVMatchInstance",
    )


class ToolParameterKVMatchSpec(proto.Message):
    r"""Spec for tool parameter key value match metric.

    Attributes:
        use_strict_string_match (bool):
            Optional. Whether to use STRICT string match
            on parameter values.
    """

    use_strict_string_match: bool = proto.Field(
        proto.BOOL,
        number=1,
    )


class ToolParameterKVMatchInstance(proto.Message):
    r"""Spec for tool parameter key value match instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ToolParameterKVMatchResults(proto.Message):
    r"""Results for tool parameter key value match metric.

    Attributes:
        tool_parameter_kv_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchMetricValue]):
            Output only. Tool parameter key value match
            metric values.
    """

    tool_parameter_kv_match_metric_values: MutableSequence[
        "ToolParameterKVMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ToolParameterKVMatchMetricValue",
    )


class ToolParameterKVMatchMetricValue(proto.Message):
    r"""Tool parameter key value match metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Tool parameter key value match
            score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class CometInput(proto.Message):
    r"""Input for Comet metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.CometSpec):
            Required. Spec for comet metric.
        instance (google.cloud.aiplatform_v1beta1.types.CometInstance):
            Required. Comet instance.
    """

    metric_spec: "CometSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CometSpec",
    )
    instance: "CometInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="CometInstance",
    )


class CometSpec(proto.Message):
    r"""Spec for Comet metric.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        version (google.cloud.aiplatform_v1beta1.types.CometSpec.CometVersion):
            Required. Which version to use for
            evaluation.

            This field is a member of `oneof`_ ``_version``.
        source_language (str):
            Optional. Source language in BCP-47 format.
        target_language (str):
            Optional. Target language in BCP-47 format.
            Covers both prediction and reference.
    """

    class CometVersion(proto.Enum):
        r"""Comet version options.

        Values:
            COMET_VERSION_UNSPECIFIED (0):
                Comet version unspecified.
            COMET_22_SRC_REF (2):
                Comet 22 for translation + source + reference
                (source-reference-combined).
        """

        COMET_VERSION_UNSPECIFIED = 0
        COMET_22_SRC_REF = 2

    version: CometVersion = proto.Field(
        proto.ENUM,
        number=1,
        optional=True,
        enum=CometVersion,
    )
    source_language: str = proto.Field(
        proto.STRING,
        number=2,
    )
    target_language: str = proto.Field(
        proto.STRING,
        number=3,
    )


class CometInstance(proto.Message):
    r"""Spec for Comet instance - The fields used for evaluation are
    dependent on the comet version.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        source (str):
            Optional. Source text in original language.

            This field is a member of `oneof`_ ``_source``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    source: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )


class CometResult(proto.Message):
    r"""Spec for Comet result - calculates the comet score for the
    given instance using the version specified in the spec.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Comet score. Range depends on
            version.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class MetricxInput(proto.Message):
    r"""Input for MetricX metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.MetricxSpec):
            Required. Spec for Metricx metric.
        instance (google.cloud.aiplatform_v1beta1.types.MetricxInstance):
            Required. Metricx instance.
    """

    metric_spec: "MetricxSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="MetricxSpec",
    )
    instance: "MetricxInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="MetricxInstance",
    )


class MetricxSpec(proto.Message):
    r"""Spec for MetricX metric.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        version (google.cloud.aiplatform_v1beta1.types.MetricxSpec.MetricxVersion):
            Required. Which version to use for
            evaluation.

            This field is a member of `oneof`_ ``_version``.
        source_language (str):
            Optional. Source language in BCP-47 format.
        target_language (str):
            Optional. Target language in BCP-47 format.
            Covers both prediction and reference.
    """

    class MetricxVersion(proto.Enum):
        r"""MetricX Version options.

        Values:
            METRICX_VERSION_UNSPECIFIED (0):
                MetricX version unspecified.
            METRICX_24_REF (1):
                MetricX 2024 (2.6) for translation +
                reference (reference-based).
            METRICX_24_SRC (2):
                MetricX 2024 (2.6) for translation + source
                (QE).
            METRICX_24_SRC_REF (3):
                MetricX 2024 (2.6) for translation + source +
                reference (source-reference-combined).
        """

        METRICX_VERSION_UNSPECIFIED = 0
        METRICX_24_REF = 1
        METRICX_24_SRC = 2
        METRICX_24_SRC_REF = 3

    version: MetricxVersion = proto.Field(
        proto.ENUM,
        number=1,
        optional=True,
        enum=MetricxVersion,
    )
    source_language: str = proto.Field(
        proto.STRING,
        number=2,
    )
    target_language: str = proto.Field(
        proto.STRING,
        number=3,
    )


class MetricxInstance(proto.Message):
    r"""Spec for MetricX instance - The fields used for evaluation
    are dependent on the MetricX version.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        source (str):
            Optional. Source text in original language.

            This field is a member of `oneof`_ ``_source``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    source: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )


class MetricxResult(proto.Message):
    r"""Spec for MetricX result - calculates the MetricX score for
    the given instance using the version specified in the spec.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. MetricX score. Range depends on
            version.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class RubricBasedInstructionFollowingInput(proto.Message):
    r"""Instance and metric spec for RubricBasedInstructionFollowing
    metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.RubricBasedInstructionFollowingSpec):
            Required. Spec for
            RubricBasedInstructionFollowing metric.
        instance (google.cloud.aiplatform_v1beta1.types.RubricBasedInstructionFollowingInstance):
            Required. Instance for
            RubricBasedInstructionFollowing metric.
    """

    metric_spec: "RubricBasedInstructionFollowingSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RubricBasedInstructionFollowingSpec",
    )
    instance: "RubricBasedInstructionFollowingInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="RubricBasedInstructionFollowingInstance",
    )


class RubricBasedInstructionFollowingInstance(proto.Message):
    r"""Instance for RubricBasedInstructionFollowing metric - one
    instance corresponds to one row in an evaluation dataset.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        json_instance (str):
            Required. Instance specified as a json string. String
            key-value pairs are expected in the json_instance to render
            RubricBasedInstructionFollowing prompt templates.

            This field is a member of `oneof`_ ``instance``.
    """

    json_instance: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="instance",
    )


class RubricBasedInstructionFollowingSpec(proto.Message):
    r"""Spec for RubricBasedInstructionFollowing metric - returns
    rubrics and verdicts corresponding to rubrics along with overall
    score.

    """


class RubricBasedInstructionFollowingResult(proto.Message):
    r"""Result for RubricBasedInstructionFollowing metric.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Overall score for the
            instruction following.

            This field is a member of `oneof`_ ``_score``.
        rubric_critique_results (MutableSequence[google.cloud.aiplatform_v1beta1.types.RubricCritiqueResult]):
            Output only. List of per rubric critique
            results.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    rubric_critique_results: MutableSequence["RubricCritiqueResult"] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message="RubricCritiqueResult",
        )
    )


class RubricCritiqueResult(proto.Message):
    r"""Rubric critique result.

    Attributes:
        rubric (str):
            Output only. Rubric to be evaluated.
        verdict (bool):
            Output only. Verdict for the rubric - true if
            the rubric is met, false otherwise.
    """

    rubric: str = proto.Field(
        proto.STRING,
        number=1,
    )
    verdict: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class TrajectoryExactMatchInput(proto.Message):
    r"""Instances and metric spec for TrajectoryExactMatch metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.TrajectoryExactMatchSpec):
            Required. Spec for TrajectoryExactMatch
            metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryExactMatchInstance]):
            Required. Repeated TrajectoryExactMatch
            instance.
    """

    metric_spec: "TrajectoryExactMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TrajectoryExactMatchSpec",
    )
    instances: MutableSequence["TrajectoryExactMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="TrajectoryExactMatchInstance",
    )


class TrajectoryExactMatchSpec(proto.Message):
    r"""Spec for TrajectoryExactMatch metric - returns 1 if tool
    calls in the reference trajectory exactly match the predicted
    trajectory, else 0.

    """


class TrajectoryExactMatchInstance(proto.Message):
    r"""Spec for TrajectoryExactMatch instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        predicted_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for predicted tool call
            trajectory.

            This field is a member of `oneof`_ ``_predicted_trajectory``.
        reference_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for reference tool call
            trajectory.

            This field is a member of `oneof`_ ``_reference_trajectory``.
    """

    predicted_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=1,
        optional=True,
        message="Trajectory",
    )
    reference_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message="Trajectory",
    )


class TrajectoryExactMatchResults(proto.Message):
    r"""Results for TrajectoryExactMatch metric.

    Attributes:
        trajectory_exact_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryExactMatchMetricValue]):
            Output only. TrajectoryExactMatch metric
            values.
    """

    trajectory_exact_match_metric_values: MutableSequence[
        "TrajectoryExactMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TrajectoryExactMatchMetricValue",
    )


class TrajectoryExactMatchMetricValue(proto.Message):
    r"""TrajectoryExactMatch metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. TrajectoryExactMatch score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class TrajectoryInOrderMatchInput(proto.Message):
    r"""Instances and metric spec for TrajectoryInOrderMatch metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.TrajectoryInOrderMatchSpec):
            Required. Spec for TrajectoryInOrderMatch
            metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryInOrderMatchInstance]):
            Required. Repeated TrajectoryInOrderMatch
            instance.
    """

    metric_spec: "TrajectoryInOrderMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TrajectoryInOrderMatchSpec",
    )
    instances: MutableSequence["TrajectoryInOrderMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="TrajectoryInOrderMatchInstance",
    )


class TrajectoryInOrderMatchSpec(proto.Message):
    r"""Spec for TrajectoryInOrderMatch metric - returns 1 if tool
    calls in the reference trajectory appear in the predicted
    trajectory in the same order, else 0.

    """


class TrajectoryInOrderMatchInstance(proto.Message):
    r"""Spec for TrajectoryInOrderMatch instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        predicted_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for predicted tool call
            trajectory.

            This field is a member of `oneof`_ ``_predicted_trajectory``.
        reference_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for reference tool call
            trajectory.

            This field is a member of `oneof`_ ``_reference_trajectory``.
    """

    predicted_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=1,
        optional=True,
        message="Trajectory",
    )
    reference_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message="Trajectory",
    )


class TrajectoryInOrderMatchResults(proto.Message):
    r"""Results for TrajectoryInOrderMatch metric.

    Attributes:
        trajectory_in_order_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryInOrderMatchMetricValue]):
            Output only. TrajectoryInOrderMatch metric
            values.
    """

    trajectory_in_order_match_metric_values: MutableSequence[
        "TrajectoryInOrderMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TrajectoryInOrderMatchMetricValue",
    )


class TrajectoryInOrderMatchMetricValue(proto.Message):
    r"""TrajectoryInOrderMatch metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. TrajectoryInOrderMatch score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class TrajectoryAnyOrderMatchInput(proto.Message):
    r"""Instances and metric spec for TrajectoryAnyOrderMatch metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.TrajectoryAnyOrderMatchSpec):
            Required. Spec for TrajectoryAnyOrderMatch
            metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryAnyOrderMatchInstance]):
            Required. Repeated TrajectoryAnyOrderMatch
            instance.
    """

    metric_spec: "TrajectoryAnyOrderMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TrajectoryAnyOrderMatchSpec",
    )
    instances: MutableSequence["TrajectoryAnyOrderMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="TrajectoryAnyOrderMatchInstance",
    )


class TrajectoryAnyOrderMatchSpec(proto.Message):
    r"""Spec for TrajectoryAnyOrderMatch metric - returns 1 if all
    tool calls in the reference trajectory appear in the predicted
    trajectory in any order, else 0.

    """


class TrajectoryAnyOrderMatchInstance(proto.Message):
    r"""Spec for TrajectoryAnyOrderMatch instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        predicted_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for predicted tool call
            trajectory.

            This field is a member of `oneof`_ ``_predicted_trajectory``.
        reference_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for reference tool call
            trajectory.

            This field is a member of `oneof`_ ``_reference_trajectory``.
    """

    predicted_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=1,
        optional=True,
        message="Trajectory",
    )
    reference_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message="Trajectory",
    )


class TrajectoryAnyOrderMatchResults(proto.Message):
    r"""Results for TrajectoryAnyOrderMatch metric.

    Attributes:
        trajectory_any_order_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryAnyOrderMatchMetricValue]):
            Output only. TrajectoryAnyOrderMatch metric
            values.
    """

    trajectory_any_order_match_metric_values: MutableSequence[
        "TrajectoryAnyOrderMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TrajectoryAnyOrderMatchMetricValue",
    )


class TrajectoryAnyOrderMatchMetricValue(proto.Message):
    r"""TrajectoryAnyOrderMatch metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. TrajectoryAnyOrderMatch score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class TrajectoryPrecisionInput(proto.Message):
    r"""Instances and metric spec for TrajectoryPrecision metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.TrajectoryPrecisionSpec):
            Required. Spec for TrajectoryPrecision
            metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryPrecisionInstance]):
            Required. Repeated TrajectoryPrecision
            instance.
    """

    metric_spec: "TrajectoryPrecisionSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TrajectoryPrecisionSpec",
    )
    instances: MutableSequence["TrajectoryPrecisionInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="TrajectoryPrecisionInstance",
    )


class TrajectoryPrecisionSpec(proto.Message):
    r"""Spec for TrajectoryPrecision metric - returns a float score
    based on average precision of individual tool calls.

    """


class TrajectoryPrecisionInstance(proto.Message):
    r"""Spec for TrajectoryPrecision instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        predicted_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for predicted tool call
            trajectory.

            This field is a member of `oneof`_ ``_predicted_trajectory``.
        reference_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for reference tool call
            trajectory.

            This field is a member of `oneof`_ ``_reference_trajectory``.
    """

    predicted_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=1,
        optional=True,
        message="Trajectory",
    )
    reference_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message="Trajectory",
    )


class TrajectoryPrecisionResults(proto.Message):
    r"""Results for TrajectoryPrecision metric.

    Attributes:
        trajectory_precision_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryPrecisionMetricValue]):
            Output only. TrajectoryPrecision metric
            values.
    """

    trajectory_precision_metric_values: MutableSequence[
        "TrajectoryPrecisionMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TrajectoryPrecisionMetricValue",
    )


class TrajectoryPrecisionMetricValue(proto.Message):
    r"""TrajectoryPrecision metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. TrajectoryPrecision score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class TrajectoryRecallInput(proto.Message):
    r"""Instances and metric spec for TrajectoryRecall metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.TrajectoryRecallSpec):
            Required. Spec for TrajectoryRecall metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryRecallInstance]):
            Required. Repeated TrajectoryRecall instance.
    """

    metric_spec: "TrajectoryRecallSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TrajectoryRecallSpec",
    )
    instances: MutableSequence["TrajectoryRecallInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="TrajectoryRecallInstance",
    )


class TrajectoryRecallSpec(proto.Message):
    r"""Spec for TrajectoryRecall metric - returns a float score
    based on average recall of individual tool calls.

    """


class TrajectoryRecallInstance(proto.Message):
    r"""Spec for TrajectoryRecall instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        predicted_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for predicted tool call
            trajectory.

            This field is a member of `oneof`_ ``_predicted_trajectory``.
        reference_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for reference tool call
            trajectory.

            This field is a member of `oneof`_ ``_reference_trajectory``.
    """

    predicted_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=1,
        optional=True,
        message="Trajectory",
    )
    reference_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=2,
        optional=True,
        message="Trajectory",
    )


class TrajectoryRecallResults(proto.Message):
    r"""Results for TrajectoryRecall metric.

    Attributes:
        trajectory_recall_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectoryRecallMetricValue]):
            Output only. TrajectoryRecall metric values.
    """

    trajectory_recall_metric_values: MutableSequence["TrajectoryRecallMetricValue"] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message="TrajectoryRecallMetricValue",
        )
    )


class TrajectoryRecallMetricValue(proto.Message):
    r"""TrajectoryRecall metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. TrajectoryRecall score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class TrajectorySingleToolUseInput(proto.Message):
    r"""Instances and metric spec for TrajectorySingleToolUse metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.TrajectorySingleToolUseSpec):
            Required. Spec for TrajectorySingleToolUse
            metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectorySingleToolUseInstance]):
            Required. Repeated TrajectorySingleToolUse
            instance.
    """

    metric_spec: "TrajectorySingleToolUseSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="TrajectorySingleToolUseSpec",
    )
    instances: MutableSequence["TrajectorySingleToolUseInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="TrajectorySingleToolUseInstance",
    )


class TrajectorySingleToolUseSpec(proto.Message):
    r"""Spec for TrajectorySingleToolUse metric - returns 1 if tool
    is present in the predicted trajectory, else 0.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tool_name (str):
            Required. Spec for tool name to be checked
            for in the predicted trajectory.

            This field is a member of `oneof`_ ``_tool_name``.
    """

    tool_name: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )


class TrajectorySingleToolUseInstance(proto.Message):
    r"""Spec for TrajectorySingleToolUse instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        predicted_trajectory (google.cloud.aiplatform_v1beta1.types.Trajectory):
            Required. Spec for predicted tool call
            trajectory.

            This field is a member of `oneof`_ ``_predicted_trajectory``.
    """

    predicted_trajectory: "Trajectory" = proto.Field(
        proto.MESSAGE,
        number=1,
        optional=True,
        message="Trajectory",
    )


class TrajectorySingleToolUseResults(proto.Message):
    r"""Results for TrajectorySingleToolUse metric.

    Attributes:
        trajectory_single_tool_use_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.TrajectorySingleToolUseMetricValue]):
            Output only. TrajectorySingleToolUse metric
            values.
    """

    trajectory_single_tool_use_metric_values: MutableSequence[
        "TrajectorySingleToolUseMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="TrajectorySingleToolUseMetricValue",
    )


class TrajectorySingleToolUseMetricValue(proto.Message):
    r"""TrajectorySingleToolUse metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. TrajectorySingleToolUse score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class Trajectory(proto.Message):
    r"""Spec for trajectory.

    Attributes:
        tool_calls (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolCall]):
            Required. Tool calls in the trajectory.
    """

    tool_calls: MutableSequence["ToolCall"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ToolCall",
    )


class ToolCall(proto.Message):
    r"""Spec for tool call.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tool_name (str):
            Required. Spec for tool name

            This field is a member of `oneof`_ ``_tool_name``.
        tool_input (str):
            Optional. Spec for tool input

            This field is a member of `oneof`_ ``_tool_input``.
    """

    tool_name: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    tool_input: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ContentMap(proto.Message):
    r"""Map of placeholder in metric prompt template to contents of
    model input.

    Attributes:
        values (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ContentMap.Contents]):
            Optional. Map of placeholder to contents.
    """

    class Contents(proto.Message):
        r"""Repeated Content type.

        Attributes:
            contents (MutableSequence[google.cloud.aiplatform_v1beta1.types.Content]):
                Optional. Repeated contents.
        """

        contents: MutableSequence[content.Content] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=content.Content,
        )

    values: MutableMapping[str, Contents] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=1,
        message=Contents,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
