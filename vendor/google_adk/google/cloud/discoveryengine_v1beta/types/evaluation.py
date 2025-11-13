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

from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore

from google.cloud.discoveryengine_v1beta.types import search_service

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1beta",
    manifest={
        "Evaluation",
        "QualityMetrics",
    },
)


class Evaluation(proto.Message):
    r"""An evaluation is a single execution (or run) of an evaluation
    process. It encapsulates the state of the evaluation and the
    resulting data.

    Attributes:
        name (str):
            Identifier. The full resource name of the
            [Evaluation][google.cloud.discoveryengine.v1beta.Evaluation],
            in the format of
            ``projects/{project}/locations/{location}/evaluations/{evaluation}``.

            This field must be a UTF-8 encoded string with a length
            limit of 1024 characters.
        evaluation_spec (google.cloud.discoveryengine_v1beta.types.Evaluation.EvaluationSpec):
            Required. The specification of the
            evaluation.
        quality_metrics (google.cloud.discoveryengine_v1beta.types.QualityMetrics):
            Output only. The metrics produced by the evaluation,
            averaged across all
            [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery]s
            in the
            [SampleQuerySet][google.cloud.discoveryengine.v1beta.SampleQuerySet].

            Only populated when the evaluation's state is SUCCEEDED.
        state (google.cloud.discoveryengine_v1beta.types.Evaluation.State):
            Output only. The state of the evaluation.
        error (google.rpc.status_pb2.Status):
            Output only. The error that occurred during
            evaluation. Only populated when the evaluation's
            state is FAILED.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp the
            [Evaluation][google.cloud.discoveryengine.v1beta.Evaluation]
            was created at.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp the
            [Evaluation][google.cloud.discoveryengine.v1beta.Evaluation]
            was completed at.
        error_samples (MutableSequence[google.rpc.status_pb2.Status]):
            Output only. A sample of errors encountered
            while processing the request.
    """

    class State(proto.Enum):
        r"""Describes the state of an evaluation.

        Values:
            STATE_UNSPECIFIED (0):
                The evaluation is unspecified.
            PENDING (1):
                The service is preparing to run the
                evaluation.
            RUNNING (2):
                The evaluation is in progress.
            SUCCEEDED (3):
                The evaluation completed successfully.
            FAILED (4):
                The evaluation failed.
        """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        FAILED = 4

    class EvaluationSpec(proto.Message):
        r"""Describes the specification of the evaluation.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            search_request (google.cloud.discoveryengine_v1beta.types.SearchRequest):
                Required. The search request that is used to perform the
                evaluation.

                Only the following fields within SearchRequest are
                supported; if any other fields are provided, an UNSUPPORTED
                error will be returned:

                - [SearchRequest.serving_config][google.cloud.discoveryengine.v1beta.SearchRequest.serving_config]
                - [SearchRequest.branch][google.cloud.discoveryengine.v1beta.SearchRequest.branch]
                - [SearchRequest.canonical_filter][google.cloud.discoveryengine.v1beta.SearchRequest.canonical_filter]
                - [SearchRequest.query_expansion_spec][google.cloud.discoveryengine.v1beta.SearchRequest.query_expansion_spec]
                - [SearchRequest.spell_correction_spec][google.cloud.discoveryengine.v1beta.SearchRequest.spell_correction_spec]
                - [SearchRequest.content_search_spec][google.cloud.discoveryengine.v1beta.SearchRequest.content_search_spec]
                - [SearchRequest.user_pseudo_id][google.cloud.discoveryengine.v1beta.SearchRequest.user_pseudo_id]

                This field is a member of `oneof`_ ``search_spec``.
            query_set_spec (google.cloud.discoveryengine_v1beta.types.Evaluation.EvaluationSpec.QuerySetSpec):
                Required. The specification of the query set.
        """

        class QuerySetSpec(proto.Message):
            r"""Describes the specification of the query set.

            Attributes:
                sample_query_set (str):
                    Required. The full resource name of the
                    [SampleQuerySet][google.cloud.discoveryengine.v1beta.SampleQuerySet]
                    used for the evaluation, in the format of
                    ``projects/{project}/locations/{location}/sampleQuerySets/{sampleQuerySet}``.
            """

            sample_query_set: str = proto.Field(
                proto.STRING,
                number=1,
            )

        search_request: search_service.SearchRequest = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="search_spec",
            message=search_service.SearchRequest,
        )
        query_set_spec: "Evaluation.EvaluationSpec.QuerySetSpec" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="Evaluation.EvaluationSpec.QuerySetSpec",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    evaluation_spec: EvaluationSpec = proto.Field(
        proto.MESSAGE,
        number=2,
        message=EvaluationSpec,
    )
    quality_metrics: "QualityMetrics" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="QualityMetrics",
    )
    state: State = proto.Field(
        proto.ENUM,
        number=4,
        enum=State,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=5,
        message=status_pb2.Status,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    error_samples: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message=status_pb2.Status,
    )


class QualityMetrics(proto.Message):
    r"""Describes the metrics produced by the evaluation.

    Attributes:
        doc_recall (google.cloud.discoveryengine_v1beta.types.QualityMetrics.TopkMetrics):
            Recall per document, at various top-k cutoff levels.

            Recall is the fraction of relevant documents retrieved out
            of all relevant documents.

            Example (top-5):

            - For a single
              [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery],
              If 3 out of 5 relevant documents are retrieved in the
              top-5, recall@5 = 3/5 = 0.6
        doc_precision (google.cloud.discoveryengine_v1beta.types.QualityMetrics.TopkMetrics):
            Precision per document, at various top-k cutoff levels.

            Precision is the fraction of retrieved documents that are
            relevant.

            Example (top-5):

            - For a single
              [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery],
              If 4 out of 5 retrieved documents in the top-5 are
              relevant, precision@5 = 4/5 = 0.8
        doc_ndcg (google.cloud.discoveryengine_v1beta.types.QualityMetrics.TopkMetrics):
            Normalized discounted cumulative gain (NDCG) per document,
            at various top-k cutoff levels.

            NDCG measures the ranking quality, giving higher relevance
            to top results.

            Example (top-3): Suppose
            [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery]
            with three retrieved documents (D1, D2, D3) and binary
            relevance judgements (1 for relevant, 0 for not relevant):

            Retrieved: [D3 (0), D1 (1), D2 (1)] Ideal: [D1 (1), D2 (1),
            D3 (0)]

            Calculate NDCG@3 for each
            [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery]:
            \* DCG@3: 0/log2(1+1) + 1/log2(2+1) + 1/log2(3+1) = 1.13 \*
            Ideal DCG@3: 1/log2(1+1) + 1/log2(2+1) + 0/log2(3+1) = 1.63
            \* NDCG@3: 1.13/1.63 = 0.693
        page_recall (google.cloud.discoveryengine_v1beta.types.QualityMetrics.TopkMetrics):
            Recall per page, at various top-k cutoff levels.

            Recall is the fraction of relevant pages retrieved out of
            all relevant pages.

            Example (top-5):

            - For a single
              [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery],
              if 3 out of 5 relevant pages are retrieved in the top-5,
              recall@5 = 3/5 = 0.6
        page_ndcg (google.cloud.discoveryengine_v1beta.types.QualityMetrics.TopkMetrics):
            Normalized discounted cumulative gain (NDCG) per page, at
            various top-k cutoff levels.

            NDCG measures the ranking quality, giving higher relevance
            to top results.

            Example (top-3): Suppose
            [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery]
            with three retrieved pages (P1, P2, P3) and binary relevance
            judgements (1 for relevant, 0 for not relevant):

            Retrieved: [P3 (0), P1 (1), P2 (1)] Ideal: [P1 (1), P2 (1),
            P3 (0)]

            Calculate NDCG@3 for
            [SampleQuery][google.cloud.discoveryengine.v1beta.SampleQuery]:
            \* DCG@3: 0/log2(1+1) + 1/log2(2+1) + 1/log2(3+1) = 1.13 \*
            Ideal DCG@3: 1/log2(1+1) + 1/log2(2+1) + 0/log2(3+1) = 1.63
            \* NDCG@3: 1.13/1.63 = 0.693
    """

    class TopkMetrics(proto.Message):
        r"""Stores the metric values at specific top-k levels.

        Attributes:
            top_1 (float):
                The top-1 value.
            top_3 (float):
                The top-3 value.
            top_5 (float):
                The top-5 value.
            top_10 (float):
                The top-10 value.
        """

        top_1: float = proto.Field(
            proto.DOUBLE,
            number=1,
        )
        top_3: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )
        top_5: float = proto.Field(
            proto.DOUBLE,
            number=3,
        )
        top_10: float = proto.Field(
            proto.DOUBLE,
            number=4,
        )

    doc_recall: TopkMetrics = proto.Field(
        proto.MESSAGE,
        number=1,
        message=TopkMetrics,
    )
    doc_precision: TopkMetrics = proto.Field(
        proto.MESSAGE,
        number=2,
        message=TopkMetrics,
    )
    doc_ndcg: TopkMetrics = proto.Field(
        proto.MESSAGE,
        number=3,
        message=TopkMetrics,
    )
    page_recall: TopkMetrics = proto.Field(
        proto.MESSAGE,
        number=4,
        message=TopkMetrics,
    )
    page_ndcg: TopkMetrics = proto.Field(
        proto.MESSAGE,
        number=5,
        message=TopkMetrics,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
