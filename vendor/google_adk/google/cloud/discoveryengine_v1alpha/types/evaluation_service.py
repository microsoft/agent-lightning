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

from google.cloud.discoveryengine_v1alpha.types import evaluation as gcd_evaluation
from google.cloud.discoveryengine_v1alpha.types import sample_query as gcd_sample_query

__protobuf__ = proto.module(
    package="google.cloud.discoveryengine.v1alpha",
    manifest={
        "GetEvaluationRequest",
        "ListEvaluationsRequest",
        "ListEvaluationsResponse",
        "CreateEvaluationRequest",
        "CreateEvaluationMetadata",
        "ListEvaluationResultsRequest",
        "ListEvaluationResultsResponse",
    },
)


class GetEvaluationRequest(proto.Message):
    r"""Request message for
    [EvaluationService.GetEvaluation][google.cloud.discoveryengine.v1alpha.EvaluationService.GetEvaluation]
    method.

    Attributes:
        name (str):
            Required. Full resource name of
            [Evaluation][google.cloud.discoveryengine.v1alpha.Evaluation],
            such as
            ``projects/{project}/locations/{location}/evaluations/{evaluation}``.

            If the caller does not have permission to access the
            [Evaluation][google.cloud.discoveryengine.v1alpha.Evaluation],
            regardless of whether or not it exists, a PERMISSION_DENIED
            error is returned.

            If the requested
            [Evaluation][google.cloud.discoveryengine.v1alpha.Evaluation]
            does not exist, a NOT_FOUND error is returned.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListEvaluationsRequest(proto.Message):
    r"""Request message for
    [EvaluationService.ListEvaluations][google.cloud.discoveryengine.v1alpha.EvaluationService.ListEvaluations]
    method.

    Attributes:
        parent (str):
            Required. The parent location resource name, such as
            ``projects/{project}/locations/{location}``.

            If the caller does not have permission to list
            [Evaluation][google.cloud.discoveryengine.v1alpha.Evaluation]s
            under this location, regardless of whether or not this
            location exists, a ``PERMISSION_DENIED`` error is returned.
        page_size (int):
            Maximum number of
            [Evaluation][google.cloud.discoveryengine.v1alpha.Evaluation]s
            to return. If unspecified, defaults to 100. The maximum
            allowed value is 1000. Values above 1000 will be coerced to
            1000.

            If this field is negative, an ``INVALID_ARGUMENT`` error is
            returned.
        page_token (str):
            A page token
            [ListEvaluationsResponse.next_page_token][google.cloud.discoveryengine.v1alpha.ListEvaluationsResponse.next_page_token],
            received from a previous
            [EvaluationService.ListEvaluations][google.cloud.discoveryengine.v1alpha.EvaluationService.ListEvaluations]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [EvaluationService.ListEvaluations][google.cloud.discoveryengine.v1alpha.EvaluationService.ListEvaluations]
            must match the call that provided the page token. Otherwise,
            an ``INVALID_ARGUMENT`` error is returned.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListEvaluationsResponse(proto.Message):
    r"""Response message for
    [EvaluationService.ListEvaluations][google.cloud.discoveryengine.v1alpha.EvaluationService.ListEvaluations]
    method.

    Attributes:
        evaluations (MutableSequence[google.cloud.discoveryengine_v1alpha.types.Evaluation]):
            The
            [Evaluation][google.cloud.discoveryengine.v1alpha.Evaluation]s.
        next_page_token (str):
            A token that can be sent as
            [ListEvaluationsRequest.page_token][google.cloud.discoveryengine.v1alpha.ListEvaluationsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    @property
    def raw_page(self):
        return self

    evaluations: MutableSequence[gcd_evaluation.Evaluation] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gcd_evaluation.Evaluation,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CreateEvaluationRequest(proto.Message):
    r"""Request message for
    [EvaluationService.CreateEvaluation][google.cloud.discoveryengine.v1alpha.EvaluationService.CreateEvaluation]
    method.

    Attributes:
        parent (str):
            Required. The parent resource name, such as
            ``projects/{project}/locations/{location}``.
        evaluation (google.cloud.discoveryengine_v1alpha.types.Evaluation):
            Required. The
            [Evaluation][google.cloud.discoveryengine.v1alpha.Evaluation]
            to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    evaluation: gcd_evaluation.Evaluation = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gcd_evaluation.Evaluation,
    )


class CreateEvaluationMetadata(proto.Message):
    r"""Metadata for
    [EvaluationService.CreateEvaluation][google.cloud.discoveryengine.v1alpha.EvaluationService.CreateEvaluation]
    method.

    """


class ListEvaluationResultsRequest(proto.Message):
    r"""Request message for
    [EvaluationService.ListEvaluationResults][google.cloud.discoveryengine.v1alpha.EvaluationService.ListEvaluationResults]
    method.

    Attributes:
        evaluation (str):
            Required. The evaluation resource name, such as
            ``projects/{project}/locations/{location}/evaluations/{evaluation}``.

            If the caller does not have permission to list
            [EvaluationResult][] under this evaluation, regardless of
            whether or not this evaluation set exists, a
            ``PERMISSION_DENIED`` error is returned.
        page_size (int):
            Maximum number of [EvaluationResult][] to return. If
            unspecified, defaults to 100. The maximum allowed value is
            1000. Values above 1000 will be coerced to 1000.

            If this field is negative, an ``INVALID_ARGUMENT`` error is
            returned.
        page_token (str):
            A page token
            [ListEvaluationResultsResponse.next_page_token][google.cloud.discoveryengine.v1alpha.ListEvaluationResultsResponse.next_page_token],
            received from a previous
            [EvaluationService.ListEvaluationResults][google.cloud.discoveryengine.v1alpha.EvaluationService.ListEvaluationResults]
            call. Provide this to retrieve the subsequent page.

            When paginating, all other parameters provided to
            [EvaluationService.ListEvaluationResults][google.cloud.discoveryengine.v1alpha.EvaluationService.ListEvaluationResults]
            must match the call that provided the page token. Otherwise,
            an ``INVALID_ARGUMENT`` error is returned.
    """

    evaluation: str = proto.Field(
        proto.STRING,
        number=1,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=2,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ListEvaluationResultsResponse(proto.Message):
    r"""Response message for
    [EvaluationService.ListEvaluationResults][google.cloud.discoveryengine.v1alpha.EvaluationService.ListEvaluationResults]
    method.

    Attributes:
        evaluation_results (MutableSequence[google.cloud.discoveryengine_v1alpha.types.ListEvaluationResultsResponse.EvaluationResult]):
            The
            [EvaluationResult][google.cloud.discoveryengine.v1alpha.ListEvaluationResultsResponse.EvaluationResult]s.
        next_page_token (str):
            A token that can be sent as
            [ListEvaluationResultsRequest.page_token][google.cloud.discoveryengine.v1alpha.ListEvaluationResultsRequest.page_token]
            to retrieve the next page. If this field is omitted, there
            are no subsequent pages.
    """

    class EvaluationResult(proto.Message):
        r"""Represents the results of an evaluation for a single
        [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery].

        Attributes:
            sample_query (google.cloud.discoveryengine_v1alpha.types.SampleQuery):
                Output only. The
                [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery]
                that was evaluated.
            quality_metrics (google.cloud.discoveryengine_v1alpha.types.QualityMetrics):
                Output only. The metrics produced by the evaluation, for a
                given
                [SampleQuery][google.cloud.discoveryengine.v1alpha.SampleQuery].
        """

        sample_query: gcd_sample_query.SampleQuery = proto.Field(
            proto.MESSAGE,
            number=1,
            message=gcd_sample_query.SampleQuery,
        )
        quality_metrics: gcd_evaluation.QualityMetrics = proto.Field(
            proto.MESSAGE,
            number=2,
            message=gcd_evaluation.QualityMetrics,
        )

    @property
    def raw_page(self):
        return self

    evaluation_results: MutableSequence[EvaluationResult] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=EvaluationResult,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
