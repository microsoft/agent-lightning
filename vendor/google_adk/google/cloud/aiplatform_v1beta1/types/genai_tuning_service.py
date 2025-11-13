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

from google.cloud.aiplatform_v1beta1.types import io
from google.cloud.aiplatform_v1beta1.types import operation
from google.cloud.aiplatform_v1beta1.types import tuning_job as gca_tuning_job


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateTuningJobRequest",
        "GetTuningJobRequest",
        "ListTuningJobsRequest",
        "ListTuningJobsResponse",
        "CancelTuningJobRequest",
        "RebaseTunedModelRequest",
        "RebaseTunedModelOperationMetadata",
    },
)


class CreateTuningJobRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.CreateTuningJob][google.cloud.aiplatform.v1beta1.GenAiTuningService.CreateTuningJob].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            TuningJob in. Format:
            ``projects/{project}/locations/{location}``
        tuning_job (google.cloud.aiplatform_v1beta1.types.TuningJob):
            Required. The TuningJob to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tuning_job: gca_tuning_job.TuningJob = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_tuning_job.TuningJob,
    )


class GetTuningJobRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.GetTuningJob][google.cloud.aiplatform.v1beta1.GenAiTuningService.GetTuningJob].

    Attributes:
        name (str):
            Required. The name of the TuningJob resource. Format:
            ``projects/{project}/locations/{location}/tuningJobs/{tuning_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListTuningJobsRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.ListTuningJobs][google.cloud.aiplatform.v1beta1.GenAiTuningService.ListTuningJobs].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            TuningJobs from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Optional. The standard list filter.
        page_size (int):
            Optional. The standard list page size.
        page_token (str):
            Optional. The standard list page token. Typically obtained
            via [ListTuningJob.next_page_token][] of the previous
            GenAiTuningService.ListTuningJob][] call.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )


class ListTuningJobsResponse(proto.Message):
    r"""Response message for
    [GenAiTuningService.ListTuningJobs][google.cloud.aiplatform.v1beta1.GenAiTuningService.ListTuningJobs]

    Attributes:
        tuning_jobs (MutableSequence[google.cloud.aiplatform_v1beta1.types.TuningJob]):
            List of TuningJobs in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListTuningJobsRequest.page_token][google.cloud.aiplatform.v1beta1.ListTuningJobsRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    tuning_jobs: MutableSequence[gca_tuning_job.TuningJob] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_tuning_job.TuningJob,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class CancelTuningJobRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.CancelTuningJob][google.cloud.aiplatform.v1beta1.GenAiTuningService.CancelTuningJob].

    Attributes:
        name (str):
            Required. The name of the TuningJob to cancel. Format:
            ``projects/{project}/locations/{location}/tuningJobs/{tuning_job}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class RebaseTunedModelRequest(proto.Message):
    r"""Request message for
    [GenAiTuningService.RebaseTunedModel][google.cloud.aiplatform.v1beta1.GenAiTuningService.RebaseTunedModel].

    Attributes:
        parent (str):
            Required. The resource name of the Location into which to
            rebase the Model. Format:
            ``projects/{project}/locations/{location}``
        tuned_model_ref (google.cloud.aiplatform_v1beta1.types.TunedModelRef):
            Required. TunedModel reference to retrieve
            the legacy model information.
        tuning_job (google.cloud.aiplatform_v1beta1.types.TuningJob):
            Optional. The TuningJob to be updated. Users
            can use this TuningJob field to overwrite tuning
            configs.
        artifact_destination (google.cloud.aiplatform_v1beta1.types.GcsDestination):
            Optional. The Google Cloud Storage location
            to write the artifacts.
        deploy_to_same_endpoint (bool):
            Optional. By default, bison to gemini
            migration will always create new model/endpoint,
            but for gemini-1.0 to gemini-1.5 migration, we
            default deploy to the same endpoint. See details
            in this Section.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    tuned_model_ref: gca_tuning_job.TunedModelRef = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_tuning_job.TunedModelRef,
    )
    tuning_job: gca_tuning_job.TuningJob = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gca_tuning_job.TuningJob,
    )
    artifact_destination: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=4,
        message=io.GcsDestination,
    )
    deploy_to_same_endpoint: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


class RebaseTunedModelOperationMetadata(proto.Message):
    r"""Runtime operation information for
    [GenAiTuningService.RebaseTunedModel][google.cloud.aiplatform.v1beta1.GenAiTuningService.RebaseTunedModel].

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1beta1.types.GenericOperationMetadata):
            The common part of the operation generic
            information.
    """

    generic_metadata: operation.GenericOperationMetadata = proto.Field(
        proto.MESSAGE,
        number=1,
        message=operation.GenericOperationMetadata,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
