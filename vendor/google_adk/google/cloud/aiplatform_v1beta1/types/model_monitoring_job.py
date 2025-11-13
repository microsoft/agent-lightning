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

from google.cloud.aiplatform_v1beta1.types import job_state
from google.cloud.aiplatform_v1beta1.types import (
    model_monitoring_spec as gca_model_monitoring_spec,
)
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from google.type import interval_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ModelMonitoringJob",
        "ModelMonitoringJobExecutionDetail",
    },
)


class ModelMonitoringJob(proto.Message):
    r"""Represents a model monitoring job that analyze dataset using
    different monitoring algorithm.

    Attributes:
        name (str):
            Output only. Resource name of a ModelMonitoringJob. Format:
            ``projects/{project_id}/locations/{location_id}/modelMonitors/{model_monitor_id}/modelMonitoringJobs/{model_monitoring_job_id}``
        display_name (str):
            The display name of the ModelMonitoringJob.
            The name can be up to 128 characters long and
            can consist of any UTF-8.
        model_monitoring_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringSpec):
            Monitoring monitoring job spec. It outlines
            the specifications for monitoring objectives,
            notifications, and result exports. If left
            blank, the default monitoring specifications
            from the top-level resource 'ModelMonitor' will
            be applied. If provided, we will use the
            specification defined here rather than the
            default one.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ModelMonitoringJob was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ModelMonitoringJob was updated most recently.
        state (google.cloud.aiplatform_v1beta1.types.JobState):
            Output only. The state of the monitoring job.

            - When the job is still creating, the state will be
              'JOB_STATE_PENDING'.
            - Once the job is successfully created, the state will be
              'JOB_STATE_RUNNING'.
            - Once the job is finished, the state will be one of
              'JOB_STATE_FAILED', 'JOB_STATE_SUCCEEDED',
              'JOB_STATE_PARTIALLY_SUCCEEDED'.
        schedule (str):
            Output only. Schedule resource name. It will
            only appear when this job is triggered by a
            schedule.
        job_execution_detail (google.cloud.aiplatform_v1beta1.types.ModelMonitoringJobExecutionDetail):
            Output only. Execution results for all the
            monitoring objectives.
        schedule_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ModelMonitoringJob was scheduled. It will only
            appear when this job is triggered by a schedule.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    model_monitoring_spec: gca_model_monitoring_spec.ModelMonitoringSpec = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gca_model_monitoring_spec.ModelMonitoringSpec,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    state: job_state.JobState = proto.Field(
        proto.ENUM,
        number=6,
        enum=job_state.JobState,
    )
    schedule: str = proto.Field(
        proto.STRING,
        number=7,
    )
    job_execution_detail: "ModelMonitoringJobExecutionDetail" = proto.Field(
        proto.MESSAGE,
        number=8,
        message="ModelMonitoringJobExecutionDetail",
    )
    schedule_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )


class ModelMonitoringJobExecutionDetail(proto.Message):
    r"""Represent the execution details of the job.

    Attributes:
        baseline_datasets (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringJobExecutionDetail.ProcessedDataset]):
            Processed baseline datasets.
        target_datasets (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringJobExecutionDetail.ProcessedDataset]):
            Processed target datasets.
        objective_status (MutableMapping[str, google.rpc.status_pb2.Status]):
            Status of data processing for each monitoring
            objective. Key is the objective.
        error (google.rpc.status_pb2.Status):
            Additional job error status.
    """

    class ProcessedDataset(proto.Message):
        r"""Processed dataset information.

        Attributes:
            location (str):
                Actual data location of the processed
                dataset.
            time_range (google.type.interval_pb2.Interval):
                Dataset time range information if any.
        """

        location: str = proto.Field(
            proto.STRING,
            number=1,
        )
        time_range: interval_pb2.Interval = proto.Field(
            proto.MESSAGE,
            number=2,
            message=interval_pb2.Interval,
        )

    baseline_datasets: MutableSequence[ProcessedDataset] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=ProcessedDataset,
    )
    target_datasets: MutableSequence[ProcessedDataset] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=ProcessedDataset,
    )
    objective_status: MutableMapping[str, status_pb2.Status] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=3,
        message=status_pb2.Status,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=4,
        message=status_pb2.Status,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
