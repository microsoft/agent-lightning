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

from google.cloud.aiplatform_v1beta1.types import feature_monitor
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "FeatureMonitorJob",
    },
)


class FeatureMonitorJob(proto.Message):
    r"""Vertex AI Feature Monitor Job.

    Attributes:
        name (str):
            Identifier. Name of the FeatureMonitorJob. Format:
            ``projects/{project}/locations/{location}/featureGroups/{feature_group}/featureMonitors/{feature_monitor}/featureMonitorJobs/{feature_monitor_job}``.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            FeatureMonitorJob was created. Creation of a
            FeatureMonitorJob means that the job is pending
            / waiting for sufficient resources but may not
            have started running yet.
        final_status (google.rpc.status_pb2.Status):
            Output only. Final status of the
            FeatureMonitorJob.
        job_summary (google.cloud.aiplatform_v1beta1.types.FeatureMonitorJob.JobSummary):
            Output only. Summary from the
            FeatureMonitorJob.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize your FeatureMonitorJob.

            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            on and examples of labels. No more than 64 user
            labels can be associated with one
            FeatureMonitor(System labels are excluded)."
            System reserved label keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable.
        description (str):
            Optional. Description of the FeatureMonitor.
        drift_base_feature_monitor_job_id (int):
            Output only. FeatureMonitorJob ID comparing
            to which the drift is calculated.
        drift_base_snapshot_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Data snapshot time comparing to
            which the drift is calculated.
        feature_selection_config (google.cloud.aiplatform_v1beta1.types.FeatureSelectionConfig):
            Output only. Feature selection config used
            when creating FeatureMonitorJob.
        trigger_type (google.cloud.aiplatform_v1beta1.types.FeatureMonitorJob.FeatureMonitorJobTrigger):
            Output only. Trigger type of the Feature
            Monitor Job.
    """

    class FeatureMonitorJobTrigger(proto.Enum):
        r"""Choices of the trigger type.

        Values:
            FEATURE_MONITOR_JOB_TRIGGER_UNSPECIFIED (0):
                Trigger type unspecified.
            FEATURE_MONITOR_JOB_TRIGGER_PERIODIC (1):
                Triggered by periodic schedule.
            FEATURE_MONITOR_JOB_TRIGGER_ON_DEMAND (2):
                Triggered on demand by
                CreateFeatureMonitorJob request.
        """

        FEATURE_MONITOR_JOB_TRIGGER_UNSPECIFIED = 0
        FEATURE_MONITOR_JOB_TRIGGER_PERIODIC = 1
        FEATURE_MONITOR_JOB_TRIGGER_ON_DEMAND = 2

    class JobSummary(proto.Message):
        r"""Summary from the FeatureMonitorJob.

        Attributes:
            total_slot_ms (int):
                Output only. BigQuery slot milliseconds
                consumed.
            feature_stats_and_anomalies (MutableSequence[google.cloud.aiplatform_v1beta1.types.FeatureStatsAndAnomaly]):
                Output only. Features and their stats and
                anomalies
        """

        total_slot_ms: int = proto.Field(
            proto.INT64,
            number=1,
        )
        feature_stats_and_anomalies: MutableSequence[
            feature_monitor.FeatureStatsAndAnomaly
        ] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message=feature_monitor.FeatureStatsAndAnomaly,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    final_status: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=3,
        message=status_pb2.Status,
    )
    job_summary: JobSummary = proto.Field(
        proto.MESSAGE,
        number=4,
        message=JobSummary,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=5,
    )
    description: str = proto.Field(
        proto.STRING,
        number=6,
    )
    drift_base_feature_monitor_job_id: int = proto.Field(
        proto.INT64,
        number=7,
    )
    drift_base_snapshot_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    feature_selection_config: feature_monitor.FeatureSelectionConfig = proto.Field(
        proto.MESSAGE,
        number=9,
        message=feature_monitor.FeatureSelectionConfig,
    )
    trigger_type: FeatureMonitorJobTrigger = proto.Field(
        proto.ENUM,
        number=10,
        enum=FeatureMonitorJobTrigger,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
