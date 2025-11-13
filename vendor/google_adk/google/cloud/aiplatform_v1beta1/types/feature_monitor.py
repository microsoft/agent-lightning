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

from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.type import interval_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "FeatureMonitor",
        "ScheduleConfig",
        "FeatureSelectionConfig",
        "FeatureStatsAndAnomaly",
        "FeatureStatsAndAnomalySpec",
    },
)


class FeatureMonitor(proto.Message):
    r"""Vertex AI Feature Monitor.

    Attributes:
        name (str):
            Identifier. Name of the FeatureMonitor. Format:
            ``projects/{project}/locations/{location}/featureGroups/{featureGroup}/featureMonitors/{featureMonitor}``
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            FeatureMonitor was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            FeatureMonitor was last updated.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize your FeatureMonitor.

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
        schedule_config (google.cloud.aiplatform_v1beta1.types.ScheduleConfig):
            Required. Schedule config for the
            FeatureMonitor.
        feature_selection_config (google.cloud.aiplatform_v1beta1.types.FeatureSelectionConfig):
            Required. Feature selection config for the
            FeatureMonitor.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=4,
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
    schedule_config: "ScheduleConfig" = proto.Field(
        proto.MESSAGE,
        number=7,
        message="ScheduleConfig",
    )
    feature_selection_config: "FeatureSelectionConfig" = proto.Field(
        proto.MESSAGE,
        number=8,
        message="FeatureSelectionConfig",
    )


class ScheduleConfig(proto.Message):
    r"""Schedule configuration for the FeatureMonitor.

    Attributes:
        cron (str):
            Cron schedule (https://en.wikipedia.org/wiki/Cron) to launch
            scheduled runs. To explicitly set a timezone to the cron
            tab, apply a prefix in the cron tab:
            "CRON_TZ=${IANA_TIME_ZONE}" or "TZ=${IANA_TIME_ZONE}". The
            ${IANA_TIME_ZONE} may only be a valid string from IANA time
            zone database. For example, "CRON_TZ=America/New_York 1 \*
            \* \* \*", or "TZ=America/New_York 1 \* \* \* \*".
    """

    cron: str = proto.Field(
        proto.STRING,
        number=1,
    )


class FeatureSelectionConfig(proto.Message):
    r"""Feature selection configuration for the FeatureMonitor.

    Attributes:
        feature_configs (MutableSequence[google.cloud.aiplatform_v1beta1.types.FeatureSelectionConfig.FeatureConfig]):
            Optional. A list of features to be monitored
            and each feature's drift threshold.
    """

    class FeatureConfig(proto.Message):
        r"""Feature configuration.

        Attributes:
            feature_id (str):
                Required. The ID of the feature resource.
                Final component of the Feature's resource name.
            drift_threshold (float):
                Optional. Drift threshold. If calculated
                difference with baseline data larger than
                threshold, it will be considered as the feature
                has drift. If not present, the threshold will be
                default to 0.3.
        """

        feature_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        drift_threshold: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )

    feature_configs: MutableSequence[FeatureConfig] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=FeatureConfig,
    )


class FeatureStatsAndAnomaly(proto.Message):
    r"""Stats and Anomaly generated by FeatureMonitorJobs. Anomaly
    only includes Drift.

    Attributes:
        feature_id (str):
            Feature Id.
        feature_stats (google.protobuf.struct_pb2.Value):
            Feature stats. e.g. histogram buckets.
            In the format of
            tensorflow.metadata.v0.DatasetFeatureStatistics.
        distribution_deviation (float):
            Deviation from the current stats to baseline
            stats.
              1. For categorical feature, the distribution
                distance is calculated by      L-inifinity
                norm.
              2. For numerical feature, the distribution
                distance is calculated by
                Jensen–Shannon divergence.
        drift_detection_threshold (float):
            This is the threshold used when detecting drifts, which is
            set in
            FeatureMonitor.FeatureSelectionConfig.FeatureConfig.drift_threshold
        drift_detected (bool):
            If set to true, indicates current stats is
            detected as and comparing with baseline stats.
        stats_time (google.protobuf.timestamp_pb2.Timestamp):
            The timestamp we take snapshot for feature
            values to generate stats.
        feature_monitor_job_id (int):
            The ID of the FeatureMonitorJob that
            generated this FeatureStatsAndAnomaly.
        feature_monitor_id (str):
            The ID of the FeatureMonitor that this
            FeatureStatsAndAnomaly generated according to.
    """

    feature_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    feature_stats: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Value,
    )
    distribution_deviation: float = proto.Field(
        proto.DOUBLE,
        number=3,
    )
    drift_detection_threshold: float = proto.Field(
        proto.DOUBLE,
        number=4,
    )
    drift_detected: bool = proto.Field(
        proto.BOOL,
        number=5,
    )
    stats_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    feature_monitor_job_id: int = proto.Field(
        proto.INT64,
        number=7,
    )
    feature_monitor_id: str = proto.Field(
        proto.STRING,
        number=8,
    )


class FeatureStatsAndAnomalySpec(proto.Message):
    r"""Defines how to select FeatureStatsAndAnomaly to be populated
    in response. If set, retrieves FeatureStatsAndAnomaly generated
    by FeatureMonitors based on this spec.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        latest_stats_count (int):
            Optional. If set, returns the most recent count of stats.
            Valid value is [0, 100]. If stats_time_range is set, return
            most recent count of stats within the stats_time_range.

            This field is a member of `oneof`_ ``_latest_stats_count``.
        stats_time_range (google.type.interval_pb2.Interval):
            Optional. If set, return all stats generated between
            [start_time, end_time). If latest_stats_count is set, return
            the most recent count of stats within the stats_time_range.
    """

    latest_stats_count: int = proto.Field(
        proto.INT32,
        number=1,
        optional=True,
    )
    stats_time_range: interval_pb2.Interval = proto.Field(
        proto.MESSAGE,
        number=2,
        message=interval_pb2.Interval,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
