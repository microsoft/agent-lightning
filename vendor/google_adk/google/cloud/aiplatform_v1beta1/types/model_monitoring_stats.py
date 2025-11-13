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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ModelMonitoringStats",
        "ModelMonitoringStatsDataPoint",
        "ModelMonitoringTabularStats",
        "SearchModelMonitoringStatsFilter",
    },
)


class ModelMonitoringStats(proto.Message):
    r"""Represents the collection of statistics for a metric.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tabular_stats (google.cloud.aiplatform_v1beta1.types.ModelMonitoringTabularStats):
            Generated tabular statistics.

            This field is a member of `oneof`_ ``stats``.
    """

    tabular_stats: "ModelMonitoringTabularStats" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="stats",
        message="ModelMonitoringTabularStats",
    )


class ModelMonitoringStatsDataPoint(proto.Message):
    r"""Represents a single statistics data point.

    Attributes:
        current_stats (google.cloud.aiplatform_v1beta1.types.ModelMonitoringStatsDataPoint.TypedValue):
            Statistics from current dataset.
        baseline_stats (google.cloud.aiplatform_v1beta1.types.ModelMonitoringStatsDataPoint.TypedValue):
            Statistics from baseline dataset.
        threshold_value (float):
            Threshold value.
        has_anomaly (bool):
            Indicate if the statistics has anomaly.
        model_monitoring_job (str):
            Model monitoring job resource name.
        schedule (str):
            Schedule resource name.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Statistics create time.
        algorithm (str):
            Algorithm used to calculated the metrics, eg:
            jensen_shannon_divergence, l_infinity.
    """

    class TypedValue(proto.Message):
        r"""Typed value of the statistics.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            double_value (float):
                Double.

                This field is a member of `oneof`_ ``value``.
            distribution_value (google.cloud.aiplatform_v1beta1.types.ModelMonitoringStatsDataPoint.TypedValue.DistributionDataValue):
                Distribution.

                This field is a member of `oneof`_ ``value``.
        """

        class DistributionDataValue(proto.Message):
            r"""Summary statistics for a population of values.

            Attributes:
                distribution (google.protobuf.struct_pb2.Value):
                    Predictive monitoring drift distribution in
                    ``tensorflow.metadata.v0.DatasetFeatureStatistics`` format.
                distribution_deviation (float):
                    Distribution distance deviation from the current dataset's
                    statistics to baseline dataset's statistics.

                    - For categorical feature, the distribution distance is
                      calculated by L-inifinity norm or Jensen–Shannon
                      divergence.
                    - For numerical feature, the distribution distance is
                      calculated by Jensen–Shannon divergence.
            """

            distribution: struct_pb2.Value = proto.Field(
                proto.MESSAGE,
                number=1,
                message=struct_pb2.Value,
            )
            distribution_deviation: float = proto.Field(
                proto.DOUBLE,
                number=2,
            )

        double_value: float = proto.Field(
            proto.DOUBLE,
            number=1,
            oneof="value",
        )
        distribution_value: (
            "ModelMonitoringStatsDataPoint.TypedValue.DistributionDataValue"
        ) = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="value",
            message="ModelMonitoringStatsDataPoint.TypedValue.DistributionDataValue",
        )

    current_stats: TypedValue = proto.Field(
        proto.MESSAGE,
        number=1,
        message=TypedValue,
    )
    baseline_stats: TypedValue = proto.Field(
        proto.MESSAGE,
        number=2,
        message=TypedValue,
    )
    threshold_value: float = proto.Field(
        proto.DOUBLE,
        number=3,
    )
    has_anomaly: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    model_monitoring_job: str = proto.Field(
        proto.STRING,
        number=5,
    )
    schedule: str = proto.Field(
        proto.STRING,
        number=6,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    algorithm: str = proto.Field(
        proto.STRING,
        number=8,
    )


class ModelMonitoringTabularStats(proto.Message):
    r"""A collection of data points that describes the time-varying
    values of a tabular metric.

    Attributes:
        stats_name (str):
            The stats name.
        objective_type (str):
            One of the supported monitoring objectives:
            ``raw-feature-drift`` ``prediction-output-drift``
            ``feature-attribution``
        data_points (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringStatsDataPoint]):
            The data points of this time series. When
            listing time series, points are returned in
            reverse time order.
    """

    stats_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    objective_type: str = proto.Field(
        proto.STRING,
        number=2,
    )
    data_points: MutableSequence["ModelMonitoringStatsDataPoint"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="ModelMonitoringStatsDataPoint",
    )


class SearchModelMonitoringStatsFilter(proto.Message):
    r"""Filter for searching ModelMonitoringStats.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tabular_stats_filter (google.cloud.aiplatform_v1beta1.types.SearchModelMonitoringStatsFilter.TabularStatsFilter):
            Tabular statistics filter.

            This field is a member of `oneof`_ ``filter``.
    """

    class TabularStatsFilter(proto.Message):
        r"""Tabular statistics filter.

        Attributes:
            stats_name (str):
                If not specified, will return all the stats_names.
            objective_type (str):
                One of the supported monitoring objectives:
                ``raw-feature-drift`` ``prediction-output-drift``
                ``feature-attribution``
            model_monitoring_job (str):
                From a particular monitoring job.
            model_monitoring_schedule (str):
                From a particular monitoring schedule.
            algorithm (str):
                Specify the algorithm type used for distance calculation,
                eg: jensen_shannon_divergence, l_infinity.
        """

        stats_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        objective_type: str = proto.Field(
            proto.STRING,
            number=2,
        )
        model_monitoring_job: str = proto.Field(
            proto.STRING,
            number=3,
        )
        model_monitoring_schedule: str = proto.Field(
            proto.STRING,
            number=4,
        )
        algorithm: str = proto.Field(
            proto.STRING,
            number=5,
        )

    tabular_stats_filter: TabularStatsFilter = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="filter",
        message=TabularStatsFilter,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
