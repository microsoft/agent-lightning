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
        "ModelMonitoringAlertCondition",
        "ModelMonitoringAnomaly",
        "ModelMonitoringAlert",
    },
)


class ModelMonitoringAlertCondition(proto.Message):
    r"""Monitoring alert triggered condition.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        threshold (float):
            A condition that compares a stats value
            against a threshold. Alert will be triggered if
            value above the threshold.

            This field is a member of `oneof`_ ``condition``.
    """

    threshold: float = proto.Field(
        proto.DOUBLE,
        number=1,
        oneof="condition",
    )


class ModelMonitoringAnomaly(proto.Message):
    r"""Represents a single model monitoring anomaly.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tabular_anomaly (google.cloud.aiplatform_v1beta1.types.ModelMonitoringAnomaly.TabularAnomaly):
            Tabular anomaly.

            This field is a member of `oneof`_ ``anomaly``.
        model_monitoring_job (str):
            Model monitoring job resource name.
        algorithm (str):
            Algorithm used to calculated the metrics, eg:
            jensen_shannon_divergence, l_infinity.
    """

    class TabularAnomaly(proto.Message):
        r"""Tabular anomaly details.

        Attributes:
            anomaly_uri (str):
                Additional anomaly information. e.g. Google
                Cloud Storage uri.
            summary (str):
                Overview of this anomaly.
            anomaly (google.protobuf.struct_pb2.Value):
                Anomaly body.
            trigger_time (google.protobuf.timestamp_pb2.Timestamp):
                The time the anomaly was triggered.
            condition (google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlertCondition):
                The alert condition associated with this
                anomaly.
        """

        anomaly_uri: str = proto.Field(
            proto.STRING,
            number=1,
        )
        summary: str = proto.Field(
            proto.STRING,
            number=2,
        )
        anomaly: struct_pb2.Value = proto.Field(
            proto.MESSAGE,
            number=3,
            message=struct_pb2.Value,
        )
        trigger_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=4,
            message=timestamp_pb2.Timestamp,
        )
        condition: "ModelMonitoringAlertCondition" = proto.Field(
            proto.MESSAGE,
            number=5,
            message="ModelMonitoringAlertCondition",
        )

    tabular_anomaly: TabularAnomaly = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="anomaly",
        message=TabularAnomaly,
    )
    model_monitoring_job: str = proto.Field(
        proto.STRING,
        number=2,
    )
    algorithm: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ModelMonitoringAlert(proto.Message):
    r"""Represents a single monitoring alert. This is currently used
    in the SearchModelMonitoringAlerts api, thus the alert wrapped
    in this message belongs to the resource asked in the request.

    Attributes:
        stats_name (str):
            The stats name.
        objective_type (str):
            One of the supported monitoring objectives:
            ``raw-feature-drift`` ``prediction-output-drift``
            ``feature-attribution``
        alert_time (google.protobuf.timestamp_pb2.Timestamp):
            Alert creation time.
        anomaly (google.cloud.aiplatform_v1beta1.types.ModelMonitoringAnomaly):
            Anomaly details.
    """

    stats_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    objective_type: str = proto.Field(
        proto.STRING,
        number=2,
    )
    alert_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    anomaly: "ModelMonitoringAnomaly" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="ModelMonitoringAnomaly",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
