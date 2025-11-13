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

from google.cloud.aiplatform_v1beta1.types import explanation
from google.cloud.aiplatform_v1beta1.types import io
from google.cloud.aiplatform_v1beta1.types import machine_resources
from google.cloud.aiplatform_v1beta1.types import model_monitoring_alert
from google.type import interval_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ModelMonitoringSpec",
        "ModelMonitoringObjectiveSpec",
        "ModelMonitoringOutputSpec",
        "ModelMonitoringInput",
        "ModelMonitoringNotificationSpec",
    },
)


class ModelMonitoringSpec(proto.Message):
    r"""Monitoring monitoring job spec. It outlines the
    specifications for monitoring objectives, notifications, and
    result exports.

    Attributes:
        objective_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveSpec):
            The monitoring objective spec.
        notification_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringNotificationSpec):
            The model monitoring notification spec.
        output_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringOutputSpec):
            The Output destination spec for metrics,
            error logs, etc.
    """

    objective_spec: "ModelMonitoringObjectiveSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ModelMonitoringObjectiveSpec",
    )
    notification_spec: "ModelMonitoringNotificationSpec" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ModelMonitoringNotificationSpec",
    )
    output_spec: "ModelMonitoringOutputSpec" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="ModelMonitoringOutputSpec",
    )


class ModelMonitoringObjectiveSpec(proto.Message):
    r"""Monitoring objectives spec.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        tabular_objective (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveSpec.TabularObjective):
            Tabular monitoring objective.

            This field is a member of `oneof`_ ``objective``.
        explanation_spec (google.cloud.aiplatform_v1beta1.types.ExplanationSpec):
            The explanation spec.
            This spec is required when the objectives spec
            includes feature attribution objectives.
        baseline_dataset (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput):
            Baseline dataset.
            It could be the training dataset or production
            serving dataset from a previous period.
        target_dataset (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput):
            Target dataset.
    """

    class DataDriftSpec(proto.Message):
        r"""Data drift monitoring spec.
        Data drift measures the distribution distance between the
        current dataset and a baseline dataset. A typical use case is to
        detect data drift between the recent production serving dataset
        and the training dataset, or to compare the recent production
        dataset with a dataset from a previous period.

        Attributes:
            features (MutableSequence[str]):
                Feature names / Prediction output names
                interested in monitoring. These should be a
                subset of the input feature names or prediction
                output names specified in the monitoring schema.
                If the field is not specified all features /
                prediction outputs outlied in the monitoring
                schema will be used.
            categorical_metric_type (str):
                Supported metrics type:

                - l_infinity
                - jensen_shannon_divergence
            numeric_metric_type (str):
                Supported metrics type:

                - jensen_shannon_divergence
            default_categorical_alert_condition (google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlertCondition):
                Default alert condition for all the
                categorical features.
            default_numeric_alert_condition (google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlertCondition):
                Default alert condition for all the numeric
                features.
            feature_alert_conditions (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlertCondition]):
                Per feature alert condition will override
                default alert condition.
        """

        features: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )
        categorical_metric_type: str = proto.Field(
            proto.STRING,
            number=2,
        )
        numeric_metric_type: str = proto.Field(
            proto.STRING,
            number=3,
        )
        default_categorical_alert_condition: (
            model_monitoring_alert.ModelMonitoringAlertCondition
        ) = proto.Field(
            proto.MESSAGE,
            number=4,
            message=model_monitoring_alert.ModelMonitoringAlertCondition,
        )
        default_numeric_alert_condition: (
            model_monitoring_alert.ModelMonitoringAlertCondition
        ) = proto.Field(
            proto.MESSAGE,
            number=5,
            message=model_monitoring_alert.ModelMonitoringAlertCondition,
        )
        feature_alert_conditions: MutableMapping[
            str, model_monitoring_alert.ModelMonitoringAlertCondition
        ] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=6,
            message=model_monitoring_alert.ModelMonitoringAlertCondition,
        )

    class FeatureAttributionSpec(proto.Message):
        r"""Feature attribution monitoring spec.

        Attributes:
            features (MutableSequence[str]):
                Feature names interested in monitoring.
                These should be a subset of the input feature
                names specified in the monitoring schema. If the
                field is not specified all features outlied in
                the monitoring schema will be used.
            default_alert_condition (google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlertCondition):
                Default alert condition for all the features.
            feature_alert_conditions (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ModelMonitoringAlertCondition]):
                Per feature alert condition will override
                default alert condition.
            batch_explanation_dedicated_resources (google.cloud.aiplatform_v1beta1.types.BatchDedicatedResources):
                The config of resources used by the Model Monitoring during
                the batch explanation for non-AutoML models. If not set,
                ``n1-standard-2`` machine type will be used by default.
        """

        features: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )
        default_alert_condition: (
            model_monitoring_alert.ModelMonitoringAlertCondition
        ) = proto.Field(
            proto.MESSAGE,
            number=2,
            message=model_monitoring_alert.ModelMonitoringAlertCondition,
        )
        feature_alert_conditions: MutableMapping[
            str, model_monitoring_alert.ModelMonitoringAlertCondition
        ] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=3,
            message=model_monitoring_alert.ModelMonitoringAlertCondition,
        )
        batch_explanation_dedicated_resources: (
            machine_resources.BatchDedicatedResources
        ) = proto.Field(
            proto.MESSAGE,
            number=4,
            message=machine_resources.BatchDedicatedResources,
        )

    class TabularObjective(proto.Message):
        r"""Tabular monitoring objective.

        Attributes:
            feature_drift_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveSpec.DataDriftSpec):
                Input feature distribution drift monitoring
                spec.
            prediction_output_drift_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveSpec.DataDriftSpec):
                Prediction output distribution drift
                monitoring spec.
            feature_attribution_spec (google.cloud.aiplatform_v1beta1.types.ModelMonitoringObjectiveSpec.FeatureAttributionSpec):
                Feature attribution monitoring spec.
        """

        feature_drift_spec: "ModelMonitoringObjectiveSpec.DataDriftSpec" = proto.Field(
            proto.MESSAGE,
            number=10,
            message="ModelMonitoringObjectiveSpec.DataDriftSpec",
        )
        prediction_output_drift_spec: "ModelMonitoringObjectiveSpec.DataDriftSpec" = (
            proto.Field(
                proto.MESSAGE,
                number=11,
                message="ModelMonitoringObjectiveSpec.DataDriftSpec",
            )
        )
        feature_attribution_spec: (
            "ModelMonitoringObjectiveSpec.FeatureAttributionSpec"
        ) = proto.Field(
            proto.MESSAGE,
            number=12,
            message="ModelMonitoringObjectiveSpec.FeatureAttributionSpec",
        )

    tabular_objective: TabularObjective = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="objective",
        message=TabularObjective,
    )
    explanation_spec: explanation.ExplanationSpec = proto.Field(
        proto.MESSAGE,
        number=3,
        message=explanation.ExplanationSpec,
    )
    baseline_dataset: "ModelMonitoringInput" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="ModelMonitoringInput",
    )
    target_dataset: "ModelMonitoringInput" = proto.Field(
        proto.MESSAGE,
        number=5,
        message="ModelMonitoringInput",
    )


class ModelMonitoringOutputSpec(proto.Message):
    r"""Specification for the export destination of monitoring
    results, including metrics, logs, etc.

    Attributes:
        gcs_base_directory (google.cloud.aiplatform_v1beta1.types.GcsDestination):
            Google Cloud Storage base folder path for
            metrics, error logs, etc.
    """

    gcs_base_directory: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=1,
        message=io.GcsDestination,
    )


class ModelMonitoringInput(proto.Message):
    r"""Model monitoring data input spec.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        columnized_dataset (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput.ModelMonitoringDataset):
            Columnized dataset.

            This field is a member of `oneof`_ ``dataset``.
        batch_prediction_output (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput.BatchPredictionOutput):
            Vertex AI Batch prediction Job.

            This field is a member of `oneof`_ ``dataset``.
        vertex_endpoint_logs (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput.VertexEndpointLogs):
            Vertex AI Endpoint request & response
            logging.

            This field is a member of `oneof`_ ``dataset``.
        time_interval (google.type.interval_pb2.Interval):
            The time interval (pair of start_time and end_time) for
            which results should be returned.

            This field is a member of `oneof`_ ``time_spec``.
        time_offset (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput.TimeOffset):
            The time offset setting for which results
            should be returned.

            This field is a member of `oneof`_ ``time_spec``.
    """

    class ModelMonitoringDataset(proto.Message):
        r"""Input dataset spec.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            vertex_dataset (str):
                Resource name of the Vertex AI managed
                dataset.

                This field is a member of `oneof`_ ``data_location``.
            gcs_source (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource):
                Google Cloud Storage data source.

                This field is a member of `oneof`_ ``data_location``.
            bigquery_source (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringBigQuerySource):
                BigQuery data source.

                This field is a member of `oneof`_ ``data_location``.
            timestamp_field (str):
                The timestamp field. Usually for serving
                data.
        """

        class ModelMonitoringGcsSource(proto.Message):
            r"""Dataset spec for data stored in Google Cloud Storage.

            Attributes:
                gcs_uri (str):
                    Google Cloud Storage URI to the input
                    file(s). May contain wildcards. For more
                    information on wildcards, see
                    https://cloud.google.com/storage/docs/wildcards.
                format_ (google.cloud.aiplatform_v1beta1.types.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat):
                    Data format of the dataset.
            """

            class DataFormat(proto.Enum):
                r"""Supported data format.

                Values:
                    DATA_FORMAT_UNSPECIFIED (0):
                        Data format unspecified, used when this field
                        is unset.
                    CSV (1):
                        CSV files.
                    TF_RECORD (2):
                        TfRecord files
                    JSONL (3):
                        JsonL files.
                """

                DATA_FORMAT_UNSPECIFIED = 0
                CSV = 1
                TF_RECORD = 2
                JSONL = 3

            gcs_uri: str = proto.Field(
                proto.STRING,
                number=1,
            )
            format_: "ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat" = proto.Field(
                proto.ENUM,
                number=2,
                enum="ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat",
            )

        class ModelMonitoringBigQuerySource(proto.Message):
            r"""Dataset spec for data sotred in BigQuery.

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                table_uri (str):
                    BigQuery URI to a table, up to 2000 characters long. All the
                    columns in the table will be selected. Accepted forms:

                    - BigQuery path. For example:
                      ``bq://projectId.bqDatasetId.bqTableId``.

                    This field is a member of `oneof`_ ``connection``.
                query (str):
                    Standard SQL to be used instead of the ``table_uri``.

                    This field is a member of `oneof`_ ``connection``.
            """

            table_uri: str = proto.Field(
                proto.STRING,
                number=1,
                oneof="connection",
            )
            query: str = proto.Field(
                proto.STRING,
                number=2,
                oneof="connection",
            )

        vertex_dataset: str = proto.Field(
            proto.STRING,
            number=1,
            oneof="data_location",
        )
        gcs_source: (
            "ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource"
        ) = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="data_location",
            message="ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource",
        )
        bigquery_source: (
            "ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringBigQuerySource"
        ) = proto.Field(
            proto.MESSAGE,
            number=6,
            oneof="data_location",
            message="ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringBigQuerySource",
        )
        timestamp_field: str = proto.Field(
            proto.STRING,
            number=7,
        )

    class BatchPredictionOutput(proto.Message):
        r"""Data from Vertex AI Batch prediction job output.

        Attributes:
            batch_prediction_job (str):
                Vertex AI Batch prediction job resource name. The job must
                match the model version specified in
                [ModelMonitor].[model_monitoring_target].
        """

        batch_prediction_job: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class VertexEndpointLogs(proto.Message):
        r"""Data from Vertex AI Endpoint request response logging.

        Attributes:
            endpoints (MutableSequence[str]):
                List of endpoint resource names. The endpoints must enable
                the logging with the
                [Endpoint].[request_response_logging_config], and must
                contain the deployed model corresponding to the model
                version specified in
                [ModelMonitor].[model_monitoring_target].
        """

        endpoints: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    class TimeOffset(proto.Message):
        r"""Time offset setting.

        Attributes:
            offset (str):
                [offset] is the time difference from the cut-off time. For
                scheduled jobs, the cut-off time is the scheduled time. For
                non-scheduled jobs, it's the time when the job was created.
                Currently we support the following format: 'w|W': Week,
                'd|D': Day, 'h|H': Hour E.g. '1h' stands for 1 hour, '2d'
                stands for 2 days.
            window (str):
                [window] refers to the scope of data selected for analysis.
                It allows you to specify the quantity of data you wish to
                examine. Currently we support the following format: 'w|W':
                Week, 'd|D': Day, 'h|H': Hour E.g. '1h' stands for 1 hour,
                '2d' stands for 2 days.
        """

        offset: str = proto.Field(
            proto.STRING,
            number=1,
        )
        window: str = proto.Field(
            proto.STRING,
            number=2,
        )

    columnized_dataset: ModelMonitoringDataset = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="dataset",
        message=ModelMonitoringDataset,
    )
    batch_prediction_output: BatchPredictionOutput = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="dataset",
        message=BatchPredictionOutput,
    )
    vertex_endpoint_logs: VertexEndpointLogs = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="dataset",
        message=VertexEndpointLogs,
    )
    time_interval: interval_pb2.Interval = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="time_spec",
        message=interval_pb2.Interval,
    )
    time_offset: TimeOffset = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="time_spec",
        message=TimeOffset,
    )


class ModelMonitoringNotificationSpec(proto.Message):
    r"""Notification spec(email, notification channel) for model
    monitoring statistics/alerts.

    Attributes:
        email_config (google.cloud.aiplatform_v1beta1.types.ModelMonitoringNotificationSpec.EmailConfig):
            Email alert config.
        enable_cloud_logging (bool):
            Dump the anomalies to Cloud Logging. The anomalies will be
            put to json payload encoded from proto
            [google.cloud.aiplatform.logging.ModelMonitoringAnomaliesLogEntry][].
            This can be further sinked to Pub/Sub or any other services
            supported by Cloud Logging.
        notification_channel_configs (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelMonitoringNotificationSpec.NotificationChannelConfig]):
            Notification channel config.
    """

    class EmailConfig(proto.Message):
        r"""The config for email alerts.

        Attributes:
            user_emails (MutableSequence[str]):
                The email addresses to send the alerts.
        """

        user_emails: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    class NotificationChannelConfig(proto.Message):
        r"""Google Cloud Notification Channel config.

        Attributes:
            notification_channel (str):
                Resource names of the NotificationChannels. Must be of the
                format
                ``projects/<project_id_or_number>/notificationChannels/<channel_id>``
        """

        notification_channel: str = proto.Field(
            proto.STRING,
            number=1,
        )

    email_config: EmailConfig = proto.Field(
        proto.MESSAGE,
        number=1,
        message=EmailConfig,
    )
    enable_cloud_logging: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    notification_channel_configs: MutableSequence[NotificationChannelConfig] = (
        proto.RepeatedField(
            proto.MESSAGE,
            number=3,
            message=NotificationChannelConfig,
        )
    )


__all__ = tuple(sorted(__protobuf__.manifest))
