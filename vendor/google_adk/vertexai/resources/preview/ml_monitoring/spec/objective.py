# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from typing import Dict, List, Optional

from google.cloud.aiplatform.compat.types import (
    explanation_v1beta1 as explanation,
    machine_resources_v1beta1 as machine_resources,
    model_monitoring_alert_v1beta1 as model_monitoring_alert,
    model_monitoring_spec_v1beta1 as model_monitoring_spec,
)

from google.protobuf import timestamp_pb2
from google.type import interval_pb2


TF_RECORD = "tf-record"
CSV = "csv"
JSONL = "jsonl"
JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"
L_INFINITY = "l_infinity"
SUPPORTED_NUMERIC_METRICS = [JENSEN_SHANNON_DIVERGENCE]
SUPPORTED_CATEGORICAL_METRICS = [JENSEN_SHANNON_DIVERGENCE, L_INFINITY]


class DataDriftSpec:
    """Data drift monitoring spec.

    Data drift measures the distribution distance between the current dataset
    and a baseline dataset. A typical use case is to detect data drift between
    the recent production serving dataset and the training dataset, or to
    compare the recent production dataset with a dataset from a previous period.

    Example:
        feature_drift_spec=DataDriftSpec(
                features=["feature1"]
                categorical_metric_type="l_infinity",
                numeric_metric_type="jensen_shannon_divergence",
                default_categorical_alert_threshold=0.01,
                default_numeric_alert_threshold=0.02,
                feature_alert_thresholds={"feature1":0.02, "feature2":0.01},
        )

    Attributes:
        features (List[str]):
            Optional. Feature names / Prediction output names interested in
            monitoring. These should be a subset of the input feature names or
            prediction output names specified in the monitoring schema.
            If not specified, all features / prediction outputs outlied in the
            monitoring schema will be used.
        categorical_metric_type (str):
            Optional. Supported metrics type: l_infinity, jensen_shannon_divergence
        numeric_metric_type (str):
            Optional. Supported metrics type: jensen_shannon_divergence
        default_categorical_alert_threshold (float):
            Optional. Default alert threshold for all the categorical features.
        default_numeric_alert_threshold (float):
            Optional. Default alert threshold for all the numeric features.
        feature_alert_thresholds (Dict[str, float]):
            Optional. Per feature alert threshold will override default alert
            threshold.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        categorical_metric_type: Optional[str] = L_INFINITY,
        numeric_metric_type: Optional[str] = JENSEN_SHANNON_DIVERGENCE,
        default_categorical_alert_threshold: Optional[float] = None,
        default_numeric_alert_threshold: Optional[float] = None,
        feature_alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.features = features
        self.categorical_metric_type = categorical_metric_type
        self.numeric_metric_type = numeric_metric_type
        self.default_categorical_alert_threshold = default_categorical_alert_threshold
        self.default_numeric_alert_threshold = default_numeric_alert_threshold
        self.feature_alert_thresholds = feature_alert_thresholds

    def _as_proto(
        self,
    ) -> model_monitoring_spec.ModelMonitoringObjectiveSpec.DataDriftSpec:
        """Converts DataDriftSpec to a proto message.

        Returns:
           The GAPIC representation of the data drift spec.
        """
        user_default_categorical_alert_threshold = None
        user_default_numeric_alert_threshold = None
        user_alert_thresholds = None
        user_features = None
        if self.numeric_metric_type not in SUPPORTED_NUMERIC_METRICS:
            raise ValueError(
                f"The numeric metric type is not supported"
                f" {self.numeric_metric_type}"
            )
        user_numeric_metric_type = self.numeric_metric_type
        if self.categorical_metric_type not in SUPPORTED_CATEGORICAL_METRICS:
            raise ValueError(
                f"The categorical metric type is not supported"
                f" {self.categorical_metric_type}"
            )
        user_categorical_metric_type = self.categorical_metric_type
        if self.default_categorical_alert_threshold:
            user_default_categorical_alert_threshold = (
                model_monitoring_alert.ModelMonitoringAlertCondition(
                    threshold=self.default_categorical_alert_threshold
                )
            )
        if self.default_numeric_alert_threshold:
            user_default_numeric_alert_threshold = (
                model_monitoring_alert.ModelMonitoringAlertCondition(
                    threshold=self.default_numeric_alert_threshold
                )
            )
        if self.feature_alert_thresholds:
            user_alert_thresholds = {}
            for feature in self.feature_alert_thresholds:
                user_alert_thresholds.update(
                    {
                        feature: model_monitoring_alert.ModelMonitoringAlertCondition(
                            threshold=self.feature_alert_thresholds[feature]
                        )
                    }
                )
        if self.features:
            user_features = self.features
        return model_monitoring_spec.ModelMonitoringObjectiveSpec.DataDriftSpec(
            default_categorical_alert_condition=user_default_categorical_alert_threshold,
            default_numeric_alert_condition=user_default_numeric_alert_threshold,
            categorical_metric_type=user_categorical_metric_type,
            numeric_metric_type=user_numeric_metric_type,
            feature_alert_conditions=user_alert_thresholds,
            features=user_features,
        )


class FeatureAttributionSpec:
    """Feature attribution spec.

    Example:
        feature_attribution_spec=FeatureAttributionSpec(
                features=["feature1"]
                default_alert_threshold=0.01,
                feature_alert_thresholds={"feature1":0.02, "feature2":0.01},
                batch_dedicated_resources=BatchDedicatedResources(
                    starting_replica_count=1,
                    max_replica_count=2,
                    machine_spec=my_machine_spec,
                ),
        )

    Attributes:
        features (List[str]):
            Optional. Input feature names interested in monitoring. These should
            be a subset of the input feature names specified in the monitoring
            schema.
            If not specified, all features outlied in the monitoring schema will
            be used.
        default_alert_threshold (float):
            Optional. Default alert threshold for all the features.
        feature_alert_thresholds (Dict[str, float]):
            Optional. Per feature alert threshold will override default alert
            threshold.
        batch_dedicated_resources (machine_resources.BatchDedicatedResources):
            Optional. The config of resources used by the Model Monitoring during
            the batch explanation for non-AutoML models. If not set, `n1-standard-2`
            machine type will be used by default.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        default_alert_threshold: Optional[float] = None,
        feature_alert_thresholds: Optional[Dict[str, float]] = None,
        batch_dedicated_resources: Optional[
            machine_resources.BatchDedicatedResources
        ] = None,
    ):
        self.features = features
        self.default_alert_threshold = default_alert_threshold
        self.feature_alert_thresholds = feature_alert_thresholds
        self.batch_dedicated_resources = batch_dedicated_resources

    def _as_proto(
        self,
    ) -> model_monitoring_spec.ModelMonitoringObjectiveSpec.FeatureAttributionSpec:
        """Converts FeatureAttributionSpec to a proto message.

        Returns:
           The GAPIC representation of the feature attribution spec.
        """
        user_default_alert_threshold = None
        user_alert_thresholds = None
        user_features = None
        if self.default_alert_threshold:
            user_default_alert_threshold = (
                model_monitoring_alert.ModelMonitoringAlertCondition(
                    threshold=self.default_alert_threshold
                )
            )
        if self.feature_alert_thresholds:
            user_alert_thresholds = {}
            for feature in self.feature_alert_thresholds:
                user_alert_thresholds.update(
                    {
                        feature: model_monitoring_alert.ModelMonitoringAlertCondition(
                            threshold=self.feature_alert_thresholds[feature]
                        )
                    }
                )
        if self.features:
            user_features = self.features
        return (
            model_monitoring_spec.ModelMonitoringObjectiveSpec.FeatureAttributionSpec(
                default_alert_condition=user_default_alert_threshold,
                feature_alert_conditions=user_alert_thresholds,
                features=user_features,
                batch_explanation_dedicated_resources=self.batch_dedicated_resources,
            )
        )


class MonitoringInput:
    """Model monitoring data input spec.

    Attributes:
        vertex_dataset (str):
            Optional. Resource name of the Vertex AI managed dataset.
            Format: ``projects/{project}/locations/{location}/datasets/{dataset}``
            At least one source of dataset should be provided, and if one of the
            fields is set, no need to set other sources
            (vertex_dataset, gcs_uri, table_uri, query, batch_prediction_job,
            endpoints).
        gcs_uri (str):
            Optional. Google Cloud Storage URI to the input file(s). May contain
            wildcards.
        data_format (str):
            Optional. Data format of Google Cloud Storage file(s). Should be
            provided if a gcs_uri is set.
            Supported formats:
                "csv", "jsonl", "tf-record"
        table_uri (str):
            Optonal. BigQuery URI to a table, up to 2000 characters long.
            All the columns in the table will be selected. Accepted forms:

            -  BigQuery path. For example:
                ``bq://projectId.bqDatasetId.bqTableId``.
        query (str):
            Optional. Standard SQL for BigQuery to be used instead of the
            ``table_uri``.
        timestamp_field (str):
            Optional. The timestamp field in the dataset.
            the ``timestamp_field`` must be specified if you'd like to use
            ``start_time``, ``end_time``, ``offset`` or ``window``.
            If you use ``query`` to specify the dataset, make sure the
            ``timestamp_field`` is in the selection fields.
        batch_prediction_job (str):
            Optional. Vertex AI Batch Prediction Job resource name.
            Format: ``projects/{project}/locations/{location}/batchPredictionJobs/{batch_prediction_job}``
        endpoints (List[str]):
            Optional. List of Vertex AI Endpoint resource names.
            Format: ``projects/{project}/locations/{location}/endpoints/{endpoint}``
        start_time (timestamp_pb2.Timestamp):
            Optional. Inclusive start of the time interval for which results
            should be returned. Should be set together with ``end_time``.
        end_time (timestamp_pb2.Timestamp):
            Optional. Exclusive end of the time interval for which results
            should be returned. Should be set together with ``start_time`.`
        offset (str):
            Optional. Offset is the time difference from the cut-off time.
            For scheduled jobs, the cut-off time is the scheduled time.
            For non-scheduled jobs, it's the time when the job was created.
            Currently we support the following format:
            'w|W': Week, 'd|D': Day, 'h|H': Hour
            E.g. '1h' stands for 1 hour, '2d' stands for 2 days.
        window (str):
            Optional. Window refers to the scope of data selected for analysis.
            It allows you to specify the quantity of data you wish to examine.
            It refers to the data time window prior to the cut-off time or the
            cut-off time minus the offset.
            Currently we support the following format:
            'w|W': Week, 'd|D': Day, 'h|H': Hour
            E.g. '1h' stands for 1 hour, '2d' stands for 2 days.
    """

    def __init__(
        self,
        vertex_dataset: Optional[str] = None,
        gcs_uri: Optional[str] = None,
        data_format: Optional[str] = None,
        table_uri: Optional[str] = None,
        query: Optional[str] = None,
        timestamp_field: Optional[str] = None,
        batch_prediction_job: Optional[str] = None,
        endpoints: Optional[List[str]] = None,
        start_time: Optional[timestamp_pb2.Timestamp] = None,
        end_time: Optional[timestamp_pb2.Timestamp] = None,
        offset: Optional[str] = None,
        window: Optional[str] = None,
    ):
        self.vertex_dataset = vertex_dataset
        self.gcs_uri = gcs_uri
        self.data_format = data_format
        self.table_uri = table_uri
        self.query = query
        self.timestamp_field = timestamp_field
        self.batch_prediction_job = batch_prediction_job
        self.endpoints = endpoints
        self.start_time = start_time
        self.end_time = end_time
        self.offset = offset
        self.window = window

    def _as_proto(self) -> model_monitoring_spec.ModelMonitoringInput:
        """Converts ModelMonitoringInput to a proto message.

        Returns:
           The GAPIC representation of the model monitoring input.
        """
        user_time_interval = None
        user_time_spec = None
        if self.offset or self.window:
            user_time_spec = model_monitoring_spec.ModelMonitoringInput.TimeOffset(
                offset=self.offset if self.offset else None,
                window=self.window if self.window else None,
            )
        elif self.start_time or self.end_time:
            user_time_interval = interval_pb2.Interval(
                start_time=self.start_time if self.start_time else None,
                end_time=self.end_time if self.end_time else None,
            )
        if self.vertex_dataset or self.gcs_uri or self.table_uri or self.query:
            user_vertex_dataset = None
            user_gcs_source = None
            user_bigquery_source = None
            if self.vertex_dataset:
                user_vertex_dataset = self.vertex_dataset
            elif self.gcs_uri:
                if not self.data_format:
                    raise ValueError("`data_format` must be provided with gcs uri.")
                if self.data_format == CSV:
                    user_data_format = (
                        model_monitoring_spec.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat.CSV
                    )
                elif self.data_format == JSONL:
                    user_data_format = (
                        model_monitoring_spec.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat.JSONL
                    )
                elif self.data_format == TF_RECORD:
                    user_data_format = (
                        model_monitoring_spec.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat.TF_RECORD
                    )
                else:
                    raise ValueError(
                        (
                            "Unsupported value in data format. `data_format` "
                            "must be one of %s, %s, or %s"
                        )
                        % (TF_RECORD, CSV, JSONL)
                    )
                user_gcs_source = model_monitoring_spec.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource(
                    gcs_uri=self.gcs_uri,
                    format_=user_data_format,
                )
            elif self.table_uri or self.query:
                user_bigquery_source = model_monitoring_spec.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringBigQuerySource(
                    table_uri=self.table_uri,
                    query=self.query,
                )
            else:
                raise ValueError(
                    ("At least one source of dataset must" " be provided.")
                )
            user_model_monitoring_dataset = (
                model_monitoring_spec.ModelMonitoringInput.ModelMonitoringDataset(
                    vertex_dataset=user_vertex_dataset,
                    gcs_source=user_gcs_source,
                    bigquery_source=user_bigquery_source,
                    timestamp_field=self.timestamp_field,
                )
            )
            return model_monitoring_spec.ModelMonitoringInput(
                columnized_dataset=user_model_monitoring_dataset,
                time_offset=user_time_spec,
                time_interval=user_time_interval,
            )
        elif self.batch_prediction_job:
            user_batch_prediction_output = (
                model_monitoring_spec.ModelMonitoringInput.BatchPredictionOutput(
                    batch_prediction_job=self.batch_prediction_job,
                )
            )
            return model_monitoring_spec.ModelMonitoringInput(
                batch_prediction_output=user_batch_prediction_output,
                time_offset=user_time_spec,
                time_interval=user_time_interval,
            )
        elif self.endpoints:
            user_vertex_endpoint_logs = (
                model_monitoring_spec.ModelMonitoringInput.VertexEndpointLogs(
                    endpoints=self.endpoints,
                )
            )
            return model_monitoring_spec.ModelMonitoringInput(
                vertex_endpoint_logs=user_vertex_endpoint_logs,
                time_offset=user_time_spec,
                time_interval=user_time_interval,
            )
        else:
            raise ValueError("At least one source of dataInput must be provided.")


class TabularObjective:
    """Initializer for TabularObjective.

    Attributes:
        feature_drift_spec (DataDriftSpec):
            Optional. Input feature distribution drift monitoring spec.
        prediction_output_drift_spec (DataDriftSpec):
            Optional. Prediction output distribution drift monitoring spec.
        feature_attribution_spec (FeatureAttributionSpec):
            Optional. Feature attribution monitoring spec.
    """

    def __init__(
        self,
        feature_drift_spec: Optional[DataDriftSpec] = None,
        prediction_output_drift_spec: Optional[DataDriftSpec] = None,
        feature_attribution_spec: Optional[FeatureAttributionSpec] = None,
    ):
        self.feature_drift_spec = feature_drift_spec
        self.prediction_output_drift_spec = prediction_output_drift_spec
        self.feature_attribution_spec = feature_attribution_spec

    def _as_proto(
        self,
    ) -> model_monitoring_spec.ModelMonitoringObjectiveSpec.TabularObjective:
        """Converts TabularObjective to a proto message.

        Returns:
           The GAPIC representation of the model monitoring tabular objective.
        """
        user_feature_drift_spec = None
        user_prediction_output_drift_spec = None
        user_feature_attribution_spec = None
        if self.feature_drift_spec:
            user_feature_drift_spec = self.feature_drift_spec._as_proto()
        if self.prediction_output_drift_spec:
            user_prediction_output_drift_spec = (
                self.prediction_output_drift_spec._as_proto()
            )
        if self.feature_attribution_spec:
            user_feature_attribution_spec = self.feature_attribution_spec._as_proto()
        return model_monitoring_spec.ModelMonitoringObjectiveSpec.TabularObjective(
            feature_drift_spec=user_feature_drift_spec,
            prediction_output_drift_spec=user_prediction_output_drift_spec,
            feature_attribution_spec=user_feature_attribution_spec,
        )


class ObjectiveSpec:
    """Initializer for ObjectiveSpec.

    Args:
        baseline_dataset (MonitoringInput):
            Required. Baseline datasets that are used by all the monitoring
            objectives. It could be the training dataset or production serving
            dataset from a previous period.
        target_dataset (MonitoringInput):
            Required. Target dataset for monitoring analysis, it's used by all
            the monitoring objectives.
        tabular_objective (TabularObjective):
            Optional. The tabular monitoring objective.
        explanation_spec (explanation.ExplanationSpec):
            Optional. The explanation spec. This spec is required when the
            objectives spec includes feature attribution objectives.
    """

    def __init__(
        self,
        baseline_dataset: MonitoringInput,
        target_dataset: MonitoringInput,
        tabular_objective: Optional[TabularObjective] = None,
        explanation_spec: Optional[explanation.ExplanationSpec] = None,
    ):
        self.baseline = baseline_dataset
        self.target = target_dataset
        self.tabular_objective = tabular_objective
        self.explanation_spec = explanation_spec

    def _as_proto(self) -> model_monitoring_spec.ModelMonitoringObjectiveSpec:
        """Converts ModelMonitoringObjectiveSpec to a proto message.

        Returns:
           The GAPIC representation of the model monitoring objective config.
        """
        user_tabular_objective = None
        if not self.baseline or not self.target:
            raise ValueError("At least one objective must be provided.")
        if self.tabular_objective:
            user_tabular_objective = self.tabular_objective._as_proto()
        return model_monitoring_spec.ModelMonitoringObjectiveSpec(
            tabular_objective=user_tabular_objective,
            explanation_spec=self.explanation_spec if self.explanation_spec else None,
            target_dataset=self.target._as_proto(),
            baseline_dataset=self.baseline._as_proto(),
        )
