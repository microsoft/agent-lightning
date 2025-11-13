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

import copy
import dataclasses
import json
import re
import time
from typing import Any, Dict, List, Optional

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base, initializer, utils
from google.cloud.aiplatform.compat.types import (
    explanation_v1beta1 as explanation,
    job_state_v1beta1 as gca_job_state,
    model_monitor_v1beta1 as gca_model_monitor_compat,
    model_monitoring_alert_v1beta1 as model_monitoring_alert,
    model_monitoring_job_v1beta1 as gca_model_monitoring_job_compat,
    model_monitoring_service_v1beta1 as model_monitoring_service,
    model_monitoring_spec_v1beta1 as model_monitoring_spec,
    model_monitoring_stats_v1beta1 as model_monitoring_stats,
    schedule_service_v1beta1 as gca_schedule_service,
    schedule_v1beta1 as gca_schedule,
)
from vertexai.resources.preview.ml_monitoring.spec import (
    notification,
    objective,
    output,
    schema,
)
import proto

from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
from google.type import interval_pb2
from google.protobuf import text_format

try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import tensorflow_data_validation as tfdv
except ImportError:
    tfdv = None
try:
    from tensorflow_metadata.proto.v0 import statistics_pb2
    from tensorflow_metadata.proto.v0 import anomalies_pb2
except ImportError:
    statistics_pb2 = None
    anomalies_pb2 = None

_LOGGER = base.Logger(__name__)

_JOB_COMPLETE_STATES = (
    gca_job_state.JobState.JOB_STATE_SUCCEEDED,
    gca_job_state.JobState.JOB_STATE_FAILED,
    gca_job_state.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
)

_JOB_ERROR_STATES = (gca_job_state.JobState.JOB_STATE_FAILED,)

# _block_until_complete wait times
_JOB_WAIT_TIME = 5  # start at five seconds
_LOG_WAIT_TIME = 5
_MAX_WAIT_TIME = 60 * 5  # 5 minute wait
_WAIT_TIME_MULTIPLIER = 2  # scale wait by 2 every iteration


def _visualize_stats(baseline_stats_output: str, target_stats_output: str) -> None:
    """Visualizes the model monitoring stats from output directory."""
    import tensorflow as tf

    if not statistics_pb2:
        raise TypeError("statistics_pb2 should be installed to visualize the results")
    if not tf.io.gfile.exists(target_stats_output):
        raise ValueError("No stats were generated.")
    if tf.io.gfile.exists(baseline_stats_output):
        with tf.io.gfile.GFile(
            baseline_stats_output, "rb"
        ) as baseline, tf.io.gfile.GFile(target_stats_output, "rb") as target:
            baseline_combined_stats = statistics_pb2.DatasetFeatureStatisticsList()
            baseline_combined_stats.ParseFromString(baseline.read())
            target_combined_stats = statistics_pb2.DatasetFeatureStatisticsList()
            target_combined_stats.ParseFromString(target.read())
            baseline.close()
            target.close()
            tfdv.visualize_statistics(
                lhs_statistics=baseline_combined_stats,
                rhs_statistics=target_combined_stats,
                lhs_name="Baseline Stats",
                rhs_name="Target Stats",
            )
    else:
        with tf.io.gfile.GFile(target_stats_output, "rb") as target:
            target_combined_stats = statistics_pb2.DatasetFeatureStatisticsList()
            target_combined_stats.ParseFromString(target.read())
            target.close()
            tfdv.visualize_statistics(target_combined_stats)


def _visualize_anomalies(anomalies_output: str) -> None:
    """Visualizes the model monitoring anomalies from output directory."""
    import tensorflow as tf

    if not anomalies_pb2:
        raise TypeError("anomalies_pb2 should be installed to visualize the results")
    with tf.io.gfile.GFile(anomalies_output, "r") as f:
        anomalies = anomalies_pb2.Anomalies()
        text_format.Merge(f.read(), anomalies)
        f.close()
        tfdv.display_anomalies(anomalies)


def _visualize_feature_attribution(feature_attribution_output: str) -> None:
    """Visualizes the model monitoring feature attribution from output directory."""
    import tensorflow as tf

    with tf.io.gfile.GFile(feature_attribution_output, "r") as f:
        print(json.dumps(json.loads(f.read()), indent=4))


def _feature_drift_stats_output_path(output_directory: str, job_id: str) -> (str, str):
    """Returns the baseline and target output paths for the model monitoring feature drift stats."""
    return (
        f"{output_directory}/tabular/jobs/{job_id}/feature_drift/baseline/statistics",
        f"{output_directory}/tabular/jobs/{job_id}/feature_drift/target/statistics",
    )


def _feature_drift_anomalies_output_path(output_directory: str, job_id: str) -> str:
    """Returns the output path for the model monitoring anomalies."""
    return f"{output_directory}/tabular/jobs/{job_id}/feature_drift/anomalies.textproto"


def _prediction_output_stats_output_path(
    output_directory: str, job_id: str
) -> (str, str):
    """Returns the baseline and target output paths for the model monitoring prediction output stats."""
    return (
        f"{output_directory}/tabular/jobs/{job_id}/output_drift/baseline/statistics",
        f"{output_directory}/tabular/jobs/{job_id}/output_drift/target/statistics",
    )


def _prediction_output_anomalies_output_path(output_directory: str, job_id: str) -> str:
    """Returns the output path for the model monitoring anomalies."""
    return f"{output_directory}/tabular/jobs/{job_id}/output_drift/anomalies.textproto"


def _feature_attribution_target_stats_output_path(
    output_directory: str, job_id: str
) -> str:
    """Returns the output path for the model monitoring stats."""
    return f"{output_directory}/tabular/jobs/{job_id}/xai/target/feature_score.json"


def _feature_attribution_baseline_stats_output_path(
    output_directory: str, job_id: str
) -> str:
    """Returns the output path for the model monitoring anomalies."""
    return f"{output_directory}/tabular/jobs/{job_id}/xai/baseline/feature_score.json"


def _transform_schema_pandas(
    dataset: Dict[str, str],
    feature_fields: Optional[List[str]] = None,
    ground_truth_fields: Optional[List[str]] = None,
    prediction_fields: Optional[List[str]] = None,
) -> schema.ModelMonitoringSchema:
    """Transforms the pandas schema to model monitoring schema."""
    ground_truth_fields_list = list()
    prediction_fields_list = list()
    feature_fields_list = list()
    pandas_integer_types = ["integer", "Int32", "Int64", "UInt32", "UInt64"]
    pandas_string_types = [
        "string",
        "bytes",
        "date",
        "time",
        "datetime64",
        "datetime",
        "mixed-integer",
        "inteval",
        "Interval",
    ]
    pandas_float_types = [
        "floating",
        "decimal",
        "mixed-integer-float",
        "Float32",
        "Float64",
    ]
    for field in dataset:
        infer_type = dataset[field]
        if infer_type in pandas_string_types:
            data_type = "string"
        elif infer_type in pandas_integer_types:
            data_type = "integer"
        elif infer_type in pandas_float_types:
            data_type = "float"
        elif infer_type == "boolean":
            data_type = "boolean"
        elif infer_type == "categorical" or infer_type == "category":
            data_type = "categorical"
        else:
            raise ValueError(f"Unsupported data type: {infer_type}")
        if ground_truth_fields and field in ground_truth_fields:
            ground_truth_fields_list.append(
                schema.FieldSchema(name=field, data_type=data_type, repeated=False)
            )
        elif prediction_fields and field in prediction_fields:
            prediction_fields_list.append(
                schema.FieldSchema(name=field, data_type=data_type, repeated=False)
            )
        elif (feature_fields and field in feature_fields) or not feature_fields:
            feature_fields_list.append(
                schema.FieldSchema(name=field, data_type=data_type, repeated=False)
            )
    return schema.ModelMonitoringSchema(
        ground_truth_fields=ground_truth_fields_list if ground_truth_fields else None,
        prediction_fields=prediction_fields_list if prediction_fields else None,
        feature_fields=feature_fields_list,
    )


def _transform_field_schema(
    field_schema: gca_model_monitor_compat.ModelMonitoringSchema.FieldSchema,
) -> Dict[str, Any]:
    result = dict()
    result["name"] = field_schema.name
    result["data_type"] = field_schema.data_type
    result["repeated"] = field_schema.repeated
    return result


def _get_schedule_name(schedule_name: str) -> str:
    if schedule_name:
        client = initializer.global_config.create_client(
            client_class=utils.ScheduleClientWithOverride,
        )
        if client.parse_schedule_path(schedule_name):
            return schedule_name
        elif re.match("^{}$".format("[0-9]{0,127}"), schedule_name):
            return client.schedule_path(
                project=initializer.global_config.project,
                location=initializer.global_config.location,
                schedule=schedule_name,
            )
        else:
            raise ValueError(
                "schedule name must be of the format `projects/{project}/locations/{location}/schedules/{schedule}` or `{schedule}`"
            )
    return schedule_name


def _get_model_monitoring_job_name(
    model_monitoring_job_name: str,
    model_monitor_name: str,
) -> str:
    if model_monitoring_job_name:
        client = initializer.global_config.create_client(
            client_class=utils.ModelMonitoringClientWithOverride,
        )
        if client.parse_model_monitoring_job_path(model_monitoring_job_name):
            return model_monitoring_job_name
        elif re.match("^{}$".format("[0-9]{0,127}"), model_monitoring_job_name):
            model_monitor_name = model_monitor_name.split("/")[-1]
            return client.model_monitoring_job_path(
                project=initializer.global_config.project,
                location=initializer.global_config.location,
                model_monitor=model_monitor_name,
                model_monitoring_job=model_monitoring_job_name,
            )
        else:
            raise ValueError(
                "model monitoring job name must be of the format `projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}` or `{model_monitoring_job}`"
            )
    return model_monitoring_job_name


@dataclasses.dataclass
class MetricsSearchResponse:
    """MetricsSearchResponse represents a response of the search metrics request.

    Attributes:
        monitoring_stats (List[model_monitoring_stats.ModelMonitoringStats]):
            Stats retrieved for requested objectives.
        next_page_token (str):
            The page token that can be used by the next call.
    """

    next_page_token: str
    _search_metrics_response: Any
    monitoring_stats: List[model_monitoring_stats.ModelMonitoringStats] = (
        dataclasses.field(default_factory=list)
    )

    @property
    def raw_search_metrics_response(
        self,
    ) -> model_monitoring_service.SearchModelMonitoringStatsResponse:
        """Raw search metrics response."""
        return self._search_metrics_response


# TODO: b/307946658 - Return a dict or a new dataclass for search_alert
@dataclasses.dataclass
class AlertsSearchResponse:
    """AlertsSearchResponse represents a response of the search alerts request.

    Attributes:
        next_page_token (str):
            The page token that can be used by the next call.
        model_monitoring_alerts (List[model_monitoring_alert.ModelMonitoringAlert]):
            Alerts retrieved for requested objectives.
        total_alerts (int):
            Total number of alerts retrieved for requested objectives.
    """

    next_page_token: str
    _search_alerts_response: Any
    total_alerts: int
    model_monitoring_alerts: List[model_monitoring_alert.ModelMonitoringAlert] = (
        dataclasses.field(default_factory=list)
    )

    @property
    def raw_search_alerts_response(
        self,
    ) -> model_monitoring_service.SearchModelMonitoringAlertsResponse:
        """Raw search metrics response."""
        return self._search_alerts_response


@dataclasses.dataclass
class ListJobsResponse:
    """ListJobsResponse represents a response of the list jobs request.

    Attributes:
        list_jobs (List[model_monitoring_job.ModelMonitoringJob]):
            Jobs retrieved for request.
        next_page_token (str):
            The page token that can be used by the next call.
    """

    next_page_token: str
    _list_jobs_response: Any
    list_jobs: List[gca_model_monitoring_job_compat.ModelMonitoringJob] = (
        dataclasses.field(default_factory=list)
    )

    @property
    def raw_list_jobs_response(
        self,
    ) -> model_monitoring_service.ListModelMonitoringJobsResponse:
        """Raw list jobs response."""
        return self._list_jobs_response


@dataclasses.dataclass
class ListSchedulesResponse:
    """ListSchedulesResponse represents a response of the list jobs request.

    Attributes:
        list_schedules (List[schedule.Schedule]):
            Jobs retrieved for request.
        next_page_token (str):
            The page token that can be used by the next call.
    """

    next_page_token: str
    _list_schedules_response: Any
    list_schedules: List[gca_schedule.Schedule] = dataclasses.field(
        default_factory=list
    )

    @property
    def raw_list_schedules_response(self) -> gca_schedule_service.ListSchedulesResponse:
        """Raw list jobs response."""
        return self._list_schedules_response


class ModelMonitor(base.VertexAiResourceNounWithFutureManager):
    """Initializer for ModelMonitor.

    Args:
        model_monitor_name (str):
            Required. A fully-qualified model monitor resource name or model
            monitor ID.
            Example: "projects/123/locations/us-central1/modelMonitors/456" or
            "456" when project and location are initialized or passed.
        project (str):
            Required. Project to retrieve model monitor from. If not set,
            project set in aiplatform.init will be used.
        location (str):
            Required. Location to retrieve model monitor from. If not set,
            location set in aiplatform.init will be used.
        credentials (auth_credentials.Credentials):
            Optional. Custom credentials to use to retrieve this model monitor.
            Overrides credentials set in aiplatform.init.
    """

    client_class = utils.ModelMonitoringClientWithOverride
    _resource_noun = "modelMonitors"
    _getter_method = "get_model_monitor"
    _list_method = "list_model_monitors"
    _delete_method = "delete_model_monitor"
    _parse_resource_name_method = "parse_model_monitor_path"
    _format_resource_name_method = "model_monitor_path"

    def __init__(
        self,
        model_monitor_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=model_monitor_name,
        )
        self._gca_resource = self._get_gca_resource(resource_name=model_monitor_name)

    @classmethod
    def create(
        cls,
        model_name: str,
        model_version_id: str,
        training_dataset: Optional[objective.MonitoringInput] = None,
        display_name: Optional[str] = None,
        model_monitoring_schema: Optional[schema.ModelMonitoringSchema] = None,
        tabular_objective_spec: Optional[objective.TabularObjective] = None,
        output_spec: Optional[output.OutputSpec] = None,
        notification_spec: Optional[notification.NotificationSpec] = None,
        explanation_spec: Optional[explanation.ExplanationSpec] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        model_monitor_id: Optional[str] = None,
    ) -> "ModelMonitor":
        """Creates a new ModelMonitor.

        Args:
            model_name (str):
                Required. A model resource name as model monitoring target.
                Format: ``projects/{project}/locations/{location}/models/{model}``
            model_version_id (str):
                Required. Model version id.
            training_dataset (objective.MonitoringInput):
                Optional. Training dataset used to train the model. It can serve
                as a baseline dataset to identify changes in production.
            display_name (str):
                Optional. The user-defined name of the ModelMonitor.
                The name can be up to 128 characters long and can comprise any
                UTF-8 character.
                Display name of the ModelMonitor.
            model_monitoring_schema (schema.ModelMonitoringSchema):
                Required for most models, but optional for Vertex AI AutoML
                Tables unless the schema information is not available.
                The Monitoring Schema specifies the model's features, prediction
                outputs and ground truth properties. It is used to extract
                pertinent data from the dataset and to process features based on
                their properties. Make sure that the schema aligns with your
                dataset, if it does not, Vertex AI will be unable to extract
                data form the dataset.
            tabular_objective_spec (objective.TabularObjective):
                Optional. The default tabular monitoring objective spec for
                the model monitor. It can be overriden in the ModelMonitoringJob
                objective spec.
            output_spec (output.OutputSpec):
                Optional. The default monitoring metrics/logs export spec, it
                can be overriden in the ModelMonitoringJob output spec.
                If not specified, a default Google Cloud Storage bucket will be
                created under your project.
            notification_spec (notification.NotificationSpec):
                Optional. The default notification spec for monitoring result.
                It can be overriden in the ModelMonitoringJob notification spec.
            explanation_spec (explanation.ExplanationSpec):
                Optional. The default explanation spec for feature attribution
                monitoring. It can be overriden in the ModelMonitoringJob
                explanation spec.
            project (str):
                Optional. Project to retrieve model monitor from. If not set,
                project set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve model monitor from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this model monitor.
                Overrides credentials set in aiplatform.init.
            model_monitor_id (str):
                Optional. The unique ID of the model monitor, which will become
                the final component of the model monitor resource name.
                If not specified, it will be generated by Vertex AI.

        Returns:
            ModelMonitor: The model monitor that was created.
        """
        api_client = initializer.global_config.create_client(
            client_class=cls.client_class,
            credentials=credentials,
            location_override=location,
        )

        if display_name:
            utils.validate_display_name(display_name)
        else:
            display_name = cls._generate_display_name()

        project = project or initializer.global_config.project
        location = location or initializer.global_config.location

        user_monitoring_target = gca_model_monitor_compat.ModelMonitor.ModelMonitoringTarget(
            vertex_model=gca_model_monitor_compat.ModelMonitor.ModelMonitoringTarget.VertexModelSource(
                model=model_name, model_version_id=model_version_id
            )
        )

        operation_future = api_client.create_model_monitor(
            request=model_monitoring_service.CreateModelMonitorRequest(
                parent=initializer.global_config.common_location_path(
                    project=project, location=location
                ),
                model_monitor=gca_model_monitor_compat.ModelMonitor(
                    display_name=display_name,
                    model_monitoring_target=user_monitoring_target,
                    training_dataset=(
                        training_dataset._as_proto() if training_dataset else None
                    ),
                    model_monitoring_schema=(
                        model_monitoring_schema._as_proto()
                        if model_monitoring_schema
                        else None
                    ),
                    tabular_objective=(
                        tabular_objective_spec._as_proto()
                        if tabular_objective_spec
                        else None
                    ),
                    notification_spec=(
                        notification_spec._as_proto() if notification_spec else None
                    ),
                    output_spec=output_spec._as_proto() if output_spec else None,
                    explanation_spec=explanation_spec,
                ),
                model_monitor_id=model_monitor_id,
            ),
        )
        _LOGGER.log_create_with_lro(cls, operation_future)
        created_model_monitor = operation_future.result(timeout=None)
        _LOGGER.log_create_complete(cls, created_model_monitor, "model_monitor")
        self = cls._construct_sdk_resource_from_gapic(
            gapic_resource=created_model_monitor,
            project=project,
            location=location,
            credentials=credentials,
        )
        model_monitor_id = self._gca_resource.name.split("/")[-1]
        _LOGGER.info(
            f"https://console.cloud.google.com/vertex-ai/model-monitoring/locations/{location}/model-monitors/{model_monitor_id}?project={project}"
        )
        return self

    def update(
        self,
        display_name: Optional[str] = None,
        training_dataset: Optional[objective.MonitoringInput] = None,
        model_monitoring_schema: Optional[schema.ModelMonitoringSchema] = None,
        tabular_objective_spec: Optional[objective.TabularObjective] = None,
        output_spec: Optional[output.OutputSpec] = None,
        notification_spec: Optional[notification.NotificationSpec] = None,
        explanation_spec: Optional[explanation.ExplanationSpec] = None,
    ) -> "ModelMonitor":
        """Updates an existing ModelMonitor.

        Args:
            display_name (str):
                Optional. The user-defined name of the ModelMonitor.
                The name can be up to 128 characters long and can comprise any
                UTF-8 character.
                Display name of the ModelMonitor.
            training_dataset (objective.MonitoringInput):
                Optional. Training dataset used to train the model. It can serve
                as a baseline dataset to identify changes in production.
            model_monitoring_schema (schema.ModelMonitoringSchema):
                Optional. The Monitoring Schema specifies the model's features,
                prediction outputs and ground truth properties. It is used to
                extract pertinent data from the dataset and to process features
                based on their properties. Make sure that the schema aligns with
                your dataset, if it does not, Vertex AI will be unable to
                extract data form the dataset.
            tabular_objective_spec (objective.TabularObjective):
                Optional. The default tabular monitoring objective spec for
                the model monitor. It can be overriden in the ModelMonitoringJob
                objective spec.
            output_spec (output.OutputSpec):
                Optional. The default monitoring metrics/logs export spec, it
                can be overriden in the ModelMonitoringJob output spec.
            notification_spec (notification.NotificationSpec):
                Optional. The default notification spec for monitoring result.
                It can be overriden in the ModelMonitoringJob notification spec.
            explanation_spec (explanation.ExplanationSpec):
                Optional. The default explanation spec for feature attribution
                monitoring. It can be overriden in the ModelMonitoringJob
                explanation spec.

        Returns:
            ModelMonitor: The updated model monitor.
        """
        self._sync_gca_resource()
        current_monitor = copy.deepcopy(self._gca_resource)
        update_mask: List[str] = []
        if display_name is not None:
            update_mask.append("display_name")
            current_monitor.display_name = display_name
        if training_dataset is not None:
            update_mask.append("training_dataset")
            current_monitor.training_dataset = training_dataset._as_proto()
        if model_monitoring_schema is not None:
            update_mask.append("model_monitoring_schema")
            current_monitor.model_monitoring_schema = (
                model_monitoring_schema._as_proto()
            )
        if tabular_objective_spec is not None:
            update_mask.append("tabular_objective")
            current_monitor.tabular_objective = tabular_objective_spec._as_proto()
        if output_spec is not None:
            update_mask.append("output_spec")
            current_monitor.output_spec = output_spec._as_proto()
        if notification_spec is not None:
            update_mask.append("notification_spec")
            current_monitor.notification_spec = notification_spec._as_proto()
        if explanation_spec is not None:
            update_mask.append("explanation_spec")
            current_monitor.explanation_spec = explanation_spec
        lro = self.api_client.update_model_monitor(
            model_monitor=current_monitor,
            update_mask=field_mask_pb2.FieldMask(paths=update_mask),
        )
        self._gca_resource = lro.result()
        return self

    @base.optional_sync()
    def delete(self, force: bool = False, sync: bool = True) -> None:
        """Force delete the model monitor.

        Args:
            force (bool):
                Required. If force is set to True, all schedules on this
                ModelMonitor will be deleted first. Default is False.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
                Default is True.
        """
        _LOGGER.log_action_start_against_resource("Deleting", "", self)
        lro = self.api_client.delete_model_monitor(
            request=model_monitoring_service.DeleteModelMonitorRequest(
                name=self._gca_resource.name, force=force
            )
        )
        _LOGGER.log_action_started_against_resource_with_lro(
            "Delete", "", self.__class__, lro
        )
        _LOGGER.log_action_completed_against_resource("deleted.", "", self)

    def create_schedule(
        self,
        cron: str,
        target_dataset: objective.MonitoringInput,
        display_name: Optional[str] = None,
        model_monitoring_job_display_name: Optional[str] = None,
        start_time: Optional[timestamp_pb2.Timestamp] = None,
        end_time: Optional[timestamp_pb2.Timestamp] = None,
        tabular_objective_spec: Optional[objective.TabularObjective] = None,
        baseline_dataset: Optional[objective.MonitoringInput] = None,
        output_spec: Optional[output.OutputSpec] = None,
        notification_spec: Optional[notification.NotificationSpec] = None,
        explanation_spec: Optional[explanation.ExplanationSpec] = None,
    ) -> "gca_schedule.Schedule":
        """Creates a new Scheduled run for model monitoring job.

        Args:
            cron (str):
                Required. Cron schedule (https://en.wikipedia.org/wiki/Cron) to
                launch scheduled runs. To explicitly set a timezone to the cron
                tab, apply a prefix in the cron tab: "CRON_TZ=${IANA_TIME_ZONE}"
                or "TZ=${IANA_TIME_ZONE}". The ${IANA_TIME_ZONE} may only be a
                valid string from IANA time zone database. For example,
                "CRON_TZ=America/New_York 1 * * * *", or
                "TZ=America/New_York 1 * * * *".
            target_dataset (objective.MonitoringInput):
                Required. The target dataset for analysis.
            display_name (str):
                Optional. The user-defined name of the Schedule.
                The name can be up to 128 characters long and can be consist of
                any UTF-8 characters.
                Display name of the Schedule.
            model_monitoring_job_display_name (str):
                Optional. The user-defined name of the ModelMonitoringJob.
                The name can be up to 128 characters long and can be consist of
                any UTF-8 characters.
                Display name of the ModelMonitoringJob.
            start_time (timestamp_pb2.Timestamp):
                Optional. Timestamp after which the first run can be scheduled.
                Default to Schedule create time if not specified.
            end_time (timestamp_pb2.Timestamp):
                Optional. Timestamp after which no new runs can be scheduled.
                If specified, The schedule will be completed when the end_time
                is reached.
                If not specified, new runs will keep getting scheduled until
                this Schedule is paused or deleted. Already scheduled runs will
                be allowed to complete. Unset if not specified.
            tabular_objective_spec (objective.TabularObjective):
                Optional. The tabular monitoring objective spec. If not set,
                the default tabular objective spec in ModelMonitor will be
                used. You must either set here or set the default one in the
                ModelMonitor.
            baseline_dataset (objective.MonitoringInput):
                Optional. The baseline dataset for monitoring job.
                If not set, the training dataset in ModelMonitor will be
                used as baseline dataset.
            output_spec (output.OutputSpec):
                Optional. The monitoring metrics/logs export spec.
                If not set, will use the default output_spec defined in
                ModelMonitor.
            notification_spec (notification.NotificationSpec):
                Optional. The notification spec for monitoring result.
                If not set, will use the default notification_spec defined in
                ModelMonitor.
            explanation_spec (explanation.ExplanationSpec):
                Optional. The explanation spec for feature attribution
                monitoring.
                If not set, will use the default explanation_spec defined in
                ModelMonitor.

        Returns:
            Schedule: The created schedule.
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ScheduleClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )

        model_monitor_name = utils.full_resource_name(
            resource_name=self._gca_resource.name,
            resource_noun=self._resource_noun,
            parse_resource_name_method=self._parse_resource_name,
            format_resource_name_method=self._format_resource_name,
            project=self.project,
            location=self.location,
        )

        schedule_request = gca_schedule_service.CreateScheduleRequest(
            parent=initializer.global_config.common_location_path(
                project=self.project, location=self.location
            ),
            schedule=gca_schedule.Schedule(
                display_name=display_name,
                start_time=start_time,
                end_time=end_time,
                cron=cron,
                create_model_monitoring_job_request=model_monitoring_service.CreateModelMonitoringJobRequest(
                    parent=model_monitor_name,
                    model_monitoring_job=gca_model_monitoring_job_compat.ModelMonitoringJob(
                        display_name=model_monitoring_job_display_name,
                        model_monitoring_spec=model_monitoring_spec.ModelMonitoringSpec(
                            objective_spec=model_monitoring_spec.ModelMonitoringObjectiveSpec(
                                tabular_objective=(
                                    tabular_objective_spec._as_proto()
                                    if tabular_objective_spec
                                    else self._gca_resource.tabular_objective
                                ),
                                target_dataset=target_dataset._as_proto(),
                                baseline_dataset=(
                                    baseline_dataset._as_proto()
                                    if baseline_dataset
                                    else self._gca_resource.training_dataset
                                ),
                                explanation_spec=(
                                    explanation_spec
                                    if explanation_spec
                                    else self._gca_resource.explanation_spec
                                ),
                            ),
                            output_spec=(
                                output_spec._as_proto()
                                if output_spec
                                else self._gca_resource.output_spec
                            ),
                            notification_spec=(
                                notification_spec._as_proto()
                                if notification_spec
                                else self._gca_resource.notification_spec
                            ),
                        ),
                    ),
                ),
                max_concurrent_run_count=1,
            ),
        )
        created_schedule = api_client.select_version("v1beta1").create_schedule(
            request=schedule_request
        )
        _LOGGER.log_create_complete(gca_schedule.Schedule, created_schedule, "schedule")
        return created_schedule

    def update_schedule(
        self,
        schedule_name: str,
        display_name: Optional[str] = None,
        model_monitoring_job_display_name: Optional[str] = None,
        cron: Optional[str] = None,
        baseline_dataset: Optional[objective.MonitoringInput] = None,
        target_dataset: Optional[objective.MonitoringInput] = None,
        tabular_objective_spec: Optional[objective.TabularObjective] = None,
        output_spec: Optional[output.OutputSpec] = None,
        notification_spec: Optional[notification.NotificationSpec] = None,
        explanation_spec: Optional[explanation.ExplanationSpec] = None,
        end_time: Optional[timestamp_pb2.Timestamp] = None,
    ) -> "gca_schedule.Schedule":
        """Updates an existing Schedule.

        Args:
            schedule_name (str):
                Required. The resource name of schedule that needs to be updated.
                Format: ``projects/{project}/locations/{location}/schedules/{schedule}``
                or
                ``{schedule}``
            display_name (str):
                Optional. The user-defined name of the Schedule.
                The name can be up to 128 characters long and can be consist of
                any UTF-8 characters.
                Display name of the Schedule.
            model_monitoring_job_display_name (str):
                Optional. The user-defined display name of the ModelMonitoringJob
                that needs to be updated.
            cron (str):
                Optional. Cron schedule (https://en.wikipedia.org/wiki/Cron) to
                launch scheduled runs. To explicitly set a timezone to the cron
                tab, apply a prefix in the cron tab: "CRON_TZ=${IANA_TIME_ZONE}"
                or "TZ=${IANA_TIME_ZONE}". The ${IANA_TIME_ZONE} may only be a
                valid string from IANA time zone database. For example,
                "CRON_TZ=America/New_York 1 * * * *", or
                "TZ=America/New_York 1 * * * *".
            baseline_dataset (objective.MonitoringInput):
                Optional. The baseline dataset for monitoring job.
            target_dataset (objective.MonitoringInput):
                Optional. The target dataset for analysis.
            tabular_objective_spec (objective.TabularObjective):
                Optional. The tabular monitoring objective spec.
            output_spec (output.OutputSpec):
                Optional. The monitoring metrics/logs export spec.
            notification_spec (notification.NotificationSpec):
                Optional. The notification spec for monitoring result.
            explanation_spec (explanation.ExplanationSpec):
                Optional. The explanation spec for feature attribution
                monitoring.
            end_time (timestamp_pb2.Timestamp):
                Optional. Timestamp after which no new runs can be scheduled.

        Returns:
            Schedule: The updated schedule.
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ScheduleClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )

        model_monitor_name = utils.full_resource_name(
            resource_name=self._gca_resource.name,
            resource_noun=self._resource_noun,
            parse_resource_name_method=self._parse_resource_name,
            format_resource_name_method=self._format_resource_name,
            project=self.project,
            location=self.location,
        )
        schedule_name = _get_schedule_name(schedule_name)
        current_schedule = copy.deepcopy(self.get_schedule(schedule_name=schedule_name))
        update_mask = []
        if display_name is not None:
            update_mask.append("display_name")
            current_schedule.display_name = display_name
        if cron is not None:
            update_mask.append("cron")
            current_schedule.cron = cron
        if end_time is not None:
            update_mask.append("end_time")
            current_schedule.end_time = end_time
        current_job_request = current_schedule.create_model_monitoring_job_request
        current_spec = current_job_request.model_monitoring_job.model_monitoring_spec
        if (
            tabular_objective_spec is not None
            or output_spec is not None
            or notification_spec is not None
            or model_monitoring_job_display_name is not None
            or baseline_dataset is not None
            or target_dataset is not None
        ):
            update_mask.append("create_model_monitoring_job_request")
            updated_model_monitoring_spec = model_monitoring_spec.ModelMonitoringSpec(
                objective_spec=model_monitoring_spec.ModelMonitoringObjectiveSpec(
                    tabular_objective=(
                        tabular_objective_spec._as_proto()
                        if tabular_objective_spec
                        else current_spec.objective_spec.tabular_objective
                    ),
                    baseline_dataset=(
                        baseline_dataset._as_proto()
                        if baseline_dataset
                        else current_spec.objective_spec.baseline
                    ),
                    target_dataset=(
                        target_dataset._as_proto()
                        if target_dataset
                        else current_spec.objective_spec.target
                    ),
                    explanation_spec=(
                        explanation_spec
                        if explanation_spec
                        else current_spec.objective_spec.explanation_spec
                    ),
                ),
                output_spec=(
                    output_spec._as_proto() if output_spec else current_spec.output_spec
                ),
                notification_spec=(
                    notification_spec._as_proto()
                    if notification_spec
                    else current_spec.notification_spec
                ),
            )
            current_schedule.create_model_monitoring_job_request = model_monitoring_service.CreateModelMonitoringJobRequest(
                parent=model_monitor_name,
                model_monitoring_job=gca_model_monitoring_job_compat.ModelMonitoringJob(
                    display_name=(
                        model_monitoring_job_display_name
                        if model_monitoring_job_display_name
                        else current_job_request.model_monitoring_job.display_name
                    ),
                    model_monitoring_spec=updated_model_monitoring_spec,
                ),
            )
        return api_client.select_version("v1beta1").update_schedule(
            schedule=current_schedule,
            update_mask=field_mask_pb2.FieldMask(paths=update_mask),
        )

    def delete_schedule(self, schedule_name: str) -> None:
        """Deletes an existing Schedule.

        Args:
            schedule_name (str):
                Required. The resource name of schedule that needs to be deleted.
                Format: ``projects/{project}/locations/{location}/schedules/{schedule}``
                or
                ``{schedule}``
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ScheduleClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )
        schedule_name = _get_schedule_name(schedule_name)
        return api_client.select_version("v1beta1").delete_schedule(name=schedule_name)

    def pause_schedule(self, schedule_name: str) -> None:
        """Pauses an existing Schedule.

        Args:
            schedule_name (str):
                Required. The resource name of schedule that needs to be paused.
                Format: ``projects/{project}/locations/{location}/schedules/{schedule}``
                or
                ``{schedule}``
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ScheduleClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )
        schedule_name = _get_schedule_name(schedule_name)
        return api_client.select_version("v1beta1").pause_schedule(name=schedule_name)

    def resume_schedule(self, schedule_name: str) -> None:
        """Resumes an existing Schedule.

        Args:
            schedule_name (str):
                Required. The resource name of schedule that needs to be resumed.
                Format: ``projects/{project}/locations/{location}/schedules/{schedule}``
                or
                ``{schedule}``
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ScheduleClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )
        schedule_name = _get_schedule_name(schedule_name)
        return api_client.select_version("v1beta1").resume_schedule(name=schedule_name)

    def get_schedule(self, schedule_name: str) -> "gca_schedule.Schedule":
        """Gets an existing Schedule.

        Args:
            schedule_name (str):
                Required. The resource name of schedule that needs to be fetched.
                Format: ``projects/{project}/locations/{location}/schedules/{schedule}``
                or
                ``{schedule}``

        Returns:
            Schedule: The schedule requested.
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ScheduleClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )
        schedule_name = _get_schedule_name(schedule_name)
        return api_client.select_version("v1beta1").get_schedule(name=schedule_name)

    def list_schedules(
        self,
        filter: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> "ListSchedulesResponse.list_schedules":
        """List Schedules.

        Args:
            filter (str):
                Optional. Lists the Schedules that match the filter expression.
                The
                following fields are supported:

                -  ``display_name``: Supports ``=``, ``!=`` comparisons, and
                   ``:`` wildcard.
                -  ``state``: Supports ``=`` and ``!=`` comparisons.
                -  ``request``: Supports existence of the <request_type>
                   check. (e.g. ``create_pipeline_job_request:*`` -->
                   Schedule has create_pipeline_job_request).
                -  ``create_time``: Supports ``=``, ``!=``, ``<``, ``>``,
                   ``<=``, and ``>=`` comparisons. Values must be in RFC
                   3339 format.
                -  ``start_time``: Supports ``=``, ``!=``, ``<``, ``>``,
                   ``<=``, and ``>=`` comparisons. Values must be in RFC
                   3339 format.
                -  ``end_time``: Supports ``=``, ``!=``, ``<``, ``>``,
                   ``<=``, ``>=`` comparisons and ``:*`` existence check.
                   Values must be in RFC 3339 format.
                -  ``next_run_time``: Supports ``=``, ``!=``, ``<``, ``>``,
                   ``<=``, and ``>=`` comparisons. Values must be in RFC
                   3339 format.

                Filter expressions can be combined together using logical
                operators (``NOT``, ``AND`` & ``OR``). The syntax to define
                filter expression is based on https://google.aip.dev/160.
            page_size (int):
                Optional. The standard page list size.
            page_token (str):
                Optional. A page token received from a previous call.

        Returns:
            MetricsSearchResponse: The model monitoring stats results.
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ScheduleClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )

        filter = (
            f"{filter} AND create_model_monitoring_job_request.parent={self._gca_resource.name}"
            if filter
            else f"create_model_monitoring_job_request.parent={self._gca_resource.name}"
        )
        list_schedules_response = (
            api_client.select_version("v1beta1")
            .list_schedules(
                request=gca_schedule_service.ListSchedulesRequest(
                    parent=f"projects/{self.project}/locations/{self.location}",
                    filter=filter,
                    page_size=page_size,
                    page_token=page_token,
                )
            )
            ._response
        )
        return ListSchedulesResponse(
            list_schedules=list_schedules_response.schedules,
            next_page_token=list_schedules_response.next_page_token,
            _list_schedules_response=list_schedules_response,
        ).list_schedules

    def run(
        self,
        target_dataset: objective.MonitoringInput,
        display_name: Optional[str] = None,
        model_monitoring_job_id: Optional[str] = None,
        sync: Optional[bool] = False,
        tabular_objective_spec: Optional[objective.TabularObjective] = None,
        baseline_dataset: Optional[objective.MonitoringInput] = None,
        output_spec: Optional[output.OutputSpec] = None,
        notification_spec: Optional[notification.NotificationSpec] = None,
        explanation_spec: Optional[explanation.ExplanationSpec] = None,
    ) -> "ModelMonitoringJob":
        """Creates a new ModelMonitoringJob.

        Args:
            target_dataset (objective.MonitoringInput):
                Required. The target dataset for analysis.
            display_name (str):
                Optional. The user-defined name of the ModelMonitoringJob.
                The name can be up to 128 characters long and can comprise any
                UTF-8 character.
                Display name of the ModelMonitoringJob.
            model_monitoring_job_id (str):
                Optional. The unique ID of the model monitoring job run, which
                will become the final component of the model monitoring job
                resource name. The maximum length is 63 characters, and valid
                characters are /^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/.
                If not specified, it will be generated by Vertex AI.
            sync (bool):
                Whether to execute this method synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed. Default is False.
            tabular_objective_spec (objective.TabularObjective):
                Optional. The tabular monitoring objective spec for the model
                monitoring job.
            baseline_dataset (objective.MonitoringInput):
                Optional. The baseline dataset for monitoring job.
                If not set, the training dataset in ModelMonitor will be
                used as baseline dataset.
            output_spec (output.OutputSpec):
                Optional. The monitoring metrics/logs export spec.
                If not set, will use the default output_spec defined in
                ModelMonitor.
            notification_spec (notification.NotificationSpec):
                Optional. The notification spec for monitoring result.
                If not set, will use the default notification_spec defined in
                ModelMonitor.
            explanation_config (explanation.ExplanationSpec):
                Optional. The explanation spec for feature attribution
                monitoring.
                If not set, will use the default explanation_spec defined in
                ModelMonitor.

        Returns:
            ModelMonitoringJob: The model monitoring job that was created.
        """
        model_monitor_name = utils.full_resource_name(
            resource_name=self._gca_resource.name,
            resource_noun=self._resource_noun,
            parse_resource_name_method=self._parse_resource_name,
            format_resource_name_method=self._format_resource_name,
            project=self.project,
            location=self.location,
        )

        return ModelMonitoringJob.create(
            model_monitor_name=model_monitor_name,
            project=self.project,
            location=self.location,
            credentials=self.credentials,
            display_name=display_name,
            target_dataset=target_dataset,
            baseline_dataset=baseline_dataset,
            model_monitoring_job_id=model_monitoring_job_id,
            tabular_objective_spec=tabular_objective_spec,
            output_spec=output_spec,
            notification_spec=notification_spec,
            explanation_spec=explanation_spec,
            sync=sync,
        )

    def search_metrics(
        self,
        stats_name: Optional[str] = None,
        objective_type: Optional[str] = None,
        model_monitoring_job_name: Optional[str] = None,
        schedule_name: Optional[str] = None,
        algorithm: Optional[str] = None,
        start_time: Optional[timestamp_pb2.Timestamp] = None,
        end_time: Optional[timestamp_pb2.Timestamp] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> "MetricsSearchResponse.monitoring_stats":
        """Search ModelMonitoringStats.

        Args:
            stats_name (str):
                Optional. The stats name filter for the search, if not set, all
                stats will be returned.
                For tabular model it's the feature name.
            objective_type (str):
                Optional. One of the supported monitoring objectives:
                    `raw-feature-drift`
                    `prediction-output-drift`
                    `feature-attribution`
            model_monitoring_job_name (str):
                Optional. The resource name of a particular model monitoring
                job that the user wants to search metrics result from.
                Format:
                ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
            schedule_name (str):
                Optional. The resource name of a particular model monitoring
                schedule that the user wants to search metrics result from.
                Format: ``projects/{project}/locations/{location}/schedules/{schedule}``
            algorithm (str):
                Optional. The algorithm type filter for the search, eg:
                jensen_shannon_divergence, l_infinity.
            start_time (timestamp_pb2.Timestamp):
                Optional. Inclusive start of the time interval for which results
                should be returned.
            end_time (timestamp_pb2.Timestamp):
                Optional. Exclusive end of the time interval for which results
                should be returned.
            page_size (int):
                Optional. The standard page list size.
            page_token (str):
                Optional. A page token received from a previous call.

        Returns:
            MetricsSearchResponse: The model monitoring stats results.
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ModelMonitoringClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )

        user_time_interval = (
            interval_pb2.Interval(start_time=start_time, end_time=end_time)
            if start_time or end_time
            else None
        )
        model_monitoring_stats_response = api_client.search_model_monitoring_stats(
            request=model_monitoring_service.SearchModelMonitoringStatsRequest(
                model_monitor=self._gca_resource.name,
                stats_filter=model_monitoring_stats.SearchModelMonitoringStatsFilter(
                    tabular_stats_filter=model_monitoring_stats.SearchModelMonitoringStatsFilter.TabularStatsFilter(
                        stats_name=stats_name,
                        objective_type=objective_type,
                        model_monitoring_job=model_monitoring_job_name,
                        model_monitoring_schedule=schedule_name,
                        algorithm=algorithm,
                    ),
                ),
                time_interval=user_time_interval,
                page_size=page_size,
                page_token=page_token,
            ),
        )._response
        return MetricsSearchResponse(
            monitoring_stats=model_monitoring_stats_response.monitoring_stats,
            next_page_token=model_monitoring_stats_response.next_page_token,
            _search_metrics_response=model_monitoring_stats_response,
        ).monitoring_stats

    def search_alerts(
        self,
        stats_name: Optional[str] = None,
        objective_type: Optional[str] = None,
        model_monitoring_job_name: Optional[str] = None,
        start_time: Optional[timestamp_pb2.Timestamp] = None,
        end_time: Optional[timestamp_pb2.Timestamp] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search ModelMonitoringAlerts.

        Args:
            stats_name (str):
                Optional. The stats name filter for the search, if not set, all
                stats will be returned.
                For tabular models, provide the name of the feature to return
                alerts from.
            objective_type (str):
                Optional. Return alerts from one of the supported monitoring
                objectives:
                    `raw-feature-drift`
                    `prediction-output-drift`
                    `feature-attribution`
            model_monitoring_job_name (str):
                Optional. The resource name of a particular model monitoring
                job that the user wants to search metrics result from.
                Format:
                ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
            start_time (timestamp_pb2.Timestamp):
                Optional. Inclusive start of the time interval for which alerts
                should be returned.
            end_time (timestamp_pb2.Timestamp):
                Optional. Exclusive end of the time interval for which alerts
                should be returned.
            page_size (int):
                Optional. The standard page list size.
            page_token (str):
                Optional. A page token received from a previous call.

        Returns:
            AlertsSearchResponse: The model monitoring alerts results.
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ModelMonitoringClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )

        user_time_interval = (
            interval_pb2.Interval(start_time=start_time, end_time=end_time)
            if start_time or end_time
            else None
        )
        model_monitoring_alerts_response = api_client.search_model_monitoring_alerts(
            request=model_monitoring_service.SearchModelMonitoringAlertsRequest(
                model_monitor=self._gca_resource.name,
                stats_name=stats_name,
                objective_type=objective_type,
                model_monitoring_job=model_monitoring_job_name,
                alert_time_interval=user_time_interval,
                page_size=page_size,
                page_token=page_token,
            ),
        )._response
        alert_response = AlertsSearchResponse(
            model_monitoring_alerts=model_monitoring_alerts_response.model_monitoring_alerts,
            next_page_token=model_monitoring_alerts_response.next_page_token,
            total_alerts=model_monitoring_alerts_response.total_number_alerts,
            _search_alerts_response=model_monitoring_alerts_response,
        )
        return {
            "total_number_alerts": alert_response.total_alerts,
            "model_monitoring_alerts": alert_response.model_monitoring_alerts,
        }

    def list_jobs(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> "ListJobsResponse.list_jobs":
        """List ModelMonitoringJobs.

        Args:
            page_size (int):
                Optional. The standard page list size.
            page_token (str):
                Optional. A page token received from a previous call.

        Returns:
            ListJobsResponse.list_jobs: The list model monitoring jobs responses.
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ModelMonitoringClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )

        model_monitor_name = utils.full_resource_name(
            resource_name=self._gca_resource.name,
            resource_noun=self._resource_noun,
            parse_resource_name_method=self._parse_resource_name,
            format_resource_name_method=self._format_resource_name,
            project=self.project,
            location=self.location,
        )

        list_jobs_response = api_client.list_model_monitoring_jobs(
            request=model_monitoring_service.ListModelMonitoringJobsRequest(
                parent=model_monitor_name,
                page_size=page_size,
                page_token=page_token,
            )
        )._response
        return ListJobsResponse(
            list_jobs=list_jobs_response.model_monitoring_jobs,
            next_page_token=list_jobs_response.next_page_token,
            _list_jobs_response=list_jobs_response,
        ).list_jobs

    def delete_model_monitoring_job(self, model_monitoring_job_name: str) -> None:
        """Delete a model monitoring job.

        Args:

            model_monitoring_job_name (str):
                Required. The resource name of the model monitoring job that
                needs to be deleted.
                Format:
                ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
                or
                ``{model_monitoring_job}``
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ModelMonitoringClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )
        job_resource_name = _get_model_monitoring_job_name(
            model_monitoring_job_name, self._gca_resource.name
        )
        api_client.delete_model_monitoring_job(name=job_resource_name)

    def get_model_monitoring_job(
        self, model_monitoring_job_name: str
    ) -> "ModelMonitoringJob":
        """Get the specified ModelMonitoringJob.

        Args:
            model_monitoring_job_name (str):
                Required. The resource name of the ModelMonitoringJob that is needed.
                Format:
                ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
                or
                ``{model_monitoring_job}``

        Returns:
            ModelMonitoringJob: The model monitoring job get.
        """
        self.wait()
        job_resource_name = _get_model_monitoring_job_name(
            model_monitoring_job_name, self._gca_resource.name
        )
        return ModelMonitoringJob(
            model_monitoring_job_name=job_resource_name,
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )

    def show_feature_drift_stats(self, model_monitoring_job_name: str) -> None:
        """The method to visualize the feature drift result from a model monitoring job as a histogram chart and a table.

        Args:
            model_monitoring_job_name (str):
                Required. The resource name of model monitoring job to show the
                drift stats from.
                Format: ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
                or
                ``{model_monitoring_job}``
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ModelMonitoringClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )
        if model_monitoring_job_name.startswith("projects/"):
            job_resource_name = model_monitoring_job_name
            job_id = model_monitoring_job_name.split("/")[-1]
        else:
            job_resource_name = f"{self._gca_resource.name}/modelMonitoringJobs/{model_monitoring_job_name}"
            job_id = model_monitoring_job_name
        job = api_client.get_model_monitoring_job(name=job_resource_name)
        output_directory = (
            job.model_monitoring_spec.output_spec.gcs_base_directory.output_uri_prefix
        )
        target_output, baseline_output = _feature_drift_stats_output_path(
            output_directory, job_id
        )
        anomoaly_output = _feature_drift_anomalies_output_path(output_directory, job_id)
        _visualize_stats(baseline_output, target_output)
        _visualize_anomalies(anomoaly_output)

    def get_schema(self) -> gca_model_monitor_compat.ModelMonitoringSchema:
        """Get the schema of the model monitor."""
        self._sync_gca_resource()
        return self._gca_resource.model_monitoring_schema

    def show_output_drift_stats(self, model_monitoring_job_name: str) -> None:
        """The method to visualize the prediction output drift result from a model monitoring job as a histogram chart and a table.

        Args:
            model_monitoring_job_name (str):
                Required. The resource name of model monitoring job to show the
                drift stats from.
                Format: ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
                or
                ``{model_monitoring_job}``
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ModelMonitoringClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )
        if model_monitoring_job_name.startswith("projects/"):
            job_resource_name = model_monitoring_job_name
            job_id = model_monitoring_job_name.split("/")[-1]
        else:
            job_resource_name = f"{self._gca_resource.name}/modelMonitoringJobs/{model_monitoring_job_name}"
            job_id = model_monitoring_job_name
        job = api_client.get_model_monitoring_job(name=job_resource_name)
        output_directory = (
            job.model_monitoring_spec.output_spec.gcs_base_directory.output_uri_prefix
        )
        target_output, baseline_output = _prediction_output_stats_output_path(
            output_directory, job_id
        )
        anomoaly_output = _prediction_output_anomalies_output_path(
            output_directory, job_id
        )
        _visualize_stats(baseline_output, target_output)
        _visualize_anomalies(anomoaly_output)

    def show_feature_attribution_drift_stats(
        self, model_monitoring_job_name: str
    ) -> None:
        """The method to visualize the feature attribution drift result from a model monitoring job as a histogram chart and a table.

        Args:
            model_monitoring_job_name (str):
                Required. The resource name of model monitoring job to show the
                feature attribution drift stats from.
                Format:
                ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}/modelMonitoringJobs/{model_monitoring_job}``
                or
                ``{model_monitoring_job}``
        """
        api_client = initializer.global_config.create_client(
            client_class=utils.ModelMonitoringClientWithOverride,
            credentials=self.credentials,
            location_override=self.location,
        )
        if model_monitoring_job_name.startswith("projects/"):
            job_resource_name = model_monitoring_job_name
            job_id = model_monitoring_job_name.split("/")[-1]
        else:
            job_resource_name = f"{self._gca_resource.name}/modelMonitoringJobs/{model_monitoring_job_name}"
            job_id = model_monitoring_job_name
        job = api_client.get_model_monitoring_job(name=job_resource_name)
        output_directory = (
            job.model_monitoring_spec.output_spec.gcs_base_directory.output_uri_prefix
        )
        target_stats_output = _feature_attribution_target_stats_output_path(
            output_directory, job_id
        )
        baseline_stats_output = _feature_attribution_baseline_stats_output_path(
            output_directory, job_id
        )
        _visualize_feature_attribution(baseline_stats_output)
        _visualize_feature_attribution(target_stats_output)


class ModelMonitoringJob(base.VertexAiStatefulResource):
    r"""Initializer for ModelMonitoringJob.

       Example Usage:

            my_monitoring_job = aiplatform.ModelMonitoringJob(
                model_monitoring_job_name='projects/123/locations/us-central1/modelMonitors/\
                my_model_monitor_id/modelMonitoringJobs/my_monitoring_job_id'
            )
            or
            my_monitoring_job = aiplatform.aiplatform.ModelMonitoringJob(
                model_monitoring_job_name='my_monitoring_job_id',
                model_monitor_id='my_model_monitor_id',
            )
       Args:
            model_monitoring_job_name (str):
                Required. The resource name for the Model Monitoring Job if
                provided alone, or the model monitoring job id if provided with
                model_monitor_id.
            model_monitor_id (str):
                Optional. The model monitor id depends on the way of initializing
                ModelMonitoringJob.
            project (str):
                Required. Project to retrieve endpoint from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Required. Location to retrieve endpoint from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to init model monitoring job.
                Overrides credentials set in aiplatform.init.
    """

    client_class = utils.ModelMonitoringClientWithOverride
    _resource_noun = "modelMonitoringJobs"
    _getter_method = "get_model_monitoring_job"
    _list_method = "list_model_monitoring_jobs"
    _delete_method = "delete_model_monitoring_job"
    _parse_resource_name_method = "parse_model_monitoring_job_path"
    _format_resource_name_method = "model_monitoring_job_path"

    # Required by the done() method
    _valid_done_states = _JOB_COMPLETE_STATES

    def __init__(
        self,
        model_monitoring_job_name: str,
        model_monitor_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=model_monitoring_job_name,
        )
        self._gca_resource = self._get_gca_resource(
            resource_name=model_monitoring_job_name,
            parent_resource_name_fields=(
                {ModelMonitor._resource_noun: model_monitor_id}
                if model_monitor_id
                else model_monitor_id
            ),
        )

    @property
    def state(self) -> gca_job_state.JobState:
        """Fetch Job again and return the current JobState.

        Returns:
            state (job_state.JobState):
                Enum that describes the state of a Model Monitoring Job.
        """

        # Fetch the Job again for most up-to-date job state
        self._sync_gca_resource()
        return self._gca_resource.state

    @classmethod
    def _construct_sdk_resource_from_gapic(
        cls,
        gapic_resource: proto.Message,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "ModelMonitoringJob":
        """Given a GAPIC ModelMonitoringJob object, return the SDK representation.

        Args:
            gapic_resource (proto.Message):
                A GAPIC representation of a ModelMonitoringJob resource, usually retrieved by a get_* or in a list_* API call.
            project (str):
                Optional. Project to construct ModelMonitoringJob object from. If not set, project set in aiplatform.init will be used.
            location (str):
                Optional. Location to construct ModelMonitoringJob object from. If not set, location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to construct ModelMonitoringJob. Overrides credentials set in aiplatform.init.

        Returns:
            ModelMonitoringJob: The model monitoring job that was created.
        """
        model_monitoring_job = super()._construct_sdk_resource_from_gapic(
            gapic_resource=gapic_resource,
            project=project,
            location=location,
            credentials=credentials,
        )

        return model_monitoring_job

    def _block_until_complete(self) -> None:
        """Helper method to block and check on job until complete."""
        # Used these numbers so failures surface fast
        wait = _JOB_WAIT_TIME  # start at five seconds
        log_wait = _LOG_WAIT_TIME
        max_wait = _MAX_WAIT_TIME  # 5 minute wait
        multiplier = _WAIT_TIME_MULTIPLIER  # scale wait by 2 every iteration

        previous_time = time.time()
        while not self.done():
            current_time = time.time()
            if current_time - previous_time >= log_wait:
                _LOGGER.info(
                    "%s %s current state:\n%s"
                    % (
                        self.__class__.__name__,
                        self._gca_resource.name,
                        self._gca_resource.state,
                    )
                )
                log_wait = min(log_wait * multiplier, max_wait)
                previous_time = current_time
            time.sleep(wait)

        # Error is only populated when the job state is JOB_STATE_FAILED.
        if self._gca_resource.state in _JOB_ERROR_STATES:
            raise RuntimeError(
                "Job failed with:\n%s" % self._gca_resource.job_execution_detail.error
            )
        elif (
            self._gca_resource.state
            == gca_job_state.JobState.JOB_STATE_PARTIALLY_SUCCEEDED
        ):
            obj_status_msg = ""
            for (
                obj,
                status,
            ) in self._gca_resource.job_execution_detail.objective_status.items():
                obj_status_msg += f"{obj}: {status}\n"
            _LOGGER.warning("Job partially succeeded:\n%s" % obj_status_msg)
        else:
            _LOGGER.log_action_completed_against_resource("run", "completed", self)

    @classmethod
    def create(
        cls,
        model_monitor_name: str = None,
        target_dataset: objective.MonitoringInput = None,
        display_name: Optional[str] = None,
        model_monitoring_job_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        baseline_dataset: Optional[objective.MonitoringInput] = None,
        tabular_objective_spec: Optional[objective.TabularObjective] = None,
        output_spec: Optional[output.OutputSpec] = None,
        notification_spec: Optional[notification.NotificationSpec] = None,
        explanation_spec: Optional[explanation.ExplanationSpec] = None,
        sync: bool = False,
    ) -> "ModelMonitoringJob":
        """Creates a new ModelMonitoringJob.

        Args:
            model_monitor_name (str):
                Required. The parent model monitor resource name. Format:
                ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}``
            target_dataset (objective.MonitoringInput):
                Required. The target dataset for analysis.
            display_name (str):
                Optional. The user-defined name of the ModelMonitoringJob.
                The name can be up to 128 characters long and can comprise any
                UTF-8 character.
            model_monitoring_job_id (str):
                Optional. The unique ID of the model monitoring job run, which
                will become the final component of the model monitoring job
                resource name. The maximum length is 63 characters, and valid
                characters are /^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/.
                If not specified, it will be generated by Vertex AI.
            project (str):
                Optional. Project to retrieve endpoint from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve endpoint from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create model monitoring job.
                Overrides credentials set in aiplatform.init.
            baseline_dataset (objective.MonitoringInput):
                Optional. The baseline dataset for monitoring job.
                If not set, the training dataset in ModelMonitor will be
                used as baseline dataset.
            output_spec (output.OutputSpec):
                Optional. The monitoring metrics/logs export spec.
                If not set, will use the default output_spec defined in
                ModelMonitor.
            notification_spec (notification.NotificationSpec):
                Optional. The notification spec for monitoring result.
                If not set, will use the default notification_spec defined in
                ModelMonitor.
            explanation_spec (explanation.ExplanationSpec):
                Optional. The explanation spec for feature attribution
                monitoring.
                If not set, will use the default explanation_spec defined in
                ModelMonitor.
            sync (bool):
                Required. Whether to execute this method synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed. Default is False.
        Returns:
            ModelMonitoringJob: The model monitoring job that was created.
        """
        if not display_name:
            display_name = cls._generate_display_name()
        utils.validate_display_name(display_name)

        project = project or initializer.global_config.project
        location = location or initializer.global_config.location

        parent = utils.full_resource_name(
            resource_name=model_monitor_name,
            resource_noun=ModelMonitor._resource_noun,
            parse_resource_name_method=ModelMonitor._parse_resource_name,
            format_resource_name_method=ModelMonitor._format_resource_name,
            project=project,
            location=location,
        )

        gca_model_monitoring_job = gca_model_monitoring_job_compat.ModelMonitoringJob(
            display_name=display_name,
            model_monitoring_spec=model_monitoring_spec.ModelMonitoringSpec(
                objective_spec=model_monitoring_spec.ModelMonitoringObjectiveSpec(
                    tabular_objective=(
                        tabular_objective_spec._as_proto()
                        if tabular_objective_spec
                        else None
                    ),
                    baseline_dataset=(
                        baseline_dataset._as_proto() if baseline_dataset else None
                    ),
                    target_dataset=(
                        target_dataset._as_proto() if target_dataset else None
                    ),
                    explanation_spec=explanation_spec,
                ),
                output_spec=(output_spec._as_proto() if output_spec else None),
                notification_spec=(
                    notification_spec._as_proto() if notification_spec else None
                ),
            ),
        )
        empty_model_monitoring_job = cls._empty_constructor(
            project=project,
            location=location,
            credentials=credentials,
        )
        return cls._submit_job(
            model_monitor_name=parent,
            empty_model_monitoring_job=empty_model_monitoring_job,
            gca_model_monitoring_job=gca_model_monitoring_job,
            model_monitoring_job_id=model_monitoring_job_id,
            sync=sync,
            project=project,
            location=location,
            credentials=credentials,
        )

    @classmethod
    @base.optional_sync(return_input_arg="empty_model_monitoring_job")
    def _submit_job(
        cls,
        model_monitor_name: str,
        empty_model_monitoring_job: "ModelMonitoringJob",
        gca_model_monitoring_job: gca_model_monitoring_job_compat.ModelMonitoringJob,
        sync: bool = False,
        model_monitoring_job_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "ModelMonitoringJob":
        """Submits a new ModelMonitoringJob.

        Args:
            model_monitor_name (str):
                Required. The parent model monitor resource name. Format:
                ``projects/{project}/locations/{location}/modelMonitors/{model_monitor}``
            empty_model_monitoring_job (ModelMonitoringJob):
                Required. ModelMonitoringJob without _gca_resource populated.
            gca_model_monitoring_job (gca_model_monitoring_job_compat.ModelMonitoringJob):
                Required. a model monitoring job proto for creating a model monitoring job on Vertex AI.
            sync (bool):
                Required. Whether to execute this method synchronously. If False, this
                method will be executed in concurrent Future and any downstream
                object will be immediately returned and synced when the Future
                has completed. Default is False.
            model_monitoring_job_id (str):
                Optional. The unique ID of the model monitoring job run, which
                will become the final component of the model monitoring job
                resource name. The maximum length is 63 characters, and valid
                characters are /^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/.
                If not specified, it will be generated by Vertex AI.
            project (str):
                Optional. Project to retrieve endpoint from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve endpoint from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create model monitoring job.
                Overrides credentials set in aiplatform.init.

        Returns:
            ModelMonitoringJob: The model monitoring job that was created.
        """
        api_client = initializer.global_config.create_client(
            client_class=cls.client_class,
            credentials=credentials,
            location_override=location,
        )
        _LOGGER.log_create_with_lro(cls)
        created_model_monitoring_job = api_client.create_model_monitoring_job(
            request=model_monitoring_service.CreateModelMonitoringJobRequest(
                parent=model_monitor_name,
                model_monitoring_job=gca_model_monitoring_job,
                model_monitoring_job_id=model_monitoring_job_id,
            ),
        )
        empty_model_monitoring_job._gca_resource = created_model_monitoring_job
        model_monitoring_job = cls._construct_sdk_resource_from_gapic(
            gapic_resource=created_model_monitoring_job,
            project=project,
            location=location,
            credentials=credentials,
        )
        _LOGGER.log_create_complete(
            cls, created_model_monitoring_job, "model_monitoring_job"
        )
        model_monitoring_job._block_until_complete()
        return model_monitoring_job

    def delete(self) -> None:
        """Deletes an Model Monitoring Job."""
        self.api_client.delete_model_monitoring_job(name=self._gca_resource.name)
