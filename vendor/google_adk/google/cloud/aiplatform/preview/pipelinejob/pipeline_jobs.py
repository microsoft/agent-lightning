# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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

import datetime
import re
from typing import Any, Dict, List, Optional

from google.auth import credentials as auth_credentials
from google.cloud import aiplatform_v1beta1
from google.cloud.aiplatform import base
from google.cloud.aiplatform import compat
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import pipeline_job_schedules
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.constants import pipeline as pipeline_constants
from google.cloud.aiplatform.metadata import constants as metadata_constants
from google.cloud.aiplatform.metadata import experiment_resources
from google.cloud.aiplatform.pipeline_jobs import (
    PipelineJob as PipelineJobGa,
)
from google.cloud.aiplatform_v1.services.pipeline_service import (
    PipelineServiceClient as PipelineServiceClientGa,
)

from google.protobuf import json_format


_LOGGER = base.Logger(__name__)

# Pattern for valid names used as a Vertex resource name.
_VALID_NAME_PATTERN = pipeline_constants._VALID_NAME_PATTERN

# Pattern for an Artifact Registry URL.
_VALID_AR_URL = pipeline_constants._VALID_AR_URL

# Pattern for any JSON or YAML file over HTTPS.
_VALID_HTTPS_URL = pipeline_constants._VALID_HTTPS_URL


def _get_current_time() -> datetime.datetime:
    """Gets the current timestamp."""
    return datetime.datetime.now()


def _set_enable_caching_value(
    pipeline_spec: Dict[str, Any], enable_caching: bool
) -> None:
    """Sets pipeline tasks caching options.

    Args:
        pipeline_spec (Dict[str, Any]):
            Required. The dictionary of pipeline spec.
        enable_caching (bool):
            Required. Whether to enable caching.
    """
    for component in [pipeline_spec["root"]] + list(
        pipeline_spec["components"].values()
    ):
        if "dag" in component:
            for task in component["dag"]["tasks"].values():
                task["cachingOptions"] = {"enableCache": enable_caching}


class _PipelineJob(
    PipelineJobGa,
    experiment_loggable_schemas=(
        experiment_resources._ExperimentLoggableSchema(
            title=metadata_constants.SYSTEM_PIPELINE_RUN
        ),
    ),
):
    """Preview PipelineJob resource for Vertex AI."""

    def __init__(
        self,
        display_name: str,
        template_path: str,
        job_id: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        parameter_values: Optional[Dict[str, Any]] = None,
        input_artifacts: Optional[Dict[str, str]] = None,
        enable_caching: Optional[bool] = None,
        encryption_spec_key_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        failure_policy: Optional[str] = None,
        enable_preflight_validations: Optional[bool] = False,
        default_runtime: Optional[Dict[str, Any]] = None,
    ):
        """Retrieves a PipelineJob resource and instantiates its
        representation.

        Args:
            display_name (str):
                Required. The user-defined name of this Pipeline.
            template_path (str):
                Required. The path of PipelineJob or PipelineSpec JSON or YAML file. It
                can be a local path, a Google Cloud Storage URI (e.g. "gs://project.name"),
                an Artifact Registry URI (e.g.
                "https://us-central1-kfp.pkg.dev/proj/repo/pack/latest"), or an HTTPS URI.
            job_id (str):
                Optional. The unique ID of the job run.
                If not specified, pipeline name + timestamp will be used.
            pipeline_root (str):
                Optional. The root of the pipeline outputs. If not set, the staging bucket
                set in aiplatform.init will be used. If that's not set a pipeline-specific
                artifacts bucket will be used.
            parameter_values (Dict[str, Any]):
                Optional. The mapping from runtime parameter names to its values that
                control the pipeline run.
            input_artifacts (Dict[str, str]):
                Optional. The mapping from the runtime parameter name for this artifact to its resource id.
                For example: "vertex_model":"456". Note: full resource name ("projects/123/locations/us-central1/metadataStores/default/artifacts/456") cannot be used.
            enable_caching (bool):
                Optional. Whether to turn on caching for the run.

                If this is not set, defaults to the compile time settings, which
                are True for all tasks by default, while users may specify
                different caching options for individual tasks.

                If this is set, the setting applies to all tasks in the pipeline.

                Overrides the compile time settings.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the job. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If this is set, then all
                resources created by the PipelineJob will
                be encrypted with the provided encryption key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The user defined metadata to organize PipelineJob.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this PipelineJob.
                Overrides credentials set in aiplatform.init.
            project (str):
                Optional. The project that you want to run this PipelineJob in. If not set,
                the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create PipelineJob. If not set,
                location set in aiplatform.init will be used.
            failure_policy (str):
                Optional. The failure policy - "slow" or "fast".
                Currently, the default of a pipeline is that the pipeline will continue to
                run until no more tasks can be executed, also known as
                PIPELINE_FAILURE_POLICY_FAIL_SLOW (corresponds to "slow").
                However, if a pipeline is set to
                PIPELINE_FAILURE_POLICY_FAIL_FAST (corresponds to "fast"),
                it will stop scheduling any new tasks when a task has failed. Any
                scheduled tasks will continue to completion.
            enable_preflight_validations (bool):
                Optional. Whether to enable preflight validations or not.
            default_runtime (Dict[str, Any]):
                Optional. Specifies the runtime for the entire pipeline.
                All tasks will use the configured runtime unless overridden at the task level.
                If not provided, Vertex Training Custom Job (on-demand) will be used as the default runtime.

                Supported Runtimes:
                - Custom Job(On-Demand) Runtime: Default if default_runtime is not provided or None.
                - Persistent Resource Runtime: To use a persistent resource as the runtime, see reference configuration below:
                    default_runtime = {
                        "persistentResourceRuntimeDetail": {
                            "persistentResourceName": "projects/my-project/locations/my-location/persistentResources/my-persistent",
                            "taskResourceUnavailableWaitTimeMs": 1000,  # Time (ms) to wait if resource is unavailable
                            "taskResourceUnavailableTimeoutBehavior": "FAIL",  # Behavior if resource is unavailable after wait
                        }
                    }
                For more information, please see https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/projects.locations.pipelineJobs#PipelineJob.DefaultRuntime.
        Raises:
            ValueError: If job_id or labels have incorrect format.
        """
        super().__init__(
            display_name=display_name,
            template_path=template_path,
            job_id=job_id,
            pipeline_root=pipeline_root,
            parameter_values=parameter_values,
            input_artifacts=input_artifacts,
            enable_caching=enable_caching,
            encryption_spec_key_name=encryption_spec_key_name,
            labels=labels,
            credentials=credentials,
            project=project,
            location=location,
            failure_policy=failure_policy,
        )

        # needs to rebuild the v1beta version of pipeline_job and runtime_config
        pipeline_json = utils.yaml_utils.load_yaml(
            template_path, self.project, self.credentials
        )

        # Pipeline_json can be either PipelineJob or PipelineSpec.
        if pipeline_json.get("pipelineSpec") is not None:
            pipeline_job = pipeline_json
            pipeline_root = (
                pipeline_root
                or pipeline_job["pipelineSpec"].get("defaultPipelineRoot")
                or pipeline_job["runtimeConfig"].get("gcsOutputDirectory")
                or initializer.global_config.staging_bucket
            )
        else:
            pipeline_job = {
                "pipelineSpec": pipeline_json,
                "runtimeConfig": {},
            }
            pipeline_root = (
                pipeline_root
                or pipeline_job["pipelineSpec"].get("defaultPipelineRoot")
                or initializer.global_config.staging_bucket
            )
        pipeline_root = (
            pipeline_root
            or utils.gcs_utils.generate_gcs_directory_for_pipeline_artifacts(
                project=project,
                location=location,
            )
        )
        builder = utils.pipeline_utils.PipelineRuntimeConfigBuilder.from_job_spec_json(
            pipeline_job
        )
        builder.update_pipeline_root(pipeline_root)
        builder.update_runtime_parameters(parameter_values)
        builder.update_input_artifacts(input_artifacts)

        builder.update_failure_policy(failure_policy)
        builder.update_default_runtime(default_runtime)
        runtime_config_dict = builder.build()
        runtime_config = aiplatform_v1beta1.PipelineJob.RuntimeConfig()._pb
        json_format.ParseDict(runtime_config_dict, runtime_config)

        pipeline_name = pipeline_job["pipelineSpec"]["pipelineInfo"]["name"]
        self.job_id = job_id or "{pipeline_name}-{timestamp}".format(
            pipeline_name=re.sub("[^-0-9a-z]+", "-", pipeline_name.lower())
            .lstrip("-")
            .rstrip("-"),
            timestamp=_get_current_time().strftime("%Y%m%d%H%M%S"),
        )
        if not _VALID_NAME_PATTERN.match(self.job_id):
            raise ValueError(
                f"Generated job ID: {self.job_id} is illegal as a Vertex pipelines job ID. "
                "Expecting an ID following the regex pattern "
                f'"{_VALID_NAME_PATTERN.pattern[1:-1]}"'
            )

        if enable_caching is not None:
            _set_enable_caching_value(pipeline_job["pipelineSpec"], enable_caching)

        pipeline_job_args = {
            "display_name": display_name,
            "pipeline_spec": pipeline_job["pipelineSpec"],
            "labels": labels,
            "runtime_config": runtime_config,
            "encryption_spec": initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            ),
            "preflight_validations": enable_preflight_validations,
        }

        if _VALID_AR_URL.match(template_path) or _VALID_HTTPS_URL.match(template_path):
            pipeline_job_args["template_uri"] = template_path

        self._v1_beta1_pipeline_job = aiplatform_v1beta1.PipelineJob(
            **pipeline_job_args
        )

    def create_schedule(
        self,
        cron_expression: str,
        display_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        allow_queueing: bool = False,
        max_run_count: Optional[int] = None,
        max_concurrent_run_count: int = 1,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
    ) -> "pipeline_job_schedules.PipelineJobSchedule":  # noqa: F821
        """Creates a PipelineJobSchedule directly from a PipelineJob.

        Example Usage:

        pipeline_job = aiplatform.PipelineJob(
            display_name='job_display_name',
            template_path='your_pipeline.yaml',
        )
        pipeline_job.run()
        pipeline_job_schedule = pipeline_job.create_schedule(
            cron_expression='* * * * *',
            display_name='schedule_display_name',
        )

        Args:
            cron_expression (str):
                Required. Time specification (cron schedule expression) to launch scheduled runs.
                To explicitly set a timezone to the cron tab, apply a prefix: "CRON_TZ=${IANA_TIME_ZONE}" or "TZ=${IANA_TIME_ZONE}".
                The ${IANA_TIME_ZONE} may only be a valid string from IANA time zone database.
                For example, "CRON_TZ=America/New_York 1 * * * *", or "TZ=America/New_York 1 * * * *".
            display_name (str):
                Required. The user-defined name of this PipelineJobSchedule.
            start_time (str):
                Optional. Timestamp after which the first run can be scheduled.
                If unspecified, it defaults to the schedule creation timestamp.
            end_time (str):
                Optional. Timestamp after which no more runs will be scheduled.
                If unspecified, then runs will be scheduled indefinitely.
            allow_queueing (bool):
                Optional. Whether new scheduled runs can be queued when max_concurrent_runs limit is reached.
            max_run_count (int):
                Optional. Maximum run count of the schedule.
                If specified, The schedule will be completed when either started_run_count >= max_run_count or when end_time is reached.
                Must be positive and <= 2^63-1.
            max_concurrent_run_count (int):
                Optional. Maximum number of runs that can be started concurrently for this PipelineJobSchedule.
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the network set in aiplatform.init will be used.
                Otherwise, the job is not peered with any network.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.

        Returns:
            A Vertex AI PipelineJobSchedule.
        """
        return super().create_schedule(
            cron=cron_expression,
            display_name=display_name,
            start_time=start_time,
            end_time=end_time,
            allow_queueing=allow_queueing,
            max_run_count=max_run_count,
            max_concurrent_run_count=max_concurrent_run_count,
            service_account=service_account,
            network=network,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    def batch_delete(
        cls,
        names: List[str],
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> aiplatform_v1beta1.BatchDeletePipelineJobsResponse:
        """
        Example Usage:
          aiplatform.init(
            project='your_project_name',
            location='your_location',
          )
          aiplatform.PipelineJob.batch_delete(
            names=['pipeline_job_name', 'pipeline_job_name2']
          )

        Args:
            names (List[str]):
                Required. The fully-qualified resource name or ID of the
                Pipeline Jobs to batch delete. Example:
                "projects/123/locations/us-central1/pipelineJobs/456"
                or "456" when project and location are initialized or passed.
            project (str):
                Optional. Project containing the Pipeline Jobs to
                batch delete. If not set, the project given to `aiplatform.init`
                will be used.
            location (str):
                Optional. Location containing the Pipeline Jobs to
                batch delete. If not set, the location given to `aiplatform.init`
                will be used.

        Returns:
          BatchDeletePipelineJobsResponse contains PipelineJobs deleted.
        """
        user_project = project or initializer.global_config.project
        user_location = location or initializer.global_config.location
        parent = initializer.global_config.common_location_path(
            project=user_project, location=user_location
        )
        pipeline_jobs_names = [
            utils.full_resource_name(
                resource_name=name,
                resource_noun="pipelineJobs",
                parse_resource_name_method=PipelineServiceClientGa.parse_pipeline_job_path,
                format_resource_name_method=PipelineServiceClientGa.pipeline_job_path,
                project=user_project,
                location=user_location,
            )
            for name in names
        ]
        request = aiplatform_v1beta1.BatchDeletePipelineJobsRequest(
            parent=parent, names=pipeline_jobs_names
        )
        client = cls._instantiate_client(
            location=user_location,
            appended_user_agent=["preview-pipeline-jobs-batch-delete"],
        )
        v1beta1_client = client.select_version(compat.V1BETA1)
        operation = v1beta1_client.batch_delete_pipeline_jobs(request)
        return operation.result()

    def submit(
        self,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        reserved_ip_ranges: Optional[List[str]] = None,
        create_request_timeout: Optional[float] = None,
        job_id: Optional[str] = None,
    ) -> None:
        """Run this configured PipelineJob.

        Args:
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.

                Private services access must already be configured for the network.
                If left unspecified, the network set in aiplatform.init will be used.
                Otherwise, the job is not peered with any network.
            reserved_ip_ranges (List[str]):
                Optional. A list of names for the reserved IP ranges under the VPC
                network that can be used for this PipelineJob's workload. For example: ['vertex-ai-ip-range'].

                If left unspecified, the job will be deployed to any IP ranges under
                the provided VPC network.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            job_id (str):
                Optional. The ID to use for the PipelineJob, which will become the final
                component of the PipelineJob name. If not provided, an ID will be
                automatically generated.
        """
        network = network or initializer.global_config.network
        service_account = service_account or initializer.global_config.service_account
        gca_resouce = self._v1_beta1_pipeline_job

        if service_account:
            gca_resouce.service_account = service_account

        if network:
            gca_resouce.network = network

        if reserved_ip_ranges:
            gca_resouce.reserved_ip_ranges = reserved_ip_ranges
        user_project = initializer.global_config.project
        user_location = initializer.global_config.location
        parent = initializer.global_config.common_location_path(
            project=user_project, location=user_location
        )

        client = self._instantiate_client(
            location=user_location,
            appended_user_agent=["preview-pipeline-job-submit"],
        )
        v1beta1_client = client.select_version(compat.V1BETA1)

        _LOGGER.log_create_with_lro(self.__class__)

        request = aiplatform_v1beta1.CreatePipelineJobRequest(
            parent=parent,
            pipeline_job=self._v1_beta1_pipeline_job,
            pipeline_job_id=job_id or self.job_id,
        )

        response = v1beta1_client.create_pipeline_job(request=request)

        self._gca_resource = response

        _LOGGER.log_create_complete_with_getter(
            self.__class__, self._gca_resource, "pipeline_job"
        )

        _LOGGER.info("View Pipeline Job:\n%s" % self._dashboard_uri())

    def rerun(
        self,
        original_pipelinejob_name: str,
        pipeline_task_rerun_configs: Optional[
            List[aiplatform_v1beta1.PipelineTaskRerunConfig]
        ] = None,
        parameter_values: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        reserved_ip_ranges: Optional[List[str]] = None,
    ) -> None:
        """Rerun a PipelineJob.

        Args:
            original_pipelinejob_name (str):
                Required. The name of the original PipelineJob.
            pipeline_task_rerun_configs (List[aiplatform_v1beta1.PipelineTaskRerunConfig]):
                Optional. The list of PipelineTaskRerunConfig to specify the tasks to rerun.
            parameter_values (Dict[str, Any]):
                Optional. The parameter values to override the original PipelineJob.
            job_id (str):
                Optional. The ID to use for the PipelineJob, which will become the final
                component of the PipelineJob name. If not provided, an ID will be
                automatically generated.
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.

                Private services access must already be configured for the network.
                If left unspecified, the network set in aiplatform.init will be used.
                Otherwise, the job is not peered with any network.
            reserved_ip_ranges (List[str]):
                Optional. A list of names for the reserved IP ranges under the VPC
                network that can be used for this PipelineJob's workload. For example: ['vertex-ai-ip-range'].

                If left unspecified, the job will be deployed to any IP ranges under
                the provided VPC network.
        """
        network = network or initializer.global_config.network
        service_account = service_account or initializer.global_config.service_account
        gca_resouce = self._v1_beta1_pipeline_job

        if service_account:
            gca_resouce.service_account = service_account

        if network:
            gca_resouce.network = network

        if reserved_ip_ranges:
            gca_resouce.reserved_ip_ranges = reserved_ip_ranges
        user_project = initializer.global_config.project
        user_location = initializer.global_config.location
        parent = initializer.global_config.common_location_path(
            project=user_project, location=user_location
        )

        client = self._instantiate_client(
            location=user_location,
            appended_user_agent=["preview-pipeline-job-submit"],
        )
        v1beta1_client = client.select_version(compat.V1BETA1)

        _LOGGER.log_create_with_lro(self.__class__)

        pipeline_job = self._v1_beta1_pipeline_job
        try:
            get_request = aiplatform_v1beta1.GetPipelineJobRequest(
                name=original_pipelinejob_name
            )
            original_pipeline_job = v1beta1_client.get_pipeline_job(request=get_request)
            pipeline_job.original_pipeline_job_id = int(
                original_pipeline_job.labels["vertex-ai-pipelines-run-billing-id"]
            )
            original_pipeline_task_details = (
                original_pipeline_job.job_detail.task_details
            )
        except Exception as e:
            raise ValueError(
                f"Failed to get original pipeline job: {original_pipelinejob_name}"
            ) from e

        task_id_to_task_rerun_config = {}
        for task_rerun_config in pipeline_task_rerun_configs:
            task_id_to_task_rerun_config[task_rerun_config.task_id] = task_rerun_config

        pipeline_job.pipeline_task_rerun_configs = []
        for task_detail in original_pipeline_task_details:
            if task_detail.task_id in task_id_to_task_rerun_config:
                task_rerun_config = task_id_to_task_rerun_config[task_detail.task_id]
                if task_detail.task_unique_name:
                    task_rerun_config.task_name = task_detail.task_unique_name
                pipeline_job.pipeline_task_rerun_configs.append(task_rerun_config)
            else:
                pipeline_job.pipeline_task_rerun_configs.append(
                    aiplatform_v1beta1.PipelineTaskRerunConfig(
                        task_id=task_detail.task_id,
                        task_name=task_detail.task_unique_name,
                        skip_task=task_detail.state
                        == aiplatform_v1beta1.PipelineTaskDetail.State.SUCCEEDED,
                    )
                )

        if parameter_values:
            runtime_config = self._v1_beta1_pipeline_job.runtime_config
            runtime_config.parameter_values = parameter_values

        pipeline_name = self._v1_beta1_pipeline_job.display_name

        job_id = job_id or "{pipeline_name}-{timestamp}".format(
            pipeline_name=re.sub("[^-0-9a-z]+", "-", pipeline_name.lower())
            .lstrip("-")
            .rstrip("-"),
            timestamp=_get_current_time().strftime("%Y%m%d%H%M%S"),
        )

        request = aiplatform_v1beta1.CreatePipelineJobRequest(
            parent=parent,
            pipeline_job=self._v1_beta1_pipeline_job,
            pipeline_job_id=job_id,
        )

        response = v1beta1_client.create_pipeline_job(request=request)

        self._gca_resource = response

        _LOGGER.log_create_complete_with_getter(
            self.__class__, self._gca_resource, "pipeline_job"
        )

        _LOGGER.info("View Pipeline Job:\n%s" % self._dashboard_uri())
