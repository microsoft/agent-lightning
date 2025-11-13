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

import copy
from typing import Dict, List, Optional, Sequence, Union
import uuid

from google.api_core import retry
from google.auth import credentials as auth_credentials
from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import compat
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import jobs
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    custom_job_v1beta1 as gca_custom_job_compat,
    hyperparameter_tuning_job_v1beta1 as gca_hyperparameter_tuning_job_compat,
    job_state as gca_job_state,
    job_state_v1beta1 as gca_job_state_v1beta1,
    study_v1beta1,
)
from google.cloud.aiplatform.compat.types import (
    execution_v1beta1 as gcs_execution_compat,
)
from google.cloud.aiplatform.compat.types import io_v1beta1 as gca_io_compat
from google.cloud.aiplatform.metadata import constants as metadata_constants
from google.cloud.aiplatform import hyperparameter_tuning
from google.cloud.aiplatform.utils import console_utils
import proto

from google.protobuf import duration_pb2  # type: ignore


_LOGGER = base.Logger(__name__)
_DEFAULT_RETRY = retry.Retry()
# TODO(b/242108750): remove temporary logic once model monitoring for batch prediction is GA
_JOB_COMPLETE_STATES = (
    gca_job_state.JobState.JOB_STATE_SUCCEEDED,
    gca_job_state.JobState.JOB_STATE_FAILED,
    gca_job_state.JobState.JOB_STATE_CANCELLED,
    gca_job_state.JobState.JOB_STATE_PAUSED,
    gca_job_state_v1beta1.JobState.JOB_STATE_SUCCEEDED,
    gca_job_state_v1beta1.JobState.JOB_STATE_FAILED,
    gca_job_state_v1beta1.JobState.JOB_STATE_CANCELLED,
    gca_job_state_v1beta1.JobState.JOB_STATE_PAUSED,
)

_JOB_ERROR_STATES = (
    gca_job_state.JobState.JOB_STATE_FAILED,
    gca_job_state.JobState.JOB_STATE_CANCELLED,
    gca_job_state_v1beta1.JobState.JOB_STATE_FAILED,
    gca_job_state_v1beta1.JobState.JOB_STATE_CANCELLED,
)

# _block_until_complete wait times
_JOB_WAIT_TIME = 5  # start at five seconds
_LOG_WAIT_TIME = 5
_MAX_WAIT_TIME = 60 * 5  # 5 minute wait
_WAIT_TIME_MULTIPLIER = 2  # scale wait by 2 every iteration


class CustomJob(jobs.CustomJob):
    """Deprecated. Vertex AI Custom Job (preview)."""

    def __init__(
        self,
        # TODO(b/223262536): Make display_name parameter fully optional in next major release
        display_name: str,
        worker_pool_specs: Union[
            List[Dict], List[gca_custom_job_compat.WorkerPoolSpec]
        ],
        base_output_dir: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
        staging_bucket: Optional[str] = None,
        persistent_resource_id: Optional[str] = None,
    ):
        """Deprecated. Please use the GA (non-preview) version of this class.

        Constructs a Custom Job with Worker Pool Specs.

        ```
        Example usage:
        worker_pool_specs = [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-4",
                        "accelerator_type": "NVIDIA_TESLA_K80",
                        "accelerator_count": 1,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": container_image_uri,
                        "command": [],
                        "args": [],
                    },
                }
            ]

        my_job = aiplatform.preview.jobs.CustomJob(
            display_name='my_job',
            worker_pool_specs=worker_pool_specs,
            labels={'my_key': 'my_value'},
        )

        my_job.run()
        ```


        For more information on configuring worker pool specs please visit:
        https://cloud.google.com/ai-platform-unified/docs/training/create-custom-job


        Args:
            display_name (str):
                Required. The user-defined name of the HyperparameterTuningJob.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            worker_pool_specs (Union[List[Dict], List[aiplatform.gapic.WorkerPoolSpec]]):
                Required. The spec of the worker pools including machine type and Docker image.
                Can provided as a list of dictionaries or list of WorkerPoolSpec proto messages.
            base_output_dir (str):
                Optional. GCS output directory of job. If not provided a
                timestamped directory in the staging directory will be used.
            project (str):
                Optional.Project to run the custom job in. Overrides project set in aiplatform.init.
            location (str):
                Optional.Location to run the custom job in. Overrides location set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional.Custom credentials to use to run call custom job service. Overrides
                credentials set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize CustomJobs.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            encryption_spec_key_name (str):
                Optional.Customer-managed encryption key name for a
                CustomJob. If this is set, then all resources
                created by the CustomJob will be encrypted with
                the provided encryption key.
            staging_bucket (str):
                Optional. Bucket for produced custom job artifacts. Overrides
                staging_bucket set in aiplatform.init.
            persistent_resource_id (str):
                Optional. The ID of the PersistentResource in the same Project
                and Location. If this is specified, the job will be run on
                existing machines held by the PersistentResource instead of
                on-demand short-live machines. The network and CMEK configs on
                the job should be consistent with those on the PersistentResource,
                otherwise, the job will be rejected.

        Raises:
            RuntimeError: If staging bucket was not set using aiplatform.init
                and a staging bucket was not passed in.
        """

        super().__init__(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
            base_output_dir=base_output_dir,
            project=project,
            location=location,
            credentials=credentials,
            labels=labels,
            encryption_spec_key_name=encryption_spec_key_name,
            staging_bucket=staging_bucket,
        )

        staging_bucket = staging_bucket or initializer.global_config.staging_bucket

        if not staging_bucket:
            raise RuntimeError(
                "staging_bucket should be passed to CustomJob constructor or "
                "should be set using aiplatform.init(staging_bucket='gs://my-bucket')"
            )

        if labels:
            utils.validate_labels(labels)

        # default directory if not given
        base_output_dir = base_output_dir or utils._timestamped_gcs_dir(
            staging_bucket, "aiplatform-custom-job"
        )

        if not display_name:
            display_name = self.__class__._generate_display_name()

        self._gca_resource = gca_custom_job_compat.CustomJob(
            display_name=display_name,
            job_spec=gca_custom_job_compat.CustomJobSpec(
                worker_pool_specs=worker_pool_specs,
                base_output_directory=gca_io_compat.GcsDestination(
                    output_uri_prefix=base_output_dir
                ),
                persistent_resource_id=persistent_resource_id,
            ),
            labels=labels,
            encryption_spec=initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name,
                select_version=compat.V1BETA1,
            ),
        )

        self._experiment = None
        self._experiment_run = None
        self._enable_autolog = False

    def _get_gca_resource(
        self,
        resource_name: str,
        parent_resource_name_fields: Optional[Dict[str, str]] = None,
    ) -> proto.Message:
        """Returns GAPIC service representation of client class resource.

        Args:
            resource_name (str): Required. A fully-qualified resource name or ID.
            parent_resource_name_fields (Dict[str,str]):
                Optional. Mapping of parent resource name key to values. These
                will be used to compose the resource name if only resource ID is given.
                Should not include project and location.
        """
        resource_name = utils.full_resource_name(
            resource_name=resource_name,
            resource_noun=self._resource_noun,
            parse_resource_name_method=self._parse_resource_name,
            format_resource_name_method=self._format_resource_name,
            project=self.project,
            location=self.location,
            parent_resource_name_fields=parent_resource_name_fields,
            resource_id_validator=self._resource_id_validator,
        )

        return getattr(self.api_client.select_version("v1beta1"), self._getter_method)(
            name=resource_name, retry=_DEFAULT_RETRY
        )

    def submit(
        self,
        *,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        timeout: Optional[int] = None,
        restart_job_on_worker_restart: bool = False,
        enable_web_access: bool = False,
        experiment: Optional[Union["aiplatform.Experiment", str]] = None,
        experiment_run: Optional[Union["aiplatform.ExperimentRun", str]] = None,
        tensorboard: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
        disable_retries: bool = False,
        max_wait_duration: Optional[int] = None,
    ) -> None:
        """Submit the configured CustomJob.

        Args:
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
            timeout (int):
                The maximum job running time in seconds. The default is 7 days.
            restart_job_on_worker_restart (bool):
                Restarts the entire CustomJob if a worker
                gets restarted. This feature can be used by
                distributed training jobs that are not resilient
                to workers leaving and joining a job.
            enable_web_access (bool):
                Whether you want Vertex AI to enable interactive shell access
                to training containers.
                https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell
            experiment (Union[aiplatform.Experiment, str]):
                Optional. The instance or name of an Experiment resource to which
                this CustomJob will upload training parameters and metrics.

                `service_account` is required with provided `experiment`.
                For more information on configuring your service account please visit:
                https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-training
            experiment_run (Union[aiplatform.ExperimentRun, str]):
                Optional. The instance or name of an ExperimentRun resource to which
                this CustomJob will upload training parameters and metrics.
                This arg can only be set when `experiment` is set. If 'experiment'
                is set but 'experiment_run` is not, an ExperimentRun resource
                will still be auto-generated.
            tensorboard (str):
                Optional. The name of a Vertex AI
                [Tensorboard][google.cloud.aiplatform.v1beta1.Tensorboard]
                resource to which this CustomJob will upload Tensorboard
                logs. Format:
                ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``

                The training script should write Tensorboard to following Vertex AI environment
                variable:

                AIP_TENSORBOARD_LOG_DIR

                `service_account` is required with provided `tensorboard`.
                For more information on configuring your service account please visit:
                https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-training
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            disable_retries (bool):
                Indicates if the job should retry for internal errors after the
                job starts running. If True, overrides
                `restart_job_on_worker_restart` to False.
            max_wait_duration (int):
                This is the maximum duration that a job will wait for the
                requested resources to be provisioned in seconds. If set to 0,
                the job will wait indefinitely. The default is 30 minutes.

        Raises:
            ValueError:
                If both `experiment` and `tensorboard` are specified or if
                `enable_autolog` is True in `CustomJob.from_local_script` but
                `experiment` is not specified or the specified experiment
                doesn't have a backing tensorboard.
        """
        if experiment and tensorboard:
            raise ValueError("'experiment' and 'tensorboard' cannot be set together.")
        if self._enable_autolog and (not experiment):
            raise ValueError(
                "'experiment' is required since you've enabled autolog in 'from_local_script'."
            )
        if service_account:
            self._gca_resource.job_spec.service_account = service_account

        if network:
            self._gca_resource.job_spec.network = network

        if (
            timeout
            or restart_job_on_worker_restart
            or disable_retries
            or max_wait_duration
        ):
            timeout = duration_pb2.Duration(seconds=timeout) if timeout else None
            max_wait_duration = (
                duration_pb2.Duration(seconds=max_wait_duration)
                if max_wait_duration
                else None
            )
            self._gca_resource.job_spec.scheduling = gca_custom_job_compat.Scheduling(
                timeout=timeout,
                restart_job_on_worker_restart=restart_job_on_worker_restart,
                disable_retries=disable_retries,
                max_wait_duration=max_wait_duration,
            )

        if enable_web_access:
            self._gca_resource.job_spec.enable_web_access = enable_web_access

        if tensorboard:
            self._gca_resource.job_spec.tensorboard = tensorboard

        # TODO(b/275105711) Update implementation after experiment/run in the proto
        if experiment:
            # short-term solution to set experiment/experimentRun in SDK
            if isinstance(experiment, aiplatform.Experiment):
                self._experiment = experiment
                # convert the Experiment instance to string to be passed to env
                experiment = experiment.name
            else:
                self._experiment = aiplatform.Experiment.get(experiment_name=experiment)
            if not self._experiment:
                raise ValueError(
                    f"Experiment '{experiment}' doesn't exist. "
                    "Please call aiplatform.init(experiment='my-exp') to create an experiment."
                )
            elif (
                not self._experiment.backing_tensorboard_resource_name
                and self._enable_autolog
            ):
                raise ValueError(
                    f"Experiment '{experiment}' doesn't have a backing tensorboard resource, "
                    "which is required by the experiment autologging feature. "
                    "Please call Experiment.assign_backing_tensorboard('my-tb-resource-name')."
                )

            # if run name is not specified, auto-generate one
            if not experiment_run:
                experiment_run = (
                    # TODO(b/223262536)Once display_name is optional this run name
                    # might be invalid as well.
                    f"{self._gca_resource.display_name}-{uuid.uuid4().hex[0:5]}"
                )

            # get or create the experiment run for the job
            if isinstance(experiment_run, aiplatform.ExperimentRun):
                self._experiment_run = experiment_run
                # convert the ExperimentRun instance to string to be passed to env
                experiment_run = experiment_run.name
            else:
                self._experiment_run = aiplatform.ExperimentRun.get(
                    run_name=experiment_run,
                    experiment=self._experiment,
                )
            if not self._experiment_run:
                self._experiment_run = aiplatform.ExperimentRun.create(
                    run_name=experiment_run,
                    experiment=self._experiment,
                )
            self._experiment_run.update_state(
                gcs_execution_compat.Execution.State.RUNNING
            )

            worker_pool_specs = self._gca_resource.job_spec.worker_pool_specs
            for spec in worker_pool_specs:
                if not spec:
                    continue

                if "python_package_spec" in spec:
                    container_spec = spec.python_package_spec
                else:
                    container_spec = spec.container_spec

                experiment_env = [
                    {
                        "name": metadata_constants.ENV_EXPERIMENT_KEY,
                        "value": experiment,
                    },
                    {
                        "name": metadata_constants.ENV_EXPERIMENT_RUN_KEY,
                        "value": experiment_run,
                    },
                ]
                if "env" in container_spec:
                    container_spec.env.extend(experiment_env)
                else:
                    container_spec.env = experiment_env

        _LOGGER.log_create_with_lro(self.__class__)

        self._gca_resource = self.api_client.select_version(
            "v1beta1"
        ).create_custom_job(
            parent=self._parent,
            custom_job=self._gca_resource,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_complete_with_getter(
            self.__class__, self._gca_resource, "custom_job"
        )

        _LOGGER.info("View Custom Job:\n%s" % self._dashboard_uri())

        if tensorboard:
            _LOGGER.info(
                "View Tensorboard:\n%s"
                % console_utils.custom_job_tensorboard_console_uri(
                    tensorboard, self.resource_name
                )
            )

        if experiment:
            custom_job = {
                metadata_constants._CUSTOM_JOB_RESOURCE_NAME: self.resource_name,
                metadata_constants._CUSTOM_JOB_CONSOLE_URI: self._dashboard_uri(),
            }

            run_context = self._experiment_run._metadata_node
            custom_jobs = run_context._gca_resource.metadata.get(
                metadata_constants._CUSTOM_JOB_KEY
            )
            if custom_jobs:
                custom_jobs.append(custom_job)
            else:
                custom_jobs = [custom_job]
            run_context.update({metadata_constants._CUSTOM_JOB_KEY: custom_jobs})


class HyperparameterTuningJob(jobs.HyperparameterTuningJob):
    """Deprecated. Vertex AI Hyperparameter Tuning Job (preview)."""

    def __init__(
        self,
        # TODO(b/223262536): Make display_name parameter fully optional in next major release
        display_name: str,
        custom_job: CustomJob,
        metric_spec: Dict[str, str],
        parameter_spec: Dict[str, hyperparameter_tuning._ParameterSpec],
        max_trial_count: int,
        parallel_trial_count: int,
        max_failed_trial_count: int = 0,
        search_algorithm: Optional[str] = None,
        measurement_selection: Optional[str] = "best",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        labels: Optional[Dict[str, str]] = None,
        encryption_spec_key_name: Optional[str] = None,
    ):
        """Deprecated. Please use the GA (non-preview) version of this class.

        Configures a HyperparameterTuning Job.

        Example usage:

        ```
        from google.cloud.aiplatform import hyperparameter_tuning as hpt

        worker_pool_specs = [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-4",
                        "accelerator_type": "NVIDIA_TESLA_K80",
                        "accelerator_count": 1,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": container_image_uri,
                        "command": [],
                        "args": [],
                    },
                }
            ]

        custom_job = aiplatform.preview.jobs.CustomJob(
            display_name='my_job',
            worker_pool_specs=worker_pool_specs,
            labels={'my_key': 'my_value'},
            persistent_resource_id='my_persistent_resource',
        )


        hp_job = aiplatform.preview.jobs.HyperparameterTuningJob(
            display_name='hp-test',
            custom_job=job,
            metric_spec={
                'loss': 'minimize',
            },
            parameter_spec={
                'lr': hpt.DoubleParameterSpec(min=0.001, max=0.1, scale='log'),
                'units': hpt.IntegerParameterSpec(min=4, max=128, scale='linear'),
                'activation': hpt.CategoricalParameterSpec(values=['relu', 'selu']),
                'batch_size': hpt.DiscreteParameterSpec(values=[128, 256], scale='linear')
            },
            max_trial_count=128,
            parallel_trial_count=8,
            labels={'my_key': 'my_value'},
            )

        hp_job.run()

        print(hp_job.trials)
        ```


        For more information on using hyperparameter tuning please visit:
        https://cloud.google.com/ai-platform-unified/docs/training/using-hyperparameter-tuning

        Args:
            display_name (str):
                Required. The user-defined name of the HyperparameterTuningJob.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            custom_job (aiplatform.preview.jobs.CustomJob):
                Required. Configured CustomJob. The worker pool spec from this custom job
                applies to the CustomJobs created in all the trials. A persistent_resource_id can be
                specified on the custom job to be used when running this Hyperparameter Tuning job.
            metric_spec: Dict[str, str]
                Required. Dictionary representing metrics to optimize. The dictionary key is the metric_id,
                which is reported by your training job, and the dictionary value is the
                optimization goal of the metric('minimize' or 'maximize'). example:

                metric_spec = {'loss': 'minimize', 'accuracy': 'maximize'}

            parameter_spec (Dict[str, hyperparameter_tuning._ParameterSpec]):
                Required. Dictionary representing parameters to optimize. The dictionary key is the metric_id,
                which is passed into your training job as a command line key word argument, and the
                dictionary value is the parameter specification of the metric.


                from google.cloud.aiplatform import hyperparameter_tuning as hpt

                parameter_spec={
                    'decay': hpt.DoubleParameterSpec(min=1e-7, max=1, scale='linear'),
                    'learning_rate': hpt.DoubleParameterSpec(min=1e-7, max=1, scale='linear')
                    'batch_size': hpt.DiscreteParamterSpec(values=[4, 8, 16, 32, 64, 128], scale='linear')
                }

                Supported parameter specifications can be found until aiplatform.hyperparameter_tuning.
                These parameter specification are currently supported:
                DoubleParameterSpec, IntegerParameterSpec, CategoricalParameterSpace, DiscreteParameterSpec

            max_trial_count (int):
                Required. The desired total number of Trials.
            parallel_trial_count (int):
                Required. The desired number of Trials to run in parallel.
            max_failed_trial_count (int):
                Optional. The number of failed Trials that need to be
                seen before failing the HyperparameterTuningJob.
                If set to 0, Vertex AI decides how many Trials
                must fail before the whole job fails.
            search_algorithm (str):
                The search algorithm specified for the Study.
                Accepts one of the following:
                    `None` - If you do not specify an algorithm, your job uses
                    the default Vertex AI algorithm. The default algorithm
                    applies Bayesian optimization to arrive at the optimal
                    solution with a more effective search over the parameter space.

                    'grid' - A simple grid search within the feasible space. This
                    option is particularly useful if you want to specify a quantity
                    of trials that is greater than the number of points in the
                    feasible space. In such cases, if you do not specify a grid
                    search, the Vertex AI default algorithm may generate duplicate
                    suggestions. To use grid search, all parameter specs must be
                    of type `IntegerParameterSpec`, `CategoricalParameterSpace`,
                    or `DiscreteParameterSpec`.

                    'random' - A simple random search within the feasible space.
            measurement_selection (str):
                This indicates which measurement to use if/when the service
                automatically selects the final measurement from previously reported
                intermediate measurements.

                Accepts: 'best', 'last'

                Choose this based on two considerations:
                A) Do you expect your measurements to monotonically improve? If so,
                choose 'last'. On the other hand, if you're in a situation
                where your system can "over-train" and you expect the performance to
                get better for a while but then start declining, choose
                'best'. B) Are your measurements significantly noisy
                and/or irreproducible? If so, 'best' will tend to be
                over-optimistic, and it may be better to choose 'last'. If
                both or neither of (A) and (B) apply, it doesn't matter which
                selection type is chosen.
            project (str):
                Optional. Project to run the HyperparameterTuningjob in. Overrides project set in aiplatform.init.
            location (str):
                Optional. Location to run the HyperparameterTuning in. Overrides location set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to run call HyperparameterTuning service. Overrides
                credentials set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to
                organize HyperparameterTuningJobs.
                Label keys and values can be no longer than 64
                characters (Unicode codepoints), can only
                contain lowercase letters, numeric characters,
                underscores and dashes. International characters
                are allowed.
                See https://goo.gl/xmQnxf for more information
                and examples of labels.
            encryption_spec_key_name (str):
                Optional. Customer-managed encryption key options for a
                HyperparameterTuningJob. If this is set, then
                all resources created by the
                HyperparameterTuningJob will be encrypted with
                the provided encryption key.
        """

        super(jobs.HyperparameterTuningJob, self).__init__(
            project=project, location=location, credentials=credentials
        )

        metrics = [
            study_v1beta1.StudySpec.MetricSpec(metric_id=metric_id, goal=goal.upper())
            for metric_id, goal in metric_spec.items()
        ]

        parameters = [
            parameter._to_parameter_spec_v1beta1(parameter_id=parameter_id)
            for parameter_id, parameter in parameter_spec.items()
        ]

        study_spec = study_v1beta1.StudySpec(
            metrics=metrics,
            parameters=parameters,
            algorithm=hyperparameter_tuning.SEARCH_ALGORITHM_TO_PROTO_VALUE[
                search_algorithm
            ],
            measurement_selection_type=hyperparameter_tuning.MEASUREMENT_SELECTION_TO_PROTO_VALUE[
                measurement_selection
            ],
        )

        if not display_name:
            display_name = self.__class__._generate_display_name()

        self._gca_resource = (
            gca_hyperparameter_tuning_job_compat.HyperparameterTuningJob(
                display_name=display_name,
                study_spec=study_spec,
                max_trial_count=max_trial_count,
                parallel_trial_count=parallel_trial_count,
                max_failed_trial_count=max_failed_trial_count,
                trial_job_spec=copy.deepcopy(custom_job.job_spec),
                labels=labels,
                encryption_spec=initializer.global_config.get_encryption_spec(
                    encryption_spec_key_name=encryption_spec_key_name,
                    select_version=compat.V1BETA1,
                ),
            )
        )

    def _get_gca_resource(
        self,
        resource_name: str,
        parent_resource_name_fields: Optional[Dict[str, str]] = None,
    ) -> proto.Message:
        """Returns GAPIC service representation of client class resource.

        Args:
            resource_name (str): Required. A fully-qualified resource name or ID.
            parent_resource_name_fields (Dict[str,str]):
                Optional. Mapping of parent resource name key to values. These
                will be used to compose the resource name if only resource ID is given.
                Should not include project and location.
        """
        resource_name = utils.full_resource_name(
            resource_name=resource_name,
            resource_noun=self._resource_noun,
            parse_resource_name_method=self._parse_resource_name,
            format_resource_name_method=self._format_resource_name,
            project=self.project,
            location=self.location,
            parent_resource_name_fields=parent_resource_name_fields,
            resource_id_validator=self._resource_id_validator,
        )

        return getattr(self.api_client.select_version("v1beta1"), self._getter_method)(
            name=resource_name, retry=_DEFAULT_RETRY
        )

    @base.optional_sync()
    def _run(
        self,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        timeout: Optional[int] = None,  # seconds
        restart_job_on_worker_restart: bool = False,
        enable_web_access: bool = False,
        tensorboard: Optional[str] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
        disable_retries: bool = False,
        max_wait_duration: Optional[int] = None,
    ) -> None:
        """Helper method to ensure network synchronization and to run the configured CustomJob.

        Args:
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
            timeout (int):
                Optional. The maximum job running time in seconds. The default is 7 days.
            restart_job_on_worker_restart (bool):
                Restarts the entire CustomJob if a worker
                gets restarted. This feature can be used by
                distributed training jobs that are not resilient
                to workers leaving and joining a job.
            enable_web_access (bool):
                Whether you want Vertex AI to enable interactive shell access
                to training containers.
                https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell
            tensorboard (str):
                Optional. The name of a Vertex AI
                [Tensorboard][google.cloud.aiplatform.v1beta1.Tensorboard]
                resource to which this CustomJob will upload Tensorboard
                logs. Format:
                ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``

                The training script should write Tensorboard to following Vertex AI environment
                variable:

                AIP_TENSORBOARD_LOG_DIR

                `service_account` is required with provided `tensorboard`.
                For more information on configuring your service account please visit:
                https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-training
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will unblock and it will be executed in a concurrent Future.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            disable_retries (bool):
                Indicates if the job should retry for internal errors after the
                job starts running. If True, overrides
                `restart_job_on_worker_restart` to False.
            max_wait_duration (int):
                This is the maximum duration that a job will wait for the
                requested resources to be provisioned in seconds. If set to 0,
                the job will wait indefinitely. The default is 30 minutes.
        """
        if service_account:
            self._gca_resource.trial_job_spec.service_account = service_account

        if network:
            self._gca_resource.trial_job_spec.network = network

        if (
            timeout
            or restart_job_on_worker_restart
            or disable_retries
            or max_wait_duration
        ):
            timeout = duration_pb2.Duration(seconds=timeout) if timeout else None
            max_wait_duration = (
                duration_pb2.Duration(seconds=max_wait_duration)
                if max_wait_duration
                else None
            )
            self._gca_resource.trial_job_spec.scheduling = (
                gca_custom_job_compat.Scheduling(
                    timeout=timeout,
                    restart_job_on_worker_restart=restart_job_on_worker_restart,
                    disable_retries=disable_retries,
                    max_wait_duration=max_wait_duration,
                )
            )

        if enable_web_access:
            self._gca_resource.trial_job_spec.enable_web_access = enable_web_access

        if tensorboard:
            self._gca_resource.trial_job_spec.tensorboard = tensorboard

        _LOGGER.log_create_with_lro(self.__class__)

        self._gca_resource = self.api_client.select_version(
            "v1beta1"
        ).create_hyperparameter_tuning_job(
            parent=self._parent,
            hyperparameter_tuning_job=self._gca_resource,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_complete_with_getter(
            self.__class__, self._gca_resource, "hpt_job"
        )

        _LOGGER.info("View HyperparameterTuningJob:\n%s" % self._dashboard_uri())

        if tensorboard:
            _LOGGER.info(
                "View Tensorboard:\n%s"
                % console_utils.custom_job_tensorboard_console_uri(
                    tensorboard, self.resource_name
                )
            )

        self._block_until_complete()


class BatchPredictionJob(jobs.BatchPredictionJob):
    """Vertex AI Batch Prediction Job."""

    @classmethod
    def create(
        cls,
        # TODO(b/223262536): Make the job_display_name parameter optional in the next major release
        job_display_name: str,
        model_name: Union[str, "aiplatform.Model"],
        instances_format: str = "jsonl",
        predictions_format: str = "jsonl",
        gcs_source: Optional[Union[str, Sequence[str]]] = None,
        bigquery_source: Optional[str] = None,
        gcs_destination_prefix: Optional[str] = None,
        bigquery_destination_prefix: Optional[str] = None,
        model_parameters: Optional[Dict] = None,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        starting_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
        generate_explanation: Optional[bool] = False,
        explanation_metadata: Optional["aiplatform.explain.ExplanationMetadata"] = None,
        explanation_parameters: Optional[
            "aiplatform.explain.ExplanationParameters"
        ] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
        sync: bool = True,
        create_request_timeout: Optional[float] = None,
        batch_size: Optional[int] = None,
        model_monitoring_objective_config: Optional[
            "aiplatform.model_monitoring.ObjectiveConfig"
        ] = None,
        model_monitoring_alert_config: Optional[
            "aiplatform.model_monitoring.AlertConfig"
        ] = None,
        analysis_instance_schema_uri: Optional[str] = None,
        service_account: Optional[str] = None,
        reservation_affinity_type: Optional[str] = None,
        reservation_affinity_key: Optional[str] = None,
        reservation_affinity_values: Optional[List[str]] = None,
    ) -> "BatchPredictionJob":
        """Create a batch prediction job.

        Args:
            job_display_name (str):
                Required. The user-defined name of the BatchPredictionJob.
                The name can be up to 128 characters long and can be consist
                of any UTF-8 characters.
            model_name (Union[str, aiplatform.Model]):
                Required. A fully-qualified model resource name or model ID.
                Example: "projects/123/locations/us-central1/models/456" or
                "456" when project and location are initialized or passed.
                May optionally contain a version ID or alias in
                {model_name}@{version} form.

                Or an instance of aiplatform.Model.
            instances_format (str):
                Required. The format in which instances are provided. Must be one
                of the formats listed in `Model.supported_input_storage_formats`.
                Default is "jsonl" when using `gcs_source`. If a `bigquery_source`
                is provided, this is overridden to "bigquery".
            predictions_format (str):
                Required. The format in which Vertex AI outputs the
                predictions, must be one of the formats specified in
                `Model.supported_output_storage_formats`.
                Default is "jsonl" when using `gcs_destination_prefix`. If a
                `bigquery_destination_prefix` is provided, this is overridden to
                "bigquery".
            gcs_source (Optional[Sequence[str]]):
                Google Cloud Storage URI(-s) to your instances to run
                batch prediction on. They must match `instances_format`.

            bigquery_source (Optional[str]):
                BigQuery URI to a table, up to 2000 characters long. For example:
                `bq://projectId.bqDatasetId.bqTableId`
            gcs_destination_prefix (Optional[str]):
                The Google Cloud Storage location of the directory where the
                output is to be written to. In the given directory a new
                directory is created. Its name is
                ``prediction-<model-display-name>-<job-create-time>``, where
                timestamp is in YYYY-MM-DDThh:mm:ss.sssZ ISO-8601 format.
                Inside of it files ``predictions_0001.<extension>``,
                ``predictions_0002.<extension>``, ...,
                ``predictions_N.<extension>`` are created where
                ``<extension>`` depends on chosen ``predictions_format``,
                and N may equal 0001 and depends on the total number of
                successfully predicted instances. If the Model has both
                ``instance`` and ``prediction`` schemata defined then each such
                file contains predictions as per the ``predictions_format``.
                If prediction for any instance failed (partially or
                completely), then an additional ``errors_0001.<extension>``,
                ``errors_0002.<extension>``,..., ``errors_N.<extension>``
                files are created (N depends on total number of failed
                predictions). These files contain the failed instances, as
                per their schema, followed by an additional ``error`` field
                which as value has ```google.rpc.Status`` <Status>`__
                containing only ``code`` and ``message`` fields.
            bigquery_destination_prefix (Optional[str]):
                The BigQuery project or dataset location where the output is
                to be written to. If project is provided, a new dataset is
                created with name
                ``prediction_<model-display-name>_<job-create-time>`` where
                is made BigQuery-dataset-name compatible (for example, most
                special characters become underscores), and timestamp is in
                YYYY_MM_DDThh_mm_ss_sssZ "based on ISO-8601" format. In the
                dataset two tables will be created, ``predictions``, and
                ``errors``. If the Model has both
                [instance][google.cloud.aiplatform.v1.PredictSchemata.instance_schema_uri]
                and
                [prediction][google.cloud.aiplatform.v1.PredictSchemata.parameters_schema_uri]
                schemata defined then the tables have columns as follows:
                The ``predictions`` table contains instances for which the
                prediction succeeded, it has columns as per a concatenation
                of the Model's instance and prediction schemata. The
                ``errors`` table contains rows for which the prediction has
                failed, it has instance columns, as per the instance schema,
                followed by a single "errors" column, which as values has
                [google.rpc.Status][google.rpc.Status] represented as a
                STRUCT, and containing only ``code`` and ``message``.
            model_parameters (Optional[Dict]):
                The parameters that govern the predictions. The schema of
                the parameters may be specified via the Model's `parameters_schema_uri`.
            machine_type (Optional[str]):
                The type of machine for running batch prediction on
                dedicated resources. Not specifying machine type will result in
                batch prediction job being run with automatic resources.
            accelerator_type (Optional[str]):
                The type of accelerator(s) that may be attached
                to the machine as per `accelerator_count`. Only used if
                `machine_type` is set.
            accelerator_count (Optional[int]):
                The number of accelerators to attach to the
                `machine_type`. Only used if `machine_type` is set.
            starting_replica_count (Optional[int]):
                The number of machine replicas used at the start of the batch
                operation. If not set, Vertex AI decides starting number, not
                greater than `max_replica_count`. Only used if `machine_type` is
                set.
            max_replica_count (Optional[int]):
                The maximum number of machine replicas the batch operation may
                be scaled to. Only used if `machine_type` is set.
                Default is 10.
            generate_explanation (bool):
                Optional. Generate explanation along with the batch prediction
                results. This will cause the batch prediction output to include
                explanations based on the `prediction_format`:
                    - `bigquery`: output includes a column named `explanation`. The value
                        is a struct that conforms to the [aiplatform.gapic.Explanation] object.
                    - `jsonl`: The JSON objects on each line include an additional entry
                        keyed `explanation`. The value of the entry is a JSON object that
                        conforms to the [aiplatform.gapic.Explanation] object.
                    - `csv`: Generating explanations for CSV format is not supported.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
                Optional. Explanation metadata configuration for this BatchPredictionJob.
                Can be specified only if `generate_explanation` is set to `True`.

                This value overrides the value of `Model.explanation_metadata`.
                All fields of `explanation_metadata` are optional in the request. If
                a field of the `explanation_metadata` object is not populated, the
                corresponding field of the `Model.explanation_metadata` object is inherited.
                For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
                Optional. Parameters to configure explaining for Model's predictions.
                Can be specified only if `generate_explanation` is set to `True`.

                This value overrides the value of `Model.explanation_parameters`.
                All fields of `explanation_parameters` are optional in the request. If
                a field of the `explanation_parameters` object is not populated, the
                corresponding field of the `Model.explanation_parameters` object is inherited.
                For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
            labels (Dict[str, str]):
                Optional. The labels with user-defined metadata to organize your
                BatchPredictionJobs. Label keys and values can be no longer than
                64 characters (Unicode codepoints), can only contain lowercase
                letters, numeric characters, underscores and dashes.
                International characters are allowed. See https://goo.gl/xmQnxf
                for more information and examples of labels.
            credentials (Optional[auth_credentials.Credentials]):
                Custom credentials to use to create this batch prediction
                job. Overrides credentials set in aiplatform.init.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the job. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If this is set, then all
                resources created by the BatchPredictionJob will
                be encrypted with the provided encryption key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            sync (bool):
                Whether to execute this method synchronously. If False, this method
                will be executed in concurrent Future and any downstream object will
                be immediately returned and synced when the Future has completed.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            batch_size (int):
                Optional. The number of the records (e.g. instances) of the operation given in each batch
                to a machine replica. Machine type, and size of a single record should be considered
                when setting this parameter, higher value speeds up the batch operation's execution,
                but too high value will result in a whole batch not fitting in a machine's memory,
                and the whole operation will fail.
                The default value is 64.
            model_monitoring_objective_config (aiplatform.model_monitoring.ObjectiveConfig):
                Optional. The objective config for model monitoring. Passing this parameter enables
                monitoring on the model associated with this batch prediction job.
            model_monitoring_alert_config (aiplatform.model_monitoring.EmailAlertConfig):
                Optional. Configures how model monitoring alerts are sent to the user. Right now
                only email alert is supported.
            analysis_instance_schema_uri (str):
                Optional. Only applicable if model_monitoring_objective_config is also passed.
                This parameter specifies the YAML schema file uri describing the format of a single
                instance that you want Tensorflow Data Validation (TFDV) to
                analyze. If this field is empty, all the feature data types are
                inferred from predict_instance_schema_uri, meaning that TFDV
                will use the data in the exact format as prediction request/response.
                If there are any data type differences between predict instance
                and TFDV instance, this field can be used to override the schema.
                For models trained with Vertex AI, this field must be set as all the
                fields in predict instance formatted as string.
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            reservation_affinity_type (str):
                Optional. The type of reservation affinity.
                One of NO_RESERVATION, ANY_RESERVATION, SPECIFIC_RESERVATION,
                SPECIFIC_THEN_ANY_RESERVATION, SPECIFIC_THEN_NO_RESERVATION
            reservation_affinity_key (str):
                Optional. Corresponds to the label key of a reservation resource.
                To target a SPECIFIC_RESERVATION by name, use `compute.googleapis.com/reservation-name` as the key
                and specify the name of your reservation as its value.
            reservation_affinity_values (List[str]):
                Optional. Corresponds to the label values of a reservation resource.
                This must be the full resource name of the reservation.
                Format: 'projects/{project_id_or_number}/zones/{zone}/reservations/{reservation_name}'
        Returns:
            (jobs.BatchPredictionJob):
                Instantiated representation of the created batch prediction job.
        """
        return cls._submit_impl(
            job_display_name=job_display_name,
            model_name=model_name,
            instances_format=instances_format,
            predictions_format=predictions_format,
            gcs_source=gcs_source,
            bigquery_source=bigquery_source,
            gcs_destination_prefix=gcs_destination_prefix,
            bigquery_destination_prefix=bigquery_destination_prefix,
            model_parameters=model_parameters,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            starting_replica_count=starting_replica_count,
            max_replica_count=max_replica_count,
            generate_explanation=generate_explanation,
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
            labels=labels,
            project=project,
            location=location,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            sync=sync,
            create_request_timeout=create_request_timeout,
            batch_size=batch_size,
            model_monitoring_objective_config=model_monitoring_objective_config,
            model_monitoring_alert_config=model_monitoring_alert_config,
            analysis_instance_schema_uri=analysis_instance_schema_uri,
            service_account=service_account,
            reservation_affinity_type=reservation_affinity_type,
            reservation_affinity_key=reservation_affinity_key,
            reservation_affinity_values=reservation_affinity_values,
            # Main distinction of `create` vs `submit`:
            wait_for_completion=True,
        )

    @classmethod
    def submit(
        cls,
        *,
        job_display_name: Optional[str] = None,
        model_name: Union[str, "aiplatform.Model"],
        instances_format: str = "jsonl",
        predictions_format: str = "jsonl",
        gcs_source: Optional[Union[str, Sequence[str]]] = None,
        bigquery_source: Optional[str] = None,
        gcs_destination_prefix: Optional[str] = None,
        bigquery_destination_prefix: Optional[str] = None,
        model_parameters: Optional[Dict] = None,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        starting_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
        generate_explanation: Optional[bool] = False,
        explanation_metadata: Optional["aiplatform.explain.ExplanationMetadata"] = None,
        explanation_parameters: Optional[
            "aiplatform.explain.ExplanationParameters"
        ] = None,
        labels: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
        batch_size: Optional[int] = None,
        model_monitoring_objective_config: Optional[
            "aiplatform.model_monitoring.ObjectiveConfig"
        ] = None,
        model_monitoring_alert_config: Optional[
            "aiplatform.model_monitoring.AlertConfig"
        ] = None,
        analysis_instance_schema_uri: Optional[str] = None,
        service_account: Optional[str] = None,
        reservation_affinity_type: Optional[str] = None,
        reservation_affinity_key: Optional[str] = None,
        reservation_affinity_values: Optional[List[str]] = None,
    ) -> "BatchPredictionJob":
        """Sumbit a batch prediction job (not waiting for completion).

        Args:
            job_display_name (str): Required. The user-defined name of the
              BatchPredictionJob. The name can be up to 128 characters long and
              can be consist of any UTF-8 characters.
            model_name (Union[str, aiplatform.Model]): Required. A fully-qualified
              model resource name or model ID.
                Example: "projects/123/locations/us-central1/models/456" or "456"
                  when project and location are initialized or passed. May
                  optionally contain a version ID or alias in
                  {model_name}@{version} form.  Or an instance of
                  aiplatform.Model.
            instances_format (str): Required. The format in which instances are
              provided. Must be one of the formats listed in
              `Model.supported_input_storage_formats`. Default is "jsonl" when
              using `gcs_source`. If a `bigquery_source` is provided, this is
              overridden to "bigquery".
            predictions_format (str): Required. The format in which Vertex AI
              outputs the predictions, must be one of the formats specified in
              `Model.supported_output_storage_formats`. Default is "jsonl" when
              using `gcs_destination_prefix`. If a `bigquery_destination_prefix`
              is provided, this is overridden to "bigquery".
            gcs_source (Optional[Sequence[str]]): Google Cloud Storage URI(-s) to
              your instances to run batch prediction on. They must match
              `instances_format`.
            bigquery_source (Optional[str]): BigQuery URI to a table, up to 2000
              characters long. For example: `bq://projectId.bqDatasetId.bqTableId`
            gcs_destination_prefix (Optional[str]): The Google Cloud Storage
              location of the directory where the output is to be written to. In
              the given directory a new directory is created. Its name is
              ``prediction-<model-display-name>-<job-create-time>``, where
              timestamp is in YYYY-MM-DDThh:mm:ss.sssZ ISO-8601 format. Inside of
              it files ``predictions_0001.<extension>``,
              ``predictions_0002.<extension>``, ..., ``predictions_N.<extension>``
              are created where ``<extension>`` depends on chosen
              ``predictions_format``, and N may equal 0001 and depends on the
              total number of successfully predicted instances. If the Model has
              both ``instance`` and ``prediction`` schemata defined then each such
              file contains predictions as per the ``predictions_format``. If
              prediction for any instance failed (partially or completely), then
              an additional ``errors_0001.<extension>``,
              ``errors_0002.<extension>``,..., ``errors_N.<extension>`` files are
              created (N depends on total number of failed predictions). These
              files contain the failed instances, as per their schema, followed by
              an additional ``error`` field which as value has
              ```google.rpc.Status`` <Status>`__ containing only ``code`` and
              ``message`` fields.
            bigquery_destination_prefix (Optional[str]): The BigQuery project or
              dataset location where the output is to be written to. If project is
              provided, a new dataset is created with name
              ``prediction_<model-display-name>_<job-create-time>`` where is made
              BigQuery-dataset-name compatible (for example, most special
              characters become underscores), and timestamp is in
              YYYY_MM_DDThh_mm_ss_sssZ "based on ISO-8601" format. In the dataset
              two tables will be created, ``predictions``, and ``errors``. If the
              Model has both
              [instance][google.cloud.aiplatform.v1.PredictSchemata.instance_schema_uri]
              and
              [prediction][google.cloud.aiplatform.v1.PredictSchemata.parameters_schema_uri]
              schemata defined then the tables have columns as follows: The
              ``predictions`` table contains instances for which the prediction
              succeeded, it has columns as per a concatenation of the Model's
              instance and prediction schemata. The ``errors`` table contains rows
              for which the prediction has failed, it has instance columns, as per
              the instance schema, followed by a single "errors" column, which as
              values has [google.rpc.Status][google.rpc.Status] represented as a
              STRUCT, and containing only ``code`` and ``message``.
            model_parameters (Optional[Dict]): The parameters that govern the
              predictions. The schema of the parameters may be specified via the
              Model's `parameters_schema_uri`.
            machine_type (Optional[str]): The type of machine for running batch
              prediction on dedicated resources. Not specifying machine type will
              result in batch prediction job being run with automatic resources.
            accelerator_type (Optional[str]): The type of accelerator(s) that may
              be attached to the machine as per `accelerator_count`. Only used if
              `machine_type` is set.
            accelerator_count (Optional[int]): The number of accelerators to
              attach to the `machine_type`. Only used if `machine_type` is set.
            starting_replica_count (Optional[int]): The number of machine replicas
              used at the start of the batch operation. If not set, Vertex AI
              decides starting number, not greater than `max_replica_count`. Only
              used if `machine_type` is set.
            max_replica_count (Optional[int]): The maximum number of machine
              replicas the batch operation may be scaled to. Only used if
              `machine_type` is set. Default is 10.
            generate_explanation (bool): Optional. Generate explanation along with
              the batch prediction results. This will cause the batch prediction
              output to include explanations based on the `prediction_format`: -
              `bigquery`: output includes a column named `explanation`. The value
              is a struct that conforms to the [aiplatform.gapic.Explanation]
              object. - `jsonl`: The JSON objects on each line include an
              additional entry keyed `explanation`. The value of the entry is a
              JSON object that conforms to the [aiplatform.gapic.Explanation]
              object. - `csv`: Generating explanations for CSV format is not
              supported.
            explanation_metadata (aiplatform.explain.ExplanationMetadata):
              Optional. Explanation metadata configuration for this
              BatchPredictionJob. Can be specified only if `generate_explanation`
              is set to `True`.  This value overrides the value of
              `Model.explanation_metadata`. All fields of `explanation_metadata`
              are optional in the request. If a field of the
              `explanation_metadata` object is not populated, the corresponding
              field of the `Model.explanation_metadata` object is inherited. For
              more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
            explanation_parameters (aiplatform.explain.ExplanationParameters):
              Optional. Parameters to configure explaining for Model's
              predictions. Can be specified only if `generate_explanation` is set
              to `True`.  This value overrides the value of
              `Model.explanation_parameters`. All fields of
              `explanation_parameters` are optional in the request. If a field of
              the `explanation_parameters` object is not populated, the
              corresponding field of the `Model.explanation_parameters` object is
              inherited. For more details, see `Ref docs
              <http://tinyurl.com/1an4zake>`
            labels (Dict[str, str]): Optional. The labels with user-defined
              metadata to organize your BatchPredictionJobs. Label keys and values
              can be no longer than 64 characters (Unicode codepoints), can only
              contain lowercase letters, numeric characters, underscores and
              dashes. International characters are allowed. See
              https://goo.gl/xmQnxf for more information and examples of labels.
            credentials (Optional[auth_credentials.Credentials]): Custom
              credentials to use to create this batch prediction job. Overrides
              credentials set in aiplatform.init.
            encryption_spec_key_name (Optional[str]): Optional. The Cloud KMS
              resource identifier of the customer managed encryption key used to
              protect the job. Has the
                form:
                  ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                  The key needs to be in the same region as where the compute
                  resource is created.  If this is set, then all resources created
                  by the BatchPredictionJob will be encrypted with the provided
                  encryption key.  Overrides encryption_spec_key_name set in
                  aiplatform.init.
            create_request_timeout (float): Optional. The timeout for the create
              request in seconds.
            batch_size (int): Optional. The number of the records (e.g. instances)
              of the operation given in each batch to a machine replica. Machine
              type, and size of a single record should be considered when setting
              this parameter, higher value speeds up the batch operation's
              execution, but too high value will result in a whole batch not
              fitting in a machine's memory, and the whole operation will fail.
              The default value is 64. model_monitoring_objective_config
              (aiplatform.model_monitoring.ObjectiveConfig): Optional. The
              objective config for model monitoring. Passing this parameter
              enables monitoring on the model associated with this batch
              prediction job. model_monitoring_alert_config
              (aiplatform.model_monitoring.EmailAlertConfig): Optional. Configures
              how model monitoring alerts are sent to the user. Right now only
              email alert is supported.
            analysis_instance_schema_uri (str): Optional. Only applicable if
              model_monitoring_objective_config is also passed. This parameter
              specifies the YAML schema file uri describing the format of a single
              instance that you want Tensorflow Data Validation (TFDV) to analyze.
              If this field is empty, all the feature data types are inferred from
              predict_instance_schema_uri, meaning that TFDV will use the data in
              the exact format as prediction request/response. If there are any
              data type differences between predict instance and TFDV instance,
              this field can be used to override the schema. For models trained
              with Vertex AI, this field must be set as all the fields in predict
              instance formatted as string.
            service_account (str): Optional. Specifies the service account for
              workload run-as account. Users submitting jobs must have act-as
              permission on this run-as account.
            reservation_affinity_type (str): Optional. The type of reservation
              affinity. One of NO_RESERVATION, ANY_RESERVATION,
              SPECIFIC_RESERVATION, SPECIFIC_THEN_ANY_RESERVATION,
              SPECIFIC_THEN_NO_RESERVATION
            reservation_affinity_key (str): Optional. Corresponds to the label key
              of a reservation resource. To target a SPECIFIC_RESERVATION by name,
              use `compute.googleapis.com/reservation-name` as the key and specify
              the name of your reservation as its value.
            reservation_affinity_values (List[str]): Optional. Corresponds to the
              label values of a reservation resource. This must be the full
              resource name of the reservation.
                Format:
                  'projects/{project_id_or_number}/zones/{zone}/reservations/{reservation_name}'

        Returns:
            (jobs.BatchPredictionJob):
                Instantiated representation of the created batch prediction job.
        """
        return cls._submit_impl(
            job_display_name=job_display_name,
            model_name=model_name,
            instances_format=instances_format,
            predictions_format=predictions_format,
            gcs_source=gcs_source,
            bigquery_source=bigquery_source,
            gcs_destination_prefix=gcs_destination_prefix,
            bigquery_destination_prefix=bigquery_destination_prefix,
            model_parameters=model_parameters,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            starting_replica_count=starting_replica_count,
            max_replica_count=max_replica_count,
            generate_explanation=generate_explanation,
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
            labels=labels,
            project=project,
            location=location,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            create_request_timeout=create_request_timeout,
            batch_size=batch_size,
            model_monitoring_objective_config=model_monitoring_objective_config,
            model_monitoring_alert_config=model_monitoring_alert_config,
            analysis_instance_schema_uri=analysis_instance_schema_uri,
            service_account=service_account,
            reservation_affinity_type=reservation_affinity_type,
            reservation_affinity_key=reservation_affinity_key,
            reservation_affinity_values=reservation_affinity_values,
            # Main distinction of `create` vs `submit`:
            wait_for_completion=False,
            sync=True,
        )
