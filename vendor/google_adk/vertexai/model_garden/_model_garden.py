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
# pylint: disable=bad-continuation, line-too-long, protected-access
"""Class for interacting with Model Garden OSS models."""

import datetime
import functools
import re
from typing import Dict, List, Optional, Sequence, Union

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import compat
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import models as aiplatform_models
from google.cloud.aiplatform import utils
from google.cloud.aiplatform_v1beta1 import types
from google.cloud.aiplatform_v1beta1.services import model_garden_service
from google.cloud.aiplatform_v1beta1.services import model_service
from vertexai import batch_prediction


from google.protobuf import duration_pb2


_LOGGER = base.Logger(__name__)
_DEFAULT_VERSION = compat.V1BETA1
_DEFAULT_TIMEOUT = 2 * 60 * 60  # 2 hours, same as UI one-click deployment.
_DEFAULT_RECOMMEND_SPEC_TIMEOUT = 1 * 60  # 1 minute.
_DEFAULT_EXPORT_TIMEOUT = 1 * 60 * 60  # 1 hour.
_HF_WILDCARD_FILTER = "is_hf_wildcard(true)"
_NATIVE_MODEL_FILTER = "is_hf_wildcard(false)"
_VERIFIED_DEPLOYMENT_FILTER = (
    "labels.VERIFIED_DEPLOYMENT_CONFIG=VERIFIED_DEPLOYMENT_SUCCEED"
)


def list_deployable_models(
    *, list_hf_models: bool = False, model_filter: Optional[str] = None
) -> List[str]:
    """Lists the deployable models in Model Garden.

    Args:
        list_hf_models: Whether to list the Hugging Face models.
        model_filter: Optional. A string to filter the models by.

    Returns:
        The names of the deployable models in Model Garden in the format of
        `{publisher}/{model}@{version}` or Hugging Face model ID in the format
        of `{organization}/{model}`.
    """
    filter_str = _NATIVE_MODEL_FILTER
    if list_hf_models:
        filter_str = " AND ".join([_HF_WILDCARD_FILTER, _VERIFIED_DEPLOYMENT_FILTER])
    if model_filter:
        filter_str = (
            f'{filter_str} AND (model_user_id=~"(?i).*{model_filter}.*" OR'
            f' display_name=~"(?i).*{model_filter}.*")'
        )

    request = types.ListPublisherModelsRequest(
        parent="publishers/*",
        list_all_versions=True,
        filter=filter_str,
    )
    client = initializer.global_config.create_client(
        client_class=_ModelGardenClientWithOverride,
        credentials=initializer.global_config.credentials,
        location_override="us-central1",
    )
    response = client.list_publisher_models(request)
    output = []
    for page in response.pages:
        for model in page.publisher_models:
            if model.supported_actions.multi_deploy_vertex.multi_deploy_vertex:
                output.append(
                    re.sub(r"publishers/(hf-|)|models/", "", model.name)
                    + ("" if list_hf_models else ("@" + model.version_id))
                )
    return output


def _is_hugging_face_model(model_name: str) -> bool:
    """Returns whether the model is a Hugging Face model."""
    return re.match(r"^(?P<publisher>[^/]+)/(?P<model>[^/@]+)$", model_name)


def _get_publisher_model_resource_name(publisher: str, model: str) -> str:
    """Returns the resource name.

    Args:
        publisher: Publisher of the model.
        model: Model name, may or may not include version.

    Returns:
        The resource name in the format of
            `publishers/{publisher}/models/{model_user_id}@{version_id}`.
    """
    return f"publishers/{publisher}/models/{model}"


def _reconcile_model_name(model_name: str) -> str:
    """Returns the resource name from the model name.

    Args:
        model_name: Model Garden model resource name in the format of
          `publishers/{publisher}/models/{model}@{version}`, or a simplified
          resource name in the format of `{publisher}/{model}@{version}`, or a
          Hugging Face model ID in the format of `{organization}/{model}`.

    Returns:
        The resource name in the format of
            `publishers/{publisher}/models/{model}@{version}`.
    """
    model_name = model_name.lower()  # Use lower case for Hugging Face.
    full_resource_name_match = re.match(
        r"^publishers/(?P<publisher>[^/]+)/models/(?P<model>[^@]+)@(?P<version>[^@]+)$",
        model_name,
    )
    if full_resource_name_match:
        return _get_publisher_model_resource_name(
            full_resource_name_match.group("publisher"),
            full_resource_name_match.group("model")
            + "@"
            + full_resource_name_match.group("version"),
        )
    else:
        simplified_name_match = re.match(
            r"^(?P<publisher>[^/]+)/(?P<model>[^@]+)(?:@(?P<version>.+))?$",
            model_name,
        )
        if simplified_name_match:
            if simplified_name_match.group("version"):
                return _get_publisher_model_resource_name(
                    publisher=simplified_name_match.group("publisher"),
                    model=simplified_name_match.group("model")
                    + "@"
                    + simplified_name_match.group("version"),
                )
            else:
                return _get_publisher_model_resource_name(
                    publisher=simplified_name_match.group("publisher"),
                    model=simplified_name_match.group("model"),
                )
        else:
            raise ValueError(f"`{model_name}` is not a valid Open Model name")


def _construct_serving_container_spec(
    serving_container_image_uri: Optional[str] = None,
    serving_container_predict_route: Optional[str] = None,
    serving_container_health_route: Optional[str] = None,
    serving_container_command: Optional[Sequence[str]] = None,
    serving_container_args: Optional[Sequence[str]] = None,
    serving_container_environment_variables: Optional[Dict[str, str]] = None,
    serving_container_ports: Optional[Sequence[int]] = None,
    serving_container_grpc_ports: Optional[Sequence[int]] = None,
    serving_container_deployment_timeout: Optional[int] = None,
    serving_container_shared_memory_size_mb: Optional[int] = None,
    serving_container_startup_probe_exec: Optional[Sequence[str]] = None,
    serving_container_startup_probe_period_seconds: Optional[int] = None,
    serving_container_startup_probe_timeout_seconds: Optional[int] = None,
    serving_container_health_probe_exec: Optional[Sequence[str]] = None,
    serving_container_health_probe_period_seconds: Optional[int] = None,
    serving_container_health_probe_timeout_seconds: Optional[int] = None,
) -> types.ModelContainerSpec:
    """Constructs a ServingContainerSpec from the proto."""
    env = None
    ports = None
    grpc_ports = None
    deployment_timeout = (
        duration_pb2.Duration(seconds=serving_container_deployment_timeout)
        if serving_container_deployment_timeout
        else None
    )
    startup_probe = None
    health_probe = None

    if serving_container_environment_variables:
        env = [
            types.EnvVar(name=str(key), value=str(value))
            for key, value in serving_container_environment_variables.items()
        ]
    if serving_container_ports:
        ports = [types.Port(container_port=port) for port in serving_container_ports]
    if serving_container_grpc_ports:
        grpc_ports = [
            types.Port(container_port=port) for port in serving_container_grpc_ports
        ]
    if (
        serving_container_startup_probe_exec
        or serving_container_startup_probe_period_seconds
        or serving_container_startup_probe_timeout_seconds
    ):
        startup_probe_exec = None
        if serving_container_startup_probe_exec:
            startup_probe_exec = types.Probe.ExecAction(
                command=serving_container_startup_probe_exec
            )
        startup_probe = types.Probe(
            exec=startup_probe_exec,
            period_seconds=serving_container_startup_probe_period_seconds,
            timeout_seconds=serving_container_startup_probe_timeout_seconds,
        )
    if (
        serving_container_health_probe_exec
        or serving_container_health_probe_period_seconds
        or serving_container_health_probe_timeout_seconds
    ):
        health_probe_exec = None
        if serving_container_health_probe_exec:
            health_probe_exec = types.Probe.ExecAction(
                command=serving_container_health_probe_exec
            )
        health_probe = types.Probe(
            exec=health_probe_exec,
            period_seconds=serving_container_health_probe_period_seconds,
            timeout_seconds=serving_container_health_probe_timeout_seconds,
        )

    return types.ModelContainerSpec(
        image_uri=serving_container_image_uri,
        command=serving_container_command,
        args=serving_container_args,
        env=env,
        ports=ports,
        grpc_ports=grpc_ports,
        predict_route=serving_container_predict_route,
        health_route=serving_container_health_route,
        deployment_timeout=deployment_timeout,
        shared_memory_size_mb=serving_container_shared_memory_size_mb,
        startup_probe=startup_probe,
        health_probe=health_probe,
    )


class _ModelGardenClientWithOverride(utils.ClientWithOverride):
    _is_temporary = True
    _default_version = _DEFAULT_VERSION
    _version_map = (
        (
            _DEFAULT_VERSION,
            model_garden_service.ModelGardenServiceClient,
        ),
    )


class _ModelServiceClientWithOverride(utils.ClientWithOverride):
    _is_temporary = True
    _default_version = _DEFAULT_VERSION
    _version_map = (
        (
            _DEFAULT_VERSION,
            model_service.ModelServiceClient,
        ),
    )


class OpenModel:
    """Represents a Model Garden Open model.

    Attributes:
        model_name: Model Garden model resource name in the format of
          `publishers/{publisher}/models/{model}@{version}`, or a simplified
          resource name in the format of `{publisher}/{model}@{version}`, or a
          Hugging Face model ID in the format of `{organization}/{model}`.
    """

    __module__ = "vertexai.model_garden"

    def __init__(
        self,
        model_name: str,
    ):
        r"""Initializes a Model Garden model.

        Usage:

            ```
            model = OpenModel("publishers/google/models/gemma2@gemma-2-2b-it")
            ```

        Args:
            model_name: Model Garden model resource name in the format of
              `publishers/{publisher}/models/{model}@{version}`, or a simplified
              resource name in the format of `{publisher}/{model}@{version}`, or a
              Hugging Face model ID in the format of `{organization}/{model}`.
        """
        project = initializer.global_config.project
        location = initializer.global_config.location
        credentials = initializer.global_config.credentials

        self._model_name = model_name
        self._is_hugging_face_model = _is_hugging_face_model(model_name)
        self._publisher_model_name = _reconcile_model_name(model_name)
        self._project = project
        self._location = location
        self._credentials = credentials

    @functools.cached_property
    def _model_garden_client(
        self,
    ) -> model_garden_service.ModelGardenServiceClient:
        """Returns the Model Garden client."""
        return initializer.global_config.create_client(
            client_class=_ModelGardenClientWithOverride,
            credentials=self._credentials,
            location_override=self._location,
        )

    @functools.cached_property
    def _us_central1_model_garden_client(
        self,
    ) -> model_garden_service.ModelGardenServiceClient:
        """Returns the Model Garden client in us-central1."""
        return initializer.global_config.create_client(
            client_class=_ModelGardenClientWithOverride,
            credentials=self._credentials,
            location_override="us-central1",
        )

    def export(
        self,
        target_gcs_path: str = "",
        export_request_timeout: Optional[float] = None,
    ) -> str:
        """Exports an Open Model to a google cloud storage bucket.

        Args:
            target_gcs_path: target gcs path.
            export_request_timeout: The timeout for the deploy request. Default is 2
              hours.

        Returns:
            str: the target gcs bucket where the model weights are downloaded to


        Raises:
            ValueError: If ``target_gcs_path`` is not specified
        """
        if not target_gcs_path:
            raise ValueError("target_gcs_path is required.")

        request = types.ExportPublisherModelRequest(
            parent=f"projects/{self._project}/locations/{self._location}",
            name=self._publisher_model_name,
            destination=types.GcsDestination(output_uri_prefix=target_gcs_path),
        )
        request_headers = [
            ("x-goog-user-project", "{}".format(initializer.global_config.project)),
        ]

        _LOGGER.info(f"Exporting model weights: {self._model_name}")

        operation_future = self._model_garden_client.export_publisher_model(
            request, metadata=request_headers
        )
        _LOGGER.info(f"LRO: {operation_future.operation.name}")

        _LOGGER.info(f"Start time: {datetime.datetime.now()}")
        export_publisher_model_response = operation_future.result(
            timeout=export_request_timeout or _DEFAULT_EXPORT_TIMEOUT
        )
        _LOGGER.info(f"End time: {datetime.datetime.now()}")
        _LOGGER.info(f"Response: {export_publisher_model_response}")

        return export_publisher_model_response.destination_uri

    def deploy(
        self,
        accept_eula: bool = False,
        hugging_face_access_token: Optional[str] = None,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        spot: bool = False,
        reservation_affinity_type: Optional[str] = None,
        reservation_affinity_key: Optional[str] = None,
        reservation_affinity_values: Optional[List[str]] = None,
        use_dedicated_endpoint: Optional[bool] = False,
        dedicated_endpoint_disabled: Optional[bool] = False,
        fast_tryout_enabled: Optional[bool] = False,
        system_labels: Optional[Dict[str, str]] = None,
        endpoint_display_name: Optional[str] = None,
        model_display_name: Optional[str] = None,
        deploy_request_timeout: Optional[float] = None,
        serving_container_spec: Optional[types.ModelContainerSpec] = None,
        serving_container_image_uri: Optional[str] = None,
        serving_container_predict_route: Optional[str] = None,
        serving_container_health_route: Optional[str] = None,
        serving_container_command: Optional[Sequence[str]] = None,
        serving_container_args: Optional[Sequence[str]] = None,
        serving_container_environment_variables: Optional[Dict[str, str]] = None,
        serving_container_ports: Optional[Sequence[int]] = None,
        serving_container_grpc_ports: Optional[Sequence[int]] = None,
        serving_container_deployment_timeout: Optional[int] = None,
        serving_container_shared_memory_size_mb: Optional[int] = None,
        serving_container_startup_probe_exec: Optional[Sequence[str]] = None,
        serving_container_startup_probe_period_seconds: Optional[int] = None,
        serving_container_startup_probe_timeout_seconds: Optional[int] = None,
        serving_container_health_probe_exec: Optional[Sequence[str]] = None,
        serving_container_health_probe_period_seconds: Optional[int] = None,
        serving_container_health_probe_timeout_seconds: Optional[int] = None,
    ) -> aiplatform.Endpoint:
        """Deploys an Open Model to an endpoint.

        Args:
            accept_eula (bool): Whether to accept the End User License Agreement.
            hugging_face_access_token (str): The access token to access Hugging Face
              models. Reference: https://huggingface.co/docs/hub/en/security-tokens
            machine_type (str): Optional. The type of machine. Not specifying
              machine type will result in model to be deployed with automatic
              resources.
            min_replica_count (int): Optional. The minimum number of machine
              replicas this deployed model will be always deployed on. If traffic
              against it increases, it may dynamically be deployed onto more
              replicas, and as traffic decreases, some of these extra replicas may
              be freed.
            max_replica_count (int): Optional. The maximum number of replicas this
              deployed model may be deployed on when the traffic against it
              increases. If requested value is too large, the deployment will error,
              but if deployment succeeds then the ability to scale the model to that
              many replicas is guaranteed (barring service outages). If traffic
              against the deployed model increases beyond what its replicas at
              maximum may handle, a portion of the traffic will be dropped. If this
              value is not provided, the larger value of min_replica_count or 1 will
              be used. If value provided is smaller than min_replica_count, it will
              automatically be increased to be min_replica_count.
            accelerator_type (str): Optional. Hardware accelerator type. Must also
              set accelerator_count if used. One of ACCELERATOR_TYPE_UNSPECIFIED,
              NVIDIA_TESLA_K80, NVIDIA_TESLA_P100, NVIDIA_TESLA_V100,
              NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int): Optional. The number of accelerators to attach
              to a worker replica.
            spot (bool): Optional. Whether to schedule the deployment workload on
              spot VMs.
            reservation_affinity_type (str): Optional. The type of reservation
              affinity. One of NO_RESERVATION, ANY_RESERVATION,
              SPECIFIC_RESERVATION, SPECIFIC_THEN_ANY_RESERVATION,
              SPECIFIC_THEN_NO_RESERVATION
            reservation_affinity_key (str): Optional. Corresponds to the label key
              of a reservation resource. To target a SPECIFIC_RESERVATION by name,
              use `compute.googleapis.com/reservation-name` as the key and specify
              the name of your reservation as its value.
            reservation_affinity_values (List[str]): Optional. Corresponds to the
              label values of a reservation resource. This must be the full resource
              name of the reservation.
                Format:
                  'projects/{project_id_or_number}/zones/{zone}/reservations/{reservation_name}'
            use_dedicated_endpoint (bool): Optional. Default value is False. If set
              to True, the underlying prediction call will be made using the
              dedicated endpoint dns.
            dedicated_endpoint_disabled (bool): Optional. Default value is False. If set
              to False, the underlying prediction call will be made using the
              dedicated endpoint dns. Otherwise, the prediction call will be made
              using the shared endpoint dns.
            fast_tryout_enabled (bool): Optional. Defaults to False. If True, model
              will be deployed using faster deployment path. Useful for quick
              experiments. Not for production workloads. Only available for most
              popular models with certain machine types.
            system_labels (Dict[str, str]): Optional. System labels for Model Garden
              deployments. These labels are managed by Google and for tracking
              purposes only.
            endpoint_display_name: The display name of the created endpoint.
            model_display_name: The display name of the uploaded model.
            deploy_request_timeout: The timeout for the deploy request. Default is 2
              hours.
            serving_container_spec (types.ModelContainerSpec): Optional. The
              container specification for the model instance. This specification
              overrides the default container specification and other serving
              container parameters.
            serving_container_image_uri (str): Optional. The URI of the Model
              serving container. This parameter is required if the parameter
              `local_model` is not specified.
            serving_container_predict_route (str): Optional. An HTTP path to send
              prediction requests to the container, and which must be supported by
              it. If not specified a default HTTP path will be used by Vertex AI.
            serving_container_health_route (str): Optional. An HTTP path to send
              health check requests to the container, and which must be supported by
              it. If not specified a standard HTTP path will be used by Vertex AI.
            serving_container_command: Optional[Sequence[str]]=None, The command
              with which the container is run. Not executed within a shell. The
              Docker image's ENTRYPOINT is used if this is not provided. Variable
              references $(VAR_NAME) are expanded using the container's environment.
              If a variable cannot be resolved, the reference in the input string
              will be unchanged. The $(VAR_NAME) syntax can be escaped with a double
              $$, ie: $$(VAR_NAME). Escaped references will never be expanded,
              regardless of whether the variable exists or not.
            serving_container_args: Optional[Sequence[str]]=None, The arguments to
              the command. The Docker image's CMD is used if this is not provided.
              Variable references $(VAR_NAME) are expanded using the container's
              environment. If a variable cannot be resolved, the reference in the
              input string will be unchanged. The $(VAR_NAME) syntax can be escaped
              with a double $$, ie: $$(VAR_NAME). Escaped references will never be
              expanded, regardless of whether the variable exists or not.
            serving_container_environment_variables: Optional[Dict[str, str]]=None,
              The environment variables that are to be present in the container.
              Should be a dictionary where keys are environment variable names and
              values are environment variable values for those names.
            serving_container_ports: Optional[Sequence[int]]=None, Declaration of
              ports that are exposed by the container. This field is primarily
              informational, it gives Vertex AI information about the network
              connections the container uses. Listing or not a port here has no
              impact on whether the port is actually exposed, any port listening on
              the default "0.0.0.0" address inside a container will be accessible
              from the network.
            serving_container_grpc_ports: Optional[Sequence[int]]=None, Declaration
              of ports that are exposed by the container. Vertex AI sends gRPC
              prediction requests that it receives to the first port on this list.
              Vertex AI also sends liveness and health checks to this port. If you
              do not specify this field, gRPC requests to the container will be
              disabled. Vertex AI does not use ports other than the first one
              listed. This field corresponds to the `ports` field of the Kubernetes
              Containers v1 core API.
            serving_container_deployment_timeout (int): Optional. Deployment timeout
              in seconds.
            serving_container_shared_memory_size_mb (int): Optional. The amount of
              the VM memory to reserve as the shared memory for the model in
              megabytes.
            serving_container_startup_probe_exec (Sequence[str]): Optional. Exec
              specifies the action to take. Used by startup probe. An example of
              this argument would be ["cat", "/tmp/healthy"]
            serving_container_startup_probe_period_seconds (int): Optional. How
              often (in seconds) to perform the startup probe. Default to 10
              seconds. Minimum value is 1.
            serving_container_startup_probe_timeout_seconds (int): Optional. Number
              of seconds after which the startup probe times out. Defaults to 1
              second. Minimum value is 1.
            serving_container_health_probe_exec (Sequence[str]): Optional. Exec
              specifies the action to take. Used by health probe. An example of this
              argument would be ["cat", "/tmp/healthy"]
            serving_container_health_probe_period_seconds (int): Optional. How often
              (in seconds) to perform the health probe. Default to 10 seconds.
              Minimum value is 1.
            serving_container_health_probe_timeout_seconds (int): Optional. Number
              of seconds after which the health probe times out. Defaults to 1
              second. Minimum value is 1.

        Returns:
            endpoint (aiplatform.Endpoint):
                Created endpoint.

        Raises:
            ValueError: If ``serving_container_spec`` is specified but
            ``serving_container_spec.image_uri``
                is ``None``, or if ``serving_container_spec`` is specified but other
                serving container parameters are specified.
        """
        request = types.DeployRequest(
            destination=f"projects/{self._project}/locations/{self._location}",
        )
        if self._is_hugging_face_model:
            request.hugging_face_model_id = self._model_name.lower()
        else:
            request.publisher_model_name = self._publisher_model_name

        if endpoint_display_name:
            request.endpoint_config.endpoint_display_name = endpoint_display_name
        if model_display_name:
            request.model_config.model_display_name = model_display_name

        if accept_eula:
            request.model_config.accept_eula = accept_eula

        if hugging_face_access_token:
            request.model_config.hugging_face_access_token = hugging_face_access_token

        provided_custom_machine_spec = (
            machine_type or accelerator_type or accelerator_count
        )
        if provided_custom_machine_spec:
            dedicated_resources = types.DedicatedResources(
                machine_spec=types.MachineSpec(
                    machine_type=machine_type,
                    accelerator_type=accelerator_type,
                    accelerator_count=accelerator_count,
                ),
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )
            request.deploy_config.dedicated_resources = dedicated_resources
        if spot:
            request.deploy_config.dedicated_resources.spot = True

        if reservation_affinity_type:
            request.deploy_config.dedicated_resources.machine_spec.reservation_affinity.reservation_affinity_type = (
                reservation_affinity_type
            )
        if reservation_affinity_key and reservation_affinity_values:
            request.deploy_config.dedicated_resources.machine_spec.reservation_affinity.key = (
                reservation_affinity_key
            )
            request.deploy_config.dedicated_resources.machine_spec.reservation_affinity.values = (
                reservation_affinity_values
            )

        # TODO(b/417560875): Remove this once notebooks are migrated to use dedicated_endpoint_disabled.
        if use_dedicated_endpoint:
            request.endpoint_config.dedicated_endpoint_enabled = use_dedicated_endpoint

        if dedicated_endpoint_disabled:
            request.endpoint_config.dedicated_endpoint_disabled = (
                dedicated_endpoint_disabled
            )

        if fast_tryout_enabled:
            request.deploy_config.fast_tryout_enabled = fast_tryout_enabled

        if system_labels:
            request.deploy_config.system_labels = system_labels

        if serving_container_spec:
            if not serving_container_spec.image_uri:
                raise ValueError(
                    "Serving container image uri is required for the serving container"
                    " spec."
                )
            if serving_container_image_uri:
                raise ValueError(
                    "Serving container image uri is already set in the serving"
                    " container spec."
                )
            request.model_config.container_spec = serving_container_spec

        if serving_container_image_uri:
            request.model_config.container_spec = _construct_serving_container_spec(
                serving_container_image_uri,
                serving_container_predict_route,
                serving_container_health_route,
                serving_container_command,
                serving_container_args,
                serving_container_environment_variables,
                serving_container_ports,
                serving_container_grpc_ports,
                serving_container_deployment_timeout,
                serving_container_shared_memory_size_mb,
                serving_container_startup_probe_exec,
                serving_container_startup_probe_period_seconds,
                serving_container_startup_probe_timeout_seconds,
                serving_container_health_probe_exec,
                serving_container_health_probe_period_seconds,
                serving_container_health_probe_timeout_seconds,
            )

        _LOGGER.info(f"Deploying model: {self._model_name}")

        operation_future = self._model_garden_client.deploy(request)
        _LOGGER.info(f"LRO: {operation_future.operation.name}")

        _LOGGER.info(f"Start time: {datetime.datetime.now()}")
        deploy_response = operation_future.result(
            timeout=deploy_request_timeout or _DEFAULT_TIMEOUT
        )
        _LOGGER.info(f"End time: {datetime.datetime.now()}")

        self._endpoint_name = deploy_response.endpoint
        _LOGGER.info(f"Endpoint: {self._endpoint_name}")
        endpoint = aiplatform.Endpoint._construct_sdk_resource_from_gapic(
            aiplatform_models.gca_endpoint_compat.Endpoint(name=self._endpoint_name),
        )
        return endpoint

    def list_deploy_options(
        self,
        concise: bool = False,
        serving_container_image_uri_filter: Optional[Union[str, List[str]]] = None,
        machine_type_filter: Optional[str] = None,
        accelerator_type_filter: Optional[str] = None,
    ) -> Union[str, Sequence[types.PublisherModel.CallToAction.Deploy]]:
        """Lists the verified deploy options for the model.

        Args:
            concise: If true, returns a human-readable string with container and
              machine specs.
            serving_container_image_uri_filter: If specified, only return the
              deploy options where the serving container image URI contains one of
              the specified keyword(s) (e.g., "vllm" or ["vllm", "tgi"]). The
              filter is case-insensitive.
            machine_type_filter: If specified, only return the deploy options
              where the machine type contains one of the specified keyword(s)
              (e.g., "n1" or ["n1", "g2"]). The filter is case-insensitive.
            accelerator_type_filter: If specified, only return the deploy options
              where the accelerator type contains one of the specified keyword(s)
              (e.g., "T4" or ["T4", "L4"]). The filter is case-insensitive.

        Returns:
            A list of deploy options or a concise formatted string.
        """
        request = types.GetPublisherModelRequest(
            name=self._publisher_model_name,
            is_hugging_face_model=bool(self._is_hugging_face_model),
            include_equivalent_model_garden_model_deployment_configs=True,
        )
        response = self._us_central1_model_garden_client.get_publisher_model(request)
        deploy_options = (
            response.supported_actions.multi_deploy_vertex.multi_deploy_vertex
        )

        if not deploy_options:
            raise ValueError(
                "Model does not support deployment. "
                "Use `list_deployable_models()` to find supported models."
            )

        if serving_container_image_uri_filter:
            if isinstance(serving_container_image_uri_filter, str):
                serving_container_image_uri_filter = [
                    serving_container_image_uri_filter
                ]
            serving_container_image_uri_filter = [
                f.lower() for f in serving_container_image_uri_filter
            ]
            deploy_options = [
                option
                for option in deploy_options
                if option.container_spec
                and any(
                    f in option.container_spec.image_uri.lower()
                    for f in serving_container_image_uri_filter
                )
            ]

        if machine_type_filter:
            filters = (
                [machine_type_filter]
                if isinstance(machine_type_filter, str)
                else machine_type_filter
            )
            deploy_options = [
                option
                for option in deploy_options
                if option.dedicated_resources
                and option.dedicated_resources.machine_spec
                and any(
                    f.lower()
                    in option.dedicated_resources.machine_spec.machine_type.lower()
                    for f in filters
                )
            ]

        if accelerator_type_filter:
            filters = (
                [accelerator_type_filter]
                if isinstance(accelerator_type_filter, str)
                else accelerator_type_filter
            )
            deploy_options = [
                option
                for option in deploy_options
                if option.dedicated_resources
                and option.dedicated_resources.machine_spec
                and option.dedicated_resources.machine_spec.accelerator_type
                and any(
                    f.lower()
                    in option.dedicated_resources.machine_spec.accelerator_type.name.lower()
                    for f in filters
                )
            ]

        if not deploy_options:
            raise ValueError("No deploy options found.")

        if not concise:
            return deploy_options

        def _extract_config(option):
            container = (
                option.container_spec.image_uri if option.container_spec else None
            )
            machine = (
                option.dedicated_resources.machine_spec
                if option.dedicated_resources
                else None
            )
            option_name = getattr(option, "deploy_task_name", None)

            return {
                "option_name": option_name,
                "serving_container_image_uri": container,
                "machine_type": getattr(machine, "machine_type", None),
                "accelerator_type": getattr(
                    getattr(machine, "accelerator_type", None), "name", None
                ),
                "accelerator_count": getattr(machine, "accelerator_count", None),
            }

        concise_deploy_options = [_extract_config(opt) for opt in deploy_options]
        return "\n\n".join(
            (
                f"[Option {i + 1}: {config['option_name']}]\n"
                if config.get("option_name")
                else f"[Option {i + 1}]\n"
            )
            + "\n".join(
                f'    {k}="{v}",' if k != "accelerator_count" else f"    {k}={v},"
                for k, v in config.items()
                if v is not None and k != "option_name"
            )
            for i, config in enumerate(concise_deploy_options)
        )

    def batch_predict(
        self,
        input_dataset: Union[str, List[str]],
        *,
        output_uri_prefix: Optional[str] = None,
        job_display_name: Optional[str] = None,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        starting_replica_count: Optional[int] = None,
        max_replica_count: Optional[int] = None,
    ) -> batch_prediction.BatchPredictionJob:
        """Perform batch prediction on the model.

        Args:
            input_dataset (Union[str, List[str]]): GCS URI(-s) or BigQuery URI to
              your input data to run batch prediction on. Example:
              "gs://path/to/input/data.jsonl" or
              "bq://projectId.bqDatasetId.bqTableId"
            output_uri_prefix (Optional[str]): GCS or BigQuery URI prefix for the
              output predictions. Example: "gs://path/to/output/data" or
              "bq://projectId.bqDatasetId" If not specified,
              f"{STAGING_BUCKET}/gen-ai-batch-prediction" will be used for GCS
              source and
              f"bq://projectId.gen_ai_batch_prediction.predictions_{TIMESTAMP}" will
              be used for BigQuery source.
            job_display_name (Optional[str]): The user-defined name of the
              BatchPredictionJob. The name can be up to 128 characters long and can
              be consist of any UTF-8 characters.
            machine_type (Optional[str]): The machine type for the batch prediction
              job.
            accelerator_type (Optional[str]): The accelerator type for the batch
              prediction job.
            accelerator_count (Optional[int]): The accelerator count for the batch
              prediction job.
            starting_replica_count (Optional[int]): The starting replica count for
              the batch prediction job.
            max_replica_count (Optional[int]): The maximum replica count for the
              batch prediction job.

        Returns:
            batch_prediction.BatchPredictionJob:
                The batch prediction job.
        """
        return batch_prediction.BatchPredictionJob.submit(
            source_model=self._publisher_model_name,
            input_dataset=input_dataset,
            output_uri_prefix=output_uri_prefix,
            job_display_name=job_display_name,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            starting_replica_count=starting_replica_count,
            max_replica_count=max_replica_count,
        )

    def check_license_agreement_status(self) -> bool:
        """Check whether the project has accepted the license agreement of the model.

        EULA (End User License Agreement) is a legal document that the user must
        accept before using the model. For Models having license restrictions,
        the user must accept the EULA before using the model. You can check the
        details of the License in Model Garden.

        Returns:
            bool : True if the project has accepted the End User License
            Agreement, False otherwise.
        """
        request = types.CheckPublisherModelEulaAcceptanceRequest(
            parent=f"projects/{self._project}",
            publisher_model=self._publisher_model_name,
        )
        response = self._model_garden_client.check_publisher_model_eula_acceptance(
            request
        )
        return response.publisher_model_eula_acked

    def accept_model_license_agreement(
        self,
    ) -> types.model_garden_service.PublisherModelEulaAcceptance:
        """Accepts the EULA(End User License Agreement) of the model for the project.

        For Models having license restrictions, the user must accept the EULA
        before using the model. Calling this method will mark the EULA as accepted
        for the project.

        Returns:
            types.model_garden_service.PublisherModelEulaAcceptance:
                The response of the accept_eula call, containing project number,
                model name and acceptance status.
        """
        request = types.AcceptPublisherModelEulaRequest(
            parent=f"projects/{self._project}",
            publisher_model=self._publisher_model_name,
        )
        return self._model_garden_client.accept_publisher_model_eula(request)


class CustomModel:
    """Represents a Model Garden Custom model."""

    def __init__(
        self,
        gcs_uri: Optional[str] = None,
    ):
        r"""Initializes a Model Garden Custom model.

        Usage:

            ```
            model = vertexai.CustomModel(
                gcs_uri = 'gs://tuning-job-output/node-0/checkpoints/final')
            ```

        Args:
            gcs_uri: The GCS URI of the custom model, storing weights and config
              files
        """
        if not gcs_uri:
            raise ValueError("gcs_uri must be specified.")

        project = initializer.global_config.project
        location = initializer.global_config.location
        credentials = initializer.global_config.credentials

        self._gcs_uri = gcs_uri
        self._project = project
        self._location = location
        self._credentials = credentials

    @functools.cached_property
    def _model_garden_client(
        self,
    ) -> model_garden_service.ModelGardenServiceClient:
        """Returns the Model Garden client."""
        return initializer.global_config.create_client(
            client_class=_ModelGardenClientWithOverride,
            credentials=self._credentials,
            location_override=self._location,
        )

    @functools.cached_property
    def _model_service_client(
        self,
    ) -> model_service.ModelServiceClient:
        """Returns the Model Service client."""
        return initializer.global_config.create_client(
            client_class=_ModelServiceClientWithOverride,
            credentials=self._credentials,
            location_override=self._location,
        )

    def list_deploy_options(
        self,
        available_machines: bool = True,
        filter_by_user_quota: bool = True,
        request_timeout: Optional[float] = None,
    ) -> str:
        """Lists the deploy options for the model.

        Args:
            available_machines: If true, only return the deploy options for
              available machines.
            filter_by_user_quota: If true, only return the deploy options for
              machines that the user has quota for.
            request_timeout: The timeout for the recommend spec request.
              Default is 60 seconds.

        Returns:
            str: A string of the deploy options represented by
                machine spec and container spec.

        """

        def _extract_spec(spec):
            machine_spec = spec.machine_spec
            return {
                "machine_type": getattr(machine_spec, "machine_type", None),
                "accelerator_type": getattr(
                    getattr(machine_spec, "accelerator_type", None), "name", None
                ),
                "accelerator_count": getattr(machine_spec, "accelerator_count", None),
            }

        def _extract_recommendation(recommendation):
            extracted_spec = _extract_spec(recommendation.spec)
            extracted_spec["region"] = getattr(recommendation, "region", None)
            if (
                recommendation.user_quota_state
                and recommendation.user_quota_state
                != types.RecommendSpecResponse.Recommendation.QuotaState.QUOTA_STATE_UNSPECIFIED
            ):
                extracted_spec["user_quota_state"] = getattr(
                    getattr(recommendation, "user_quota_state", None), "name", None
                )
            return extracted_spec

        request = types.RecommendSpecRequest(
            gcs_uri=self._gcs_uri,
            parent=f"projects/{self._project}/locations/{self._location}",
            check_machine_availability=available_machines,
            check_user_quota=filter_by_user_quota,
        )
        try:
            response = self._model_service_client.recommend_spec(
                request, timeout=request_timeout or _DEFAULT_RECOMMEND_SPEC_TIMEOUT
            )
            options = []
            if response.recommendations:
                options = [
                    _extract_recommendation(recommendation)
                    for recommendation in response.recommendations
                    if recommendation.spec
                ]
                if filter_by_user_quota:
                    options = [
                        option
                        for option in options
                        if option.get("user_quota_state")
                        == "QUOTA_STATE_USER_HAS_QUOTA"
                    ]
            elif response.specs:
                options = [_extract_spec(spec) for spec in response.specs if spec]
            return "\n\n".join(
                f"[Option {i + 1}]\n"
                + ",\n".join(
                    f'    {k}="{v}"' if k != "accelerator_count" else f"    {k}={v}"
                    for k, v in config.items()
                    if v is not None
                )
                for i, config in enumerate(options)
            )

        except Exception as e:
            _LOGGER.error(f"Failed to list deploy options: {e}")
            raise e

    def deploy(
        self,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        reservation_affinity_type: Optional[str] = None,
        reservation_affinity_key: Optional[str] = None,
        reservation_affinity_values: Optional[List[str]] = None,
        endpoint_display_name: Optional[str] = None,
        model_display_name: Optional[str] = None,
        enable_private_service_connect: bool = False,
        psc_project_allow_list: Optional[List[str]] = None,
        deploy_request_timeout: Optional[float] = None,
    ) -> aiplatform.Endpoint:
        """Deploys a Custom Model to an endpoint.

        Args:
            machine_type (str): Optional. The type of machine. Not specifying
              machine type will result in model to be deployed with automatic
              resources.
            min_replica_count (int): Optional. The minimum number of machine
              replicas this deployed model will be always deployed on. If traffic
              against it increases, it may dynamically be deployed onto more
              replicas, and as traffic decreases, some of these extra replicas may
              be freed.
            max_replica_count (int): Optional. The maximum number of replicas this
              deployed model may be deployed on when the traffic against it
              increases. If requested value is too large, the deployment will error,
              but if deployment succeeds then the ability to scale the model to that
              many replicas is guaranteed (barring service outages). If traffic
              against the deployed model increases beyond what its replicas at
              maximum may handle, a portion of the traffic will be dropped. If this
              value is not provided, the larger value of min_replica_count or 1 will
              be used. If value provided is smaller than min_replica_count, it will
              automatically be increased to be min_replica_count.
            accelerator_type (str): Optional. Hardware accelerator type. Must also
              set accelerator_count if used.
            accelerator_count (int): Optional. The number of accelerators to attach
              to a worker replica.
            reservation_affinity_type (str): Optional. The type of reservation
              affinity. One of NO_RESERVATION, ANY_RESERVATION,
              SPECIFIC_RESERVATION, SPECIFIC_THEN_ANY_RESERVATION,
              SPECIFIC_THEN_NO_RESERVATION
            reservation_affinity_key (str): Optional. Corresponds to the label key
              of a reservation resource. To target a SPECIFIC_RESERVATION by name,
              use `compute.googleapis.com/reservation-name` as the key and specify
              the name of your reservation as its value.
            reservation_affinity_values (List[str]): Optional. Corresponds to the
              label values of a reservation resource. This must be the full resource
              name of the reservation.
                Format:
                  'projects/{project_id_or_number}/zones/{zone}/reservations/{reservation_name}'
            endpoint_display_name: The display name of the created endpoint.
            model_display_name: The display name of the custom model.
            enable_private_service_connect (bool): Whether to enable private service
              connect.
            psc_project_allow_list (List[str]): The list of projects that are allowed to
              access the endpoint over private service connect.
            deploy_request_timeout: The timeout for the deploy request. Default is 2
              hours.

        Returns:
            endpoint (aiplatform.Endpoint):
                Created endpoint.
        """
        return self._deploy_gcs_uri(
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            reservation_affinity_type=reservation_affinity_type,
            reservation_affinity_key=reservation_affinity_key,
            reservation_affinity_values=reservation_affinity_values,
            endpoint_display_name=endpoint_display_name,
            model_display_name=model_display_name,
            enable_private_service_connect=enable_private_service_connect,
            psc_project_allow_list=psc_project_allow_list,
            deploy_request_timeout=deploy_request_timeout,
        )

    def _deploy_model_registry_model(self) -> aiplatform.Endpoint:
        """Deploys a Model Registry model to an endpoint."""
        raise NotImplementedError(
            "Not implemented yet. Please provide gcs_uri in CustomModel constructor."
        )

    def _deploy_gcs_uri(
        self,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        enable_private_service_connect: bool = False,
        psc_project_allow_list: Optional[List[str]] = None,
        reservation_affinity_type: Optional[str] = None,
        reservation_affinity_key: Optional[str] = None,
        reservation_affinity_values: Optional[List[str]] = None,
        endpoint_display_name: Optional[str] = None,
        model_display_name: Optional[str] = None,
        deploy_request_timeout: Optional[float] = None,
    ) -> aiplatform.Endpoint:
        """Deploys a Custom Model to an endpoint.

        Args:
            machine_type (str): Optional. The type of machine. Not specifying
              machine type will result in model to be deployed with automatic
              resources.
            min_replica_count (int): Optional. The minimum number of machine
              replicas this deployed model will be always deployed on. If traffic
              against it increases, it may dynamically be deployed onto more
              replicas, and as traffic decreases, some of these extra replicas may
              be freed.
            max_replica_count (int): Optional. The maximum number of replicas this
              deployed model may be deployed on when the traffic against it
              increases. If requested value is too large, the deployment will error,
              but if deployment succeeds then the ability to scale the model to that
              many replicas is guaranteed (barring service outages). If traffic
              against the deployed model increases beyond what its replicas at
              maximum may handle, a portion of the traffic will be dropped. If this
              value is not provided, the larger value of min_replica_count or 1 will
              be used. If value provided is smaller than min_replica_count, it will
              automatically be increased to be min_replica_count.
            accelerator_type (str): Optional. Hardware accelerator type. Must also
              set accelerator_count if used. One of ACCELERATOR_TYPE_UNSPECIFIED,
              NVIDIA_TESLA_K80, NVIDIA_TESLA_P100, NVIDIA_TESLA_V100,
              NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int): Optional. The number of accelerators to attach
              to a worker replica.
            reservation_affinity_type (str): Optional. The type of reservation
              affinity. One of NO_RESERVATION, ANY_RESERVATION,
              SPECIFIC_RESERVATION, SPECIFIC_THEN_ANY_RESERVATION,
              SPECIFIC_THEN_NO_RESERVATION
            reservation_affinity_key (str): Optional. Corresponds to the label key
              of a reservation resource. To target a SPECIFIC_RESERVATION by name,
              use `compute.googleapis.com/reservation-name` as the key and specify
              the name of your reservation as its value.
            reservation_affinity_values (List[str]): Optional. Corresponds to the
              label values of a reservation resource. This must be the full resource
              name of the reservation.
                Format:
                  'projects/{project_id_or_number}/zones/{zone}/reservations/{reservation_name}'
            endpoint_display_name: The display name of the created endpoint.
            model_display_name: The display name of the custom model.
            enable_private_service_connect (bool): Whether to enable private service
              connect.
            psc_project_allow_list (List[str]): The list of projects that are allowed to
              access the endpoint over private service connect.
            deploy_request_timeout: The timeout for the deploy request. Default is 2
              hours.

        Returns:
            endpoint (aiplatform.Endpoint):
                Created endpoint.
        """

        # Validation on machine type, accelerator type and count.
        # Return true if all three of them have value or are None.
        def has_all_or_none_values(var1, var2, var3) -> bool:
            return (var1 and var2 and var3) or (not var1 and not var2 and not var3)

        if not has_all_or_none_values(
            machine_type, accelerator_type, accelerator_count
        ):
            raise ValueError(
                "machine_type, accelerator_type and accelerator_count must all be provided or not provided."
            )

        request = types.DeployRequest(
            destination=f"projects/{self._project}/locations/{self._location}",
        )
        request.custom_model = types.DeployRequest.CustomModel(gcs_uri=self._gcs_uri)
        if endpoint_display_name:
            request.endpoint_config.endpoint_display_name = endpoint_display_name
        if model_display_name:
            request.model_config.model_display_name = model_display_name

        if enable_private_service_connect and psc_project_allow_list:
            request.endpoint_config.private_service_connect_config = (
                types.PrivateServiceConnectConfig(
                    enable_private_service_connect=enable_private_service_connect,
                    project_allowlist=psc_project_allow_list,
                )
            )

        if machine_type and accelerator_type and accelerator_count:
            request.deploy_config.dedicated_resources = types.DedicatedResources(
                machine_spec=types.MachineSpec(
                    machine_type=machine_type,
                    accelerator_type=accelerator_type,
                    accelerator_count=accelerator_count,
                )
            )
        if min_replica_count:
            request.deploy_config.dedicated_resources.min_replica_count = (
                min_replica_count
            )
        if max_replica_count:
            request.deploy_config.dedicated_resources.max_replica_count = (
                max_replica_count
            )

        if reservation_affinity_type:
            request.deploy_config.dedicated_resources.machine_spec.reservation_affinity.reservation_affinity_type = (
                reservation_affinity_type
            )
        if reservation_affinity_key and reservation_affinity_values:
            request.deploy_config.dedicated_resources.machine_spec.reservation_affinity.key = (
                reservation_affinity_key
            )
            request.deploy_config.dedicated_resources.machine_spec.reservation_affinity.values = (
                reservation_affinity_values
            )

        _LOGGER.info(f"Deploying custom model: {self._gcs_uri}")

        try:
            operation_future = self._model_garden_client.deploy(request)
            _LOGGER.info(f"LRO: {operation_future.operation.name}")
            deploy_response = operation_future.result(
                timeout=deploy_request_timeout or _DEFAULT_TIMEOUT
            )
            _LOGGER.info(f"End time: {datetime.datetime.now()}")
            self._endpoint_name = deploy_response.endpoint
            _LOGGER.info(f"Endpoint: {self._endpoint_name}")
            endpoint = aiplatform.Endpoint._construct_sdk_resource_from_gapic(
                aiplatform_models.gca_endpoint_compat.Endpoint(
                    name=self._endpoint_name
                ),
            )
            return endpoint
        except ValueError as e:
            _LOGGER.error(f"Failed to deploy custom model: {e}")
            raise e


class PartnerModel:
    """Represents a Model Garden Partner model."""

    def __init__(
        self,
        model_name: str,
    ):
        r"""Initializes a Model Garden partner model.

        Usage:

            ```
            model = PartnerModel("publishers/ai21/models/jamba-large-1.6@001")
            ```

        Args:
            model_name: Model Garden model resource name in the format of
              `publishers/{publisher}/models/{model}@{version}`, or a simplified
              resource name in the format of `{publisher}/{model}@{version}`.
        """
        project = initializer.global_config.project
        location = initializer.global_config.location
        credentials = initializer.global_config.credentials

        self._model_name = model_name
        self._publisher_model_name = _reconcile_model_name(model_name)
        self._project = project
        self._location = location
        self._credentials = credentials

    @functools.cached_property
    def _model_garden_client(
        self,
    ) -> model_garden_service.ModelGardenServiceClient:
        """Returns the Model Garden client."""
        return initializer.global_config.create_client(
            client_class=_ModelGardenClientWithOverride,
            credentials=self._credentials,
            location_override=self._location,
        )

    def deploy(
        self,
        machine_type: Optional[str] = None,
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None,
        endpoint_display_name: Optional[str] = None,
        model_display_name: Optional[str] = None,
        deploy_request_timeout: Optional[float] = None,
    ) -> aiplatform.Endpoint:
        """Deploys an Open Model to an endpoint.

        Args:
            machine_type (str): Optional. The type of machine. Not specifying
              machine type will result in model to be deployed with automatic
              resources.
            min_replica_count (int): Optional. The minimum number of machine
              replicas this deployed model will be always deployed on. If traffic
              against it increases, it may dynamically be deployed onto more
              replicas, and as traffic decreases, some of these extra replicas may
              be freed.
            max_replica_count (int): Optional. The maximum number of replicas this
              deployed model may be deployed on when the traffic against it
              increases. If requested value is too large, the deployment will error,
              but if deployment succeeds then the ability to scale the model to that
              many replicas is guaranteed (barring service outages). If traffic
              against the deployed model increases beyond what its replicas at
              maximum may handle, a portion of the traffic will be dropped. If this
              value is not provided, the larger value of min_replica_count or 1 will
              be used. If value provided is smaller than min_replica_count, it will
              automatically be increased to be min_replica_count.
            accelerator_type (str): Optional. Hardware accelerator type. Must also
              set accelerator_count if used. One of ACCELERATOR_TYPE_UNSPECIFIED,
              NVIDIA_TESLA_K80, NVIDIA_TESLA_P100, NVIDIA_TESLA_V100,
              NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
            accelerator_count (int): Optional. The number of accelerators to attach
              to a worker replica.
            endpoint_display_name: The display name of the created endpoint.
            model_display_name: The display name of the uploaded model.
            deploy_request_timeout: The timeout for the deploy request. Default is 2
              hours.

        Returns:
            endpoint (aiplatform.Endpoint):
                Created endpoint.
        """
        request = types.DeployRequest(
            destination=f"projects/{self._project}/locations/{self._location}",
        )
        request.publisher_model_name = self._publisher_model_name

        if endpoint_display_name:
            request.endpoint_config.endpoint_display_name = endpoint_display_name
        if model_display_name:
            request.model_config.model_display_name = model_display_name

        provided_custom_machine_spec = (
            machine_type or accelerator_type or accelerator_count
        )
        if provided_custom_machine_spec:
            dedicated_resources = types.DedicatedResources(
                machine_spec=types.MachineSpec(
                    machine_type=machine_type,
                    accelerator_type=accelerator_type,
                    accelerator_count=accelerator_count,
                ),
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )
            request.deploy_config.dedicated_resources = dedicated_resources

        _LOGGER.info(f"Deploying model: {self._model_name}")

        operation_future = self._model_garden_client.deploy(request)
        _LOGGER.info(f"LRO: {operation_future.operation.name}")

        _LOGGER.info(f"Start time: {datetime.datetime.now()}")
        deploy_response = operation_future.result(
            timeout=deploy_request_timeout or _DEFAULT_TIMEOUT
        )
        _LOGGER.info(f"End time: {datetime.datetime.now()}")

        self._endpoint_name = deploy_response.endpoint
        _LOGGER.info(f"Endpoint: {self._endpoint_name}")
        endpoint = aiplatform.Endpoint._construct_sdk_resource_from_gapic(
            aiplatform_models.gca_endpoint_compat.Endpoint(name=self._endpoint_name),
        )
        return endpoint


class Model:
    """Represents a Model Garden model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
    ):
        r"""Initializes a Model Garden model.

        Usage:

            ```
            model = Model("publishers/google/models/gemma3@gemma-3-27b-it")
            model = Model("google/gemma3@gemma-3-27b-it")
            model = Model("deepseek-ai/DeepSeek-V3-0324")
            model = Model("gs://fine-tuning-output/node-0/checkpoints/final")
            model = Model("projects/123/locations/us-central1/models/456")
            ```

        Args:
            model_name: Name of the model artifact.

            It can be:
            1. A pretrained model
              1.1 Model Garden model resource name in the format of
              `publishers/{publisher}/models/{model}@{version}`, or
              1.2 a simplified resource name in the format of `{publisher}/{model}@{version}`, or
              1.3 a Hugging Face model ID in the format of `{organization}/{model}`.
            2. A custom model weights like gs://fine-tuning-output/node-0/checkpoints/final
            3. A Model Registry model like projects/123/locations/us-central1/models/456
        """
        if not model_name:
            raise ValueError("model_name must be specified.")

        if re.match(r"^gs://", model_name):
            self._model = CustomModel(gcs_uri=model_name)
        elif re.match(r"^projects/.*/locations/.*/models/.*", model_name):
            raise NotImplementedError("Model Registry models are not supported yet.")
        else:
            self._model = OpenModel(model_name)

    def deploy(
        self,
        **kwargs,
    ) -> aiplatform.Endpoint:
        """Deploys the model to an endpoint."""
        return self._model.deploy(**kwargs)
