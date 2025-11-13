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


from concurrent import futures
import enum
import functools
import inspect
import logging
import os
import types
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from google.api_core import client_options
from google.api_core import gapic_v1
import google.auth
from google.auth import credentials as auth_credentials
from google.auth.exceptions import GoogleAuthError

from google.cloud.aiplatform import __version__
from google.cloud.aiplatform import compat
from google.cloud.aiplatform.constants import base as constants
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.metadata import metadata
from google.cloud.aiplatform.utils import resource_manager_utils
from google.cloud.aiplatform.tensorboard import tensorboard_resource
from google.cloud.aiplatform import telemetry

from google.cloud.aiplatform.compat.types import (
    encryption_spec as gca_encryption_spec_compat,
    encryption_spec_v1 as gca_encryption_spec_v1,
    encryption_spec_v1beta1 as gca_encryption_spec_v1beta1,
)

try:
    import google.auth.aio

    AsyncCredentials = google.auth.aio.credentials.Credentials
    _HAS_ASYNC_CRED_DEPS = True
except (ImportError, AttributeError):
    AsyncCredentials = Any
    _HAS_ASYNC_CRED_DEPS = False

_TVertexAiServiceClientWithOverride = TypeVar(
    "_TVertexAiServiceClientWithOverride",
    bound=utils.VertexAiServiceClientWithOverride,
)

_TOP_GOOGLE_CONSTRUCTOR_METHOD_TAG = "top_google_constructor_method"


class _Product(enum.Enum):
    """Notebook product types."""

    WORKBENCH_INSTANCE = "WORKBENCH_INSTANCE"
    COLAB_ENTERPRISE = "COLAB_ENTERPRISE"
    WORKBENCH_CUSTOM_CONTAINER = "WORKBENCH_CUSTOM_CONTAINER"


class _Config:
    """Stores common parameters and options for API calls."""

    def _set_project_as_env_var_or_google_auth_default(self):
        """Tries to set the project from the environment variable or calls google.auth.default().

        Stores the returned project and credentials as instance attributes.

        This prevents google.auth.default() from being called multiple times when
        the project and credentials have already been set.
        """

        if not self._project and not self._api_key:
            # Project is not set. Trying to get it from the environment.
            # See https://github.com/googleapis/python-aiplatform/issues/852
            # See https://github.com/googleapis/google-auth-library-python/issues/924
            # TODO: Remove when google.auth.default() learns the
            # CLOUD_ML_PROJECT_ID env variable or Vertex AI starts setting GOOGLE_CLOUD_PROJECT env variable.
            project_number = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
                "CLOUD_ML_PROJECT_ID"
            )
            if project_number:
                if not self._credentials:
                    credentials, _ = google.auth.default(
                        scopes=constants.DEFAULT_AUTHED_SCOPES
                    )
                    self._credentials = credentials
                # Try to convert project number to project ID which is more readable.
                try:
                    project_id = resource_manager_utils.get_project_id(
                        project_number=project_number,
                        credentials=self._credentials,
                    )
                    self._project = project_id
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Failed to convert project number to project ID.", exc_info=True
                    )
                    self._project = project_number
            else:
                credentials, project = google.auth.default()
                self._credentials = self._credentials or credentials
                self._project = project

        if not self._credentials and not self._api_key:
            credentials, _ = google.auth.default(scopes=constants.DEFAULT_AUTHED_SCOPES)
            self._credentials = credentials

    def __init__(self):
        self._project = None
        self._location = None
        self._staging_bucket = None
        self._credentials = None
        self._encryption_spec_key_name = None
        self._network = None
        self._service_account = None
        self._api_endpoint = None
        self._api_key = None
        self._api_transport = None
        self._request_metadata = None
        self._resource_type = None
        self._async_rest_credentials = None

    def init(
        self,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        experiment: Optional[str] = None,
        experiment_description: Optional[str] = None,
        experiment_tensorboard: Optional[
            Union[str, tensorboard_resource.Tensorboard, bool]
        ] = None,
        staging_bucket: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
        network: Optional[str] = None,
        service_account: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_transport: Optional[str] = None,
        request_metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """Updates common initialization parameters with provided options.

        Args:
            project (str): The default project to use when making API calls.
            location (str): The default location to use when making API calls. If not
                set defaults to us-central-1.
            experiment (str): Optional. The experiment name.
            experiment_description (str): Optional. The description of the experiment.
            experiment_tensorboard (Union[str, tensorboard_resource.Tensorboard, bool]):
                Optional. The Vertex AI TensorBoard instance, Tensorboard resource name,
                or Tensorboard resource ID to use as a backing Tensorboard for the provided
                experiment.

                Example tensorboard resource name format:
                "projects/123/locations/us-central1/tensorboards/456"

                If `experiment_tensorboard` is provided and `experiment` is not,
                the provided `experiment_tensorboard` will be set as the global Tensorboard.
                Any subsequent calls to aiplatform.init() with `experiment` and without
                `experiment_tensorboard` will automatically assign the global Tensorboard
                to the `experiment`.

                If `experiment_tensorboard` is ommitted or set to `True` or `None` the global
                Tensorboard will be assigned to the `experiment`. If a global Tensorboard is
                not set, the default Tensorboard instance will be used, and created if it does not exist.

                To disable creating and using Tensorboard with `experiment`, set `experiment_tensorboard` to `False`.
                Any subsequent calls to aiplatform.init() should include this setting as well.
            staging_bucket (str): The default staging bucket to use to stage artifacts
                when making API calls. In the form gs://...
            credentials (google.auth.credentials.Credentials): The default custom
                credentials to use when making API calls. If not provided credentials
                will be ascertained from the environment.
            encryption_spec_key_name (Optional[str]):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect a resource. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If set, this resource and all sub-resources will be secured by this key.
            network (str):
                Optional. The full name of the Compute Engine network to which jobs
                and resources should be peered. E.g. "projects/12345/global/networks/myVPC".
                Private services access must already be configured for the network.
                If specified, all eligible jobs and resources created will be peered
                with this VPC.
            service_account (str):
                Optional. The service account used to launch jobs and deploy models.
                Jobs that use service_account: BatchPredictionJob, CustomJob,
                PipelineJob, HyperparameterTuningJob, CustomTrainingJob,
                CustomPythonPackageTrainingJob, CustomContainerTrainingJob,
                ModelEvaluationJob.
            api_endpoint (str):
                Optional. The desired API endpoint,
                e.g., us-central1-aiplatform.googleapis.com
            api_key (str):
                Optional. The API key to use for service calls.
                NOTE: Not all services support API keys.
            api_transport (str):
                Optional. The transport method which is either 'grpc' or 'rest'.
                NOTE: "rest" transport functionality is currently in a
                beta state (preview).
            request_metadata:
                Optional. Additional gRPC metadata to send with every client request.
        Raises:
            ValueError:
                If experiment_description is provided but experiment is not.
        """
        # This method mutates state, so we need to be careful with the validation
        # First, we need to validate all passed values
        if api_transport:
            VALID_TRANSPORT_TYPES = ["grpc", "rest"]
            if api_transport not in VALID_TRANSPORT_TYPES:
                raise ValueError(
                    f"{api_transport} is not a valid transport type. "
                    + f"Valid transport types: {VALID_TRANSPORT_TYPES}"
                )
            # Raise error if api_transport other than rest is specified for usage with API key.
            elif api_key and api_transport != "rest":
                raise ValueError(f"{api_transport} is not supported with API keys. ")
        else:
            if not project and not api_transport:
                api_transport = "rest"

        if location:
            utils.validate_region(location)
            # Set api_transport as "rest" if location is "global".
            if location == "global" and not api_transport:
                self._api_transport = "rest"
            elif location == "global" and api_transport == "grpc":
                raise ValueError(
                    "api_transport cannot be 'grpc' when location is 'global'."
                )
        if experiment_description and experiment is None:
            raise ValueError(
                "Experiment needs to be set in `init` in order to add experiment"
                " descriptions."
            )

        # reset metadata_service config if project or location is updated.
        if (project and project != self._project) or (
            location and location != self._location
        ):
            if metadata._experiment_tracker.experiment_name:
                logging.info("project/location updated, reset Experiment config.")
            metadata._experiment_tracker.reset()

        if project and api_key:
            logging.info(
                "Both a project and API key have been provided. The project will take precedence over the API key."
            )

        # Then we change the main state
        if api_endpoint is not None:
            self._api_endpoint = api_endpoint
        if api_transport:
            self._api_transport = api_transport
        if project:
            self._project = project
        if location:
            self._location = location
        if staging_bucket:
            self._staging_bucket = staging_bucket
        if credentials:
            self._credentials = credentials
        if encryption_spec_key_name:
            self._encryption_spec_key_name = encryption_spec_key_name
        if network is not None:
            self._network = network
        if service_account is not None:
            self._service_account = service_account
        if request_metadata is not None:
            self._request_metadata = request_metadata
        if api_key is not None:
            self._api_key = api_key
        self._resource_type = None

        # Finally, perform secondary state updates
        if experiment_tensorboard and not isinstance(experiment_tensorboard, bool):
            metadata._experiment_tracker.set_tensorboard(
                tensorboard=experiment_tensorboard,
                project=project,
                location=location,
                credentials=credentials,
            )

        if experiment:
            metadata._experiment_tracker.set_experiment(
                experiment=experiment,
                description=experiment_description,
                backing_tensorboard=experiment_tensorboard,
            )

    def get_encryption_spec(
        self,
        encryption_spec_key_name: Optional[str],
        select_version: Optional[str] = compat.DEFAULT_VERSION,
    ) -> Optional[
        Union[
            gca_encryption_spec_v1.EncryptionSpec,
            gca_encryption_spec_v1beta1.EncryptionSpec,
        ]
    ]:
        """Creates a gca_encryption_spec.EncryptionSpec instance from the given
        key name. If the provided key name is None, it uses the default key
        name if provided.

        Args:
            encryption_spec_key_name (Optional[str]): The default encryption key name to use when creating resources.
            select_version: The default version is set to compat.DEFAULT_VERSION
        """
        kms_key_name = encryption_spec_key_name or self.encryption_spec_key_name
        encryption_spec = None
        if kms_key_name:
            gca_encryption_spec = gca_encryption_spec_compat
            if select_version == compat.V1BETA1:
                gca_encryption_spec = gca_encryption_spec_v1beta1
            encryption_spec = gca_encryption_spec.EncryptionSpec(
                kms_key_name=kms_key_name
            )
        return encryption_spec

    @property
    def api_endpoint(self) -> Optional[str]:
        """Default API endpoint, if provided."""
        return self._api_endpoint

    @property
    def api_key(self) -> Optional[str]:
        """API Key, if provided."""
        return self._api_key

    @property
    def project(self) -> str:
        """Default project."""
        if self._project:
            return self._project

        project_not_found_exception_str = (
            "Unable to find your project. Please provide a project ID by:"
            "\n- Passing a constructor argument"
            "\n- Using vertexai.init()"
            "\n- Setting project using 'gcloud config set project my-project'"
            "\n- Setting a GCP environment variable"
            "\n- To create a Google Cloud project, please follow guidance at https://developers.google.com/workspace/guides/create-project"
        )

        try:
            self._set_project_as_env_var_or_google_auth_default()
            project_id = self._project
        except GoogleAuthError as exc:
            raise GoogleAuthError(project_not_found_exception_str) from exc

        if not project_id and not self.api_key:
            raise ValueError(project_not_found_exception_str)

        return project_id

    @property
    def location(self) -> str:
        """Default location."""
        if self._location:
            return self._location

        location = os.getenv("GOOGLE_CLOUD_REGION") or os.getenv("CLOUD_ML_REGION")
        if location:
            utils.validate_region(location)
            return location

        return constants.DEFAULT_REGION

    @property
    def staging_bucket(self) -> Optional[str]:
        """Default staging bucket, if provided."""
        return self._staging_bucket

    @property
    def credentials(self) -> Optional[auth_credentials.Credentials]:
        """Default credentials."""
        if self._credentials:
            return self._credentials
        logger = logging.getLogger("google.auth._default")
        logging_warning_filter = utils.LoggingFilter(logging.WARNING)
        logger.addFilter(logging_warning_filter)
        self._set_project_as_env_var_or_google_auth_default()
        credentials = self._credentials
        logger.removeFilter(logging_warning_filter)
        return credentials

    @property
    def encryption_spec_key_name(self) -> Optional[str]:
        """Default encryption spec key name, if provided."""
        return self._encryption_spec_key_name

    @property
    def network(self) -> Optional[str]:
        """Default Compute Engine network to peer to, if provided."""
        return self._network

    @property
    def service_account(self) -> Optional[str]:
        """Default service account, if provided."""
        return self._service_account

    @property
    def experiment_name(self) -> Optional[str]:
        """Default experiment name, if provided."""
        return metadata._experiment_tracker.experiment_name

    def get_resource_type(self) -> _Product:
        """Returns the resource type from environment variables."""
        if self._resource_type:
            return self._resource_type

        vertex_product = os.getenv("VERTEX_PRODUCT")
        product_mapping = {
            "COLAB_ENTERPRISE": _Product.COLAB_ENTERPRISE,
            "WORKBENCH_CUSTOM_CONTAINER": _Product.WORKBENCH_CUSTOM_CONTAINER,
            "WORKBENCH_INSTANCE": _Product.WORKBENCH_INSTANCE,
        }

        if vertex_product in product_mapping:
            self._resource_type = product_mapping[vertex_product]

        return self._resource_type

    def get_client_options(
        self,
        location_override: Optional[str] = None,
        prediction_client: bool = False,
        api_base_path_override: Optional[str] = None,
        api_key: Optional[str] = None,
        api_path_override: Optional[str] = None,
    ) -> client_options.ClientOptions:
        """Creates GAPIC client_options using location and type.

        Args:
            location_override (str):
                Optional. Set this parameter to get client options for a location different
                from location set by initializer. Must be a GCP region supported by
                Vertex AI.
            prediction_client (str): Optional. flag to use a prediction endpoint.
            api_base_path_override (str): Optional. Override default API base path.
            api_key (str): Optional. API key to use for the client.
            api_path_override (str): Optional. Override default api path.
        Returns:
            clients_options (google.api_core.client_options.ClientOptions):
                A ClientOptions object set with regionalized API endpoint, i.e.
                { "api_endpoint": "us-central1-aiplatform.googleapis.com" } or
                { "api_endpoint": "asia-east1-aiplatform.googleapis.com" }
        """

        api_endpoint = self.api_endpoint

        if (
            api_endpoint is None
            and not self._project
            and not self._location
            and not location_override
        ) or (self._location == "global"):
            # Default endpoint is location invariant if using API key or global
            # location.
            api_endpoint = "aiplatform.googleapis.com"

        # If both project and API key are passed in, project takes precedence.
        if api_endpoint is None:
            # Form the default endpoint to use with no API key.
            if not (self.location or location_override):
                raise ValueError(
                    "No location found. Provide or initialize SDK with a location."
                )

            region = location_override or self.location
            region = region.lower()

            utils.validate_region(region)

            service_base_path = api_base_path_override or (
                constants.PREDICTION_API_BASE_PATH
                if prediction_client
                else constants.API_BASE_PATH
            )

            api_endpoint = (
                f"{region}-{service_base_path}"
                if not api_path_override
                else api_path_override
            )

        # Project/location take precedence over api_key
        if api_key and not self._project:
            return client_options.ClientOptions(
                api_endpoint=api_endpoint, api_key=api_key
            )
        return client_options.ClientOptions(api_endpoint=api_endpoint)

    def common_location_path(
        self, project: Optional[str] = None, location: Optional[str] = None
    ) -> str:
        """Get parent resource with optional project and location override.

        Args:
            project (str): GCP project. If not provided will use the current project.
            location (str): Location. If not provided will use the current location.
        Returns:
            resource_parent: Formatted parent resource string.
        """
        if location:
            utils.validate_region(location)

        return "/".join(
            [
                "projects",
                project or self.project,
                "locations",
                location or self.location,
            ]
        )

    def create_client(
        self,
        client_class: Type[_TVertexAiServiceClientWithOverride],
        credentials: Optional[auth_credentials.Credentials] = None,
        location_override: Optional[str] = None,
        prediction_client: bool = False,
        api_base_path_override: Optional[str] = None,
        api_key: Optional[str] = None,
        api_path_override: Optional[str] = None,
        appended_user_agent: Optional[List[str]] = None,
        appended_gapic_version: Optional[str] = None,
    ) -> _TVertexAiServiceClientWithOverride:
        """Instantiates a given VertexAiServiceClient with optional
        overrides.

        Args:
            client_class (utils.VertexAiServiceClientWithOverride):
                Required. A Vertex AI Service Client with optional overrides.
            credentials (auth_credentials.Credentials):
                Optional. Custom auth credentials. If not provided will use the current config.
            location_override (str): Optional. location override.
            prediction_client (str): Optional. flag to use a prediction endpoint.
            api_key (str): Optional. API key to use for the client.
            api_base_path_override (str): Optional. Override default api base path.
            api_path_override (str): Optional. Override default api path.
            appended_user_agent (List[str]):
                Optional. User agent appended in the client info. If more than one, it will be
                separated by spaces.
            appended_gapic_version (str):
                Optional. GAPIC version suffix appended in the client info.
        Returns:
            client: Instantiated Vertex AI Service client with optional overrides
        """
        gapic_version = __version__

        if appended_gapic_version:
            gapic_version = f"{gapic_version}+{appended_gapic_version}"

        try:
            caller_method = _get_top_level_google_caller_method_name()
            if caller_method:
                gapic_version += (
                    f"+{_TOP_GOOGLE_CONSTRUCTOR_METHOD_TAG}+{caller_method}"
                )
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        resource_type = self.get_resource_type()
        if resource_type:
            gapic_version += f"+environment+{resource_type.value}"

        if telemetry._tool_names_to_append:
            # Must append to gapic_version due to b/259738581.
            gapic_version = f"{gapic_version}+tools+{'+'.join(telemetry._tool_names_to_append[::-1])}"

        user_agent = f"{constants.USER_AGENT_PRODUCT}/{gapic_version}"
        if appended_user_agent:
            user_agent = f"{user_agent} {' '.join(appended_user_agent)}"

        client_info = gapic_v1.client_info.ClientInfo(
            gapic_version=gapic_version,
            user_agent=user_agent,
        )

        kwargs = {
            "credentials": credentials or self.credentials,
            "client_options": self.get_client_options(
                location_override=location_override,
                prediction_client=prediction_client,
                api_key=api_key,
                api_base_path_override=api_base_path_override,
                api_path_override=api_path_override,
            ),
            "client_info": client_info,
        }

        # Do not pass "grpc", rely on gapic defaults unless "rest" is specified
        if self._api_transport == "rest" and "Async" in client_class.__name__:
            # User requests async rest
            if self._async_rest_credentials:
                # Rest async recieves credentials from _async_rest_credentials
                kwargs["credentials"] = self._async_rest_credentials
                kwargs["transport"] = "rest_asyncio"
            else:
                # Rest async was specified, but no async credentials were set.
                # Fallback to gRPC instead.
                logging.warning(
                    "REST async clients requires async credentials set using "
                    + "aiplatform.initializer._set_async_rest_credentials().\n"
                    + "Falling back to grpc since no async rest credentials "
                    + "were detected."
                )
        elif self._api_transport == "rest":
            # User requests sync REST
            kwargs["transport"] = self._api_transport

        client = client_class(**kwargs)
        # We only wrap the client if the request_metadata is set at the creation time.
        if self._request_metadata:
            client = _ClientWrapperThatAddsDefaultMetadata(client)
        return client

    def _get_default_project_and_location(self) -> Tuple[str, str]:
        return (
            self.project,
            self.location,
        )


# Helper classes for adding default metadata to API requests.
# We're solving multiple non-trivial issues here.
# Intended behavior.
# The first big question is whether calling `vertexai.init(request_metadata=...)`
# should change the existing clients.
# This question is non-trivial. Client's client options are immutable.
# But changes to default project, location and credentials affect SDK calls immediately.
# It can be argued that default metadata should affect previously created clients.
# Implementation.
# There are 3 kinds of clients:
# 1) Raw GAPIC client (there are also different transports like "grpc" and "rest")
# 2) ClientWithOverride with _is_temporary=True
# 3) ClientWithOverride with _is_temporary=False
# While a raw client or a non-temporary ClientWithOverride object can be patched once
# (`callable._metadata for callable in client._transport._wrapped_methods.values()`),
# a temporary `ClientWithOverride` creates new client at every call and they
# need to be dynamically patched.
# The temporary `ClientWithOverride` case requires dynamic wrapping/patching.
# A client wrapper, that dynamically wraps methods to add metadata, solves all 3 cases.
class _ClientWrapperThatAddsDefaultMetadata:
    """A client wrapper that dynamically wraps methods to add default metadata."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name: str):
        result = getattr(self._client, name)
        if global_config._request_metadata and callable(result):
            func = result
            if "metadata" in inspect.signature(func).parameters:
                return _FunctionWrapperThatAddsDefaultMetadata(func)
        return result

    def select_version(self, *args, **kwargs):
        client = self._client.select_version(*args, **kwargs)
        if global_config._request_metadata:
            client = _ClientWrapperThatAddsDefaultMetadata(client)
        return client


class _FunctionWrapperThatAddsDefaultMetadata:
    """A function wrapper that wraps a function/method to add default metadata."""

    def __init__(self, func):
        self._func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        # Start with default metadata (copy it)
        metadata_list = list(global_config._request_metadata or [])
        # Add per-request metadata (overrides defaults)
        # The "metadata" argument is removed from "kwargs"
        metadata_list.extend(kwargs.pop("metadata", []))
        # Call the wrapped function with extra metadata
        return self._func(*args, **kwargs, metadata=metadata_list)


# global config to store init parameters: ie, aiplatform.init(project=..., location=...)
global_config = _Config()

global_pool = futures.ThreadPoolExecutor(
    max_workers=min(32, max(4, (os.cpu_count() or 0) * 5))
)


def _set_async_rest_credentials(credentials: AsyncCredentials):
    """Private method to set async REST credentials."""
    if global_config._api_transport != "rest":
        raise ValueError(
            "Async REST credentials can only be set when using REST transport."
        )
    elif not _HAS_ASYNC_CRED_DEPS or not isinstance(credentials, AsyncCredentials):
        raise ValueError(
            "Async REST transport requires async credentials of type"
            + f"{AsyncCredentials} which is only supported in "
            + "google-auth >= 2.35.0.\n\n"
            + "Install the following dependencies:\n"
            + "pip install google-api-core[grpc, async_rest] >= 2.21.0\n"
            + "pip install google-auth[aiohttp] >= 2.35.0\n\n"
            + "Example usage:\n"
            + "from google.auth.aio.credentials import StaticCredentials\n"
            + "async_credentials = StaticCredentials(token=YOUR_TOKEN_HERE)\n"
            + "aiplatform.initializer._set_async_rest_credentials("
            + "credentials=async_credentials)"
        )
    global_config._async_rest_credentials = credentials


def _get_function_name_from_stack_frame(frame) -> str:
    """Gates fully qualified function or method name.

    Args:
        frame: A stack frame

    Returns:
        Fully qualified function or method name
    """
    module_name = frame.f_globals["__name__"]
    function_name = frame.f_code.co_name

    # Getting the class from instance and class methods
    # We need to differentiate between function parameters and other local variables.
    if frame.f_code.co_argcount > 0:
        first_arg_name = frame.f_code.co_varnames[0]
    else:
        first_arg_name = None

    # Inferring the class based on the name of the function's first parameter.
    if first_arg_name == "self":
        f_cls = frame.f_locals["self"].__class__
    elif first_arg_name == "cls":
        f_cls = frame.f_locals["cls"]
    else:
        f_cls = None

    if f_cls:
        module_name = f_cls.__module__ or module_name
        # Not using __qualname__ since it's not affected by the __name__ changes
        class_name = f_cls.__name__
        return f"{module_name}.{class_name}.{function_name}"
    else:
        return f"{module_name}.{function_name}"


def _get_stack_frames() -> Iterator[types.FrameType]:
    """A faster version of inspect.stack().

    This function avoids the expensive inspect.getframeinfo() calls which locate
    the source code and extract the traceback context code lines.
    """
    frame = inspect.currentframe()
    while frame:
        yield frame
        frame = frame.f_back


def _get_top_level_google_caller_method_name() -> Optional[str]:
    top_level_method = None
    for frame in _get_stack_frames():
        function_name = _get_function_name_from_stack_frame(frame)
        if function_name.startswith("vertexai.") or (
            function_name.startswith("google.cloud.aiplatform.")
            and not function_name.startswith("google.cloud.aiplatform.tests")
        ):
            top_level_method = function_name
    return top_level_method
