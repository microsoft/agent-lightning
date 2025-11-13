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
import abc
import inspect
import io
import json
import logging
import os
import sys
import tarfile
import types
import typing
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from google.api_core import exceptions
from google.cloud import storage
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils as aip_utils
from google.cloud.aiplatform_v1 import types as aip_types
from google.cloud.aiplatform_v1.types import reasoning_engine_service
from vertexai.agent_engines import _utils
import httpx
import proto

from google.protobuf import field_mask_pb2


_LOGGER = _utils.LOGGER
_SUPPORTED_PYTHON_VERSIONS = ("3.9", "3.10", "3.11", "3.12", "3.13")
_DEFAULT_GCS_DIR_NAME = "agent_engine"
_BLOB_FILENAME = "agent_engine.pkl"
_REQUIREMENTS_FILE = "requirements.txt"
_EXTRA_PACKAGES_FILE = "dependencies.tar.gz"
_STANDARD_API_MODE = ""
_ASYNC_API_MODE = "async"
_STREAM_API_MODE = "stream"
_ASYNC_STREAM_API_MODE = "async_stream"
_BIDI_STREAM_API_MODE = "bidi_stream"
_A2A_EXTENSION_MODE = "a2a_extension"
_A2A_AGENT_CARD = "a2a_agent_card"
_MODE_KEY_IN_SCHEMA = "api_mode"
_METHOD_NAME_KEY_IN_SCHEMA = "name"
_DEFAULT_METHOD_NAME = "query"
_DEFAULT_ASYNC_METHOD_NAME = "async_query"
_DEFAULT_STREAM_METHOD_NAME = "stream_query"
_DEFAULT_ASYNC_STREAM_METHOD_NAME = "async_stream_query"
_DEFAULT_METHOD_RETURN_TYPE = "dict[str, Any]"
_DEFAULT_ASYNC_METHOD_RETURN_TYPE = "Coroutine[Any, Any, Any]"
_DEFAULT_STREAM_METHOD_RETURN_TYPE = "Iterable[Any]"
_DEFAULT_ASYNC_STREAM_METHOD_RETURN_TYPE = "AsyncIterable[Any]"
_DEFAULT_METHOD_DOCSTRING_TEMPLATE = """
    Runs the Agent Engine to serve the user request.
    This will be based on the `.{method_name}(...)` of the python object that
    was passed in when creating the Agent Engine. The method will invoke the
    `{default_method_name}` API client of the python object.
    Args:
        **kwargs:
            Optional. The arguments of the `.{method_name}(...)` method.
    Returns:
        {return_type}: The response from serving the user request.
"""
_FAILED_TO_REGISTER_API_METHODS_WARNING_TEMPLATE = (
    "Failed to register API methods. Please follow the guide to "
    "register the API methods: "
    "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/develop/custom#custom-methods. "
    "Error: {%s}"
)
_AGENT_FRAMEWORK_ATTR = "agent_framework"
_DEFAULT_AGENT_FRAMEWORK = "custom"
_BUILD_OPTIONS_INSTALLATION = "installation_scripts"
_DEFAULT_METHOD_NAME_MAP = {
    _STANDARD_API_MODE: _DEFAULT_METHOD_NAME,
    _ASYNC_API_MODE: _DEFAULT_ASYNC_METHOD_NAME,
    _STREAM_API_MODE: _DEFAULT_STREAM_METHOD_NAME,
    _ASYNC_STREAM_API_MODE: _DEFAULT_ASYNC_STREAM_METHOD_NAME,
}
_DEFAULT_METHOD_RETURN_TYPE_MAP = {
    _STANDARD_API_MODE: _DEFAULT_METHOD_RETURN_TYPE,
    _ASYNC_API_MODE: _DEFAULT_ASYNC_METHOD_RETURN_TYPE,
    _STREAM_API_MODE: _DEFAULT_STREAM_METHOD_RETURN_TYPE,
    _ASYNC_STREAM_API_MODE: _DEFAULT_ASYNC_STREAM_METHOD_RETURN_TYPE,
}


try:
    from google.adk.agents import BaseAgent

    ADKAgent = BaseAgent
except (ImportError, AttributeError):
    ADKAgent = None

try:
    from a2a.types import (
        AgentCard,
        TransportProtocol,
        Message,
        TaskIdParams,
        TaskQueryParams,
    )
    from a2a.client import ClientConfig, ClientFactory

    AgentCard = AgentCard
    TransportProtocol = TransportProtocol
    Message = Message
    ClientConfig = ClientConfig
    ClientFactory = ClientFactory
    TaskIdParams = TaskIdParams
    TaskQueryParams = TaskQueryParams
except (ImportError, AttributeError):
    AgentCard = None
    TransportProtocol = None
    Message = None
    ClientConfig = None
    ClientFactory = None
    TaskIdParams = None
    TaskQueryParams = None


@typing.runtime_checkable
class Queryable(Protocol):
    """Protocol for Agent Engines that can be queried."""

    @abc.abstractmethod
    def query(self, **kwargs) -> Any:
        """Runs the Agent Engine to serve the user query."""


@typing.runtime_checkable
class AsyncQueryable(Protocol):
    """Protocol for Agent Engines that can be queried asynchronously."""

    @abc.abstractmethod
    def async_query(self, **kwargs) -> Coroutine[Any, Any, Any]:
        """Runs the Agent Engine to serve the user query asynchronously."""


@typing.runtime_checkable
class AsyncStreamQueryable(Protocol):
    """Protocol for Agent Engines that can stream responses asynchronously."""

    @abc.abstractmethod
    async def async_stream_query(self, **kwargs) -> AsyncIterable[Any]:
        """Asynchronously stream responses to serve the user query."""


@typing.runtime_checkable
class StreamQueryable(Protocol):
    """Protocol for Agent Engines that can stream responses."""

    @abc.abstractmethod
    def stream_query(self, **kwargs) -> Iterable[Any]:
        """Stream responses to serve the user query."""


@typing.runtime_checkable
class BidiStreamQueryable(Protocol):
    """Protocol for Agent Engines that can stream requests and responses."""

    @abc.abstractmethod
    async def bidi_stream_query(self, **kwargs) -> AsyncIterable[Any]:
        """Asynchronously stream requests and responses to serve the user query."""


@typing.runtime_checkable
class Cloneable(Protocol):
    """Protocol for Agent Engines that can be cloned."""

    @abc.abstractmethod
    def clone(self) -> Any:
        """Return a clone of the object."""


@typing.runtime_checkable
class OperationRegistrable(Protocol):
    """Protocol for agents that have registered operations."""

    @abc.abstractmethod
    def register_operations(self, **kwargs) -> Dict[str, Sequence[str]]:
        """Register the user provided operations (modes and methods)."""


_AgentEngineInterface = Union[
    ADKAgent,
    AsyncQueryable,
    AsyncStreamQueryable,
    BidiStreamQueryable,
    OperationRegistrable,
    Queryable,
    StreamQueryable,
]


def _wrap_agent_operation(agent: Any, operation: str):
    """Wraps an agent operation into a method (works for all API modes)."""

    def _method(self, **kwargs):
        if not self._tmpl_attrs.get("agent"):
            self.set_up()
        return getattr(self._tmpl_attrs["agent"], operation)(**kwargs)

    _method.__name__ = operation
    _method.__doc__ = getattr(agent, operation).__doc__
    return _method


class ModuleAgent(Cloneable, OperationRegistrable):
    """Agent that is defined by a module and an agent name.

    This agent is instantiated by importing a module and instantiating an agent
    from that module. It also allows to register operations that are defined in
    the agent.
    """

    def __init__(
        self,
        *,
        module_name: str,
        agent_name: str,
        register_operations: Dict[str, Sequence[str]],
        sys_paths: Optional[Sequence[str]] = None,
        agent_framework: Optional[str] = None,
    ):
        """Initializes a module-based agent.

        Args:
            module_name (str):
                Required. The name of the module to import.
            agent_name (str):
                Required. The name of the agent in the module to instantiate.
            register_operations (Dict[str, Sequence[str]]):
                Required. A dictionary of API modes to a list of method names.
            sys_paths (Sequence[str]):
                Optional. The system paths to search for the module. It should
                be relative to the directory where the code will be running.
                I.e. it should correspond to the directory being passed to
                `extra_packages=...` in the create method. It will be appended
                to the system path in the sequence being specified here, and
                only be appended if it is not already in the system path.
        """
        self.agent_framework = agent_framework
        self._tmpl_attrs = {
            "module_name": module_name,
            "agent_name": agent_name,
            "register_operations": register_operations,
            "sys_paths": sys_paths,
        }

    def clone(self):
        """Return a clone of the agent."""
        return ModuleAgent(
            module_name=self._tmpl_attrs.get("module_name"),
            agent_name=self._tmpl_attrs.get("agent_name"),
            register_operations=self._tmpl_attrs.get("register_operations"),
            sys_paths=self._tmpl_attrs.get("sys_paths"),
            agent_framework=self.agent_framework,
        )

    def register_operations(self, **kwargs) -> Dict[str, Sequence[str]]:
        return self._tmpl_attrs.get("register_operations")

    def set_up(self) -> None:
        """Sets up the agent for execution of queries at runtime.

        It runs the code to import the agent from the module, and registers the
        operations of the agent.
        """
        if self._tmpl_attrs.get("sys_paths"):
            import sys

            for sys_path in self._tmpl_attrs.get("sys_paths"):
                abs_path = os.path.abspath(sys_path)
                if abs_path not in sys.path:
                    sys.path.append(abs_path)

        import importlib

        module = importlib.import_module(self._tmpl_attrs.get("module_name"))
        try:
            importlib.reload(module)
        except Exception as e:
            _LOGGER.warning(
                f"Failed to reload module {self._tmpl_attrs.get('module_name')}: {e}"
            )
        agent_name = self._tmpl_attrs.get("agent_name")
        try:
            agent = getattr(module, agent_name)
        except AttributeError as e:
            raise AttributeError(
                f"Agent {agent_name} not found in module "
                f"{self._tmpl_attrs.get('module_name')}"
            ) from e
        if not self.agent_framework:
            self.agent_framework = _get_agent_framework(agent)
        self._tmpl_attrs["agent"] = agent
        if hasattr(agent, "set_up"):
            agent.set_up()
        for operations in self.register_operations().values():
            for operation in operations:
                op = _wrap_agent_operation(agent, operation)
                setattr(self, operation, types.MethodType(op, self))


class AgentEngine(base.VertexAiResourceNounWithFutureManager):
    """Represents a Vertex AI Agent Engine resource."""

    client_class = aip_utils.AgentEngineClientWithOverride
    _resource_noun = "reasoning_engine"
    _getter_method = "get_reasoning_engine"
    _list_method = "list_reasoning_engines"
    _delete_method = "delete_reasoning_engine"
    _parse_resource_name_method = "parse_reasoning_engine_path"
    _format_resource_name_method = "reasoning_engine_path"

    def __init__(self, resource_name: str):
        """Retrieves an Agent Engine resource.

        Args:
            resource_name (str):
                Required. A fully-qualified resource name or ID such as
                "projects/123/locations/us-central1/reasoningEngines/456" or
                "456" when project and location are initialized or passed.
        """
        super().__init__(resource_name=resource_name)
        self.execution_api_client = initializer.global_config.create_client(
            client_class=aip_utils.AgentEngineExecutionClientWithOverride,
        )
        self.execution_async_client = initializer.global_config.create_client(
            client_class=aip_utils.AgentEngineExecutionAsyncClientWithOverride,
        )
        self._gca_resource = self._get_gca_resource(resource_name=resource_name)
        try:
            _register_api_methods_or_raise(self)
        except Exception as e:
            _LOGGER.warning(_FAILED_TO_REGISTER_API_METHODS_WARNING_TEMPLATE, e)
        self._operation_schemas = None

    @property
    def resource_name(self) -> str:
        """Fully-qualified resource name."""
        return self._gca_resource.name

    @classmethod
    def create(
        cls,
        agent_engine: Optional[_AgentEngineInterface] = None,
        *,
        requirements: Optional[Union[str, Sequence[str]]] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        gcs_dir_name: Optional[str] = None,
        extra_packages: Optional[Sequence[str]] = None,
        env_vars: Optional[
            Union[Sequence[str], Dict[str, Union[str, aip_types.SecretRef]]]
        ] = None,
        build_options: Optional[Dict[str, Sequence[str]]] = None,
        service_account: Optional[str] = None,
        psc_interface_config: Optional[aip_types.PscInterfaceConfig] = None,
        min_instances: Optional[int] = None,
        max_instances: Optional[int] = None,
        resource_limits: Optional[Dict[str, str]] = None,
        container_concurrency: Optional[int] = None,
        encryption_spec: Optional[aip_types.EncryptionSpec] = None,
    ) -> "AgentEngine":
        """Creates a new Agent Engine.

        The Agent Engine will be an instance of the `agent_engine` that
        was passed in, running remotely on Vertex AI.

        Sample `src_dir` contents (e.g. `./user_src_dir`):

        .. code-block:: python

            user_src_dir/
            |-- main.py
            |-- requirements.txt
            |-- user_code/
            |   |-- utils.py
            |   |-- ...
            |-- installation_scripts/
            |   |-- install_package.sh
            |   |-- ...
            |-- ...

        To build an Agent Engine with the above files, run:

        .. code-block:: python

            remote_agent = agent_engines.create(
                agent_engine=local_agent,
                requirements=[
                    # I.e. the PyPI dependencies listed in requirements.txt
                    "google-cloud-aiplatform==1.25.0",
                    "langchain==0.0.242",
                    ...
                ],
                extra_packages=[
                    "./user_src_dir/main.py", # a single file
                    "./user_src_dir/user_code", # a directory
                    ...
                ],
                build_options={
                    "installation_scripts": [
                        "./user_src_dir/installation_scripts/install_package.sh",
                        ...
                    ],
                },
            )

        Args:
            agent_engine (AgentEngineInterface):
                Optional. The Agent Engine to be created.
            requirements (Union[str, Sequence[str]]):
                Optional. The set of PyPI dependencies needed. It can either be
                the path to a single file (requirements.txt), or an ordered list
                of strings corresponding to each line of the requirements file.
            display_name (str):
                Optional. The user-defined name of the Agent Engine.
                The name can be up to 128 characters long and can comprise any
                UTF-8 character.
            description (str):
                Optional. The description of the Agent Engine.
            gcs_dir_name (str):
                Optional. The GCS bucket directory under `staging_bucket` to
                use for staging the artifacts needed.
            extra_packages (Sequence[str]):
                Optional. The set of extra user-provided packages (if any).
            env_vars (Union[Sequence[str], Dict[str, Union[str, SecretRef]]]):
                Optional. The environment variables to be set when running the
                Agent Engine. If it is a list of strings, each string should be
                a valid key to `os.environ`. If it is a dictionary, the keys are
                the environment variable names, and the values are the
                corresponding values.
            build_options (Dict[str, Sequence[str]]):
                Optional. The build options for the Agent Engine.
                The following keys are supported:
                - installation_scripts:
                    Optional. The paths to the installation scripts to be
                    executed in the Docker image.
                    The scripts must be located in the `installation_scripts`
                    subdirectory and the path must be added to `extra_packages`.
            service_account (str):
                Optional. The service account to be used for the Agent Engine.
                If not specified, the default reasoning engine service agent
                service account will be used.
            psc_interface_config (aip_types.PscInterfaceConfig):
                Optional. The Private Service Connect interface config for the
                Agent Engine.
            min_instances (int):
                Optional. The minimum number of instances to be running for the
                Agent Engine.
            max_instances (int):
                Optional. The maximum number of instances to be running for the
                Agent Engine.
            resource_limits (Dict[str, str]):
                Optional. The resource limits for the Agent Engine.
            container_concurrency (int):
                Optional. The container concurrency for the Agent Engine.
            encryption_spec (aip_types.EncryptionSpec):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                `projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key`.
                The key needs to be in the same region as the model.

        Returns:
            AgentEngine: The Agent Engine that was created.

        Raises:
            ValueError: If the `project` was not set using `vertexai.init`.
            ValueError: If the `location` was not set using `vertexai.init`.
            ValueError: If the `staging_bucket` was not set using vertexai.init.
            ValueError: If the `staging_bucket` does not start with "gs://".
            ValueError: If `extra_packages` is specified but `agent_engine` is None.
            ValueError: If `requirements` is specified but `agent_engine` is None.
            ValueError: If `env_vars` has a dictionary entry that does not
            correspond to a SecretRef.
            ValueError: If `env_vars` is a list which contains a string that
            does not exist in `os.environ`.
            TypeError: If `env_vars` is not a list of strings or a dictionary.
            TypeError: If `env_vars` has a value that is not a string or SecretRef.
            FileNotFoundError: If `extra_packages` includes a file or directory
            that does not exist.
            IOError: If requirements is a string that corresponds to a
            nonexistent file.
        """
        sys_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _validate_sys_version_or_raise(sys_version)
        gcs_dir_name = gcs_dir_name or _DEFAULT_GCS_DIR_NAME
        staging_bucket = initializer.global_config.staging_bucket

        if agent_engine is not None:
            agent_engine = _validate_agent_engine_or_raise(agent_engine)
            staging_bucket = _validate_staging_bucket_or_raise(staging_bucket)
            if _is_adk_agent(None, agent_engine):
                env_vars = _add_telemetry_enablement_env(env_vars=env_vars)

        if agent_engine is None:
            if requirements is not None:
                raise ValueError("requirements must be None if agent_engine is None.")
            if extra_packages is not None:
                raise ValueError("extra_packages must be None if agent_engine is None.")
        requirements = _validate_requirements_or_raise(
            agent_engine=agent_engine,
            requirements=requirements,
        )
        extra_packages = _validate_extra_packages_or_raise(
            extra_packages=extra_packages,
            build_options=build_options,
        )

        sdk_resource = cls.__new__(cls)
        base.VertexAiResourceNounWithFutureManager.__init__(sdk_resource)

        # Prepares the Agent Engine for creation in Vertex AI.
        # This involves packaging and uploading the artifacts for
        # agent_engine, requirements and extra_packages to
        # `staging_bucket/gcs_dir_name`.
        _prepare(
            agent_engine=agent_engine,
            requirements=requirements,
            project=sdk_resource.project,
            location=sdk_resource.location,
            staging_bucket=staging_bucket,
            gcs_dir_name=gcs_dir_name,
            extra_packages=extra_packages,
        )
        reasoning_engine = aip_types.ReasoningEngine(
            display_name=display_name,
            description=description,
            encryption_spec=encryption_spec,
        )
        if agent_engine is not None:
            # Update the package spec.
            package_spec = aip_types.ReasoningEngineSpec.PackageSpec(
                python_version=sys_version,
                pickle_object_gcs_uri="{}/{}/{}".format(
                    staging_bucket,
                    gcs_dir_name,
                    _BLOB_FILENAME,
                ),
            )
            if extra_packages:
                package_spec.dependency_files_gcs_uri = "{}/{}/{}".format(
                    staging_bucket,
                    gcs_dir_name,
                    _EXTRA_PACKAGES_FILE,
                )
            if requirements:
                package_spec.requirements_gcs_uri = "{}/{}/{}".format(
                    staging_bucket,
                    gcs_dir_name,
                    _REQUIREMENTS_FILE,
                )
            agent_engine_spec = aip_types.ReasoningEngineSpec(
                package_spec=package_spec,
            )
            if (
                env_vars
                or psc_interface_config
                or min_instances is not None
                or max_instances is not None
                or resource_limits
                or container_concurrency is not None
            ):
                deployment_spec, _ = _generate_deployment_spec_or_raise(
                    env_vars=env_vars,
                    psc_interface_config=psc_interface_config,
                    min_instances=min_instances,
                    max_instances=max_instances,
                    resource_limits=resource_limits,
                    container_concurrency=container_concurrency,
                )
                agent_engine_spec.deployment_spec = deployment_spec
            class_methods_spec = _generate_class_methods_spec_or_raise(
                agent_engine=agent_engine,
                operations=_get_registered_operations(agent_engine),
            )
            agent_engine_spec.class_methods.extend(class_methods_spec)
            if service_account:
                agent_engine_spec.service_account = service_account
            reasoning_engine.spec = agent_engine_spec
            reasoning_engine.spec.agent_framework = _get_agent_framework(agent_engine)
        operation_future = sdk_resource.api_client.create_reasoning_engine(
            parent=initializer.global_config.common_location_path(
                project=sdk_resource.project, location=sdk_resource.location
            ),
            reasoning_engine=reasoning_engine,
        )
        _LOGGER.log_create_with_lro(cls, operation_future)
        _LOGGER.info(
            f"View progress and logs at https://console.cloud.google.com/logs/query?project={sdk_resource.project}"
        )
        created_resource = operation_future.result()
        _LOGGER.info(f"{cls.__name__} created. Resource name: {created_resource.name}")
        _LOGGER.info(f"To use this {cls.__name__} in another session:")
        _LOGGER.info(
            f"agent_engine = vertexai.agent_engines.get('{created_resource.name}')"
        )
        # We use `._get_gca_resource(...)` instead of `created_resource` to
        # fully instantiate the attributes of the agent engine.
        sdk_resource._gca_resource = sdk_resource._get_gca_resource(
            resource_name=created_resource.name
        )
        sdk_resource.execution_api_client = initializer.global_config.create_client(
            client_class=aip_utils.AgentEngineExecutionClientWithOverride,
            credentials=sdk_resource.credentials,
            location_override=sdk_resource.location,
        )
        sdk_resource.execution_async_client = initializer.global_config.create_client(
            client_class=aip_utils.AgentEngineExecutionAsyncClientWithOverride,
            credentials=sdk_resource.credentials,
            location_override=sdk_resource.location,
        )
        if agent_engine is not None:
            try:
                _register_api_methods_or_raise(sdk_resource)
            except Exception as e:
                _LOGGER.warning(_FAILED_TO_REGISTER_API_METHODS_WARNING_TEMPLATE, e)
        sdk_resource._operation_schemas = None
        return sdk_resource

    def update(
        self,
        *,
        agent_engine: Optional[_AgentEngineInterface] = None,
        requirements: Optional[Union[str, Sequence[str]]] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        gcs_dir_name: Optional[str] = None,
        extra_packages: Optional[Sequence[str]] = None,
        env_vars: Optional[
            Union[Sequence[str], Dict[str, Union[str, aip_types.SecretRef]]]
        ] = None,
        build_options: Optional[Dict[str, Sequence[str]]] = None,
        service_account: Optional[str] = None,
        psc_interface_config: Optional[aip_types.PscInterfaceConfig] = None,
        min_instances: Optional[int] = None,
        max_instances: Optional[int] = None,
        resource_limits: Optional[Dict[str, str]] = None,
        container_concurrency: Optional[int] = None,
        encryption_spec: Optional[aip_types.EncryptionSpec] = None,
    ) -> "AgentEngine":
        """Updates an existing Agent Engine.

        This method updates the configuration of an existing Agent Engine
        running remotely, which is identified by its resource name.
        Unlike the `create` function which requires a `agent_engine` object,
        all arguments in this method are optional.
        This method allows you to modify individual aspects of the configuration
        by providing any of the optional arguments.

        Args:
            agent_engine (AgentEngineInterface):
                Optional. The instance to be used as the updated Agent Engine.
                If it is not specified, the existing instance will be used.
            requirements (Union[str, Sequence[str]]):
                Optional. The set of PyPI dependencies needed. It can either be
                the path to a single file (requirements.txt), or an ordered list
                of strings corresponding to each line of the requirements file.
                If it is not specified, the existing requirements will be used.
                If it is set to an empty string or list, the existing
                requirements will be removed.
            display_name (str):
                Optional. The user-defined name of the Agent Engine.
                The name can be up to 128 characters long and can comprise any
                UTF-8 character.
            description (str):
                Optional. The description of the Agent Engine.
            gcs_dir_name (str):
                Optional. The GCS bucket directory under `staging_bucket` to
                use for staging the artifacts needed.
            extra_packages (Sequence[str]):
                Optional. The set of extra user-provided packages (if any). If
                it is not specified, the existing extra packages will be used.
                If it is set to an empty list, the existing extra packages will
                be removed.
            env_vars (Union[Sequence[str], Dict[str, Union[str, SecretRef]]]):
                Optional. The environment variables to be set when running the
                Agent Engine. If it is a list of strings, each string should be
                a valid key to `os.environ`. If it is a dictionary, the keys are
                the environment variable names, and the values are the
                corresponding values.
            build_options (Dict[str, Sequence[str]]):
                Optional. The build options for the Agent Engine.
                The following keys are supported:
                - installation_scripts:
                    Optional. The paths to the installation scripts to be
                    executed in the Docker image.
                    The scripts must be located in the `installation_scripts`
                    subdirectory and the path must be added to `extra_packages`.
            service_account (str):
                Optional. The service account to be used for the Agent Engine.
                If not specified, the default reasoning engine service agent
                service account will be used.
            psc_interface_config (aip_types.PscInterfaceConfig):
                Optional. The Private Service Connect interface config for the
                Agent Engine.
            min_instances (int):
                Optional. The minimum number of instances to be running for the
                Agent Engine.
            max_instances (int):
                Optional. The maximum number of instances to be running for the
                Agent Engine.
            resource_limits (Dict[str, str]):
                Optional. The resource limits for the Agent Engine.
            container_concurrency (int):
                Optional. The container concurrency for the Agent Engine.
            encryption_spec (aip_types.EncryptionSpec):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the model. Has the
                form:
                `projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key`.
                The key needs to be in the same region as the model.

        Returns:
            AgentEngine: The Agent Engine that was updated.

        Raises:
            ValueError: If the `staging_bucket` was not set using vertexai.init.
            ValueError: If the `staging_bucket` does not start with "gs://".
            ValueError: If `env_vars` has a dictionary entry that does not
            correspond to a SecretRef.
            ValueError: If `env_vars` is a list which contains a string that
            does not exist in `os.environ`.
            TypeError: If `env_vars` is not a list of strings or a dictionary.
            TypeError: If `env_vars` has a value that is not a string or SecretRef.
            FileNotFoundError: If `extra_packages` includes a file or directory
            that does not exist.
            ValueError: if none of `display_name`, `description`, `requirements`,
            `extra_packages`, `env_vars`, or `agent_engine` were specified.
            IOError: If requirements is a string that corresponds to a
            nonexistent file.
        """
        staging_bucket = initializer.global_config.staging_bucket
        staging_bucket = _validate_staging_bucket_or_raise(staging_bucket)
        historical_operation_schemas = self.operation_schemas()
        gcs_dir_name = gcs_dir_name or _DEFAULT_GCS_DIR_NAME

        # Validate the arguments.
        if not any(
            [
                agent_engine,
                requirements,
                extra_packages,
                display_name,
                description,
                env_vars,
                build_options,
                service_account,
                psc_interface_config,
                min_instances is not None,
                max_instances is not None,
                resource_limits,
                container_concurrency is not None,
                encryption_spec,
            ]
        ):
            raise ValueError(
                "At least one of `agent_engine`, `requirements`, "
                "`extra_packages`, `display_name`, `description`, "
                "`env_vars`, `build_options`, `service_account`, "
                "`psc_interface_config`, `min_instances`, `max_instances`, "
                "`resource_limits`, `container_concurrency`, or "
                "`encryption_spec` must be specified."
            )
        if requirements is not None:
            requirements = _validate_requirements_or_raise(
                agent_engine=agent_engine,
                requirements=requirements,
            )
        if extra_packages is not None:
            extra_packages = _validate_extra_packages_or_raise(
                extra_packages=extra_packages,
                build_options=build_options,
            )
        if agent_engine is not None:
            agent_engine = _validate_agent_engine_or_raise(agent_engine)

        if _is_adk_agent(self, agent_engine):
            env_vars = _add_telemetry_enablement_env(env_vars=env_vars)

        # Prepares the Agent Engine for update in Vertex AI. This involves
        # packaging and uploading the artifacts for agent_engine, requirements
        # and extra_packages to `staging_bucket/gcs_dir_name`.
        _prepare(
            agent_engine=agent_engine,
            requirements=requirements,
            project=self.project,
            location=self.location,
            staging_bucket=staging_bucket,
            gcs_dir_name=gcs_dir_name,
            extra_packages=extra_packages,
        )
        update_request = _generate_update_request_or_raise(
            resource_name=self.resource_name,
            staging_bucket=staging_bucket,
            gcs_dir_name=gcs_dir_name,
            agent_engine=agent_engine,
            requirements=requirements,
            extra_packages=extra_packages,
            display_name=display_name,
            description=description,
            env_vars=env_vars,
            service_account=service_account,
            psc_interface_config=psc_interface_config,
            min_instances=min_instances,
            max_instances=max_instances,
            resource_limits=resource_limits,
            container_concurrency=container_concurrency,
            encryption_spec=encryption_spec,
        )
        operation_future = self.api_client.update_reasoning_engine(
            request=update_request
        )
        _LOGGER.info(
            f"Update Agent Engine backing LRO: {operation_future.operation.name}"
        )
        created_resource = operation_future.result()
        _LOGGER.info(f"Agent Engine updated. Resource name: {created_resource.name}")
        self._operation_schemas = None
        self.execution_api_client = initializer.global_config.create_client(
            client_class=aip_utils.AgentEngineExecutionClientWithOverride,
        )
        # We use `._get_gca_resource(...)` instead of `created_resource` to
        # fully instantiate the attributes of the agent engine.
        self._gca_resource = self._get_gca_resource(resource_name=self.resource_name)

        if (
            agent_engine is None
            or historical_operation_schemas == self.operation_schemas()
        ):
            # The operations of the agent engine are unchanged, so we return it.
            return self

        # If the agent engine has changed and the historical operation
        # schemas are different from the current operation schemas, we need to
        # unregister the historical operation schemas and register the current
        # operation schemas.
        _unregister_api_methods(self, historical_operation_schemas)
        try:
            _register_api_methods_or_raise(self)
        except Exception as e:
            _LOGGER.warning(_FAILED_TO_REGISTER_API_METHODS_WARNING_TEMPLATE, e)
        return self

    def delete(
        self,
        *,
        force: bool = False,
        **kwargs,
    ) -> None:
        """Deletes the ReasoningEngine.

        Args:
            force (bool):
                Optional. If set to True, child resources will also be deleted.
                Otherwise, the request will fail with FAILED_PRECONDITION error
                when the Agent Engine has undeleted child resources. Defaults to
                False.
            **kwargs (dict[str, Any]):
                Optional. Additional keyword arguments to pass to the
                delete_reasoning_engine method.
        """
        kwargs = kwargs or {}
        operation_future = self.api_client.delete_reasoning_engine(
            request=aip_types.DeleteReasoningEngineRequest(
                name=self.resource_name,
                force=force,
                **kwargs,
            ),
        )
        _LOGGER.info(
            f"Delete Agent Engine backing LRO: {operation_future.operation.name}"
        )
        operation_future.result()
        _LOGGER.info(f"Agent Engine deleted. Resource name: {self.resource_name}")

    def operation_schemas(self) -> Sequence[_utils.JsonDict]:
        """Returns the (Open)API schemas for the Agent Engine."""
        spec = _utils.to_dict(self._gca_resource.spec)
        if not hasattr(self, "_operation_schemas") or self._operation_schemas is None:
            self._operation_schemas = spec.get("class_methods", [])
        return self._operation_schemas


def _validate_sys_version_or_raise(sys_version: str) -> None:
    """Tries to validate the python system version."""
    if sys_version not in _SUPPORTED_PYTHON_VERSIONS:
        raise ValueError(
            f"Unsupported python version: {sys_version}. AgentEngine "
            f"only supports {_SUPPORTED_PYTHON_VERSIONS} at the moment."
        )
    if sys_version != f"{sys.version_info.major}.{sys.version_info.minor}":
        _LOGGER.warning(
            f"{sys_version=} is inconsistent with {sys.version_info=}. "
            "This might result in issues with deployment, and should only "
            "be used as a workaround for advanced cases."
        )


def _validate_staging_bucket_or_raise(staging_bucket: Optional[str]) -> str:
    """Tries to validate the staging bucket."""
    if not staging_bucket:
        raise ValueError("Please provide a `staging_bucket` in `vertexai.init(...)`")
    if not staging_bucket.startswith("gs://"):
        raise ValueError(f"{staging_bucket=} must start with `gs://`")
    return staging_bucket


def _validate_agent_engine_or_raise(
    agent_engine: _AgentEngineInterface,
    logger: base.Logger = _LOGGER,
) -> _AgentEngineInterface:
    """Tries to validate the agent engine.

    The agent engine must have one of the following:
    * a callable method named `query`
    * a callable method named `stream_query`
    * a callable method named `async_stream_query`
    * a callable method named `bidi_stream_query`
    * a callable method named `register_operations`

    Args:
        agent_engine: The agent engine to be validated.
        logger: The logger to use for logging.

    Returns:
        The validated agent engine.

    Raises:
        TypeError: If `agent_engine` has no callable method named `query`,
        `stream_query` or `register_operations`.
        ValueError: If `agent_engine` has an invalid `query`, `stream_query` or
        `register_operations` signature.
    """
    try:
        from google.adk.agents import BaseAgent

        if isinstance(agent_engine, BaseAgent):
            logger.info("Deploying google.adk.agents.Agent as an application.")
            from vertexai import agent_engines

            agent_engine = agent_engines.AdkApp(agent=agent_engine)
    except Exception:
        pass
    is_queryable = isinstance(agent_engine, Queryable) and callable(agent_engine.query)
    is_async_queryable = isinstance(agent_engine, AsyncQueryable) and callable(
        agent_engine.async_query
    )
    is_stream_queryable = isinstance(agent_engine, StreamQueryable) and callable(
        agent_engine.stream_query
    )
    is_async_stream_queryable = isinstance(
        agent_engine, AsyncStreamQueryable
    ) and callable(agent_engine.async_stream_query)
    is_bidi_stream_queryable = isinstance(
        agent_engine, BidiStreamQueryable
    ) and callable(agent_engine.bidi_stream_query)
    is_operation_registrable = isinstance(
        agent_engine, OperationRegistrable
    ) and callable(agent_engine.register_operations)

    if not (
        is_queryable
        or is_async_queryable
        or is_stream_queryable
        or is_operation_registrable
        or is_async_stream_queryable
        or is_bidi_stream_queryable
    ):
        raise TypeError(
            "agent_engine has none of the following callable methods: "
            "`query`, `async_query`, `stream_query`, `async_stream_query`, "
            "`bidi_stream_query` or `register_operations`."
        )

    if is_queryable:
        try:
            inspect.signature(getattr(agent_engine, "query"))
        except ValueError as err:
            raise ValueError(
                "Invalid query signature. This might be due to a missing "
                "`self` argument in the agent_engine.query method."
            ) from err

    if is_async_queryable:
        try:
            inspect.signature(getattr(agent_engine, "async_query"))
        except ValueError as err:
            raise ValueError(
                "Invalid async_query signature. This might be due to a missing "
                "`self` argument in the agent_engine.async_query method."
            ) from err

    if is_stream_queryable:
        try:
            inspect.signature(getattr(agent_engine, "stream_query"))
        except ValueError as err:
            raise ValueError(
                "Invalid stream_query signature. This might be due to a missing"
                " `self` argument in the agent_engine.stream_query method."
            ) from err

    if is_async_stream_queryable:
        try:
            inspect.signature(getattr(agent_engine, "async_stream_query"))
        except ValueError as err:
            raise ValueError(
                "Invalid async_stream_query signature. This might be due to a "
                " missing `self` argument in the "
                "agent_engine.async_stream_query method."
            ) from err

    if is_bidi_stream_queryable:
        try:
            inspect.signature(getattr(agent_engine, "bidi_stream_query"))
        except ValueError as err:
            raise ValueError(
                "Invalid bidi_stream_query signature. This might be due to a "
                " missing `self` argument in the "
                "agent_engine.bidi_stream_query method."
            ) from err

    if is_operation_registrable:
        try:
            inspect.signature(getattr(agent_engine, "register_operations"))
        except ValueError as err:
            raise ValueError(
                "Invalid register_operations signature. This might be due to a "
                "missing `self` argument in the "
                "agent_engine.register_operations method."
            ) from err

    if isinstance(agent_engine, Cloneable):
        # Avoid undeployable states.
        agent_engine = agent_engine.clone()
    return agent_engine


def _is_adk_agent(
    agent_engine_to_update: Optional[AgentEngine],
    new_agent_engine: Optional[_AgentEngineInterface],
) -> bool:
    """Checks if the agent engine is an ADK agent.

    Args:
        agent_engine_to_update: Existing agent engine, None if creating new one.
        new_agent_engine: The new agent engine to deploy. Can be None during an update, if the Python agent implementation is not provided, and should remain unchanged.

    Returns:
        True if the agent after the create/update operation, will be an ADK agent.
    """

    from vertexai.agent_engines.templates import adk

    if new_agent_engine is not None:
        return (
            getattr(new_agent_engine, "agent_framework", None)
            == adk.AdkApp.agent_framework
        )
    if agent_engine_to_update is not None:
        return (
            agent_engine_to_update.gca_resource.spec.agent_framework
            == adk.AdkApp.agent_framework
        )
    return False


EnvVars = Optional[Union[Sequence[str], Dict[str, Union[str, aip_types.SecretRef]]]]


def _add_telemetry_enablement_env(*, env_vars: EnvVars) -> EnvVars:
    """Adds telemetry enablement env var to the env vars.

    This is in order to achieve default-on telemetry.
    If the telemetry enablement env var is already set, we do not override it.

    Args:
        env_vars: The env vars to add the telemetry enablement env var to.

    Returns:
        The env vars with the telemetry enablement env var added.
    """

    GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY = (
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"
    )

    if env_vars is None:
        return {GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY: "unspecified"}
    if isinstance(env_vars, dict):
        return (
            env_vars
            if GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY in env_vars
            else env_vars | {GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY: "unspecified"}
        )
    if isinstance(env_vars, list) or isinstance(env_vars, tuple):
        if GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY not in os.environ:
            os.environ[GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY] = "unspecified"

        if isinstance(env_vars, list):
            return env_vars + [GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY]
        else:
            return env_vars + (GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY,)

    raise TypeError(
        f"env_vars must be a list, tuple or a dict, but got {type(env_vars)}."
    )


def _validate_requirements_or_raise(
    *,
    agent_engine: _AgentEngineInterface,
    requirements: Optional[Sequence[str]] = None,
    logger: logging.getLoggerClass() = _LOGGER,
) -> Sequence[str]:
    """Tries to validate the requirements."""
    if requirements is None:
        requirements = []
    elif isinstance(requirements, str):
        try:
            logger.info(f"Reading requirements from {requirements=}")
            with open(requirements) as f:
                requirements = f.read().splitlines()
                logger.info(f"Read the following lines: {requirements}")
        except IOError as err:
            raise IOError(f"Failed to read requirements from {requirements=}") from err
    requirements = _utils.validate_requirements_or_warn(
        obj=agent_engine,
        requirements=requirements,
        logger=logger,
    )
    logger.info(f"The final list of requirements: {requirements}")
    return requirements


def _validate_extra_packages_or_raise(
    extra_packages: Optional[Sequence[str]],
    build_options: Optional[Dict[str, Sequence[str]]] = None,
) -> Sequence[str]:
    """Tries to validates the extra packages."""
    extra_packages = extra_packages or []
    if build_options and _BUILD_OPTIONS_INSTALLATION in build_options:
        _utils.validate_installation_scripts_or_raise(
            script_paths=build_options[_BUILD_OPTIONS_INSTALLATION],
            extra_packages=extra_packages,
        )
    for extra_package in extra_packages:
        if not os.path.exists(extra_package):
            raise FileNotFoundError(
                f"Extra package specified but not found: {extra_package=}"
            )
    return extra_packages


def _get_gcs_bucket(
    *,
    project: str,
    location: str,
    staging_bucket: str,
    logger: base.Logger = _LOGGER,
) -> storage.Bucket:
    """Gets or creates the GCS bucket."""
    storage = _utils._import_cloud_storage_or_raise()
    storage_client = storage.Client(project=project)
    staging_bucket = staging_bucket.replace("gs://", "")
    try:
        gcs_bucket = storage_client.get_bucket(staging_bucket)
        logger.info(f"Using bucket {staging_bucket}")
    except exceptions.NotFound:
        new_bucket = storage_client.bucket(staging_bucket)
        gcs_bucket = storage_client.create_bucket(new_bucket, location=location)
        logger.info(f"Creating bucket {staging_bucket} in {location=}")
    return gcs_bucket


def _upload_agent_engine(
    *,
    agent_engine: _AgentEngineInterface,
    gcs_bucket: storage.Bucket,
    gcs_dir_name: str,
    logger: base.Logger = _LOGGER,
) -> None:
    """Uploads the agent engine to GCS."""
    cloudpickle = _utils._import_cloudpickle_or_raise()
    blob = gcs_bucket.blob(f"{gcs_dir_name}/{_BLOB_FILENAME}")
    with blob.open("wb") as f:
        try:
            cloudpickle.dump(agent_engine, f)
        except Exception as e:
            url = "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/develop/custom#deployment-considerations"
            raise TypeError(
                f"Failed to serialize agent engine. Visit {url} for details."
            ) from e
    with blob.open("rb") as f:
        try:
            _ = cloudpickle.load(f)
        except Exception as e:
            raise TypeError("Agent engine serialized to an invalid format") from e
    dir_name = f"gs://{gcs_bucket.name}/{gcs_dir_name}"
    logger.info(f"Wrote to {dir_name}/{_BLOB_FILENAME}")


def _upload_requirements(
    *,
    requirements: Sequence[str],
    gcs_bucket: storage.Bucket,
    gcs_dir_name: str,
    logger: base.Logger = _LOGGER,
) -> None:
    """Uploads the requirements file to GCS."""
    blob = gcs_bucket.blob(f"{gcs_dir_name}/{_REQUIREMENTS_FILE}")
    blob.upload_from_string("\n".join(requirements))
    dir_name = f"gs://{gcs_bucket.name}/{gcs_dir_name}"
    logger.info(f"Writing to {dir_name}/{_REQUIREMENTS_FILE}")


def _upload_extra_packages(
    *,
    extra_packages: Sequence[str],
    gcs_bucket: storage.Bucket,
    gcs_dir_name: str,
    logger: base.Logger = _LOGGER,
) -> None:
    """Uploads extra packages to GCS."""
    logger.info("Creating in-memory tarfile of extra_packages")
    tar_fileobj = io.BytesIO()
    with tarfile.open(fileobj=tar_fileobj, mode="w|gz") as tar:
        for file in extra_packages:
            tar.add(file)
    tar_fileobj.seek(0)
    blob = gcs_bucket.blob(f"{gcs_dir_name}/{_EXTRA_PACKAGES_FILE}")
    blob.upload_from_string(tar_fileobj.read())
    dir_name = f"gs://{gcs_bucket.name}/{gcs_dir_name}"
    logger.info(f"Writing to {dir_name}/{_EXTRA_PACKAGES_FILE}")


def _prepare(
    agent_engine: Optional[_AgentEngineInterface],
    requirements: Optional[Sequence[str]],
    extra_packages: Optional[Sequence[str]],
    project: str,
    location: str,
    staging_bucket: str,
    gcs_dir_name: str,
    logger: base.Logger = _LOGGER,
) -> None:
    """Prepares the agent engine for creation or updates in Vertex AI.

    This involves packaging and uploading artifacts to Cloud Storage. Note that
    1. This does not actually update the Agent Engine in Vertex AI.
    2. This will only generate and upload a pickled object if specified.
    3. This will only generate and upload the dependencies.tar.gz file if
    extra_packages is non-empty.

    Args:
        agent_engine: The agent engine to be prepared.
        requirements (Sequence[str]): The set of PyPI dependencies needed.
        extra_packages (Sequence[str]): The set of extra user-provided packages.
        project (str): The project for the staging bucket.
        location (str): The location for the staging bucket.
        staging_bucket (str): The staging bucket name in the form "gs://...".
        gcs_dir_name (str): The GCS bucket directory under `staging_bucket` to
            use for staging the artifacts needed.
    """
    if agent_engine is None:
        return
    gcs_bucket = _get_gcs_bucket(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        logger=logger,
    )
    _upload_agent_engine(
        agent_engine=agent_engine,
        gcs_bucket=gcs_bucket,
        gcs_dir_name=gcs_dir_name,
        logger=logger,
    )
    if requirements is not None:
        _upload_requirements(
            requirements=requirements,
            gcs_bucket=gcs_bucket,
            gcs_dir_name=gcs_dir_name,
            logger=logger,
        )
    if extra_packages is not None:
        _upload_extra_packages(
            extra_packages=extra_packages,
            gcs_bucket=gcs_bucket,
            gcs_dir_name=gcs_dir_name,
            logger=logger,
        )


def _update_deployment_spec_with_env_vars_dict_or_raise(
    *,
    deployment_spec: aip_types.ReasoningEngineSpec.DeploymentSpec,
    env_vars: Dict[str, Union[str, aip_types.SecretRef]],
) -> None:
    for key, value in env_vars.items():
        if isinstance(value, Dict):
            try:
                secret_ref = _utils.to_proto(value, aip_types.SecretRef())
            except Exception as e:
                raise ValueError(f"Failed to convert to secret ref: {value}") from e
            deployment_spec.secret_env.append(
                aip_types.SecretEnvVar(name=key, secret_ref=secret_ref)
            )
        elif isinstance(value, aip_types.SecretRef):
            deployment_spec.secret_env.append(
                aip_types.SecretEnvVar(name=key, secret_ref=value)
            )
        elif isinstance(value, str):
            deployment_spec.env.append(aip_types.EnvVar(name=key, value=value))
        else:
            raise TypeError(
                f"Unknown value type in env_vars for {key}. "
                f"Must be a str or SecretRef: {value}"
            )


def _update_deployment_spec_with_env_vars_list_or_raise(
    *,
    deployment_spec: aip_types.ReasoningEngineSpec.DeploymentSpec,
    env_vars: Sequence[str],
) -> None:
    for env_var in env_vars:
        if env_var not in os.environ:
            raise ValueError(f"Env var not found in os.environ: {env_var}.")
        deployment_spec.env.append(
            aip_types.EnvVar(name=env_var, value=os.environ[env_var])
        )


def _generate_deployment_spec_or_raise(
    *,
    env_vars: Optional[
        Union[Sequence[str], Dict[str, Union[str, aip_types.SecretRef]]]
    ] = None,
    psc_interface_config: Optional[aip_types.PscInterfaceConfig] = None,
    min_instances: Optional[int] = None,
    max_instances: Optional[int] = None,
    resource_limits: Optional[Dict[str, str]] = None,
    container_concurrency: Optional[int] = None,
) -> Tuple[aip_types.ReasoningEngineSpec.DeploymentSpec, List[str]]:
    deployment_spec = aip_types.ReasoningEngineSpec.DeploymentSpec()
    update_masks = []
    if env_vars:
        deployment_spec.env = []
        deployment_spec.secret_env = []
        if isinstance(env_vars, Dict):
            _update_deployment_spec_with_env_vars_dict_or_raise(
                deployment_spec=deployment_spec,
                env_vars=env_vars,
            )
        elif isinstance(env_vars, Sequence):
            _update_deployment_spec_with_env_vars_list_or_raise(
                deployment_spec=deployment_spec,
                env_vars=env_vars,
            )
        else:
            raise TypeError(
                f"env_vars must be a list, tuple or a dict, but got {type(env_vars)}."
            )
        if deployment_spec.env:
            update_masks.append("spec.deployment_spec.env")
        if deployment_spec.secret_env:
            update_masks.append("spec.deployment_spec.secret_env")
    if psc_interface_config:
        deployment_spec.psc_interface_config = psc_interface_config
        update_masks.append("spec.deployment_spec.psc_interface_config")
    if min_instances is not None:
        deployment_spec.min_instances = min_instances
        update_masks.append("spec.deployment_spec.min_instances")
    if max_instances is not None:
        deployment_spec.max_instances = max_instances
        update_masks.append("spec.deployment_spec.max_instances")
    if resource_limits:
        deployment_spec.resource_limits = resource_limits
        update_masks.append("spec.deployment_spec.resource_limits")
    if container_concurrency is not None:
        deployment_spec.container_concurrency = container_concurrency
        update_masks.append("spec.deployment_spec.container_concurrency")
    return deployment_spec, update_masks


def _get_agent_framework(
    agent_engine: _AgentEngineInterface,
) -> str:
    if (
        hasattr(agent_engine, _AGENT_FRAMEWORK_ATTR)
        and getattr(agent_engine, _AGENT_FRAMEWORK_ATTR) is not None
        and isinstance(getattr(agent_engine, _AGENT_FRAMEWORK_ATTR), str)
    ):
        return getattr(agent_engine, _AGENT_FRAMEWORK_ATTR)
    return _DEFAULT_AGENT_FRAMEWORK


def _generate_update_request_or_raise(
    *,
    resource_name: str,
    staging_bucket: str,
    gcs_dir_name: str = _DEFAULT_GCS_DIR_NAME,
    agent_engine: Optional[_AgentEngineInterface] = None,
    requirements: Optional[Union[str, Sequence[str]]] = None,
    extra_packages: Optional[Sequence[str]] = None,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    env_vars: Optional[
        Union[Sequence[str], Dict[str, Union[str, aip_types.SecretRef]]]
    ] = None,
    service_account: Optional[str] = None,
    psc_interface_config: Optional[aip_types.PscInterfaceConfig] = None,
    min_instances: Optional[int] = None,
    max_instances: Optional[int] = None,
    resource_limits: Optional[Dict[str, str]] = None,
    container_concurrency: Optional[int] = None,
    encryption_spec: Optional[str] = None,
) -> reasoning_engine_service.UpdateReasoningEngineRequest:
    """Tries to generate the update request for the agent engine."""
    is_spec_update = False
    update_masks: List[str] = []
    agent_engine_spec = aip_types.ReasoningEngineSpec()
    package_spec = aip_types.ReasoningEngineSpec.PackageSpec()
    if requirements is not None:
        is_spec_update = True
        update_masks.append("spec.package_spec.requirements_gcs_uri")
        package_spec.requirements_gcs_uri = "{}/{}/{}".format(
            staging_bucket,
            gcs_dir_name,
            _REQUIREMENTS_FILE,
        )
    if extra_packages is not None:
        is_spec_update = True
        update_masks.append("spec.package_spec.dependency_files_gcs_uri")
        package_spec.dependency_files_gcs_uri = "{}/{}/{}".format(
            staging_bucket,
            gcs_dir_name,
            _EXTRA_PACKAGES_FILE,
        )
    if agent_engine is not None:
        is_spec_update = True
        update_masks.append("spec.package_spec.pickle_object_gcs_uri")
        package_spec.pickle_object_gcs_uri = "{}/{}/{}".format(
            staging_bucket,
            gcs_dir_name,
            _BLOB_FILENAME,
        )
        class_methods_spec = _generate_class_methods_spec_or_raise(
            agent_engine=agent_engine,
            operations=_get_registered_operations(agent_engine),
        )
        agent_engine_spec.class_methods.extend(class_methods_spec)
        update_masks.append("spec.class_methods")
        agent_engine_spec.agent_framework = _get_agent_framework(agent_engine)
        update_masks.append("spec.agent_framework")
    if (
        env_vars is not None
        or psc_interface_config
        or min_instances is not None
        or max_instances is not None
        or resource_limits
        or container_concurrency is not None
    ):
        is_spec_update = True
        deployment_spec, deployment_update_masks = _generate_deployment_spec_or_raise(
            env_vars=env_vars,
            psc_interface_config=psc_interface_config,
            min_instances=min_instances,
            max_instances=max_instances,
            resource_limits=resource_limits,
            container_concurrency=container_concurrency,
        )
        update_masks.extend(deployment_update_masks)
        agent_engine_spec.deployment_spec = deployment_spec
    if service_account is not None:
        is_spec_update = True
        update_masks.append("spec.service_account")
        agent_engine_spec.service_account = service_account

    agent_engine_message = aip_types.ReasoningEngine(name=resource_name)
    if is_spec_update:
        if package_spec:
            agent_engine_spec.package_spec = package_spec
        agent_engine_message.spec = agent_engine_spec
    if display_name:
        agent_engine_message.display_name = display_name
        update_masks.append("display_name")
    if description:
        agent_engine_message.description = description
        update_masks.append("description")
    if encryption_spec:
        agent_engine_message.encryption_spec = encryption_spec
        update_masks.append("encryption_spec")
    if not update_masks:
        raise ValueError(
            "At least one of `agent_engine`, `requirements`, `extra_packages`, "
            "`display_name`, `description`, `env_vars`, or "
            "`encryption_spec` must be specified."
        )
    return reasoning_engine_service.UpdateReasoningEngineRequest(
        reasoning_engine=agent_engine_message,
        update_mask=field_mask_pb2.FieldMask(paths=update_masks),
    )


def _wrap_query_operation(method_name: str) -> Callable[..., _utils.JsonDict]:
    """Wraps an Agent Engine method, creating a callable for `query` API.

    This function creates a callable object that executes the specified
    Agent Engine method using the `query` API.  It handles the creation of
    the API request and the processing of the API response.

    Args:
        method_name: The name of the Agent Engine method to call.
        doc: Documentation string for the method.

    Returns:
        A callable object that executes the method on the Agent Engine via
        the `query` API.
    """

    def _method(self, **kwargs) -> _utils.JsonDict:
        response = self.execution_api_client.query_reasoning_engine(
            request=aip_types.QueryReasoningEngineRequest(
                name=self.resource_name,
                input=kwargs,
                class_method=method_name,
            ),
        )
        output = _utils.to_dict(response)
        return output.get("output", output)

    return _method


def _wrap_async_query_operation(method_name: str) -> Callable[..., Coroutine]:
    """Wraps an Agent Engine method, creating an async callable for `query` API.

    This function creates a callable object that executes the specified
    Agent Engine method asynchronously using the `query` API. It handles the
    creation of the API request and the processing of the API response.

    Args:
        method_name: The name of the Agent Engine method to call.
        doc: Documentation string for the method.

    Returns:
        A callable object that executes the method on the Agent Engine via
        the `query` API.
    """

    async def _method(self, **kwargs) -> _utils.JsonDict:
        response = await self.execution_async_client.query_reasoning_engine(
            request=aip_types.QueryReasoningEngineRequest(
                name=self.resource_name,
                input=kwargs,
                class_method=method_name,
            ),
        )
        output = _utils.to_dict(response)
        return output.get("output", output)

    return _method


def _wrap_stream_query_operation(*, method_name: str) -> Callable[..., Iterable[Any]]:
    """Wraps an Agent Engine method, creating a callable for `stream_query` API.

    This function creates a callable object that executes the specified
    Agent Engine method using the `stream_query` API.  It handles the
    creation of the API request and the processing of the API response.

    Args:
        method_name: The name of the Agent Engine method to call.
        doc: Documentation string for the method.

    Returns:
        A callable object that executes the method on the Agent Engine via
        the `stream_query` API.
    """

    def _method(self, **kwargs) -> Iterable[Any]:
        response = self.execution_api_client.stream_query_reasoning_engine(
            request=aip_types.StreamQueryReasoningEngineRequest(
                name=self.resource_name,
                input=kwargs,
                class_method=method_name,
            ),
        )
        for chunk in response:
            for parsed_json in _utils.yield_parsed_json(chunk):
                if parsed_json is not None:
                    yield parsed_json

    return _method


def _wrap_async_stream_query_operation(
    *, method_name: str
) -> Callable[..., AsyncIterable[Any]]:
    """Wraps an Agent Engine method, creating an async callable for `stream_query` API.

    This function creates a callable object that executes the specified
    Agent Engine method using the `stream_query` API.  It handles the
    creation of the API request and the processing of the API response.

    Args:
        method_name: The name of the Agent Engine method to call.
        doc: Documentation string for the method.

    Returns:
        A callable object that executes the method on the Agent Engine via
        the `stream_query` API.
    """

    async def _method(self, **kwargs) -> AsyncIterable[Any]:
        response = self.execution_api_client.stream_query_reasoning_engine(
            request=aip_types.StreamQueryReasoningEngineRequest(
                name=self.resource_name,
                input=kwargs,
                class_method=method_name,
            ),
        )
        for chunk in response:
            for parsed_json in _utils.yield_parsed_json(chunk):
                if parsed_json is not None:
                    yield parsed_json

    return _method


def _wrap_bidi_stream_query_operation(
    *, method_name: str
) -> Callable[..., AsyncIterable[Any]]:
    """Wraps an Agent Engine method, creating an async callable for `bidi_stream_query` API.

    This function creates a callable object that executes the specified
    Agent Engine method using the `bidi_stream_query` API.  It handles the
    creation of the API request and the processing of the API response.

    Args:
        method_name: The name of the Agent Engine method to call.

    Returns:
        A callable object that executes the method on the Agent Engine via
        the `bidi_stream_query` API.
    """

    async def _method(self, **kwargs) -> AsyncIterable[Any]:
        # Agent Engine bidi streaming query execution should use GenAI SDK Agent
        # Engine live API client directly.
        raise NotImplementedError(
            f"{method_name} is not implemented, please use GenAI SDK Agent "
            "Enginve live API client instead."
        )


def _wrap_a2a_operation(method_name: str, agent_card: str) -> Callable[..., list]:
    """Wraps an Agent Engine method, creating a callable for A2A API.

    Args:
        method_name: The name of the Agent Engine method to call.
        agent_card: The agent card to use for the A2A API call.
            Example:
                {'additionalInterfaces': None,
                'capabilities': {'extensions': None,
                'pushNotifications': None,
                'stateTransitionHistory': None,
                'streaming': False},
                'defaultInputModes': ['text'],
                'defaultOutputModes': ['text'],
                'description': (
                    'A helpful assistant agent that can answer questions.'
                ),
                'documentationUrl': None,
                'iconUrl': None,
                'name': 'Q&A Agent',
                'preferredTransport': 'JSONRPC',
                'protocolVersion': '0.3.0',
                'provider': None,
                'security': None,
                'securitySchemes': None,
                'signatures': None,
                'skills': [{
                    'description': (
                        'A helpful assistant agent that can answer questions.'
                    ),
                    'examples': ['Who is leading 2025 F1 Standings?',
                        'Where can i find an active volcano?'],
                    'id': 'question_answer',
                'inputModes': None,
                'name': 'Q&A Agent',
                'outputModes': None,
                'security': None,
                'tags': ['Question-Answer']}],
                'supportsAuthenticatedExtendedCard': True,
                'url': 'http://localhost:8080/',
                'version': '1.0.0'}
    Returns:
        A callable object that executes the method on the Agent Engine via
        the A2A API.
    """

    async def _method(self, **kwargs) -> Any:
        """Wraps an Agent Engine method, creating a callable for A2A API."""
        a2a_agent_card = AgentCard(**json.loads(agent_card))

        # A2A + AE integration currently only supports Rest API.
        if (
            a2a_agent_card.preferred_transport
            and a2a_agent_card.preferred_transport != TransportProtocol.http_json
        ):
            raise ValueError(
                "Only HTTP+JSON is supported for preferred transport on agent card "
            )

        # Set preferred transport to HTTP+JSON if not set.
        if not hasattr(a2a_agent_card, "preferred_transport"):
            a2a_agent_card.preferred_transport = TransportProtocol.http_json

        # AE cannot support streaming yet. Turn off streaming for now.
        if a2a_agent_card.capabilities and a2a_agent_card.capabilities.streaming:
            raise ValueError(
                "Streaming is not supported in Agent Engine, please change "
                "a2a_agent_card.capabilities.streaming to False."
            )

        if not hasattr(a2a_agent_card.capabilities, "streaming"):
            a2a_agent_card.capabilities.streaming = False

        # agent_card is set on the class_methods before set_up is invoked.
        # Ensure that the agent_card url is set correctly before the client is created.
        a2a_agent_card.url = f"https://{initializer.global_config.api_endpoint}/v1beta1/{self.resource_name}/a2a"

        # Using a2a client, inject the auth token from the global config.
        config = ClientConfig(
            supported_transports=[
                TransportProtocol.http_json,
            ],
            use_client_preference=True,
            httpx_client=httpx.AsyncClient(
                headers={
                    "Authorization": (
                        f"Bearer {initializer.global_config.credentials.token}"
                    )
                }
            ),
        )
        factory = ClientFactory(config)
        client = factory.create(a2a_agent_card)

        # kokoro job uses python 3.9, replaced match with if else.
        if method_name == "on_message_send":
            response = client.send_message(Message(**kwargs))
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
            return chunks
        elif method_name == "on_get_task":
            response = await client.get_task(TaskQueryParams(**kwargs))
        elif method_name == "on_cancel_task":
            response = await client.cancel_task(TaskIdParams(**kwargs))
        elif method_name == "handle_authenticated_agent_card":
            response = await client.get_card()
        else:
            raise ValueError(f"Unknown method name: {method_name}")

        return response

    return _method


def _unregister_api_methods(
    obj: "AgentEngine", operation_schemas: Sequence[_utils.JsonDict]
):
    """Unregisters Agent Engine API methods based on operation schemas.

    This function iterates through operation schemas provided by the
    AgentEngine object.  Each schema defines an API mode and method name.
    It dynamically unregisters methods on the AgentEngine object. This
    should only be used when updating the object.

    Args:
        obj: The AgentEngine object to augment with API methods.
        operation_schemas: The operation schemas to use for method unregistration.
    """
    for operation_schema in operation_schemas:
        if "name" in operation_schema:
            method_name = operation_schema.get("name")
            if hasattr(obj, method_name):
                delattr(obj, method_name)


def _register_api_methods_or_raise(
    obj: "AgentEngine",
    wrap_operation_fn: Optional[
        dict[str, Callable[[str, str], Callable[..., Any]]]
    ] = None,
):
    """Registers Agent Engine API methods based on operation schemas.

    This function iterates through operation schemas provided by the
    AgentEngine object.  Each schema defines an API mode and method name.
    It dynamically creates and registers methods on the AgentEngine object
    to handle API calls based on the specified API mode.
    Currently, only standard API mode `` is supported.

    Args:
        obj: The AgentEngine object to augment with API methods.
        wrap_operation_fn: A dictionary of API modes and method wrapping
            functions.

    Raises:
        ValueError: If the API mode is not supported or if the operation schema
        is missing any required fields (e.g. `api_mode` or `name`).
    """
    for operation_schema in obj.operation_schemas():
        if _MODE_KEY_IN_SCHEMA not in operation_schema:
            raise ValueError(
                f"Operation schema {operation_schema} does not"
                f" contain an `{_MODE_KEY_IN_SCHEMA}` field."
            )
        api_mode = operation_schema.get(_MODE_KEY_IN_SCHEMA)
        if _METHOD_NAME_KEY_IN_SCHEMA not in operation_schema:
            raise ValueError(
                f"Operation schema {operation_schema} does not"
                f" contain a `{_METHOD_NAME_KEY_IN_SCHEMA}` field."
            )
        method_name = operation_schema.get(_METHOD_NAME_KEY_IN_SCHEMA)
        method_description = operation_schema.get(
            "description",
            _DEFAULT_METHOD_DOCSTRING_TEMPLATE.format(
                method_name=method_name,
                default_method_name=_DEFAULT_METHOD_NAME_MAP.get(
                    api_mode, _DEFAULT_METHOD_NAME
                ),
                return_type=_DEFAULT_METHOD_RETURN_TYPE_MAP.get(
                    api_mode,
                    _DEFAULT_METHOD_RETURN_TYPE,
                ),
            ),
        )
        _wrap_operation_map = {
            _STANDARD_API_MODE: _wrap_query_operation,
            _ASYNC_API_MODE: _wrap_async_query_operation,
            _STREAM_API_MODE: _wrap_stream_query_operation,
            _ASYNC_STREAM_API_MODE: _wrap_async_stream_query_operation,
            _BIDI_STREAM_API_MODE: _wrap_bidi_stream_query_operation,
            _A2A_EXTENSION_MODE: _wrap_a2a_operation,
        }
        if isinstance(wrap_operation_fn, dict) and api_mode in wrap_operation_fn:
            # Override the default function with user-specified function if it exists.
            _wrap_operation = wrap_operation_fn[api_mode]
        elif api_mode in _wrap_operation_map:
            _wrap_operation = _wrap_operation_map[api_mode]
        else:
            supported_api_modes = ", ".join(
                f"`{mode}`" for mode in sorted(_wrap_operation_map.keys())
            )
            raise ValueError(
                f"Unsupported api mode: `{api_mode}`,"
                f" Supported modes are: {supported_api_modes}."
            )

        # Bind the method to the object.
        if api_mode == _A2A_EXTENSION_MODE:
            agent_card = operation_schema.get(_A2A_AGENT_CARD)
            method = _wrap_operation(method_name=method_name, agent_card=agent_card)
        else:
            method = _wrap_operation(method_name=method_name)
        method.__name__ = method_name
        method.__doc__ = method_description
        setattr(obj, method_name, types.MethodType(method, obj))


def _get_registered_operations(
    agent_engine: _AgentEngineInterface,
) -> Dict[str, List[str]]:
    """Retrieves registered operations for a AgentEngine."""
    if isinstance(agent_engine, OperationRegistrable):
        return agent_engine.register_operations()

    operations = {}
    if isinstance(agent_engine, Queryable):
        operations[_STANDARD_API_MODE] = [_DEFAULT_METHOD_NAME]
    if isinstance(agent_engine, AsyncQueryable):
        operations[_ASYNC_API_MODE] = [_DEFAULT_ASYNC_METHOD_NAME]
    if isinstance(agent_engine, StreamQueryable):
        operations[_STREAM_API_MODE] = [_DEFAULT_STREAM_METHOD_NAME]
    if isinstance(agent_engine, AsyncStreamQueryable):
        operations[_ASYNC_STREAM_API_MODE] = [_DEFAULT_ASYNC_STREAM_METHOD_NAME]
    return operations


def _generate_class_methods_spec_or_raise(
    *,
    agent_engine: _AgentEngineInterface,
    operations: Dict[str, List[str]],
    logger: base.Logger = _LOGGER,
) -> List[proto.Message]:
    """Generates a ReasoningEngineSpec based on the registered operations.

    Args:
        agent_engine: The AgentEngine instance.
        operations: A dictionary of API modes and method names.

    Returns:
        A list of ReasoningEngineSpec.ClassMethod messages.

    Raises:
        ValueError: If a method defined in `register_operations` is not found on
        the AgentEngine.
    """
    if isinstance(agent_engine, ModuleAgent):
        # We do a dry-run of setting up the agent engine to have the operations
        # needed for registration.
        agent_engine = agent_engine.clone()
        try:
            agent_engine.set_up()
        except Exception as e:
            raise ValueError(
                f"Failed to set up agent engine {agent_engine}: {e}"
            ) from e
    class_methods_spec = []
    for mode, method_names in operations.items():
        for method_name in method_names:
            if mode == _BIDI_STREAM_API_MODE:
                _LOGGER.warning(
                    "Bidi stream API mode is not supported yet in Vertex SDK, "
                    "please use the GenAI SDK instead. Skipping "
                    f"method {method_name}."
                )
                continue
            if not hasattr(agent_engine, method_name):
                raise ValueError(
                    f"Method `{method_name}` defined in `register_operations`"
                    " not found on AgentEngine."
                )

            method = getattr(agent_engine, method_name)
            try:
                schema_dict = _utils.generate_schema(method, schema_name=method_name)
            except Exception as e:
                logger.warning(f"failed to generate schema for {method_name}: {e}")
                continue

            class_method = _utils.to_proto(schema_dict)
            class_method[_MODE_KEY_IN_SCHEMA] = mode
            # A2A agent card is a special case, when running in A2A mode,
            if hasattr(agent_engine, "agent_card"):
                class_method[_A2A_AGENT_CARD] = getattr(
                    agent_engine, "agent_card"
                ).model_dump_json()
            class_methods_spec.append(class_method)

    return class_methods_spec


def _class_methods_to_class_methods_spec(
    class_methods: List[dict[str, Any]],
) -> List[proto.Message]:
    """Converts a list of class methods to a list of ReasoningEngineSpec.ClassMethod messages."""
    return [_utils.to_proto(class_method) for class_method in class_methods]
