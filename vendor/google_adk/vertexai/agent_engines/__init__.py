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
"""Classes and functions for working with agent engines."""

from typing import Dict, Iterable, Optional, Sequence, Union

from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils as aip_utils
from google.cloud.aiplatform_v1 import types as aip_types

# We just want to re-export certain classes
# pylint: disable=g-multiple-import,g-importing-member
from vertexai.agent_engines._agent_engines import (
    _AgentEngineInterface,
    AgentEngine,
    Cloneable,
    ModuleAgent,
    OperationRegistrable,
    Queryable,
    AsyncQueryable,
    StreamQueryable,
    AsyncStreamQueryable,
)
from vertexai.agent_engines.templates.adk import (
    AdkApp,
)
from vertexai.agent_engines.templates.ag2 import (
    AG2Agent,
)
from vertexai.agent_engines.templates.langchain import (
    LangchainAgent,
)
from vertexai.agent_engines.templates.langgraph import (
    LanggraphAgent,
)


_LOGGER = base.Logger(__name__)


def get(resource_name: str) -> AgentEngine:
    """Retrieves an Agent Engine resource.

    Args:
        resource_name (str):
            Required. A fully-qualified resource name or ID such as
            "projects/123/locations/us-central1/reasoningEngines/456" or
            "456" when project and location are initialized or passed.
    """
    return AgentEngine(resource_name)


def create(
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
) -> AgentEngine:
    """Creates a new Agent Engine.

    The Agent Engine will be an instance of the `agent_engine` that
    was passed in, running remotely on Vertex AI.

    Sample ``src_dir`` contents (e.g. ``./user_src_dir``):

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
                "installation": [
                    "./user_src_dir/installation_scripts/install_package.sh",
                    ...
                ],
            },
        )

    Args:
        agent_engine (AgentEngineInterface):
            Required. The Agent Engine to be created.
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
            Optional. The build options for the Agent Engine. This includes
            options such as installation scripts.
        service_account (str):
            Optional. The service account to be used for the Agent Engine. If
            not specified, the default reasoning engine service agent service
            account will be used.
        psc_interface_config (PscInterfaceConfig):
            Optional. The PSC interface config for the Agent Engine. If not
            specified, the default PSC interface config will be used.
        min_instances (int):
            Optional. The minimum number of instances to run the Agent Engine.
            If not specified, the default value will be used.
        max_instances (int):
            Optional. The maximum number of instances to run the Agent Engine.
            If not specified, the default value will be used.
        resource_limits (Dict[str, str]):
            Optional. The resource limits for the Agent Engine. If not
            specified, the default value will be used.
        container_concurrency (int):
            Optional. The container concurrency for the Agent Engine. If not
            specified, the default value will be used.
        encryption_spec (EncryptionSpec):
            Optional. The encryption spec for the Agent Engine. If not
            specified, the default encryption spec will be used.

    Returns:
        AgentEngine: The Agent Engine that was created.

    Raises:
        ValueError: If the `project` was not set using `vertexai.init`.
        ValueError: If the `location` was not set using `vertexai.init`.
        ValueError: If the `staging_bucket` was not set using vertexai.init.
        ValueError: If the `staging_bucket` does not start with "gs://".
        FileNotFoundError: If `extra_packages` includes a file or directory
        that does not exist.
        IOError: If requirements is a string that corresponds to a
        nonexistent file.
    """
    return AgentEngine.create(
        agent_engine=agent_engine,
        requirements=requirements,
        display_name=display_name,
        description=description,
        gcs_dir_name=gcs_dir_name,
        extra_packages=extra_packages,
        env_vars=env_vars,
        build_options=build_options,
        service_account=service_account,
        psc_interface_config=psc_interface_config,
        min_instances=min_instances,
        max_instances=max_instances,
        resource_limits=resource_limits,
        container_concurrency=container_concurrency,
        encryption_spec=encryption_spec,
    )


def list(*, filter: str = "") -> Iterable[AgentEngine]:
    """List all instances of Agent Engine matching the filter.

    Example Usage:

    .. code-block:: python
        import vertexai
        from vertexai import agent_engines

        vertexai.init(project="my_project", location="us-central1")
        agent_engines.list(filter='display_name="My Custom Agent"')

    Args:
        filter (str):
            Optional. An expression for filtering the results of the request.
            For field names both snake_case and camelCase are supported.

    Returns:
        Iterable[AgentEngine]: An iterable of Agent Engines matching the filter.
    """
    api_client = initializer.global_config.create_client(
        client_class=aip_utils.AgentEngineClientWithOverride,
    )
    for agent in api_client.list_reasoning_engines(
        request=aip_types.ListReasoningEnginesRequest(
            parent=initializer.global_config.common_location_path(),
            filter=filter,
        )
    ):
        yield AgentEngine(agent.name)


def delete(
    resource_name: str,
    *,
    force: bool = False,
    **kwargs,
) -> None:
    """Delete an Agent Engine resource.

    Args:
        resource_name (str):
            Required. The name of the Agent Engine to be deleted. Format:
            `projects/{project}/locations/{location}/reasoningEngines/{resource_id}`
        force (bool):
            Optional. If set to True, child resources will also be deleted.
            Otherwise, the request will fail with FAILED_PRECONDITION error
            when the Agent Engine has undeleted child resources. Defaults to
            False.
        **kwargs (dict[str, Any]):
            Optional. Additional keyword arguments to pass to the
            delete_reasoning_engine method.
    """
    api_client = initializer.global_config.create_client(
        client_class=aip_utils.AgentEngineClientWithOverride,
    )
    _LOGGER.info(f"Deleting AgentEngine resource: {resource_name}")
    operation_future = api_client.delete_reasoning_engine(
        request=aip_types.DeleteReasoningEngineRequest(
            name=resource_name,
            force=force,
            **(kwargs or {}),
        )
    )
    _LOGGER.info(f"Delete AgentEngine backing LRO: {operation_future.operation.name}")
    operation_future.result()
    _LOGGER.info(f"AgentEngine resource deleted: {resource_name}")


def update(
    resource_name: str,
    *,
    agent_engine: Optional[Union[Queryable, OperationRegistrable]] = None,
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

    This method updates the configuration of a deployed Agent Engine, identified
    by its resource name. Unlike the `create` function which requires an
    `agent_engine` object, all arguments in this method are optional. This
    method allows you to modify individual aspects of the configuration by
    providing any of the optional arguments.

    Args:
        resource_name (str):
            Required. The name of the Agent Engine to be updated. Format:
            `projects/{project}/locations/{location}/reasoningEngines/{resource_id}`.
        agent_engine (AgentEngineInterface):
            Optional. The instance to be used as the updated Agent Engine. If it
            is not specified, the existing instance will be used.
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
            Optional. The build options for the Agent Engine. This includes
            options such as installation scripts.
        service_account (str):
            Optional. The service account to be used for the Agent Engine. If
            not specified, the default reasoning engine service agent service
            account will be used.
        min_instances (int):
            Optional. The minimum number of instances to run the Agent Engine.
            If not specified, the default value will be used.
        max_instances (int):
            Optional. The maximum number of instances to run the Agent Engine.
            If not specified, the default value will be used.
        resource_limits (Dict[str, str]):
            Optional. The resource limits for the Agent Engine. If not
            specified, the default value will be used.
        container_concurrency (int):
            Optional. The container concurrency for the Agent Engine. If not
            specified, the default value will be used.
        encryption_spec (EncryptionSpec):
            Optional. The encryption spec for the Agent Engine. If not
            specified, the default encryption spec will be used.

    Returns:
        AgentEngine: The Agent Engine that was updated.

    Raises:
        ValueError: If the `staging_bucket` was not set using vertexai.init.
        ValueError: If the `staging_bucket` does not start with "gs://".
        FileNotFoundError: If `extra_packages` includes a file or directory
        that does not exist.
        ValueError: if none of `display_name`, `description`,
        `requirements`, `extra_packages`, `agent_engine`, or `build_options`
        were specified.
        IOError: If requirements is a string that corresponds to a
        nonexistent file.
    """
    agent = get(resource_name)
    return agent.update(
        agent_engine=agent_engine,
        requirements=requirements,
        display_name=display_name,
        description=description,
        gcs_dir_name=gcs_dir_name,
        extra_packages=extra_packages,
        env_vars=env_vars,
        build_options=build_options,
        service_account=service_account,
        psc_interface_config=psc_interface_config,
        min_instances=min_instances,
        max_instances=max_instances,
        resource_limits=resource_limits,
        container_concurrency=container_concurrency,
        encryption_spec=encryption_spec,
    )


__all__ = (
    # Resources
    "AgentEngine",
    # Protocols
    "Cloneable",
    "OperationRegistrable",
    "Queryable",
    "AsyncQueryable",
    "StreamQueryable",
    "AsyncStreamQueryable",
    # Methods
    "create",
    "delete",
    "get",
    "list",
    "update",
    # Templates
    "AdkApp",
    "ModuleAgent",
    "LangchainAgent",
    "LanggraphAgent",
    "AG2Agent",
)
