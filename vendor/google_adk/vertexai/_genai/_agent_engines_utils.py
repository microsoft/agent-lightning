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
"""Utility functions for agent engines."""

import abc
import asyncio
import base64
from importlib import metadata as importlib_metadata
import inspect
import io
import json
import logging
import os
import sys
import tarfile
import time
import types
import typing
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    TypedDict,
    Union,
)

import httpx

import proto

from google.api_core import exceptions
from google.genai import types as google_genai_types
from google.protobuf import struct_pb2
from google.protobuf import json_format

from . import types as genai_types


try:
    _BUILTIN_MODULE_NAMES: Sequence[str] = sys.builtin_module_names
except AttributeError:
    _BUILTIN_MODULE_NAMES: Sequence[str] = []  # type: ignore[no-redef]

try:
    _PACKAGE_DISTRIBUTIONS: Mapping[
        str, Sequence[str]
    ] = importlib_metadata.packages_distributions()  # type: ignore[attr-defined]
except AttributeError:
    _PACKAGE_DISTRIBUTIONS: Mapping[str, Sequence[str]] = {}  # type: ignore[no-redef]

try:
    # sys.stdlib_module_names is available from Python 3.10 onwards.
    _STDLIB_MODULE_NAMES: frozenset[str] = sys.stdlib_module_names  # type: ignore[attr-defined]
except AttributeError:
    _STDLIB_MODULE_NAMES: frozenset[str] = frozenset()  # type: ignore[no-redef]


try:
    from google.cloud import storage  # type: ignore[attr-defined]

    _StorageBucket: type[Any] = storage.Bucket
except (ImportError, AttributeError):
    _StorageBucket: type[Any] = Any  # type: ignore[no-redef]


try:
    import packaging

    _SpecifierSet: type[Any] = packaging.specifiers.SpecifierSet
except (ImportError, AttributeError):
    _SpecifierSet: type[Any] = Any


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

_ACTIONS_KEY = "actions"
_ACTION_APPEND = "append"
_AGENT_FRAMEWORK_ATTR = "agent_framework"
_ASYNC_API_MODE = "async"
_ASYNC_STREAM_API_MODE = "async_stream"
_BIDI_STREAM_API_MODE = "bidi_stream"
_BASE_MODULES = set(_BUILTIN_MODULE_NAMES + tuple(_STDLIB_MODULE_NAMES))
_BLOB_FILENAME = "agent_engine.pkl"
_DEFAULT_AGENT_FRAMEWORK = "custom"
_SUPPORTED_AGENT_FRAMEWORKS = frozenset(
    [
        "google-adk",
        "langchain",
        "langgraph",
        "ag2",
        "llama-index",
        "custom",
    ]
)
_DEFAULT_ASYNC_METHOD_NAME = "async_query"
_DEFAULT_ASYNC_METHOD_RETURN_TYPE = "Coroutine[Any]"
_DEFAULT_ASYNC_STREAM_METHOD_NAME = "async_stream_query"
_DEFAULT_ASYNC_STREAM_METHOD_RETURN_TYPE = "AsyncIterable[Any]"
_DEFAULT_GCS_DIR_NAME = "agent_engine"
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
_DEFAULT_METHOD_NAME = "query"
_DEFAULT_METHOD_RETURN_TYPE = "dict[str, Any]"
_DEFAULT_STREAM_METHOD_RETURN_TYPE = "Iterable[Any]"
_DEFAULT_REQUIRED_PACKAGES = frozenset(["cloudpickle", "pydantic"])
_DEFAULT_STREAM_METHOD_NAME = "stream_query"
_DEFAULT_BIDI_STREAM_METHOD_NAME = "bidi_stream_query"
_EXTRA_PACKAGES_FILE = "dependencies.tar.gz"
_FAILED_TO_REGISTER_API_METHODS_WARNING_TEMPLATE = (
    "Failed to register API methods. Please follow the guide to "
    "register the API methods: "
    "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/develop/custom#custom-methods. "
    "Error: {%s}"
)
_INSTALLATION_SUBDIR = "installation_scripts"
_METHOD_NAME_KEY_IN_SCHEMA = "name"
_MODE_KEY_IN_SCHEMA = "api_mode"
_REQUIREMENTS_FILE = "requirements.txt"
_STANDARD_API_MODE = ""
_STREAM_API_MODE = "stream"
_A2A_EXTENSION_MODE = "a2a_extension"
_A2A_AGENT_CARD = "a2a_agent_card"
_WARNINGS_KEY = "warnings"
_WARNING_MISSING = "missing"
_WARNING_INCOMPATIBLE = "incompatible"

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


logger = logging.getLogger("vertexai_genai.agentengines")


@typing.runtime_checkable
class Queryable(Protocol):
    """Protocol for Agent Engines that can be queried."""

    @abc.abstractmethod
    def query(self, **kwargs):  # type: ignore[no-untyped-def]
        """Runs the Agent Engine to serve the user query."""


@typing.runtime_checkable
class AsyncQueryable(Protocol):
    """Protocol for Agent Engines that can be queried asynchronously."""

    @abc.abstractmethod
    def async_query(self, **kwargs):  # type: ignore[no-untyped-def]
        """Runs the Agent Engine to serve the user query asynchronously."""


@typing.runtime_checkable
class AsyncStreamQueryable(Protocol):
    """Protocol for Agent Engines that can stream responses asynchronously."""

    @abc.abstractmethod
    async def async_stream_query(self, **kwargs) -> AsyncIterator[Any]:  # type: ignore[no-untyped-def]
        """Asynchronously stream responses to serve the user query."""


@typing.runtime_checkable
class StreamQueryable(Protocol):
    """Protocol for Agent Engines that can stream responses."""

    @abc.abstractmethod
    def stream_query(self, **kwargs) -> Iterator[Any]:  # type: ignore[no-untyped-def]
        """Stream responses to serve the user query."""


@typing.runtime_checkable
class BidiStreamQueryable(Protocol):
    """Protocol for Agent Engines that can stream requests and responses."""

    @abc.abstractmethod
    async def bidi_stream_query(self, input_queue: asyncio.Queue) -> AsyncIterator[Any]:
        """Stream requests and responses to serve the user queries."""


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


try:
    from google.adk.agents import BaseAgent

    ADKAgent: type[Any] = BaseAgent
except (ImportError, AttributeError):
    ADKAgent: type[Any] = Any

_AgentEngineInterface = Union[
    ADKAgent,
    AsyncQueryable,
    AsyncStreamQueryable,
    OperationRegistrable,
    Queryable,
    StreamQueryable,
    BidiStreamQueryable,
]


class _ModuleAgentAttributes(TypedDict, total=False):
    module_name: str
    agent_name: str
    register_operations: Dict[str, Sequence[str]]
    sys_paths: Optional[Sequence[str]]


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
        self._tmpl_attrs: _ModuleAgentAttributes = {
            "module_name": module_name,
            "agent_name": agent_name,
            "register_operations": register_operations,
            "sys_paths": sys_paths,
        }

    def clone(self) -> "ModuleAgent":
        """Return a clone of the agent."""
        return ModuleAgent(
            module_name=self._tmpl_attrs.get("module_name"),
            agent_name=self._tmpl_attrs.get("agent_name"),
            register_operations=self._tmpl_attrs.get("register_operations"),
            sys_paths=self._tmpl_attrs.get("sys_paths"),
        )

    def register_operations(self) -> Dict[str, Sequence[str]]:
        self._tmpl_attrs.get("register_operations")

    def set_up(self) -> None:
        """Sets up the agent for execution of queries at runtime.

        It runs the code to import the agent from the module, and registers the
        operations of the agent.
        """
        sys_paths = self._tmpl_attrs.get("sys_paths")
        if isinstance(sys_paths, Sequence):
            import sys

            for sys_path in sys_paths:
                abs_path = os.path.abspath(sys_path)
                if abs_path not in sys.path:
                    sys.path.append(abs_path)

        import importlib

        module = importlib.import_module(self._tmpl_attrs.get("module_name"))
        try:
            importlib.reload(module)
        except Exception as e:
            logger.warning(
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
        self._tmpl_attrs["agent"] = agent
        if hasattr(agent, "set_up"):
            agent.set_up()
        for operations in self.register_operations().values():
            for operation in operations:
                op = _wrap_agent_operation(agent=agent, operation=operation)
                setattr(self, operation, types.MethodType(op, self))


class _RequirementsValidationActions(TypedDict):
    append: Set[str]


class _RequirementsValidationWarnings(TypedDict):
    missing: Set[str]
    incompatible: Set[str]


class _RequirementsValidationResult(TypedDict):
    warnings: _RequirementsValidationWarnings
    actions: _RequirementsValidationActions


AgentEngineOperationUnion = Union[
    genai_types.AgentEngineOperation,
    genai_types.AgentEngineMemoryOperation,
    genai_types.AgentEngineGenerateMemoriesOperation,
]


class GetOperationFunction(Protocol):
    def __call__(self, *, operation_name: str, **kwargs) -> AgentEngineOperationUnion:
        pass


class GetAsyncOperationFunction(Protocol):
    async def __call__(
        self, *, operation_name: str, **kwargs
    ) -> Awaitable[AgentEngineOperationUnion]:
        pass


async def _await_async_operation(
    *,
    operation_name: str,
    get_operation_fn: GetAsyncOperationFunction,
    poll_interval_seconds: float = 10,
) -> Any:
    """Waits for the operation for creating an agent engine to complete.

    Args:
        operation_name (str):
            Required. The name of the operation for creating the Agent Engine.
        poll_interval_seconds (float):
            The number of seconds to wait between each poll.
        get_operation_fn (Callable[[str], Awaitable[Any]]):
            Optional. The async function to use for getting the operation. If not
            provided, `self._get_agent_operation` will be used.

    Returns:
        The operation that has completed (i.e. `operation.done==True`).
    """
    operation = await get_operation_fn(operation_name=operation_name)
    while not operation.done:
        await asyncio.sleep(poll_interval_seconds)
        operation = await get_operation_fn(operation_name=operation.name)

    return operation


def _await_operation(
    *,
    operation_name: str,
    get_operation_fn: GetOperationFunction,
    poll_interval_seconds: float = 10,
) -> Any:
    """Waits for the operation for creating an agent engine to complete.

    Args:
        operation_name (str):
            Required. The name of the operation for creating the Agent Engine.
        poll_interval_seconds (float):
            The number of seconds to wait between each poll.
        get_operation_fn (Callable[[str], Any]):
            Optional. The function to use for getting the operation. If not
            provided, `self._get_agent_operation` will be used.

    Returns:
        The operation that has completed (i.e. `operation.done==True`).
    """
    operation = get_operation_fn(operation_name=operation_name)
    while not operation.done:
        time.sleep(poll_interval_seconds)
        operation = get_operation_fn(operation_name=operation.name)

    return operation


def _compare_requirements(
    *,
    requirements: Mapping[str, str],
    constraints: Union[Sequence[str], Mapping[str, "_SpecifierSet"]],
    required_packages: Optional[Iterator[str]] = None,
) -> _RequirementsValidationResult:
    """Compares the requirements with the constraints.

    Args:
        requirements (Mapping[str, str]):
            Required. The packages (and their versions) to compare with the constraints.
            This is assumed to be the result of `scan_requirements`.
        constraints (Union[Sequence[str], Mapping[str, SpecifierSet]]):
            Required. The package constraints to compare against. This is assumed
            to be the result of `parse_constraints`.
        required_packages (Iterator[str]):
            Optional. The set of packages that are required to be in the
            constraints. It defaults to the set of packages that are required
            for deployment on Agent Engine.

    Returns:
        dict[str, dict[str, Any]]: The comparison result as a dictionary containing:
            * warnings:
                * missing: The set of packages that are not in the constraints.
                * incompatible: The set of packages that are in the constraints
                    but have versions that are not in the constraint specifier.
            * actions:
                * append: The set of packages that are not in the constraints
                    but should be appended to the constraints.
    """
    packaging_version = _import_packaging_version_or_raise()
    if required_packages is None:
        required_packages = _DEFAULT_REQUIRED_PACKAGES
    result = _RequirementsValidationResult(
        warnings=_RequirementsValidationWarnings(missing=set(), incompatible=set()),
        actions=_RequirementsValidationActions(append=set()),
    )
    if isinstance(constraints, list):
        constraints = _parse_constraints(constraints=constraints)
    for package, package_version in requirements.items():
        if package not in constraints:
            result[_WARNINGS_KEY][_WARNING_MISSING].add(package)  # type: ignore[literal-required]
            if package in required_packages:  # type: ignore[operator]
                result[_ACTIONS_KEY][_ACTION_APPEND].add(  # type: ignore[literal-required]
                    f"{package}=={package_version}"
                )
            continue
        if package_version:
            package_specifier = constraints[package]  # type: ignore[call-overload]
            if not package_specifier:
                continue
            if packaging_version.Version(package_version) not in package_specifier:
                result[_WARNINGS_KEY][_WARNING_INCOMPATIBLE].add(  # type: ignore[literal-required]
                    f"{package}=={package_version} (required: {str(package_specifier)})"
                )
    return result


def _generate_class_methods_spec_or_raise(
    *,
    agent: _AgentEngineInterface,
    operations: Dict[str, List[str]],
) -> List[proto.Message]:
    """Generates a ReasoningEngineSpec based on the registered operations.

    Args:
        agent: The AgentEngine instance.
        operations: A dictionary of API modes and method names.

    Returns:
        A list of ReasoningEngineSpec.ClassMethod messages.

    Raises:
        ValueError: If a method defined in `register_operations` is not found on
        the AgentEngine.
    """
    if isinstance(agent, ModuleAgent):
        # We do a dry-run of setting up the agent engine to have the operations
        # needed for registration.
        agent: ModuleAgent = agent.clone()
        try:
            agent.set_up()
        except Exception as e:
            raise ValueError(f"Failed to set up agent {agent}: {e}") from e
    class_methods_spec = []
    for mode, method_names in operations.items():
        for method_name in method_names:
            if not hasattr(agent, method_name):
                raise ValueError(
                    f"Method `{method_name}` defined in `register_operations`"
                    " not found on agent."
                )

            method = getattr(agent, method_name)
            try:
                schema_dict = _generate_schema(method, schema_name=method_name)
            except Exception as e:
                logger.warning(f"failed to generate schema for {method_name}: {e}")
                continue

            class_method = _to_proto(schema_dict)
            class_method[_MODE_KEY_IN_SCHEMA] = mode
            if hasattr(agent, "agent_card"):
                class_method[_A2A_AGENT_CARD] = getattr(
                    agent, "agent_card"
                ).model_dump_json()
            class_methods_spec.append(class_method)

    return class_methods_spec


def _class_methods_to_class_methods_spec(
    class_methods: List[dict[str, Any]],
) -> List[proto.Message]:
    """Converts a list of class methods to a list of ReasoningEngineSpec.ClassMethod messages."""
    return [_to_proto(class_method) for class_method in class_methods]


def _is_pydantic_serializable(param: inspect.Parameter) -> bool:
    """Checks if the parameter is pydantic serializable."""

    if param.annotation == inspect.Parameter.empty:
        return True

    if "ForwardRef" in repr(param.annotation):
        return True

    if isinstance(param.annotation, str):
        return False

    pydantic = _import_pydantic_or_raise()
    try:
        pydantic.TypeAdapter(param.annotation)
        return True
    except Exception:
        return False


def _generate_schema(
    f: Callable[..., Any],
    *,
    schema_name: Optional[str] = None,
    descriptions: Mapping[str, str] = {},
    required: Sequence[str] = [],
) -> Dict[str, Any]:
    """Generates the OpenAPI Schema for a callable object.

    Only positional and keyword arguments of the function `f` will be supported
    in the OpenAPI Schema that is generated. I.e. `*args` and `**kwargs` will
    not be present in the OpenAPI schema returned from this function. For those
    cases, you can either include it in the docstring for `f`, or modify the
    OpenAPI schema returned from this function to include additional arguments.

    Args:
        f (Callable):
            Required. The function to generate an OpenAPI Schema for.
        schema_name (str):
            Optional. The name for the OpenAPI schema. If unspecified, the name
            of the Callable will be used.
        descriptions (Mapping[str, str]):
            Optional. A `{name: description}` mapping for annotating input
            arguments of the function with user-provided descriptions. It
            defaults to an empty dictionary (i.e. there will not be any
            description for any of the inputs).
        required (Sequence[str]):
            Optional. For the user to specify the set of required arguments in
            function calls to `f`. If specified, it will be automatically
            inferred from `f`.

    Returns:
        dict[str, Any]: The OpenAPI Schema for the function `f` in JSON format.
    """
    pydantic = _import_pydantic_or_raise()
    defaults = dict(inspect.signature(f).parameters)
    fields_dict = {
        name: (
            # 1. We infer the argument type here: use Any rather than None so
            # it will not try to auto-infer the type based on the default value.
            (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                and "ForwardRef" not in repr(param.annotation)
                else Any
            ),
            pydantic.Field(
                # 2. We do not support default values for now.
                # default=(
                #     param.default if param.default != inspect.Parameter.empty
                #     else None
                # ),
                # 3. We support user-provided descriptions.
                description=descriptions.get(name, None),
            ),
        )
        for name, param in defaults.items()
        # We do not support *args or **kwargs
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        )
        # For a bidi endpoint, it requires an asyncio.Queue as the input, but
        # it is not JSON serializable. We hence exclude it from the schema.
        and param.annotation != asyncio.Queue and _is_pydantic_serializable(param)
    }
    parameters = pydantic.create_model(f.__name__, **fields_dict).schema()
    # Postprocessing
    # 4. Suppress unnecessary title generation:
    #    * https://github.com/pydantic/pydantic/issues/1051
    #    * http://cl/586221780
    parameters.pop("title", "")
    for name, function_arg in parameters.get("properties", {}).items():
        function_arg.pop("title", "")
        annotation = defaults[name].annotation
        # 5. Nullable fields:
        #     * https://github.com/pydantic/pydantic/issues/1270
        #     * https://stackoverflow.com/a/58841311
        #     * https://github.com/pydantic/pydantic/discussions/4872
        if typing.get_origin(annotation) is Union and type(None) in typing.get_args(
            annotation
        ):
            # for "typing.Optional" arguments, function_arg might be a
            # dictionary like
            #
            #   {'anyOf': [{'type': 'integer'}, {'type': 'null'}]
            for schema in function_arg.pop("anyOf", []):
                schema_type = schema.get("type")
                if schema_type and schema_type != "null":
                    function_arg["type"] = schema_type
                    break
            function_arg["nullable"] = True
    # 6. Annotate required fields.
    if required:
        # We use the user-provided "required" fields if specified.
        parameters["required"] = required
    else:
        # Otherwise we infer it from the function signature.
        parameters["required"] = [
            k
            for k in defaults
            if (
                defaults[k].default == inspect.Parameter.empty
                and defaults[k].kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.POSITIONAL_ONLY,
                )
            )
        ]
    schema = dict(name=f.__name__, description=f.__doc__, parameters=parameters)
    if schema_name:
        schema["name"] = schema_name
    return schema


def _get_agent_framework(
    *,
    agent_framework: Optional[str],
    agent: _AgentEngineInterface,
) -> str:
    """Gets the agent framework to use.

    The agent framework is determined in the following order of priority:
    1. The `agent_framework` passed to this function.
    2. The `agent_framework` attribute on the `agent` object.
    3. The default framework, "custom".

    Args:
        agent_framework (str):
            The agent framework provided by the user.
        agent (_AgentEngineInterface):
            The agent engine instance.

    Returns:
        str: The name of the agent framework to use.
    """
    if agent_framework is not None and agent_framework in _SUPPORTED_AGENT_FRAMEWORKS:
        logger.info(f"Using agent framework: {agent_framework}")
        return agent_framework
    if hasattr(agent, _AGENT_FRAMEWORK_ATTR):
        agent_framework_attr = getattr(agent, _AGENT_FRAMEWORK_ATTR)
        if (
            agent_framework_attr is not None
            and isinstance(agent_framework_attr, str)
            and agent_framework_attr in _SUPPORTED_AGENT_FRAMEWORKS
        ):
            logger.info(f"Using agent framework: {agent_framework_attr}")
            return agent_framework_attr
    logger.info(
        f"The provided agent framework {agent_framework} is not supported."
        f" Defaulting to {_DEFAULT_AGENT_FRAMEWORK}."
    )
    return _DEFAULT_AGENT_FRAMEWORK


def _get_gcs_bucket(
    *,
    project: str,
    location: str,
    staging_bucket: str,
) -> _StorageBucket:
    """Gets or creates the GCS bucket."""
    storage = _import_cloud_storage_or_raise()
    storage_client = storage.Client(project=project)
    staging_bucket = staging_bucket.replace("gs://", "")
    try:
        gcs_bucket = storage_client.get_bucket(staging_bucket)
        logger.info(f"Using bucket {staging_bucket}")
    except exceptions.NotFound:
        new_bucket = storage_client.bucket(staging_bucket)
        gcs_bucket = storage_client.create_bucket(new_bucket, location=location)
        logger.info(f"Creating bucket {staging_bucket} in {location=}")
    return gcs_bucket  # type: ignore[no-any-return]


def _get_registered_operations(
    *,
    agent: _AgentEngineInterface,
) -> Dict[str, List[str]]:
    """Retrieves registered operations for a AgentEngine."""
    if isinstance(agent, OperationRegistrable):
        return agent.register_operations()

    operations = {}
    if isinstance(agent, Queryable):
        operations[_STANDARD_API_MODE] = [_DEFAULT_METHOD_NAME]
    if isinstance(agent, AsyncQueryable):
        operations[_ASYNC_API_MODE] = [_DEFAULT_ASYNC_METHOD_NAME]
    if isinstance(agent, StreamQueryable):
        operations[_STREAM_API_MODE] = [_DEFAULT_STREAM_METHOD_NAME]
    if isinstance(agent, AsyncStreamQueryable):
        operations[_ASYNC_STREAM_API_MODE] = [_DEFAULT_ASYNC_STREAM_METHOD_NAME]
    if isinstance(agent, BidiStreamQueryable):
        operations[_BIDI_STREAM_API_MODE] = [_DEFAULT_BIDI_STREAM_METHOD_NAME]
    return operations


def _import_cloudpickle_or_raise() -> types.ModuleType:
    """Tries to import the cloudpickle module."""
    try:
        import cloudpickle  # noqa:F401
    except ImportError as e:
        raise ImportError(
            "cloudpickle is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return cloudpickle  # type: ignore[no-any-return]


def _import_cloud_storage_or_raise() -> types.ModuleType:
    """Tries to import the Cloud Storage module."""
    try:
        from google.cloud import storage  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            "Cloud Storage is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return storage  # type: ignore[no-any-return]


def _import_packaging_requirements_or_raise() -> types.ModuleType:
    """Tries to import the packaging.requirements module."""
    try:
        from packaging import requirements
    except ImportError as e:
        raise ImportError(
            "packaging.requirements is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return requirements  # type: ignore[no-any-return]


def _import_packaging_version_or_raise() -> types.ModuleType:
    """Tries to import the packaging.requirements module."""
    try:
        from packaging import version
    except ImportError as e:
        raise ImportError(
            "packaging.version is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return version  # type: ignore[no-any-return]


def _import_pydantic_or_raise() -> types.ModuleType:
    """Tries to import the pydantic module."""
    try:
        import pydantic

        _ = pydantic.Field
    except AttributeError:
        from pydantic import v1 as pydantic  # type: ignore[no-redef]
    except ImportError as e:
        raise ImportError(
            "pydantic is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return pydantic  # type: ignore[no-any-return]


def _parse_constraints(
    *,
    constraints: Sequence[str],
) -> Mapping[str, Optional["_SpecifierSet"]]:
    """Parses a list of constraints into a dict of requirements.

    Args:
        constraints (list[str]):
            Required. The list of package requirements to parse. This is assumed
            to come from the `requirements.txt` file.

    Returns:
        dict[str, SpecifierSet]: The specifiers for each package.
    """
    requirements = _import_packaging_requirements_or_raise()
    result: Dict[str, Optional[_SpecifierSet]] = {}
    for constraint in constraints:
        try:
            if constraint.endswith(".whl"):
                constraint = os.path.basename(constraint)
            requirement = requirements.Requirement(constraint)
        except Exception as e:
            logger.warning(f"Failed to parse constraint: {constraint}. Exception: {e}")
            continue
        result[requirement.name] = requirement.specifier or None
    return result


def _prepare(
    *,
    agent: Optional[_AgentEngineInterface],
    requirements: Optional[Sequence[str]],
    extra_packages: Optional[Sequence[str]],
    project: str,
    location: str,
    staging_bucket: str,
    gcs_dir_name: str,
) -> None:
    """Prepares the agent engine for creation or updates in Vertex AI.

    This involves packaging and uploading artifacts to Cloud Storage. Note that
    1. This does not actually update the Agent Engine in Vertex AI.
    2. This will only generate and upload a pickled object if specified.
    3. This will only generate and upload the dependencies.tar.gz file if
    extra_packages is non-empty.

    Args:
        agent: The agent engine to be prepared.
        requirements (Sequence[str]): The set of PyPI dependencies needed.
        extra_packages (Sequence[str]): The set of extra user-provided packages.
        project (str): The project for the staging bucket.
        location (str): The location for the staging bucket.
        staging_bucket (str): The staging bucket name in the form "gs://...".
        gcs_dir_name (str): The GCS bucket directory under `staging_bucket` to
            use for staging the artifacts needed.
    """
    if agent is None:
        return
    gcs_bucket = _get_gcs_bucket(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
    )
    _upload_agent_engine(
        agent=agent,
        gcs_bucket=gcs_bucket,
        gcs_dir_name=gcs_dir_name,
    )
    if requirements is not None:
        _upload_requirements(
            requirements=requirements,
            gcs_bucket=gcs_bucket,
            gcs_dir_name=gcs_dir_name,
        )
    if extra_packages is not None:
        _upload_extra_packages(
            extra_packages=extra_packages,
            gcs_bucket=gcs_bucket,
            gcs_dir_name=gcs_dir_name,
        )


def _register_api_methods_or_raise(
    *,
    agent_engine: genai_types.AgentEngine,
    wrap_operation_fn: Optional[
        dict[str, Callable[[str, str], Callable[..., Any]]]
    ] = None,
) -> None:
    """Registers Agent Engine API methods based on operation schemas.

    This function iterates through operation schemas provided by the
    `agent_engine`.  Each schema defines an API mode and method name.
    It dynamically creates and registers methods on the `agent_engine`
    to handle API calls based on the specified API mode.
    Currently, only standard API mode `` is supported.

    Args:
        agent_engine: The AgentEngine to augment with API methods.
        wrap_operation_fn: A dictionary of API modes and method wrapping
            functions.

    Raises:
        ValueError: If the API mode is not supported or if the operation schema
        is missing any required fields (e.g. `api_mode` or `name`).
    """
    operation_schemas = agent_engine.operation_schemas()
    if not operation_schemas:
        return
    for operation_schema in operation_schemas:
        if _MODE_KEY_IN_SCHEMA not in operation_schema:
            raise ValueError(
                f"Operation schema {operation_schema} does not"
                f" contain an `{_MODE_KEY_IN_SCHEMA}` field."
            )
        api_mode = operation_schema.get(_MODE_KEY_IN_SCHEMA)
        # For bidi stream api mode, we don't need to wrap the operation.
        if api_mode == _BIDI_STREAM_API_MODE:
            continue

        if _METHOD_NAME_KEY_IN_SCHEMA not in operation_schema:
            raise ValueError(
                f"Operation schema {operation_schema} does not"
                f" contain a `{_METHOD_NAME_KEY_IN_SCHEMA}` field."
            )
        method_name = operation_schema.get(_METHOD_NAME_KEY_IN_SCHEMA)
        if not isinstance(method_name, str):
            raise ValueError(
                "Operation schema has a non-string value for"
                f" `{_METHOD_NAME_KEY_IN_SCHEMA}`: {method_name}"
            )
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
            _A2A_EXTENSION_MODE: _wrap_a2a_operation,
        }
        if isinstance(wrap_operation_fn, dict) and api_mode in wrap_operation_fn:
            # Override the default function with user-specified function if it exists.
            _wrap_operation = wrap_operation_fn[api_mode]  # type: ignore[assignment]
        elif api_mode in _wrap_operation_map:
            _wrap_operation = _wrap_operation_map[api_mode]  # type: ignore[assignment]
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
            method = _wrap_operation(
                method_name=method_name, agent_card=agent_card
            )  # type: ignore[call-arg]
        else:
            method = _wrap_operation(method_name=method_name)  # type: ignore[call-arg]
        method.__name__ = method_name
        if method_description and isinstance(method_description, str):
            method.__doc__ = method_description
        setattr(agent_engine, method_name, types.MethodType(method, agent_engine))


def _scan_requirements(
    *,
    obj: Any,
    ignore_modules: Optional[Sequence[str]] = None,
    package_distributions: Optional[Mapping[str, Sequence[str]]] = None,
    inspect_getmembers_kwargs: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, str]:
    """Scans the object for modules and returns the requirements discovered.

    This is not a comprehensive scan of the object, and only detects for common
    cases based on the members of the object returned by `dir(obj)`.

    Args:
        obj (Any):
            Required. The object to scan for package requirements.
        ignore_modules (Sequence[str]):
            Optional. The set of modules to ignore. It defaults to the set of
            built-in and stdlib modules.
        package_distributions (Mapping[str, Sequence[str]]):
            Optional. The mapping of module names to the set of packages that
            contain them. It defaults to the set of packages from
            `importlib_metadata.packages_distributions()`.
        inspect_getmembers_kwargs (Mapping[str, Any]):
            Optional. The keyword arguments to pass to `inspect.getmembers`. It
            defaults to an empty dictionary.

    Returns:
        Sequence[str]: The list of requirements that were discovered.
    """
    if ignore_modules is None:
        ignore_modules = _BASE_MODULES  # type: ignore[assignment]
    if package_distributions is None:
        package_distributions = _PACKAGE_DISTRIBUTIONS
    modules_found = set(_DEFAULT_REQUIRED_PACKAGES)
    inspect_getmembers_kwargs = inspect_getmembers_kwargs or {}
    for _, attr in inspect.getmembers(obj, **inspect_getmembers_kwargs):
        if not attr or inspect.isbuiltin(attr) or not hasattr(attr, "__module__"):
            continue
        module_name = (attr.__module__ or "").split(".")[0]
        if module_name and module_name not in ignore_modules:  # type: ignore[operator]
            for module in package_distributions.get(module_name, []):
                modules_found.add(module)
    return {module: importlib_metadata.version(module) for module in modules_found}


def _to_dict(message: proto.Message) -> Dict[str, Any]:
    """Converts the contents of the protobuf message to JSON format.

    Args:
        message (proto.Message):
            Required. The proto message to be converted to a JSON dictionary.

    Returns:
        dict[str, Any]: A dictionary containing the contents of the proto.
    """
    try:
        # Best effort attempt to convert the message into a JSON dictionary.
        result: Dict[str, Any] = json.loads(
            json_format.MessageToJson(
                message._pb,
                preserving_proto_field_name=True,
            )
        )
    except AttributeError:
        result: Dict[str, Any] = json.loads(  # type: ignore[no-redef]
            json_format.MessageToJson(
                message,
                preserving_proto_field_name=True,
            )
        )
    return result


def _to_proto(
    obj: Union[Dict[str, Any], proto.Message],
    message: Optional[proto.Message] = None,
) -> proto.Message:
    """Parses a JSON-like object into a message.

    If the object is already a message, this will return the object as-is. If
    the object is a JSON Dict, this will parse and merge the object into the
    message.

    Args:
        obj (Union[dict[str, Any], proto.Message]):
            Required. The object to convert to a proto message.
        message (proto.Message):
            Optional. A protocol buffer message to merge the obj into. It
            defaults to Struct() if unspecified.

    Returns:
        proto.Message: The same message passed as argument.
    """
    if message is None:
        message = struct_pb2.Struct()
    if isinstance(obj, (proto.Message, struct_pb2.Struct)):
        return obj
    try:
        json_format.ParseDict(obj, message._pb)
    except AttributeError:
        json_format.ParseDict(obj, message)
    return message


def _upload_agent_engine(
    *,
    agent: _AgentEngineInterface,
    gcs_bucket: _StorageBucket,
    gcs_dir_name: str,
) -> None:
    """Uploads the agent engine to GCS."""
    cloudpickle = _import_cloudpickle_or_raise()
    blob = gcs_bucket.blob(f"{gcs_dir_name}/{_BLOB_FILENAME}")  # type: ignore[attr-defined]
    with blob.open("wb") as f:
        try:
            cloudpickle.dump(agent, f)
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
    dir_name = f"gs://{gcs_bucket.name}/{gcs_dir_name}"  # type: ignore[attr-defined]
    logger.info(f"Wrote to {dir_name}/{_BLOB_FILENAME}")


def _upload_requirements(
    *,
    requirements: Sequence[str],
    gcs_bucket: _StorageBucket,
    gcs_dir_name: str,
) -> None:
    """Uploads the requirements file to GCS."""
    blob = gcs_bucket.blob(f"{gcs_dir_name}/{_REQUIREMENTS_FILE}")  # type: ignore[attr-defined]
    blob.upload_from_string("\n".join(requirements))
    dir_name = f"gs://{gcs_bucket.name}/{gcs_dir_name}"  # type: ignore[attr-defined]
    logger.info(f"Writing to {dir_name}/{_REQUIREMENTS_FILE}")


def _upload_extra_packages(
    *,
    extra_packages: Sequence[str],
    gcs_bucket: _StorageBucket,
    gcs_dir_name: str,
) -> None:
    """Uploads extra packages to GCS."""
    logger.info("Creating in-memory tarfile of extra_packages")
    tar_fileobj = io.BytesIO()
    with tarfile.open(fileobj=tar_fileobj, mode="w|gz") as tar:
        for file in extra_packages:
            tar.add(file)
    tar_fileobj.seek(0)
    blob = gcs_bucket.blob(f"{gcs_dir_name}/{_EXTRA_PACKAGES_FILE}")  # type: ignore[attr-defined]
    blob.upload_from_string(tar_fileobj.read())
    dir_name = f"gs://{gcs_bucket.name}/{gcs_dir_name}"  # type: ignore[attr-defined]
    logger.info(f"Writing to {dir_name}/{_EXTRA_PACKAGES_FILE}")


def _create_base64_encoded_tarball(
    *,
    source_packages: Sequence[str],
) -> str:
    """Creates a base64 encoded tarball from the source packages."""
    logger.info("Creating in-memory tarfile of source_packages")
    tar_fileobj = io.BytesIO()
    project_dir = os.path.realpath(os.getcwd())
    with tarfile.open(fileobj=tar_fileobj, mode="w|gz") as tar:
        for file in source_packages:
            real_file_path = os.path.realpath(file)
            if real_file_path != project_dir and not real_file_path.startswith(
                project_dir + os.sep
            ):
                raise ValueError(
                    f"File path '{file}' is outside the project directory "
                    f"'{project_dir}'."
                )
            tar.add(file)
    tar_fileobj.seek(0)
    tarball_bytes = tar_fileobj.read()
    return base64.b64encode(tarball_bytes).decode("utf-8")


def _validate_extra_packages_or_raise(
    *,
    extra_packages: Sequence[str],
    build_options: Optional[Dict[str, Sequence[str]]] = None,
) -> Sequence[str]:
    """Tries to validates the extra packages."""
    extra_packages = extra_packages or []
    if build_options and _INSTALLATION_SUBDIR in build_options:
        _validate_installation_scripts_or_raise(
            script_paths=build_options[_INSTALLATION_SUBDIR],
            extra_packages=extra_packages,
        )
    for extra_package in extra_packages:
        if not os.path.exists(extra_package):
            raise FileNotFoundError(
                f"Extra package specified but not found: {extra_package=}"
            )
    return extra_packages


def _validate_installation_scripts_or_raise(
    *,
    script_paths: Sequence[str],
    extra_packages: Sequence[str],
) -> None:
    """Validates the installation scripts' path explicitly provided by the user.

    Args:
        script_paths (Sequence[str]):
            Required. The paths to the installation scripts.
        extra_packages (Sequence[str]):
            Required. The extra packages to be updated.

    Raises:
        ValueError: If a user-defined script is not under the expected
            subdirectory, or not in `extra_packages`, or if an extra package is
            in the installation scripts subdirectory, but is not specified as an
            installation script.
    """
    for script_path in script_paths:
        if not script_path.startswith(_INSTALLATION_SUBDIR):
            logger.warning(
                f"User-defined installation script '{script_path}' is not in "
                f"the expected '{_INSTALLATION_SUBDIR}' subdirectory. "
                f"Ensure it is placed in '{_INSTALLATION_SUBDIR}' within your "
                f"`extra_packages`."
            )
            raise ValueError(
                f"Required installation script '{script_path}' "
                f"is not under '{_INSTALLATION_SUBDIR}'"
            )

        if script_path not in extra_packages:
            logger.warning(
                f"User-defined installation script '{script_path}' is not in "
                f"extra_packages. Ensure it is added to `extra_packages`."
            )
            raise ValueError(
                f"User-defined installation script '{script_path}' "
                f"does not exist in `extra_packages`"
            )

    for extra_package in extra_packages:
        if (
            extra_package.startswith(_INSTALLATION_SUBDIR)
            and extra_package not in script_paths
        ):
            logger.warning(
                f"Extra package '{extra_package}' is in the installation "
                "scripts subdirectory, but is not specified as an installation "
                "script in `build_options`. "
                "Ensure it is added to installation_scripts for "
                "automatic execution."
            )
            raise ValueError(
                f"Extra package '{extra_package}' is in the installation "
                "scripts subdirectory, but is not specified as an installation "
                "script in `build_options`."
            )
    return


def _validate_staging_bucket_or_raise(*, staging_bucket: str) -> str:
    """Tries to validate the staging bucket."""
    if not staging_bucket:
        raise ValueError(
            "Please provide a `staging_bucket` in `client.agent_engines.create(...)`."
        )
    if not staging_bucket.startswith("gs://"):
        raise ValueError(f"{staging_bucket=} must start with `gs://`")
    return staging_bucket


def _validate_requirements_or_warn(
    *,
    obj: Any,
    requirements: List[str],
) -> Mapping[str, str]:
    """Compiles the requirements into a list of requirements."""
    requirements = requirements.copy()
    try:
        current_requirements = _scan_requirements(obj=obj)
        logger.info(f"Identified the following requirements: {current_requirements}")
        constraints = _parse_constraints(constraints=requirements)
        missing_requirements = _compare_requirements(
            requirements=current_requirements,
            constraints=constraints,
        )
        for warning_type, warnings in missing_requirements.get(
            _WARNINGS_KEY, {}
        ).items():
            if warnings:
                logger.warning(
                    f"The following requirements are {warning_type}: {warnings}"
                )
        for action_type, actions in missing_requirements.get(_ACTIONS_KEY, {}).items():
            if actions and action_type == _ACTION_APPEND:
                for action in actions:
                    requirements.append(action)
                logger.info(f"The following requirements are appended: {actions}")
    except Exception as e:
        logger.warning(f"Failed to compile requirements: {e}")
    return requirements


def _validate_requirements_or_raise(
    *,
    agent: Any,
    requirements: Optional[Sequence[str]] = None,
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
    requirements = _validate_requirements_or_warn(  # type: ignore[assignment]
        obj=agent,
        requirements=requirements,
    )
    logger.info(f"The final list of requirements: {requirements}")
    return requirements


def _validate_agent_or_raise(
    *,
    agent: _AgentEngineInterface,
) -> _AgentEngineInterface:
    """Tries to validate the agent engine.

    The agent engine must have one of the following:
    * a callable method named `query`
    * a callable method named `stream_query`
    * a callable method named `async_stream_query`
    * a callable method named `bidi_stream_query`
    * a callable method named `register_operations`

    Args:
        agent: The agent to be validated.

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

        if isinstance(agent, BaseAgent):
            logger.info("Deploying google.adk.agents.Agent as an application.")
            from vertexai import agent_engines

            agent = agent_engines.AdkApp(agent=agent)
    except Exception:
        pass
    is_queryable = isinstance(agent, Queryable) and callable(agent.query)
    is_async_queryable = isinstance(agent, AsyncQueryable) and callable(
        agent.async_query
    )
    is_stream_queryable = isinstance(agent, StreamQueryable) and callable(
        agent.stream_query
    )
    is_async_stream_queryable = isinstance(agent, AsyncStreamQueryable) and callable(
        agent.async_stream_query
    )
    is_bidi_stream_queryable = isinstance(agent, BidiStreamQueryable) and callable(
        agent.bidi_stream_query
    )
    is_operation_registrable = isinstance(agent, OperationRegistrable) and callable(
        agent.register_operations
    )

    if not (
        is_queryable
        or is_async_queryable
        or is_stream_queryable
        or is_operation_registrable
        or is_bidi_stream_queryable
        or is_async_stream_queryable
    ):
        raise TypeError(
            "agent_engine has none of the following callable methods: "
            "`query`, `async_query`, `stream_query`, `async_stream_query`, "
            "`bidi_stream_query`, or `register_operations`."
        )

    if is_queryable:
        try:
            inspect.signature(getattr(agent, "query"))
        except ValueError as err:
            raise ValueError(
                "Invalid query signature. This might be due to a missing "
                "`self` argument in the agent.query method."
            ) from err

    if is_async_queryable:
        try:
            inspect.signature(getattr(agent, "async_query"))
        except ValueError as err:
            raise ValueError(
                "Invalid async_query signature. This might be due to a missing "
                "`self` argument in the agent.async_query method."
            ) from err

    if is_stream_queryable:
        try:
            inspect.signature(getattr(agent, "stream_query"))
        except ValueError as err:
            raise ValueError(
                "Invalid stream_query signature. This might be due to a missing"
                " `self` argument in the agent.stream_query method."
            ) from err

    if is_async_stream_queryable:
        try:
            inspect.signature(getattr(agent, "async_stream_query"))
        except ValueError as err:
            raise ValueError(
                "Invalid async_stream_query signature. This might be due to a "
                " missing `self` argument in the agent.async_stream_query method."
            ) from err

    if is_bidi_stream_queryable:
        try:
            inspect.signature(getattr(agent, "bidi_stream_query"))
        except ValueError as err:
            raise ValueError(
                "Invalid bidi_stream_query signature. This might be due to a "
                " missing `self` argument in the agent.bidi_stream_query method."
            ) from err

    if is_operation_registrable:
        try:
            inspect.signature(getattr(agent, "register_operations"))
        except ValueError as err:
            raise ValueError(
                "Invalid register_operations signature. This might be due to a "
                "missing `self` argument in the agent.register_operations method."
            ) from err

    if isinstance(agent, Cloneable):
        # Avoid undeployable states.
        agent = agent.clone()
    return agent


def _wrap_agent_operation(*, agent: Any, operation: str) -> Callable[..., Any]:
    """Wraps an agent operation into a method (works for all API modes)."""

    def _method(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        if not self._tmpl_attrs.get("agent"):
            self.set_up()
        return getattr(self._tmpl_attrs["agent"], operation)(**kwargs)

    _method.__name__ = operation
    _method.__doc__ = getattr(agent, operation).__doc__
    return _method


AgentEngineOperationUnion = Union[
    genai_types.AgentEngineOperation,
    genai_types.AgentEngineMemoryOperation,
    genai_types.AgentEngineGenerateMemoriesOperation,
]


class GetOperationFunction(Protocol):
    def __call__(  # noqa: E704
        self, *, operation_name: str, **kwargs
    ) -> AgentEngineOperationUnion: ...


def _wrap_query_operation(*, method_name: str) -> Callable[..., Any]:
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

    def _method(self: genai_types.AgentEngine, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        if not self.api_client:
            raise ValueError("api_client is not initialized.")
        if not self.api_resource:
            raise ValueError("api_resource is not initialized.")
        response = self.api_client._query(
            name=self.api_resource.name,
            config={
                "class_method": method_name,
                "input": kwargs,
                "include_all_fields": True,
            },
        )
        return response.output

    return _method


def _wrap_async_query_operation(
    *, method_name: str
) -> Callable[..., Coroutine[Any, Any, Any]]:
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

    async def _method(
        self: genai_types.AgentEngine, **kwargs
    ) -> Coroutine[Any, Any, Any]:  # type: ignore[no-untyped-def]
        if not self.api_async_client:
            raise ValueError("api_async_client is not initialized.")
        if not self.api_resource:
            raise ValueError("api_resource is not initialized.")
        response = await self.api_async_client._query(
            name=self.api_resource.name,
            config={
                "class_method": method_name,
                "input": kwargs,
                "include_all_fields": True,
            },
        )
        return response.output

    return _method


def _wrap_stream_query_operation(*, method_name: str) -> Callable[..., Iterator[Any]]:
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

    def _method(self: genai_types.AgentEngine, **kwargs) -> Iterator[Any]:  # type: ignore[no-untyped-def]
        if not self.api_client:
            raise ValueError("api_client is not initialized.")
        if not self.api_resource:
            raise ValueError("api_resource is not initialized.")
        for http_response in self.api_client._stream_query(
            name=self.api_resource.name,
            config={
                "class_method": method_name,
                "input": kwargs,
                "include_all_fields": True,
            },
        ):
            for line in _yield_parsed_json(http_response=http_response):
                if line is not None:
                    yield line

    return _method


def _wrap_async_stream_query_operation(
    *, method_name: str
) -> Callable[..., AsyncIterator[Any]]:
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

    async def _method(self: genai_types.AgentEngine, **kwargs) -> AsyncIterator[Any]:  # type: ignore[no-untyped-def]
        if not self.api_client:
            raise ValueError("api_client is not initialized.")
        if not self.api_resource:
            raise ValueError("api_resource is not initialized.")
        async for http_response in self.api_client._async_stream_query(
            name=self.api_resource.name,
            config={
                "class_method": method_name,
                "input": kwargs,
                "include_all_fields": True,
            },
        ):
            for line in _yield_parsed_json(http_response=http_response):
                if line is not None:
                    yield line

    return _method


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
        if not self.api_client:
            raise ValueError("api_client is not initialized.")
        if not self.api_resource:
            raise ValueError("api_resource is not initialized.")
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
        base_url = self.api_client._api_client._http_options.base_url.rstrip("/")
        api_version = self.api_client._api_client._http_options.api_version
        a2a_agent_card.url = f"{base_url}/{api_version}/{self.api_resource.name}/a2a"

        # Using a2a client, inject the auth token from the global config.
        config = ClientConfig(
            supported_transports=[
                TransportProtocol.http_json,
            ],
            use_client_preference=True,
            httpx_client=httpx.AsyncClient(
                headers={
                    "Authorization": (
                        f"Bearer {self.api_client._api_client._credentials.token}"
                    )
                }
            ),
        )
        factory = ClientFactory(config)
        client = factory.create(a2a_agent_card)

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


def _yield_parsed_json(http_response: google_genai_types.HttpResponse) -> Iterator[Any]:
    """Converts the body of the HTTP Response message to JSON format.

    Args:
        http_response (google.genai.types.HttpResponse):
            Required. The httpbody body to be converted to JSON object(s).

    Yields:
        Any: A JSON object or line of the original body or None.
    """
    if not http_response.body:
        yield None
        return

    # Handle the case of multiple dictionaries delimited by newlines.
    for line in http_response.body.split("\n"):
        if line:
            try:
                line = json.loads(line)
            except Exception as e:
                logger.warning(f"failed to parse json: {line}. Exception: {e}")
            yield line


def _validate_resource_limits_or_raise(resource_limits: dict[str, str]) -> None:
    """Validates the resource limits.

    Checks that the resource limits are a dict with 'cpu' and 'memory' keys.
    Checks that the 'cpu' value is one of 1, 2, 4, 6, 8.
    Checks that the 'memory' value is a string ending with 'Gi'.
    Checks that the memory size is smaller than 32Gi.
    Checks that the memory size requires at least the specified number of CPUs.

    Args:
        resource_limits: The resource limits to be validated.

    Raises:
        TypeError: If the resource limits are not a dict.
        KeyError: If the resource limits do not contain 'cpu' and 'memory' keys.
        ValueError: If the 'cpu' value is not one of 1, 2, 4, 6, 8.
        ValueError: If the 'memory' value is not a string ending with 'Gi'.
        ValueError: If the memory size is too large.
        ValueError: If the memory size requires more CPUs than the specified
        'cpu' value.
    """
    if not isinstance(resource_limits, dict):
        raise TypeError(f"resource_limits must be a dict. Got {type(resource_limits)}")
    if "cpu" not in resource_limits or "memory" not in resource_limits:
        raise KeyError("resource_limits must contain 'cpu' and 'memory' keys.")

    cpu = int(resource_limits["cpu"])
    memory_str = resource_limits["memory"]

    if cpu not in [1, 2, 4, 6, 8]:
        raise ValueError(
            "resource_limits['cpu'] must be one of 1, 2, 4, 6, 8. Got" f" {cpu}"
        )

    if not isinstance(memory_str, str) or not memory_str.endswith("Gi"):
        raise ValueError(
            "resource_limits['memory'] must be a string ending with 'Gi'."
            f" Got {memory_str}"
        )

    try:
        memory_gb = int(memory_str[:-2])
    except ValueError:
        raise ValueError(
            f"Invalid memory value: {memory_str}. Must be an integer"
            " followed by 'Gi'."
        )

    # https://cloud.google.com/run/docs/configuring/memory-limits
    if memory_gb > 32:
        raise ValueError(
            f"Memory size of {memory_str} is too large. Must be smaller than 32Gi."
        )
    if memory_gb > 24:
        min_cpu = 8
    elif memory_gb > 16:
        min_cpu = 6
    elif memory_gb > 8:
        min_cpu = 4
    elif memory_gb > 4:
        min_cpu = 2
    else:
        min_cpu = 1

    if cpu < min_cpu:
        raise ValueError(
            f"Memory size of {memory_str} requires at least {min_cpu} CPUs."
            f" Got {cpu}"
        )


def _is_adk_agent(agent_engine: _AgentEngineInterface) -> bool:
    """Checks if the agent engine is an ADK agent.

    Args:
        agent_engine: The agent engine to check.

    Returns:
        True if the agent engine is an ADK agent, False otherwise.
    """

    from vertexai.agent_engines.templates import adk

    return isinstance(agent_engine, adk.AdkApp)


def _add_telemetry_enablement_env(
    env_vars: Optional[Dict[str, Union[str, Any]]]
) -> Optional[Dict[str, Union[str, Any]]]:
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
    env_to_add = {GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY: "unspecified"}

    if env_vars is None:
        return env_to_add

    if not isinstance(env_vars, dict):
        raise TypeError(f"env_vars must be a dict, but got {type(env_vars)}.")

    if GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY in env_vars:
        return env_vars

    return env_vars | env_to_add
