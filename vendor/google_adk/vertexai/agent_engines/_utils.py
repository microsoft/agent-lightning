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
import dataclasses
import inspect
import json
import os
import sys
import types
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypedDict,
    Union,
)
from importlib import metadata as importlib_metadata

import proto

from google.cloud.aiplatform import base
from google.api import httpbody_pb2
from google.protobuf import struct_pb2
from google.protobuf import json_format

try:
    # For LangChain templates, they might not import langchain_core and get
    #   PydanticUserError: `query` is not fully defined; you should define
    #   `RunnableConfig`, then call `query.model_rebuild()`.
    import langchain_core.runnables.config

    RunnableConfig = langchain_core.runnables.config.RunnableConfig
except ImportError:
    RunnableConfig = Any

try:
    import packaging

    SpecifierSet = packaging.specifiers.SpecifierSet
except AttributeError:
    SpecifierSet = Any

try:
    _BUILTIN_MODULE_NAMES: Sequence[str] = sys.builtin_module_names
except AttributeError:
    _BUILTIN_MODULE_NAMES: Sequence[str] = []

try:
    # sys.stdlib_module_names is available from Python 3.10 onwards.
    _STDLIB_MODULE_NAMES: frozenset = sys.stdlib_module_names
except AttributeError:
    _STDLIB_MODULE_NAMES: frozenset = frozenset()

try:
    _PACKAGE_DISTRIBUTIONS: Mapping[str, Sequence[str]] = (
        importlib_metadata.packages_distributions()
    )

except AttributeError:
    _PACKAGE_DISTRIBUTIONS: Mapping[str, Sequence[str]] = {}

try:
    from autogen.agentchat import chat

    AutogenChatResult = chat.ChatResult
except ImportError:
    AutogenChatResult = Any

try:
    from autogen.io import run_response

    AutogenRunResponse = run_response.RunResponse
except ImportError:
    AutogenRunResponse = Any

try:
    import pydantic

    BaseModel = pydantic.BaseModel
except ImportError:
    BaseModel = Any

JsonDict = Dict[str, Any]


class _RequirementsValidationActions(TypedDict):
    append: Set[str]


class _RequirementsValidationWarnings(TypedDict):
    missing: Set[str]
    incompatible: Set[str]


class _RequirementsValidationResult(TypedDict):
    warnings: _RequirementsValidationWarnings
    actions: _RequirementsValidationActions


LOGGER = base.Logger("vertexai.agent_engines")

_BASE_MODULES = set(_BUILTIN_MODULE_NAMES + tuple(_STDLIB_MODULE_NAMES))
_DEFAULT_REQUIRED_PACKAGES = frozenset(["cloudpickle", "pydantic"])
_ACTIONS_KEY = "actions"
_ACTION_APPEND = "append"
_WARNINGS_KEY = "warnings"
_WARNING_MISSING = "missing"
_WARNING_INCOMPATIBLE = "incompatible"
_INSTALLATION_SUBDIR = "installation_scripts"


def to_proto(
    obj: Union[JsonDict, proto.Message],
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


def to_dict(message: proto.Message) -> JsonDict:
    """Converts the contents of the protobuf message to JSON format.

    Args:
        message (proto.Message):
            Required. The proto message to be converted to a JSON dictionary.

    Returns:
        dict[str, Any]: A dictionary containing the contents of the proto.
    """
    try:
        # Best effort attempt to convert the message into a JSON dictionary.
        result: JsonDict = json.loads(
            json_format.MessageToJson(
                message._pb,
                preserving_proto_field_name=True,
            )
        )
    except AttributeError:
        result: JsonDict = json.loads(
            json_format.MessageToJson(
                message,
                preserving_proto_field_name=True,
            )
        )
    return result


def _dataclass_to_dict_or_raise(obj: Any) -> JsonDict:
    """Converts a dataclass to a JSON dictionary.

    Args:
        obj (Any):
            Required. The dataclass to be converted to a JSON dictionary.

    Returns:
        dict[str, Any]: A dictionary containing the contents of the dataclass.

    Raises:
        TypeError: If the object is not a dataclass.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError(f"Object is not a dataclass: {obj}")
    return json.loads(json.dumps(dataclasses.asdict(obj)))


def _autogen_run_response_protocol_to_dict(
    obj: AutogenRunResponse,
) -> JsonDict:
    """Converts an AutogenRunResponse object into a JSON-serializable dictionary.

    This function takes a `RunResponseProtocol` object and transforms its
    relevant attributes into a dictionary format suitable for JSON conversion.

    The `RunResponseProtocol` defines the structure of the response object,
    which typically includes:

    *   **summary** (`Optional[str]`):
        A textual summary of the run.
    *   **messages** (`Iterable[Message]`):
        A sequence of messages exchanged during the run.
        Each message is expected to be a JSON-serializable dictionary (`Dict[str,
        Any]`).
    *   **events** (`Iterable[BaseEvent]`):
        A sequence of events that occurred during the run.
        Note: The `process()` method, if present, is called before conversion,
        which typically clears this event queue.
    *   **context_variables** (`Optional[dict[str, Any]]`):
        A dictionary containing contextual variables from the run.
    *   **last_speaker** (`Optional[Agent]`):
        The agent that produced the last message.
        The `Agent` object has attributes like `name` (Optional[str]) and
        `description` (Optional[str]).
    *   **cost** (`Optional[Cost]`):
        Information about the computational cost of the run.
        The `Cost` object inherits from `pydantic.BaseModel` and is converted
        to JSON using its `model_dump_json()` method.
    *   **process** (`Optional[Callable[[], None]]`):
        An optional function (like a console event processor) that is called
        before the conversion takes place.
        Executing this method often clears the `events` queue.

    For a detailed definition of `RunResponseProtocol` and its components, refer
    to: https://github.com/ag2ai/ag2/blob/main/autogen/io/run_response.py

    Args:
        obj (AutogenRunResponse): The AutogenRunResponse object to convert. This
            object must conform to the `RunResponseProtocol`.

    Returns:
        JsonDict: A dictionary representation of the AutogenRunResponse, ready
            to be serialized into JSON. The dictionary includes keys like
            'summary', 'messages', 'context_variables', 'last_speaker_name',
            and 'cost'.
    """
    if hasattr(obj, "process"):
        obj.process()

    last_speaker = None
    if getattr(obj, "last_speaker", None) is not None:
        last_speaker = {
            "name": getattr(obj.last_speaker, "name", None),
            "description": getattr(obj.last_speaker, "description", None),
        }

    cost = None
    if getattr(obj, "cost", None) is not None:
        if hasattr(obj.cost, "model_dump_json"):
            cost = json.loads(obj.cost.model_dump_json())
        else:
            cost = str(obj.cost)

    result = {
        "summary": getattr(obj, "summary", None),
        "messages": list(getattr(obj, "messages", [])),
        "context_variables": getattr(obj, "context_variables", None),
        "last_speaker": last_speaker,
        "cost": cost,
    }
    return json.loads(json.dumps(result))


def to_json_serializable_autogen_object(
    obj: Union[
        AutogenChatResult,
        AutogenRunResponse,
    ],
) -> JsonDict:
    """Converts an Autogen object to a JSON serializable object.

    In `ag2<=0.8.4`, `.run()` will return a `ChatResult` object.
    In `ag2>=0.8.5`, `.run()` will return a `RunResponse` object.

    Args:
        obj (Union[AutogenChatResult, AutogenRunResponse]):
            Required. The Autogen object to be converted to a JSON serializable
            object.

    Returns:
        JsonDict: A JSON serializable object.
    """
    if isinstance(obj, AutogenChatResult):
        return _dataclass_to_dict_or_raise(obj)
    return _autogen_run_response_protocol_to_dict(obj)


def yield_parsed_json(body: httpbody_pb2.HttpBody) -> Iterable[Any]:
    """Converts the contents of the httpbody message to JSON format.

    Args:
        body (httpbody_pb2.HttpBody):
            Required. The httpbody body to be converted to a JSON.

    Yields:
        Any: A JSON object or the original body if it is not JSON or None.
    """
    content_type = getattr(body, "content_type", None)
    data = getattr(body, "data", None)

    if content_type is None or data is None or "application/json" not in content_type:
        yield body
        return

    try:
        utf8_data = data.decode("utf-8")
    except Exception as e:
        LOGGER.warning(f"Failed to decode data: {data}. Exception: {e}")
        yield body
        return

    if not utf8_data:
        yield None
        return

    # Handle the case of multiple dictionaries delimited by newlines.
    for line in utf8_data.split("\n"):
        if line:
            try:
                line = json.loads(line)
            except Exception as e:
                LOGGER.warning(f"failed to parse json: {line}. Exception: {e}")
            yield line


def parse_constraints(
    constraints: Sequence[str],
) -> Mapping[str, "SpecifierSet"]:
    """Parses a list of constraints into a dict of requirements.

    Args:
        constraints (list[str]):
            Required. The list of package requirements to parse. This is assumed
            to come from the `requirements.txt` file.

    Returns:
        dict[str, SpecifierSet]: The specifiers for each package.
    """
    requirements = _import_packaging_requirements_or_raise()
    result = {}
    for constraint in constraints:
        try:
            if constraint.endswith(".whl"):
                constraint = os.path.basename(constraint)
            requirement = requirements.Requirement(constraint)
        except Exception as e:
            LOGGER.warning(f"Failed to parse constraint: {constraint}. Exception: {e}")
            continue
        result[requirement.name] = requirement.specifier or None
    return result


def validate_requirements_or_warn(
    obj: Any,
    requirements: List[str],
    logger: base.Logger = LOGGER,
) -> Mapping[str, str]:
    """Compiles the requirements into a list of requirements."""
    requirements = requirements.copy()
    try:
        current_requirements = scan_requirements(obj)
        logger.info(f"Identified the following requirements: {current_requirements}")
        constraints = parse_constraints(requirements)
        missing_requirements = compare_requirements(current_requirements, constraints)
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


def compare_requirements(
    requirements: Mapping[str, str],
    constraints: Union[Sequence[str], Mapping[str, "SpecifierSet"]],
    *,
    required_packages: Optional[Sequence[str]] = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Compares the requirements with the constraints.

    Args:
        requirements (Mapping[str, str]):
            Required. The packages (and their versions) to compare with the constraints.
            This is assumed to be the result of `scan_requirements`.
        constraints (Union[Sequence[str], Mapping[str, SpecifierSet]]):
            Required. The package constraints to compare against. This is assumed
            to be the result of `parse_constraints`.
        required_packages (Sequence[str]):
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
        constraints = parse_constraints(constraints)
    for package, package_version in requirements.items():
        if package not in constraints:
            result[_WARNINGS_KEY][_WARNING_MISSING].add(package)
            if package in required_packages:
                result[_ACTIONS_KEY][_ACTION_APPEND].add(
                    f"{package}=={package_version}"
                )
            continue
        if package_version:
            package_specifier = constraints[package]
            if not package_specifier:
                continue
            if packaging_version.Version(package_version) not in package_specifier:
                result[_WARNINGS_KEY][_WARNING_INCOMPATIBLE].add(
                    f"{package}=={package_version} (required: {str(package_specifier)})"
                )
    return result


def scan_requirements(
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
        ignore_modules = _BASE_MODULES
    if package_distributions is None:
        package_distributions = _PACKAGE_DISTRIBUTIONS
    modules_found = set(_DEFAULT_REQUIRED_PACKAGES)
    inspect_getmembers_kwargs = inspect_getmembers_kwargs or {}
    for _, attr in inspect.getmembers(obj, **inspect_getmembers_kwargs):
        if not attr or inspect.isbuiltin(attr) or not hasattr(attr, "__module__"):
            continue
        module_name = (attr.__module__ or "").split(".")[0]
        if module_name and module_name not in ignore_modules:
            for module in package_distributions.get(module_name, []):
                modules_found.add(module)
    return {module: importlib_metadata.version(module) for module in modules_found}


def _is_pydantic_serializable(param: inspect.Parameter) -> bool:
    """Checks if the parameter is pydantic serializable."""

    if param.annotation == inspect.Parameter.empty:
        return True

    if isinstance(param.annotation, str):
        return False
    pydantic = _import_pydantic_or_raise()
    try:
        pydantic.TypeAdapter(param.annotation)
        return True
    except Exception:
        return False


def generate_schema(
    f: Callable[..., Any],
    *,
    schema_name: Optional[str] = None,
    descriptions: Mapping[str, str] = {},
    required: Sequence[str] = [],
) -> JsonDict:
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
            (param.annotation if param.annotation != inspect.Parameter.empty else Any),
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
        and _is_pydantic_serializable(param)
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


def is_noop_or_proxy_tracer_provider(tracer_provider) -> bool:
    """Returns True if the tracer_provider is Proxy or NoOp."""
    opentelemetry = _import_opentelemetry_or_warn()
    ProxyTracerProvider = opentelemetry.trace.ProxyTracerProvider
    NoOpTracerProvider = opentelemetry.trace.NoOpTracerProvider
    return isinstance(tracer_provider, (NoOpTracerProvider, ProxyTracerProvider))


def dump_event_for_json(event: BaseModel) -> Dict[str, Any]:
    """Dumps an ADK event to a JSON-serializable dictionary."""
    return json.loads(event.model_dump_json(exclude_none=True))


def _import_cloud_storage_or_raise() -> types.ModuleType:
    """Tries to import the Cloud Storage module."""
    try:
        from google.cloud import storage
    except ImportError as e:
        raise ImportError(
            "Cloud Storage is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return storage


def _import_cloudpickle_or_raise() -> types.ModuleType:
    """Tries to import the cloudpickle module."""
    try:
        import cloudpickle  # noqa:F401
    except ImportError as e:
        raise ImportError(
            "cloudpickle is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return cloudpickle


def _import_pydantic_or_raise() -> types.ModuleType:
    """Tries to import the pydantic module."""
    try:
        import pydantic

        _ = pydantic.Field
    except AttributeError:
        from pydantic import v1 as pydantic
    except ImportError as e:
        raise ImportError(
            "pydantic is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return pydantic


def _import_packaging_requirements_or_raise() -> types.ModuleType:
    """Tries to import the packaging.requirements module."""
    try:
        from packaging import requirements
    except ImportError as e:
        raise ImportError(
            "packaging.requirements is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return requirements


def _import_packaging_version_or_raise() -> types.ModuleType:
    """Tries to import the packaging.requirements module."""
    try:
        from packaging import version
    except ImportError as e:
        raise ImportError(
            "packaging.version is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        ) from e
    return version


def _import_opentelemetry_or_warn() -> Optional[types.ModuleType]:
    """Tries to import the opentelemetry module."""
    try:
        import opentelemetry  # noqa:F401

        return opentelemetry
    except ImportError:
        LOGGER.warning(
            "opentelemetry-sdk is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        )
    return None


def _import_opentelemetry_sdk_trace_or_warn() -> Optional[types.ModuleType]:
    """Tries to import the opentelemetry.sdk.trace module."""
    try:
        import opentelemetry.sdk.trace  # noqa:F401

        return opentelemetry.sdk.trace
    except ImportError:
        LOGGER.warning(
            "opentelemetry-sdk is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        )
    return None


def _import_cloud_trace_v2_or_warn() -> Optional[types.ModuleType]:
    """Tries to import the google.cloud.trace_v2 module."""
    try:
        import google.cloud.trace_v2

        return google.cloud.trace_v2
    except ImportError:
        LOGGER.warning(
            "google-cloud-trace is not installed. Please call "
            "'pip install google-cloud-aiplatform[agent_engines]'."
        )
    return None


def _import_cloud_trace_exporter_or_warn() -> Optional[types.ModuleType]:
    """Tries to import the opentelemetry.exporter.cloud_trace module."""
    try:
        import opentelemetry.exporter.cloud_trace  # noqa:F401

        return opentelemetry.exporter.cloud_trace
    except ImportError:
        LOGGER.warning(
            "opentelemetry-exporter-gcp-trace is not installed. Please "
            "call 'pip install google-cloud-aiplatform[agent_engines]'."
        )
    return None


def _import_openinference_langchain_or_warn() -> Optional[types.ModuleType]:
    """Tries to import the openinference.instrumentation.langchain module."""
    try:
        import openinference.instrumentation.langchain  # noqa:F401

        return openinference.instrumentation.langchain
    except ImportError:
        LOGGER.warning(
            "openinference-instrumentation-langchain is not installed. Please "
            "call 'pip install google-cloud-aiplatform[langchain]'."
        )
    return None


def _import_openinference_autogen_or_warn() -> Optional[types.ModuleType]:
    """Tries to import the openinference.instrumentation.autogen module."""
    try:
        import openinference.instrumentation.autogen  # noqa:F401

        return openinference.instrumentation.autogen
    except ImportError:
        LOGGER.warning(
            "openinference-instrumentation-autogen is not installed. Please "
            "call 'pip install google-cloud-aiplatform[ag2]'."
        )
    return None


def _import_autogen_tools_or_warn() -> Optional[types.ModuleType]:
    """Tries to import the autogen.tools module."""
    try:
        from autogen import tools

        return tools
    except ImportError:
        LOGGER.warning(
            "autogen.tools is not installed. Please "
            "call `pip install google-cloud-aiplatform[ag2]`."
        )
    return None


def validate_installation_scripts_or_raise(
    script_paths: Sequence[str],
    extra_packages: Sequence[str],
):
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
            LOGGER.warning(
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
            LOGGER.warning(
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
            LOGGER.warning(
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
