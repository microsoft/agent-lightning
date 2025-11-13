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
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import asyncio
from collections.abc import Awaitable
import queue
import threading
import warnings

if TYPE_CHECKING:
    try:
        from google.adk.events.event import Event

        Event = Event
    except (ImportError, AttributeError):
        Event = Any

    try:
        from google.adk.apps import App

        App = App
    except (ImportError, AttributeError):
        App = Any

    try:
        from google.adk.agents import BaseAgent

        BaseAgent = BaseAgent
    except (ImportError, AttributeError):
        BaseAgent = Any

    try:
        from google.adk.plugins.base_plugin import BasePlugin

        BasePlugin = BasePlugin
    except (ImportError, AttributeError):
        BasePlugin = Any

    try:
        from google.adk.sessions import BaseSessionService

        BaseSessionService = BaseSessionService
    except (ImportError, AttributeError):
        BaseSessionService = Any

    try:
        from google.adk.artifacts import BaseArtifactService

        BaseArtifactService = BaseArtifactService
    except (ImportError, AttributeError):
        BaseArtifactService = Any

    try:
        from google.adk.memory import BaseMemoryService

        BaseMemoryService = BaseMemoryService
    except (ImportError, AttributeError):
        BaseMemoryService = Any

    try:
        from opentelemetry.sdk import trace

        TracerProvider = trace.TracerProvider
        SpanProcessor = trace.SpanProcessor
        SynchronousMultiSpanProcessor = trace.SynchronousMultiSpanProcessor
    except (ImportError, AttributeError):
        TracerProvider = Any
        SpanProcessor = Any
        SynchronousMultiSpanProcessor = Any


_DEFAULT_APP_NAME = "default-app-name"
_DEFAULT_USER_ID = "default-user-id"
_TELEMETRY_API_DISABLED_WARNING = (
    "Tracing integration for Agent Engine has migrated to a new API.\n"
    "The 'telemetry.googleapis.com' has not been enabled in project %s. \n"
    "**Impact:** Until this API is enabled, telemetry data will not be stored."
    "\n"
    "**Action:** Please enable the API by visiting "
    "https://console.developers.google.com/apis/api/telemetry.googleapis.com/overview?project=%s."
    "\n"
    "(If you enabled this API recently, you can safely ignore this warning.)"
)


def get_adk_version() -> Optional[str]:
    """Returns the version of the ADK package."""
    try:
        from google.adk import version

        return version.__version__
    except (ImportError, AttributeError):
        return None


def is_version_sufficient(version_to_check: str) -> bool:
    """Compares the existing version of ADK with the required version.

    Args:
        version_to_check: The version string to check.

    Returns:
        True if the existing version is sufficient, False otherwise.
    """
    try:
        from packaging.version import parse

        return parse(get_adk_version()) >= parse(version_to_check)
    except (AttributeError, ImportError):
        return False


class _ArtifactVersion:
    def __init__(self, **kwargs):
        self.version: Optional[str] = kwargs.get("version")
        self.data = kwargs.get("data")

    def dump(self) -> Dict[str, Any]:
        result = {}
        if self.version:
            result["version"] = self.version
        if self.data:
            result["data"] = self.data
        return result


class _Artifact:
    def __init__(self, **kwargs):
        self.file_name: Optional[str] = kwargs.get("file_name")
        self.versions: List[_ArtifactVersion] = kwargs.get("versions")

    def dump(self) -> Dict[str, Any]:
        result = {}
        if self.file_name:
            result["file_name"] = self.file_name
        if self.versions:
            result["versions"] = [version.dump() for version in self.versions]
        return result


class _Authorization:
    def __init__(self, **kwargs):
        self.access_token: Optional[str] = kwargs.get("access_token") or kwargs.get(
            "accessToken"
        )


class _StreamRunRequest:
    """Request object for `streaming_agent_run_with_events` method."""

    def __init__(self, **kwargs):
        from google.adk.events.event import Event
        from google.genai import types

        self.message: Optional[types.Content] = kwargs.get("message")
        # The new message to be processed by the agent.

        self.events: Optional[List[Event]] = kwargs.get("events")
        # List of preceding events happened in the same session.

        self.artifacts: Optional[List[_Artifact]] = kwargs.get("artifacts")
        # List of artifacts belonging to the session.

        self.authorizations: Dict[str, _Authorization] = kwargs.get(
            "authorizations", {}
        )
        # The authorizations of the user, keyed by authorization ID.

        self.user_id: Optional[str] = kwargs.get("user_id") or kwargs.get(
            "userId", _DEFAULT_USER_ID
        )
        # The user ID.

        self.session_id: Optional[str] = kwargs.get("session_id") or kwargs.get(
            "sessionId"
        )
        # The session ID.


class _StreamingRunResponse:
    """Response object for `streaming_agent_run_with_events` method.

    It contains the generated events together with the belonging artifacts.
    """

    def __init__(self, **kwargs):
        self.events: Optional[List["Event"]] = kwargs.get("events")
        # List of generated events.
        self.artifacts: Optional[List[_Artifact]] = kwargs.get("artifacts")
        # List of artifacts belonging to the session.
        self.session_id: Optional[str] = kwargs.get("session_id")
        # The session ID.

    def dump(self) -> Dict[str, Any]:
        from vertexai.agent_engines import _utils

        result = {}
        if self.events:
            result["events"] = []
            for event in self.events:
                event_dict = _utils.dump_event_for_json(event)
                event_dict["invocation_id"] = event_dict.get("invocation_id", "")
                result["events"].append(event_dict)
        if self.artifacts:
            result["artifacts"] = [artifact.dump() for artifact in self.artifacts]
        if self.session_id:
            result["session_id"] = self.session_id
        return result


def _warn(msg: str):
    if not hasattr(_warn, "_LOGGER"):
        from google.cloud.aiplatform import base

        _warn._LOGGER = base.Logger(
            __name__
        )  # pyright: ignore[reportFunctionMemberAccess]

    _warn._LOGGER.warning(msg)  # pyright: ignore[reportFunctionMemberAccess]


async def _force_flush_otel(tracing_enabled: bool, logging_enabled: bool):
    try:
        import opentelemetry.trace
        import opentelemetry._logs
    except (ImportError, AttributeError):
        _warn(
            "Could not force flush telemetry data. opentelemetry-api is not installed. Please call  'pip install google-cloud-aiplatform[agent_engines]'."
        )
        return None

    try:
        import opentelemetry.sdk.trace
        import opentelemetry.sdk._logs
    except (ImportError, AttributeError):
        _warn(
            "Could not force flush telemetry data. opentelemetry-sdk is not installed. Please call  'pip install google-cloud-aiplatform[agent_engines]'."
        )
        return None

    coros: List[Awaitable[bool]] = []

    if tracing_enabled:
        tracer_provider = opentelemetry.trace.get_tracer_provider()
        if isinstance(tracer_provider, opentelemetry.sdk.trace.TracerProvider):
            coros.append(asyncio.to_thread(tracer_provider.force_flush))

    if logging_enabled:
        logger_provider = opentelemetry._logs.get_logger_provider()
        if isinstance(logger_provider, opentelemetry.sdk._logs.LoggerProvider):
            coros.append(asyncio.to_thread(logger_provider.force_flush))

    await asyncio.gather(*coros, return_exceptions=True)


def _default_instrumentor_builder(
    project_id: str,
    *,
    enable_tracing: bool = False,
    enable_logging: bool = False,
):
    if not enable_tracing and not enable_logging:
        return None

    import os

    def _warn_missing_dependency(
        package: str,
        *,
        needed_for_logging: bool = False,
        needed_for_tracing: bool = False,
    ) -> None:
        _warn(
            f"{package} is not installed. Please call 'pip install google-cloud-aiplatform[agent_engines]'."
        )
        MISSING_TRACE_IMPORT_ERROR_MESSAGE = "proceeding with tracing disabled because not all packages (i.e. `google-cloud-trace`, `opentelemetry-sdk`, `opentelemetry-exporter-gcp-trace`) for tracing have been installed"
        MISSING_LOGGING_IMPORT_ERROR_MESSAGE = "proceeding with logging disabled because not all packages (i.e. `google-cloud-logging`, `opentelemetry-sdk`, `opentelemetry-exporter-gcp-logging`) for tracing have been installed"

        if needed_for_tracing and enable_tracing:
            _warn(MISSING_TRACE_IMPORT_ERROR_MESSAGE)
        if needed_for_logging and enable_logging:
            _warn(MISSING_LOGGING_IMPORT_ERROR_MESSAGE)
        return None

    def _detect_cloud_resource_id(project_id: str) -> Optional[str]:
        location = os.getenv("GOOGLE_CLOUD_LOCATION", None)
        agent_engine_id = os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", None)
        if all(v is not None for v in (location, agent_engine_id)):
            return f"//aiplatform.googleapis.com/projects/{project_id}/locations/{location}/reasoningEngines/{agent_engine_id}"
        return None

    try:
        import opentelemetry
        import opentelemetry.trace
        import opentelemetry._logs
        import opentelemetry._events
    except (ImportError, AttributeError):
        return _warn_missing_dependency(
            "opentelemetry-api", needed_for_tracing=True, needed_for_logging=True
        )

    try:
        import opentelemetry.sdk.resources
        import opentelemetry.sdk.trace
        import opentelemetry.sdk.trace.export
        import opentelemetry.sdk._logs
        import opentelemetry.sdk._logs.export
        import opentelemetry.sdk._events
    except (ImportError, AttributeError):
        return _warn_missing_dependency(
            "opentelemetry-sdk", needed_for_tracing=True, needed_for_logging=True
        )

    import uuid

    # Provide a set of resource attributes but allow to override them with env
    # variables like OTEL_RESOURCE_ATTRIBUTES and OTEL_SERVICE_NAME.
    cloud_resource_id = _detect_cloud_resource_id(project_id)
    resource = opentelemetry.sdk.resources.Resource.create(
        attributes={
            "gcp.project_id": project_id,
            "cloud.account.id": project_id,
            "cloud.platform": "gcp.agent_engine",
            "service.name": os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", ""),
            "service.instance.id": f"{uuid.uuid4().hex}-{os.getpid()}",
            "cloud.region": os.getenv("GOOGLE_CLOUD_LOCATION", ""),
        }
        | (
            {"cloud.resource_id": cloud_resource_id}
            if cloud_resource_id is not None
            else {}
        )
    ).merge(opentelemetry.sdk.resources.OTELResourceDetector().detect())

    if enable_tracing:
        try:
            import opentelemetry.exporter.otlp.proto.http.version
            import opentelemetry.exporter.otlp.proto.http.trace_exporter
            import google.auth.transport.requests
            from google.cloud.aiplatform import version as aip_version
        except (ImportError, AttributeError):
            return _warn_missing_dependency(
                "opentelemetry-exporter-otlp-proto-http", needed_for_tracing=True
            )

        import google.auth

        credentials, _ = google.auth.default()
        vertex_sdk_version = aip_version.__version__
        otlp_http_version = opentelemetry.exporter.otlp.proto.http.version.__version__
        user_agent = f"Vertex-Agent-Engine/{vertex_sdk_version} OTel-OTLP-Exporter-Python/{otlp_http_version}"

        span_exporter = (
            opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter(
                session=google.auth.transport.requests.AuthorizedSession(
                    credentials=credentials
                ),
                endpoint="https://telemetry.googleapis.com/v1/traces",
                headers={"User-Agent": user_agent},
            )
        )
        span_processor = opentelemetry.sdk.trace.export.BatchSpanProcessor(
            span_exporter=span_exporter,
        )
        tracer_provider = opentelemetry.trace.get_tracer_provider()
        # Get the appropriate tracer provider:
        # 1. If _TRACER_PROVIDER is already set, use that.
        # 2. Otherwise, if the OTEL_PYTHON_TRACER_PROVIDER environment
        # variable is set, use that.
        # 3. As a final fallback, use _PROXY_TRACER_PROVIDER.
        # If none of the above is set, we log a warning, and
        # create a tracer provider.
        if not tracer_provider:
            _warn(
                "No tracer provider. By default, "
                "we should get one of the following providers: "
                "OTEL_PYTHON_TRACER_PROVIDER, _TRACER_PROVIDER, "
                "or _PROXY_TRACER_PROVIDER."
            )
            tracer_provider = opentelemetry.sdk.trace.TracerProvider(resource=resource)
            opentelemetry.trace.set_tracer_provider(tracer_provider)
        # Avoids AttributeError:
        # 'ProxyTracerProvider' and 'NoOpTracerProvider' objects has no
        # attribute 'add_span_processor'.
        from vertexai.agent_engines import _utils

        if _utils.is_noop_or_proxy_tracer_provider(tracer_provider):
            tracer_provider = opentelemetry.sdk.trace.TracerProvider(resource=resource)
            opentelemetry.trace.set_tracer_provider(tracer_provider)
        # Avoids OpenTelemetry client already exists error.
        _override_active_span_processor(
            tracer_provider,
            opentelemetry.sdk.trace.SynchronousMultiSpanProcessor(),
        )
        tracer_provider.add_span_processor(span_processor)

    if enable_logging:
        try:
            import opentelemetry.exporter.cloud_logging
        except (ImportError, AttributeError):
            return _warn_missing_dependency(
                "opentelemetry-exporter-gcp-logging", needed_for_logging=True
            )

        logger_provider = opentelemetry.sdk._logs.LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            opentelemetry.sdk._logs.export.BatchLogRecordProcessor(
                opentelemetry.exporter.cloud_logging.CloudLoggingExporter(
                    project_id=project_id,
                    default_log_name=os.getenv(
                        "GCP_DEFAULT_LOG_NAME", "adk-on-agent-engine"
                    ),
                ),
            )
        )
        event_logger_provider = opentelemetry.sdk._events.EventLoggerProvider(
            logger_provider=logger_provider
        )

        opentelemetry._logs.set_logger_provider(logger_provider=logger_provider)
        opentelemetry._events.set_event_logger_provider(
            event_logger_provider=event_logger_provider
        )

    try:
        from opentelemetry.instrumentation import google_genai

        google_genai.GoogleGenAiSdkInstrumentor().instrument()
    except (ImportError, AttributeError):
        _warn(
            "telemetry enabled but proceeding without GenAI instrumentation, because not all packages (i.e. opentelemetry-instrumentation-google-genai) have been installed"
        )

    return None


def _override_active_span_processor(
    tracer_provider: "TracerProvider",
    active_span_processor: "SynchronousMultiSpanProcessor",
):
    """Overrides the active span processor.

    When working with multiple LangchainAgents in the same environment,
    it's crucial to manage trace exports carefully.
    Each agent needs its own span processor tied to a unique project ID.
    While we add a new span processor for each agent, this can lead to
    unexpected behavior.
    For instance, with two agents linked to different projects, traces from the
    second agent might be sent to both projects.
    To prevent this and guarantee traces go to the correct project, we overwrite
    the active span processor whenever a new LangchainAgent is created.

    Args:
        tracer_provider (TracerProvider):
            The tracer provider to use for the project.
        active_span_processor (SynchronousMultiSpanProcessor):
            The active span processor overrides the tracer provider's
            active span processor.
    """
    if tracer_provider._active_span_processor:
        tracer_provider._active_span_processor.shutdown()
    tracer_provider._active_span_processor = active_span_processor


def _validate_run_config(run_config: Optional[Dict[str, Any]]):
    """Validates the run config."""
    from google.adk.agents.run_config import RunConfig

    if run_config is None:
        return None
    elif isinstance(run_config, Dict):
        return RunConfig.model_validate(run_config)
    raise TypeError("run_config must be a dictionary representing a RunConfig object.")


class AdkApp:
    """An ADK Application."""

    agent_framework = "google-adk"

    def __init__(
        self,
        *,
        app: "App" = None,
        agent: "BaseAgent" = None,
        app_name: Optional[str] = None,
        plugins: Optional[List["BasePlugin"]] = None,
        enable_tracing: Optional[bool] = None,
        session_service_builder: Optional[Callable[..., "BaseSessionService"]] = None,
        artifact_service_builder: Optional[Callable[..., "BaseArtifactService"]] = None,
        memory_service_builder: Optional[Callable[..., "BaseMemoryService"]] = None,
        instrumentor_builder: Optional[Callable[..., Any]] = None,
    ):
        """An ADK Application.

        See https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/develop/adk
        for details on how to develop ADK applications on Agent Engine.

        Args:
            agent (google.adk.agents.BaseAgent):
                Required. The ADK agent to run.
            app_name (str):
                Optional. The name of the ADK application. Defaults to
                "default-app-name" when running locally, and to the
                corresponding agent engine ID when deployed on Agent Engine.
            plugins (List[BasePlugin]):
                Optional. The plugins to use for the ADK application.
                Defaults to an empty list.
            enable_tracing (bool):
                Optional. Whether to enable tracing in Cloud Trace. Defaults to
                False.
            session_service_builder (Callable[..., BaseSessionService]):
                Optional. A callable that returns an ADK session service.
                Defaults to a callable that returns InMemorySessionService
                when running locally and VertexAiSessionService when running
                on Agent Engine.
            artifact_service_builder (Callable[..., BaseArtifactService]):
                Optional. A callable that returns an ADK artifact service.
                Defaults to a callable that returns InMemoryArtifactService.
            memory_service_builder (Callable[..., BaseMemoryService]):
                Optional. A callable that returns an ADK memory service.
                Defaults to a callable that returns InMemoryMemoryService
                when running locally and VertexAiMemoryBankService when running
                on Agent Engine.
            instrumentor_builder (Callable[..., Any]):
                Optional. Callable that returns a new instrumentor. This can be
                used for customizing the instrumentation logic of the Agent.
                If not provided, a default instrumentor builder will be used.
                This parameter is ignored if `enable_tracing` is False.
        """
        import os
        from google.cloud.aiplatform import initializer

        adk_version = get_adk_version()
        if not is_version_sufficient("1.5.0"):
            msg = (
                f"Unsupported google-adk version: {adk_version}, please use "
                "google-adk>=1.5.0 for AdkApp deployment on Agent Engine."
            )
            raise ValueError(msg)

        if not agent and not app:
            raise ValueError("One of `agent` or `app` must be provided.")
        if app:
            if app_name:
                raise ValueError(
                    "When app is provided, app_name should not be provided."
                )
            if agent:
                raise ValueError("When app is provided, agent should not be provided.")
            if plugins:
                raise ValueError(
                    "When app is provided, plugins should not be provided and"
                    " should be provided in the app instead."
                )

        self._tmpl_attrs: Dict[str, Any] = {
            "project": initializer.global_config.project,
            "location": initializer.global_config.location,
            "agent": agent,
            "app": app,
            "app_name": app_name,
            "plugins": plugins,
            "enable_tracing": enable_tracing,
            "session_service_builder": session_service_builder,
            "artifact_service_builder": artifact_service_builder,
            "memory_service_builder": memory_service_builder,
            "instrumentor_builder": instrumentor_builder,
            "express_mode_api_key": (
                initializer.global_config.api_key or os.environ.get("GOOGLE_API_KEY")
            ),
        }

    async def _init_session(
        self,
        session_service: "BaseSessionService",
        artifact_service: "BaseArtifactService",
        request: _StreamRunRequest,
    ):
        """Initializes the session, and returns the session id."""
        from google.adk.events.event import Event

        session_state = None
        if request.authorizations:
            session_state = {}
            for auth_id, auth in request.authorizations.items():
                auth = _Authorization(**auth)
                session_state[auth_id] = auth.access_token

        session = await session_service.create_session(
            app_name=self._tmpl_attrs.get("app_name"),
            user_id=request.user_id,
            state=session_state,
        )
        if not session:
            raise RuntimeError("Create session failed.")
        if request.events:
            for event in request.events:
                await session_service.append_event(session, Event(**event))
        if request.artifacts:
            for artifact in request.artifacts:
                artifact = _Artifact(**artifact)
                for version_data in sorted(
                    artifact.versions, key=lambda x: x["version"]
                ):
                    version_data = _ArtifactVersion(**version_data)
                    saved_version = await artifact_service.save_artifact(
                        app_name=self._tmpl_attrs.get("app_name"),
                        user_id=request.user_id,
                        session_id=session.id,
                        filename=artifact.file_name,
                        artifact=version_data.data,
                    )
                    if saved_version != version_data.version:
                        from google.cloud.aiplatform import base

                        _LOGGER = base.Logger(__name__)
                        _LOGGER.debug(
                            "Artifact '%s' saved at version %s instead of %s",
                            artifact.file_name,
                            saved_version,
                            version_data.version,
                        )
        return session

    async def _convert_response_events(
        self,
        user_id: str,
        session_id: str,
        events: List["Event"],
        artifact_service: Optional["BaseArtifactService"],
    ) -> _StreamingRunResponse:
        """Converts the events to the streaming run response object."""
        import collections

        result = _StreamingRunResponse(
            events=events, artifacts=[], session_id=session_id
        )

        # Save the generated artifacts into the result object.
        artifact_versions = collections.defaultdict(list)
        for event in events:
            if event.actions and event.actions.artifact_delta:
                for key, version in event.actions.artifact_delta.items():
                    artifact_versions[key].append(version)

        for key, versions in artifact_versions.items():
            result.artifacts.append(
                _Artifact(
                    file_name=key,
                    versions=[
                        _ArtifactVersion(
                            version=version,
                            data=await artifact_service.load_artifact(
                                app_name=self._tmpl_attrs.get("app_name"),
                                user_id=user_id,
                                session_id=session_id,
                                filename=key,
                                version=version,
                            ),
                        )
                        for version in versions
                    ],
                )
            )

        return result.dump()

    def clone(self):
        """Returns a clone of the ADK application."""
        import copy

        return self.__class__(
            app=copy.deepcopy(self._tmpl_attrs.get("app")),
            enable_tracing=self._tmpl_attrs.get("enable_tracing"),
            agent=(
                None
                if self._tmpl_attrs.get("app")
                else copy.deepcopy(self._tmpl_attrs.get("agent"))
            ),
            app_name=(
                None
                if self._tmpl_attrs.get("app")
                else self._tmpl_attrs.get("app_name")
            ),
            plugins=(
                None
                if self._tmpl_attrs.get("app")
                else copy.deepcopy(self._tmpl_attrs.get("plugins"))
            ),
            session_service_builder=self._tmpl_attrs.get("session_service_builder"),
            artifact_service_builder=self._tmpl_attrs.get("artifact_service_builder"),
            memory_service_builder=self._tmpl_attrs.get("memory_service_builder"),
            instrumentor_builder=self._tmpl_attrs.get("instrumentor_builder"),
        )

    def set_up(self):
        """Sets up the ADK application."""
        import os
        from google.adk.runners import Runner
        from google.adk.sessions.in_memory_session_service import InMemorySessionService
        from google.adk.artifacts.in_memory_artifact_service import (
            InMemoryArtifactService,
        )
        from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
        project = self._tmpl_attrs.get("project")
        if project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project
        location = self._tmpl_attrs.get("location")
        if location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = location
        express_mode_api_key = self._tmpl_attrs.get("express_mode_api_key")
        if express_mode_api_key and not project:
            os.environ["GOOGLE_API_KEY"] = express_mode_api_key
            # Clear location and project env vars if express mode api key is provided.
            os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            location = None

        # Disable content capture in custom ADK spans unless user enabled
        # tracing explicitly with the old flag
        # (this is to preserve compatibility with old behavior).
        if self._tmpl_attrs.get("enable_tracing"):
            os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"
        else:
            os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "false"

        enable_logging = bool(self._telemetry_enabled())

        custom_instrumentor = self._tmpl_attrs.get("instrumentor_builder")

        if self._tmpl_attrs.get("enable_tracing"):
            self._warn_if_telemetry_api_disabled()

        if self._tmpl_attrs.get("enable_tracing") is False:
            _warn(
                (
                    "Your 'enable_tracing=False' setting is being deprecated "
                    "and will be removed in a future release.\n"
                    "This legacy setting overrides the new Cloud Console "
                    "toggle and environment variable controls.\n"
                    "Impact: The Cloud Console may incorrectly show telemetry "
                    "as 'On' when it is actually 'Off', and the UI toggle will "
                    "not work.\n"
                    "Action: To fix this and control telemetry, please remove "
                    "the 'enable_tracing' parameter from your deployment "
                    "code.\n"
                    "You can then use the "
                    "'GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY' "
                    "environment variable:\n"
                    "agent_engines.create(\n"
                    "  env_vars={\n"
                    '    "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": true|false\n'
                    "  }\n"
                    ")\n"
                    "or the toggle in the Cloud Console: "
                    "https://console.cloud.google.com/vertex-ai/agents."
                ),
            )

        if custom_instrumentor and self._tracing_enabled():
            self._tmpl_attrs["instrumentor"] = custom_instrumentor(project)

        if not custom_instrumentor:
            self._tmpl_attrs["instrumentor"] = _default_instrumentor_builder(
                project,
                enable_tracing=self._tracing_enabled(),
                enable_logging=enable_logging,
            )

        if not self._tmpl_attrs.get("app_name"):
            if "GOOGLE_CLOUD_AGENT_ENGINE_ID" in os.environ:
                self._tmpl_attrs["app_name"] = os.environ.get(
                    "GOOGLE_CLOUD_AGENT_ENGINE_ID",
                )
            else:
                self._tmpl_attrs["app_name"] = _DEFAULT_APP_NAME

        artifact_service_builder = self._tmpl_attrs.get("artifact_service_builder")
        if artifact_service_builder:
            self._tmpl_attrs["artifact_service"] = artifact_service_builder()
        else:
            self._tmpl_attrs["artifact_service"] = InMemoryArtifactService()

        session_service_builder = self._tmpl_attrs.get("session_service_builder")
        if session_service_builder:
            self._tmpl_attrs["session_service"] = session_service_builder()
        elif "GOOGLE_CLOUD_AGENT_ENGINE_ID" in os.environ:
            try:
                from google.adk.sessions.vertex_ai_session_service import (
                    VertexAiSessionService,
                )

                # If the express mode api key is set, it will be read from the
                # environment variable when initializing the session service.
                self._tmpl_attrs["session_service"] = VertexAiSessionService(
                    project=project,
                    location=location,
                    agent_engine_id=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"),
                )
            except (ImportError, AttributeError):
                from google.adk.sessions.vertex_ai_session_service_g3 import (
                    VertexAiSessionService,
                )

                # If the express mode api key is set, it will be read from the
                # environment variable when initializing the session service.
                self._tmpl_attrs["session_service"] = VertexAiSessionService(
                    project=project,
                    location=location,
                    agent_engine_id=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"),
                )

        else:
            self._tmpl_attrs["session_service"] = InMemorySessionService()

        memory_service_builder = self._tmpl_attrs.get("memory_service_builder")
        if memory_service_builder:
            self._tmpl_attrs["memory_service"] = memory_service_builder()
        elif "GOOGLE_CLOUD_AGENT_ENGINE_ID" in os.environ and is_version_sufficient(
            "1.5.0"
        ):
            try:
                from google.adk.memory.vertex_ai_memory_bank_service import (
                    VertexAiMemoryBankService,
                )

                # If the express mode api key is set, it will be read from the
                # environment variable when initializing the memory service.
                self._tmpl_attrs["memory_service"] = VertexAiMemoryBankService(
                    project=project,
                    location=location,
                    agent_engine_id=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"),
                )
            except (ImportError, AttributeError):
                # TODO(ysian): Handle this via _g3 import for google3.
                pass
        else:
            self._tmpl_attrs["memory_service"] = InMemoryMemoryService()

        self._tmpl_attrs["runner"] = Runner(
            app=self._tmpl_attrs.get("app"),
            agent=(
                None if self._tmpl_attrs.get("app") else self._tmpl_attrs.get("agent")
            ),
            app_name=(
                None
                if self._tmpl_attrs.get("app")
                else self._tmpl_attrs.get("app_name")
            ),
            plugins=(
                None if self._tmpl_attrs.get("app") else self._tmpl_attrs.get("plugins")
            ),
            session_service=self._tmpl_attrs.get("session_service"),
            artifact_service=self._tmpl_attrs.get("artifact_service"),
            memory_service=self._tmpl_attrs.get("memory_service"),
        )
        self._tmpl_attrs["in_memory_session_service"] = InMemorySessionService()
        self._tmpl_attrs["in_memory_artifact_service"] = InMemoryArtifactService()
        self._tmpl_attrs["in_memory_memory_service"] = InMemoryMemoryService()
        self._tmpl_attrs["in_memory_runner"] = Runner(
            app=self._tmpl_attrs.get("app"),
            app_name=(
                None
                if self._tmpl_attrs.get("app")
                else self._tmpl_attrs.get("app_name")
            ),
            agent=(
                None if self._tmpl_attrs.get("app") else self._tmpl_attrs.get("agent")
            ),
            plugins=(
                None if self._tmpl_attrs.get("app") else self._tmpl_attrs.get("plugins")
            ),
            session_service=self._tmpl_attrs.get("in_memory_session_service"),
            artifact_service=self._tmpl_attrs.get("in_memory_artifact_service"),
            memory_service=self._tmpl_attrs.get("in_memory_memory_service"),
        )

    async def async_stream_query(
        self,
        *,
        message: Union[str, Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterable[Dict[str, Any]]:
        """Streams responses asynchronously from the ADK application.

        Args:
            message (str):
                Required. The message to stream responses for.
            user_id (str):
                Required. The ID of the user.
            session_id (str):
                Optional. The ID of the session. If not provided, a new
                session will be created for the user.
            run_config (Optional[Dict[str, Any]]):
                Optional. The run config to use for the query. If you want to
                pass in a `run_config` pydantic object, you can pass in a dict
                representing it as `run_config.model_dump(mode="json")`.
            **kwargs (dict[str, Any]):
                Optional. Additional keyword arguments to pass to the
                runner.

        Yields:
            Event dictionaries asynchronously.
        """
        from vertexai.agent_engines import _utils
        from google.genai import types

        if isinstance(message, Dict):
            content = types.Content.model_validate(message)
        elif isinstance(message, str):
            content = types.Content(role="user", parts=[types.Part(text=message)])
        else:
            raise TypeError(
                "message must be a string or a dictionary representing"
                " a Content object."
            )

        if not self._tmpl_attrs.get("runner"):
            self.set_up()
        if not session_id:
            session = await self.async_create_session(user_id=user_id)
            session_id = session.id

        run_config = _validate_run_config(run_config)
        if run_config:
            events_async = self._tmpl_attrs.get("runner").run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                run_config=run_config,
                **kwargs,
            )
        else:
            events_async = self._tmpl_attrs.get("runner").run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                **kwargs,
            )

        try:
            async for event in events_async:
                # Yield the event data as a dictionary
                yield _utils.dump_event_for_json(event)
        finally:
            # Avoid telemetry data loss having to do with CPU throttling on instance turndown
            _ = await _force_flush_otel(
                tracing_enabled=self._tracing_enabled(),
                logging_enabled=bool(self._telemetry_enabled()),
            )

    def stream_query(
        self,
        *,
        message: Union[str, Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Deprecated. Use async_stream_query instead.

        Streams responses from the ADK application in response to a message.

        Args:
            message (Union[str, Dict[str, Any]]):
                Required. The message to stream responses for.
            user_id (str):
                Required. The ID of the user.
            session_id (str):
                Optional. The ID of the session. If not provided, a new
                session will be created for the user.
            run_config (Optional[Dict[str, Any]]):
                Optional. The run config to use for the query. If you want to
                pass in a `run_config` pydantic object, you can pass in a dict
                representing it as `run_config.model_dump(mode="json")`.
            **kwargs (dict[str, Any]):
                Optional. Additional keyword arguments to pass to the
                runner.

        Yields:
            The output of querying the ADK application.
        """
        warnings.warn(
            (
                "AdkApp.stream_query(...) is deprecated. "
                "Use AdkApp.async_stream_query(...) instead. See "
                "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/use/adk#stream-responses "
                "for more details."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        from vertexai.agent_engines import _utils
        from google.genai import types

        if isinstance(message, Dict):
            content = types.Content.model_validate(message)
        elif isinstance(message, str):
            content = types.Content(role="user", parts=[types.Part(text=message)])
        else:
            raise TypeError(
                "message must be a string or a dictionary representing"
                " a Content object."
            )

        if not self._tmpl_attrs.get("runner"):
            self.set_up()
        if not session_id:
            session = self.create_session(user_id=user_id)
            session_id = session.id
        run_config = _validate_run_config(run_config)
        if run_config:
            for event in self._tmpl_attrs.get("runner").run(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                run_config=run_config,
                **kwargs,
            ):
                yield _utils.dump_event_for_json(event)
        else:
            for event in self._tmpl_attrs.get("runner").run(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                **kwargs,
            ):
                yield _utils.dump_event_for_json(event)

    async def streaming_agent_run_with_events(self, request_json: str):
        """Streams responses asynchronously from the ADK application.

        In general, you should use `async_stream_query` instead, as it has a
        more structured API and works with the respective ADK services that
        you have defined for the AdkApp. This method is primarily meant for
        invocation from AgentSpace.

        Args:
            request_json (str):
                Required. The request to stream responses for.
        """

        import json
        from google.genai import types
        from google.genai.errors import ClientError

        request = _StreamRunRequest(**json.loads(request_json))
        if not self._tmpl_attrs.get("in_memory_runner"):
            self.set_up()
        if not self._tmpl_attrs.get("runner"):
            self.set_up()
        # Prepare the in-memory session.
        if not self._tmpl_attrs.get("in_memory_artifact_service"):
            self.set_up()
        if not self._tmpl_attrs.get("artifact_service"):
            self.set_up()
        if not self._tmpl_attrs.get("in_memory_session_service"):
            self.set_up()
        if not self._tmpl_attrs.get("session_service"):
            self.set_up()
        app = self._tmpl_attrs.get("app")

        # Try to get the session, if it doesn't exist, create a new one.
        if request.session_id:
            session_service = self._tmpl_attrs.get("session_service")
            artifact_service = self._tmpl_attrs.get("artifact_service")
            runner = self._tmpl_attrs.get("runner")
            try:
                session = await session_service.get_session(
                    app_name=app.name if app else self._tmpl_attrs.get("app_name"),
                    user_id=request.user_id,
                    session_id=request.session_id,
                )
            except ClientError:
                #  Fall back to create session if the session is not found.
                #  Specifying session_id on creation is not supported,
                #  so session id will be regenerated.
                session = await self._init_session(
                    session_service=session_service,
                    artifact_service=artifact_service,
                    request=request,
                )
        else:
            # Not providing a session ID will create a new in-memory session.
            session_service = self._tmpl_attrs.get("in_memory_session_service")
            artifact_service = self._tmpl_attrs.get("in_memory_artifact_service")
            runner = self._tmpl_attrs.get("in_memory_runner")
            session = await self._init_session(
                session_service=session_service,
                artifact_service=artifact_service,
                request=request,
            )
        if not session:
            raise RuntimeError("Session initialization failed.")

        # Run the agent
        message_for_agent = types.Content(**request.message)
        try:
            async for event in runner.run_async(
                user_id=request.user_id,
                session_id=session.id,
                new_message=message_for_agent,
            ):
                converted_event = await self._convert_response_events(
                    user_id=request.user_id,
                    session_id=session.id,
                    events=[event],
                    artifact_service=artifact_service,
                )
                yield converted_event
        finally:
            if session and not request.session_id:
                app = self._tmpl_attrs.get("app")
                await session_service.delete_session(
                    app_name=app.name if app else self._tmpl_attrs.get("app_name"),
                    user_id=request.user_id,
                    session_id=session.id,
                )
            # Avoid telemetry data loss having to do with CPU throttling on instance turndown
            _ = await _force_flush_otel(
                tracing_enabled=self._tracing_enabled(),
                logging_enabled=bool(self._telemetry_enabled()),
            )

    async def async_get_session(
        self,
        *,
        user_id: str,
        session_id: str,
        **kwargs,
    ):
        """Get a session for the given user.

        Args:
            user_id (str):
                Required. The ID of the user.
            session_id (str):
                Required. The ID of the session.
            **kwargs (dict[str, Any]):
                Optional. Additional keyword arguments to pass to the
                session service.

        Returns:
            Session: The session instance (if any). It returns None if the
            session is not found.

        Raises:
            RuntimeError: If the session is not found.
        """
        if not self._tmpl_attrs.get("session_service"):
            self.set_up()
        app = self._tmpl_attrs.get("app")
        session = await self._tmpl_attrs.get("session_service").get_session(
            app_name=app.name if app else self._tmpl_attrs.get("app_name"),
            user_id=user_id,
            session_id=session_id,
            **kwargs,
        )
        if not session:
            raise RuntimeError(
                "Session not found. Please create it using .create_session()"
            )
        return session

    def get_session(
        self,
        *,
        user_id: str,
        session_id: str,
        **kwargs,
    ):
        """Deprecated. Use async_get_session instead.

        Get a session for the given user.
        """
        warnings.warn(
            (
                "AdkApp.get_session(...) is deprecated. "
                "Use AdkApp.async_get_session(...) instead. See "
                "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/use/adk#get-session "
                "for more details."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        event_queue = queue.Queue(maxsize=1)

        async def _invoke_async_get_session():
            return await self.async_get_session(
                user_id=user_id, session_id=session_id, **kwargs
            )

        def _asyncio_thread_main():
            try:
                result = asyncio.run(_invoke_async_get_session())
                event_queue.put(result)
            except Exception as e:
                event_queue.put(e)

        thread = threading.Thread(target=_asyncio_thread_main)
        thread.start()

        # Wait for the thread to finish
        thread.join()
        try:
            outcome = event_queue.get(timeout=10)
        except queue.Empty:
            raise RuntimeError(
                "Session not found. Please create it using .create_session()"
            ) from None
        if isinstance(outcome, RuntimeError):
            raise outcome from None
        return outcome

    async def async_list_sessions(self, *, user_id: str, **kwargs):
        """List sessions for the given user.

        Args:
            user_id (str):
                Required. The ID of the user.
            **kwargs (dict[str, Any]):
                Optional. Additional keyword arguments to pass to the
                session service.

        Returns:
            ListSessionsResponse: The list of sessions.
        """
        if not self._tmpl_attrs.get("session_service"):
            self.set_up()
        app = self._tmpl_attrs.get("app")
        return await self._tmpl_attrs.get("session_service").list_sessions(
            app_name=app.name if app else self._tmpl_attrs.get("app_name"),
            user_id=user_id,
            **kwargs,
        )

    def list_sessions(self, *, user_id: str, **kwargs):
        """Deprecated. Use async_list_sessions instead.

        List sessions for the given user.
        """
        warnings.warn(
            (
                "AdkApp.list_sessions(...) is deprecated. "
                "Use AdkApp.async_list_sessions(...) instead. See "
                "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/use/adk#list-sessions "
                "for more details."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        event_queue = queue.Queue()

        async def _invoke_async_list_sessions():
            try:
                response = await self.async_list_sessions(user_id=user_id, **kwargs)
                event_queue.put(response)
            except RuntimeError as e:
                event_queue.put(e)

        def _asyncio_thread_main():
            try:
                asyncio.run(_invoke_async_list_sessions())
            finally:
                event_queue.put(None)

        thread = threading.Thread(target=_asyncio_thread_main)
        thread.start()
        # Wait for the thread to finish
        thread.join()
        try:
            return event_queue.get(timeout=10)
        except queue.Empty:
            raise RuntimeError("Failed to list sessions.") from None

    async def async_create_session(
        self,
        *,
        user_id: str,
        session_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Creates a new session.

        Args:
            user_id (str):
                Required. The ID of the user.
            session_id (str):
                Optional. The ID of the session. If not provided, an ID
                will be be generated for the session.
            state (dict[str, Any]):
                Optional. The initial state of the session.
            **kwargs (dict[str, Any]):
                Optional. Additional keyword arguments to pass to the
                session service.

        Returns:
            Session: The newly created session instance.
        """
        if not self._tmpl_attrs.get("session_service"):
            self.set_up()
        app = self._tmpl_attrs.get("app")
        session = await self._tmpl_attrs.get("session_service").create_session(
            app_name=app.name if app else self._tmpl_attrs.get("app_name"),
            user_id=user_id,
            session_id=session_id,
            state=state,
            **kwargs,
        )
        return session

    def create_session(
        self,
        *,
        user_id: str,
        session_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Deprecated. Use async_create_session instead.

        Creates a new session.
        """
        warnings.warn(
            (
                "AdkApp.create_session(...) is deprecated. "
                "Use AdkApp.async_create_session(...) instead. See "
                "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/use/adk#create-session "
                "for more details."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        event_queue = queue.Queue(maxsize=1)

        async def _invoke_async_create_session():
            return await self.async_create_session(
                user_id=user_id,
                session_id=session_id,
                state=state,
                **kwargs,
            )

        def _asyncio_thread_main():
            try:
                result = asyncio.run(_invoke_async_create_session())
                event_queue.put(result)
            except RuntimeError as e:
                event_queue.put(e)

        thread = threading.Thread(target=_asyncio_thread_main)
        thread.start()
        # Wait for the thread to finish
        thread.join()

        try:
            outcome = event_queue.get(timeout=10)
        except queue.Empty:
            raise RuntimeError("Failed to create session.") from None
        if isinstance(outcome, RuntimeError):
            raise outcome from None
        return outcome

    async def async_delete_session(
        self,
        *,
        user_id: str,
        session_id: str,
        **kwargs,
    ):
        """Deletes a session for the given user.

        Args:
            user_id (str):
                Required. The ID of the user.
            session_id (str):
                Required. The ID of the session.
            **kwargs (dict[str, Any]):
                Optional. Additional keyword arguments to pass to the
                session service.
        """
        if not self._tmpl_attrs.get("session_service"):
            self.set_up()
        app = self._tmpl_attrs.get("app")
        await self._tmpl_attrs.get("session_service").delete_session(
            app_name=app.name if app else self._tmpl_attrs.get("app_name"),
            user_id=user_id,
            session_id=session_id,
            **kwargs,
        )

    def delete_session(
        self,
        *,
        user_id: str,
        session_id: str,
        **kwargs,
    ):
        """Deprecated. Use async_delete_session instead.

        Deletes a session for the given user.
        """
        warnings.warn(
            (
                "AdkApp.delete_session(...) is deprecated. "
                "Use AdkApp.async_delete_session(...) instead. See "
                "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/use/adk#delete-session "
                "for more details."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        event_queue = queue.Queue(maxsize=1)

        async def _invoke_async_delete_session():
            await self.async_delete_session(
                user_id=user_id, session_id=session_id, **kwargs
            )

        def _asyncio_thread_main():
            try:
                asyncio.run(_invoke_async_delete_session())
                event_queue.put(None)
            except RuntimeError as e:
                event_queue.put(e)

        thread = threading.Thread(target=_asyncio_thread_main)
        thread.start()
        # Wait for the thread to finish
        thread.join()

        outcome = event_queue.get(timeout=10)
        if isinstance(outcome, RuntimeError):
            raise outcome from None

    async def async_add_session_to_memory(self, *, session: Dict[str, Any]):
        """Generates memories.

        Args:
            session (Dict[str, Any]):
                Required. The session to use for generating memories. It should
                be a dictionary representing an ADK Session object, e.g.
                session.model_dump(mode="json").
        """
        from google.adk.sessions.session import Session

        if isinstance(session, Dict):
            session = Session.model_validate(session)
        elif not isinstance(session, Session):
            raise TypeError("session must be a Session object.")
        if not session.events:
            # Get the latest version of the session in case it was updated.
            session = await self.async_get_session(
                user_id=session.user_id,
                session_id=session.id,
            )
        if not self._tmpl_attrs.get("memory_service"):
            self.set_up()
        return await self._tmpl_attrs.get("memory_service").add_session_to_memory(
            session=session,
        )

    async def async_search_memory(self, *, user_id: str, query: str):
        """Searches memories for the given user.

        Args:
            user_id: The id of the user.
            query: The query to match the memories on.

        Returns:
            A SearchMemoryResponse containing the matching memories.
        """
        if not self._tmpl_attrs.get("memory_service"):
            self.set_up()
        app = self._tmpl_attrs.get("app")
        return await self._tmpl_attrs.get("memory_service").search_memory(
            app_name=app.name if app else self._tmpl_attrs.get("app_name"),
            user_id=user_id,
            query=query,
        )

    def register_operations(self) -> Dict[str, List[str]]:
        """Registers the operations of the ADK application."""
        return {
            "": [
                "get_session",
                "list_sessions",
                "create_session",
                "delete_session",
            ],
            "async": [
                "async_get_session",
                "async_list_sessions",
                "async_create_session",
                "async_delete_session",
                "async_add_session_to_memory",
                "async_search_memory",
            ],
            "stream": ["stream_query"],
            "async_stream": [
                "async_stream_query",
                "streaming_agent_run_with_events",
            ],
        }

    def _telemetry_enabled(self) -> Optional[bool]:
        """Return status of telemetry enablement depending on enablement env variable.

        In detail:
        - Logging is always enabled when telemetry is enabled.
        - Tracing is enabled depending on the truth table seen in `_tracing_enabled` method, in order to not break existing user enablement.

        Returns:
            True if telemetry is enabled, False if telemetry is disabled, or None
            if telemetry enablement is not set (i.e. old deployments which don't support this env variable).
        """
        import os

        GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY = (
            "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"
        )

        env_value = os.getenv(
            GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY, "unspecified"
        ).lower()

        if env_value in ("true", "1"):
            return True
        if env_value in ("false", "0"):
            return False
        return None

    # Tracing enablement follows truth table:
    def _tracing_enabled(self) -> bool:
        """Tracing enablement follows true table:

        | enable_tracing | enable_telemetry(env) | tracing_actually_enabled |
        |----------------|-----------------------|--------------------------|
        | false          | false                 | false                    |
        | false          | true                  | false                    |
        | false          | None                  | false                    |
        | true           | false                 | false                    |
        | true           | true                  | true                     |
        | true           | None                  | true                     |
        | None(default)  | false                 | false                    |
        | None(default)  | true                  | adk_version >= 1.17      |
        | None(default)  | None                  | false                    |
        """
        enable_tracing: Optional[bool] = self._tmpl_attrs.get("enable_tracing")
        enable_telemetry: Optional[bool] = self._telemetry_enabled()

        return (enable_tracing is True and enable_telemetry is not False) or (
            enable_tracing is None
            and enable_telemetry is True
            and is_version_sufficient("1.17.0")
        )

    def _warn_if_telemetry_api_disabled(self):
        """Warn if telemetry API is disabled."""
        try:
            import google.auth.transport.requests
            import google.auth
        except (ImportError, AttributeError):
            return
        credentials, project = google.auth.default()
        session = google.auth.transport.requests.AuthorizedSession(
            credentials=credentials
        )
        r = session.post("https://telemetry.googleapis.com/v1/traces", data=None)
        if "Telemetry API has not been used in project" in r.text:
            _warn(_TELEMETRY_API_DISABLED_WARNING % (project, project))
