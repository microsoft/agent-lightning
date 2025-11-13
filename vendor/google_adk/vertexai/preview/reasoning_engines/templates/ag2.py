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
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    try:
        from autogen import agentchat

        ConversableAgent = agentchat.ConversableAgent
        ChatResult = agentchat.ChatResult
    except ImportError:
        ConversableAgent = Any

    try:
        from opentelemetry.sdk import trace

        TracerProvider = trace.TracerProvider
        SpanProcessor = trace.SpanProcessor
        SynchronousMultiSpanProcessor = trace.SynchronousMultiSpanProcessor
    except ImportError:
        TracerProvider = Any
        SpanProcessor = Any
        SynchronousMultiSpanProcessor = Any


def _prepare_runnable_kwargs(
    runnable_kwargs: Mapping[str, Any],
    system_instruction: str,
    runnable_name: str,
    llm_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Prepares the configuration for a runnable, applying defaults and enforcing constraints."""
    if runnable_kwargs is None:
        runnable_kwargs = {}

    if (
        "human_input_mode" in runnable_kwargs
        and runnable_kwargs["human_input_mode"] != "NEVER"
    ):
        from google.cloud.aiplatform import base

        _LOGGER = base.Logger(__name__)
        _LOGGER.warning(
            f"human_input_mode={runnable_kwargs['human_input_mode']}"
            "is not supported. Will be enforced to 'NEVER'."
        )
    runnable_kwargs["human_input_mode"] = "NEVER"

    if "system_message" not in runnable_kwargs and system_instruction:
        runnable_kwargs["system_message"] = system_instruction

    if "name" not in runnable_kwargs:
        runnable_kwargs["name"] = runnable_name

    if "llm_config" not in runnable_kwargs:
        runnable_kwargs["llm_config"] = llm_config

    return runnable_kwargs


def _default_runnable_builder(
    **runnable_kwargs: Any,
) -> "ConversableAgent":
    from autogen import agentchat

    return agentchat.ConversableAgent(**runnable_kwargs)


def _validate_callable_parameters_are_annotated(callable: Callable):
    """Validates that the parameters of the callable have type annotations.

    This ensures that they can be used for constructing AG2 tools that are
    usable with Gemini function calling.
    """
    import inspect

    parameters = dict(inspect.signature(callable).parameters)
    for name, parameter in parameters.items():
        if parameter.annotation == inspect.Parameter.empty:
            raise TypeError(
                f"Callable={callable.__name__} has untyped input_arg={name}. "
                f"Please specify a type when defining it, e.g. `{name}: str`."
            )


def _validate_tools(tools: Sequence[Callable[..., Any]]):
    """Validates that the tools are usable for tool calling."""
    for tool in tools:
        if isinstance(tool, Callable):
            _validate_callable_parameters_are_annotated(tool)


def _override_active_span_processor(
    tracer_provider: "TracerProvider",
    active_span_processor: "SynchronousMultiSpanProcessor",
):
    """Overrides the active span processor.

    When working with multiple AG2Agents in the same environment,
    it's crucial to manage trace exports carefully.
    Each agent needs its own span processor tied to a unique project ID.
    While we add a new span processor for each agent, this can lead to
    unexpected behavior.
    For instance, with two agents linked to different projects, traces from the
    second agent might be sent to both projects.
    To prevent this and guarantee traces go to the correct project, we overwrite
    the active span processor whenever a new AG2Agent is created.

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


class AG2Agent:
    """An AG2 Agent.

    See https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/develop/ag2
    for details.
    """

    agent_framework = "ag2"

    def __init__(
        self,
        model: str,
        runnable_name: str,
        *,
        api_type: Optional[str] = None,
        llm_config: Optional[Mapping[str, Any]] = None,
        system_instruction: Optional[str] = None,
        runnable_kwargs: Optional[Mapping[str, Any]] = None,
        runnable_builder: Optional[Callable[..., "ConversableAgent"]] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        enable_tracing: bool = False,
    ):
        """Initializes the AG2 Agent.

        Under-the-hood, assuming .set_up() is called, this will correspond to
        ```python
        # runnable_builder
        runnable = runnable_builder(
            llm_config=llm_config,
            system_message=system_instruction,
            **runnable_kwargs,
        )
        ```

        When everything is based on their default values, this corresponds to
        ```python
        # llm_config
        llm_config = {
            "config_list": [{
                "project_id":       initializer.global_config.project,
                "location":         initializer.global_config.location,
                "model":            "gemini-1.0-pro-001",
                "api_type":         "google",
            }]
        }

        # runnable_builder
        runnable = ConversableAgent(
            llm_config=llm_config,
            name="Default AG2 Agent"
            system_message="You are a helpful AI Assistant.",
            human_input_mode="NEVER",
        )
        ```

        By default, if `llm_config` is not specified, a default configuration
        will be created using the provided `model` and `api_type`.

        If `runnable_builder` is not specified, a default runnable builder will
        be used, configured with the `system_instruction`, `runnable_name` and
        `runnable_kwargs`.

        Args:
            model (str):
                Required. The name of the model (e.g. "gemini-1.0-pro").
                Used to create a default `llm_config` if one is not provided.
                This parameter is ignored if `llm_config` is provided.
            runnable_name (str):
                Required. The name of the runnable.
                This name is used as the default `runnable_kwargs["name"]`
                unless `runnable_kwargs` already contains a "name", in which
                case the provided `runnable_kwargs["name"]` will be used.
            api_type (str):
                Optional. The API type to use for the language model.
                Used to create a default `llm_config` if one is not provided.
                This parameter is ignored if `llm_config` is provided.
            llm_config (Mapping[str, Any]):
                Optional. Configuration dictionary for the language model.
                If provided, this configuration will be used directly.
                Otherwise, a default `llm_config` will be created using `model`
                and `api_type`. This `llm_config` is used as the default
                `runnable_kwargs["llm_config"]` unless `runnable_kwargs` already
                contains a "llm_config", in which case the provided
                `runnable_kwargs["llm_config"]` will be used.
            system_instruction (str):
                Optional. The system instruction for the agent.
                This instruction is used as the default
                `runnable_kwargs["system_message"]` unless `runnable_kwargs`
                already contains a "system_message", in which case the provided
                `runnable_kwargs["system_message"]` will be used.
            runnable_kwargs (Mapping[str, Any]):
                Optional. Additional keyword arguments for the constructor of
                the runnable. Details of the kwargs can be found in
                https://docs.ag2.ai/docs/api-reference/autogen/ConversableAgent.
                `runnable_kwargs` only supports `human_input_mode="NEVER"`.
                Other `human_input_mode` values will trigger a warning.
            runnable_builder (Callable[..., "ConversableAgent"]):
                Optional. Callable that returns a new runnable. This can be used
                for customizing the orchestration logic of the Agent.
                If not provided, a default runnable builder will be used.
            tools (Sequence[Callable[..., Any]]):
                Optional. The tools for the agent to be able to use. All input
                callables (e.g. function or class method) will be converted
                to a AG2 tool . Defaults to None.
            enable_tracing (bool):
                Optional. Whether to enable tracing in Cloud Trace. Defaults to
                False.
        """
        from google.cloud.aiplatform import initializer

        # Set up llm config.
        self._project = initializer.global_config.project
        self._location = initializer.global_config.location
        self._model_name = model or "gemini-1.0-pro-001"
        self._api_type = api_type or "google"
        self._llm_config = llm_config or {
            "config_list": [
                {
                    "project_id": self._project,
                    "location": self._location,
                    "model": self._model_name,
                    "api_type": self._api_type,
                }
            ]
        }
        self._system_instruction = system_instruction
        self._runnable_name = runnable_name
        self._runnable_kwargs = _prepare_runnable_kwargs(
            runnable_kwargs=runnable_kwargs,
            llm_config=self._llm_config,
            system_instruction=self._system_instruction,
            runnable_name=self._runnable_name,
        )

        self._tools = []
        if tools:
            # We validate tools at initialization for actionable feedback before
            # they are deployed.
            _validate_tools(tools)
            self._tools = tools
        self._ag2_tool_objects = []
        self._runnable = None
        self._runnable_builder = runnable_builder

        self._instrumentor = None
        self._enable_tracing = enable_tracing

    def set_up(self):
        """Sets up the agent for execution of queries at runtime.

        It initializes the runnable, binds the runnable with tools.

        This method should not be called for an object that being passed to
        the ReasoningEngine service for deployment, as it initializes clients
        that can not be serialized.
        """
        if self._enable_tracing:
            from vertexai.reasoning_engines import _utils

            cloud_trace_exporter = _utils._import_cloud_trace_exporter_or_warn()
            cloud_trace_v2 = _utils._import_cloud_trace_v2_or_warn()
            openinference_autogen = _utils._import_openinference_autogen_or_warn()
            opentelemetry = _utils._import_opentelemetry_or_warn()
            opentelemetry_sdk_trace = _utils._import_opentelemetry_sdk_trace_or_warn()
            if all(
                (
                    cloud_trace_exporter,
                    cloud_trace_v2,
                    openinference_autogen,
                    opentelemetry,
                    opentelemetry_sdk_trace,
                )
            ):
                import google.auth

                credentials, _ = google.auth.default()
                span_exporter = cloud_trace_exporter.CloudTraceSpanExporter(
                    project_id=self._project,
                    client=cloud_trace_v2.TraceServiceClient(
                        credentials=credentials.with_quota_project(self._project),
                    ),
                )
                span_processor: SpanProcessor = (
                    opentelemetry_sdk_trace.export.SimpleSpanProcessor(
                        span_exporter=span_exporter,
                    )
                )
                tracer_provider: TracerProvider = (
                    opentelemetry.trace.get_tracer_provider()
                )
                # Get the appropriate tracer provider:
                # 1. If _TRACER_PROVIDER is already set, use that.
                # 2. Otherwise, if the OTEL_PYTHON_TRACER_PROVIDER environment
                # variable is set, use that.
                # 3. As a final fallback, use _PROXY_TRACER_PROVIDER.
                # If none of the above is set, we log a warning, and
                # create a tracer provider.
                if not tracer_provider:
                    from google.cloud.aiplatform import base

                    _LOGGER = base.Logger(__name__)
                    _LOGGER.warning(
                        "No tracer provider. By default, "
                        "we should get one of the following providers: "
                        "OTEL_PYTHON_TRACER_PROVIDER, _TRACER_PROVIDER, "
                        "or _PROXY_TRACER_PROVIDER."
                    )
                    tracer_provider = opentelemetry_sdk_trace.TracerProvider()
                    opentelemetry.trace.set_tracer_provider(tracer_provider)
                # Avoids AttributeError:
                # 'ProxyTracerProvider' and 'NoOpTracerProvider' objects has no
                # attribute 'add_span_processor'.
                if _utils.is_noop_or_proxy_tracer_provider(tracer_provider):
                    tracer_provider = opentelemetry_sdk_trace.TracerProvider()
                    opentelemetry.trace.set_tracer_provider(tracer_provider)
                # Avoids OpenTelemetry client already exists error.
                _override_active_span_processor(
                    tracer_provider,
                    opentelemetry_sdk_trace.SynchronousMultiSpanProcessor(),
                )
                tracer_provider.add_span_processor(span_processor)
                # Keep the instrumentation up-to-date.
                # When creating multiple AG2Agents,
                # we need to keep the instrumentation up-to-date.
                # We deliberately override the instrument each time,
                # so that if different agents end up using different
                # instrumentations, we guarantee that the user is always
                # working with the most recent agent's instrumentation.
                self._instrumentor = openinference_autogen.AutogenInstrumentor()
                self._instrumentor.uninstrument()
                self._instrumentor.instrument()
            else:
                from google.cloud.aiplatform import base

                _LOGGER = base.Logger(__name__)
                _LOGGER.warning(
                    "enable_tracing=True but proceeding with tracing disabled "
                    "because not all packages for tracing have been installed"
                )

        # Set up tools.
        if self._tools and not self._ag2_tool_objects:
            from vertexai.reasoning_engines import _utils

            autogen_tools = _utils._import_autogen_tools_or_warn()
            if autogen_tools:
                for tool in self._tools:
                    self._ag2_tool_objects.append(autogen_tools.Tool(func_or_tool=tool))

        # Set up runnable.
        runnable_builder = self._runnable_builder or _default_runnable_builder
        self._runnable = runnable_builder(
            **self._runnable_kwargs,
        )

    def clone(self) -> "AG2Agent":
        """Returns a clone of the AG2Agent."""
        import copy

        return AG2Agent(
            model=self._model_name,
            api_type=self._api_type,
            llm_config=copy.deepcopy(self._llm_config),
            system_instruction=self._system_instruction,
            runnable_name=self._runnable_name,
            tools=copy.deepcopy(self._tools),
            runnable_kwargs=copy.deepcopy(self._runnable_kwargs),
            runnable_builder=self._runnable_builder,
            enable_tracing=self._enable_tracing,
        )

    def query(
        self,
        *,
        input: Union[str, Mapping[str, Any]],
        max_turns: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Queries the Agent with the given input.

        Args:
            input (Union[str, Mapping[str, Any]]):
                Required. The input to be passed to the Agent.
            max_turns (int):
                Optional. The maximum number of turns to run the agent for.
                If not provided, the agent will run indefinitely.
                If `max_turns` is a `float`, it will be converted to `int`
                through rounding.
            **kwargs:
                Optional. Any additional keyword arguments to be passed to the
                `.run()` method of the corresponding runnable.
                Details of the kwargs can be found in
                https://docs.ag2.ai/docs/api-reference/autogen/ConversableAgent#run.
                The `user_input` parameter defaults to `False`, and should not
                be passed through `kwargs`.

        Returns:
            The output of querying the Agent with the given input.
        """
        if isinstance(input, str):
            input = {"content": input}

        if max_turns and isinstance(max_turns, float):
            # Supporting auto-conversion float to int.
            max_turns = round(max_turns)

        if "user_input" in kwargs:
            from google.cloud.aiplatform import base

            _LOGGER = base.Logger(__name__)
            _LOGGER.warning(
                "The `user_input` parameter should not be passed through"
                "kwargs. The `user_input` defaults to `False`."
            )
            kwargs.pop("user_input")

        if not self._runnable:
            self.set_up()

        from vertexai.reasoning_engines import _utils

        # `.run()` will return a `ChatResult` object, which is a dataclass.
        # We need to convert it to a JSON-serializable object.
        # More details of `ChatResult` can be found in
        # https://docs.ag2.ai/docs/api-reference/autogen/ChatResult.
        return _utils.dataclass_to_dict(
            self._runnable.run(
                input,
                user_input=False,
                tools=self._ag2_tool_objects,
                max_turns=max_turns,
                **kwargs,
            )
        )
