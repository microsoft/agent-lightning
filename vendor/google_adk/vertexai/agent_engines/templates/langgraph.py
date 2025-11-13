# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    try:
        from langchain_core.language_models import base as lc_language_models

        BaseLanguageModel = lc_language_models.BaseLanguageModel
    except ImportError:
        BaseLanguageModel = Any

    try:
        from langchain_google_vertexai.functions_utils import _ToolsType

        _ToolLike = _ToolsType
    except ImportError:
        _ToolLike = Any

    try:
        from opentelemetry.sdk import trace

        TracerProvider = trace.TracerProvider
        SpanProcessor = trace.SpanProcessor
        SynchronousMultiSpanProcessor = trace.SynchronousMultiSpanProcessor
    except ImportError:
        TracerProvider = Any
        SpanProcessor = Any
        SynchronousMultiSpanProcessor = Any

    try:
        from langgraph_checkpoint.checkpoint import base

        BaseCheckpointSaver = base.BaseCheckpointSaver
    except ImportError:
        try:
            from langgraph.checkpoint import base

            BaseCheckpointSaver = base.BaseCheckpointSaver
        except ImportError:
            BaseCheckpointSaver = Any


def _default_model_builder(
    model_name: str,
    *,
    project: str,
    location: str,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> "BaseLanguageModel":
    """Default callable for building a language model.

    Args:
        model_name (str):
            Required. The name of the model (e.g. "gemini-1.0-pro").
        project (str):
            Required. The Google Cloud project ID.
        location (str):
            Required. The Google Cloud location.
        model_kwargs (Mapping[str, Any]):
            Optional. Additional keyword arguments for the constructor of
            chat_models.ChatVertexAI.

    Returns:
        BaseLanguageModel: The language model.
    """
    import vertexai
    from google.cloud.aiplatform import initializer
    from langchain_google_vertexai import ChatVertexAI

    model_kwargs = model_kwargs or {}
    current_project = initializer.global_config.project
    current_location = initializer.global_config.location
    vertexai.init(project=project, location=location)
    model = ChatVertexAI(model_name=model_name, **model_kwargs)
    vertexai.init(project=current_project, location=current_location)
    return model


def _default_runnable_builder(
    model: "BaseLanguageModel",
    *,
    tools: Optional[Sequence["_ToolLike"]] = None,
    checkpointer: Optional[Any] = None,
    model_tool_kwargs: Optional[Mapping[str, Any]] = None,
    runnable_kwargs: Optional[Mapping[str, Any]] = None,
):
    """Default callable for building a runnable.

    Args:
        model (BaseLanguageModel):
            Required. The language model.
        tools (Optional[Sequence[_ToolLike]]):
            Optional. The tools for the agent to be able to use.
        checkpointer (Optional[Checkpointer]):
            Optional. The checkpointer for the agent.
        model_tool_kwargs (Optional[Mapping[str, Any]]):
            Optional. Additional keyword arguments when binding tools to the model.
        runnable_kwargs (Optional[Mapping[str, Any]]):
            Optional. Additional keyword arguments for the runnable.

    Returns:
        RunnableSerializable: The runnable.
    """
    from langgraph import prebuilt as langgraph_prebuilt

    model_tool_kwargs = model_tool_kwargs or {}
    runnable_kwargs = runnable_kwargs or {}
    if tools:
        model = model.bind_tools(tools=tools, **model_tool_kwargs)
    else:
        tools = []
    if checkpointer:
        if "checkpointer" in runnable_kwargs:
            from google.cloud.aiplatform import base

            base.Logger(__name__).warning(
                "checkpointer is being specified in both checkpointer_builder "
                "and runnable_kwargs. Please specify it in only one of them. "
                "Overriding the checkpointer in runnable_kwargs."
            )
        runnable_kwargs["checkpointer"] = checkpointer
    return langgraph_prebuilt.create_react_agent(
        model,
        tools=tools,
        **runnable_kwargs,
    )


def _default_instrumentor_builder(project_id: str):
    from vertexai.agent_engines import _utils

    cloud_trace_exporter = _utils._import_cloud_trace_exporter_or_warn()
    cloud_trace_v2 = _utils._import_cloud_trace_v2_or_warn()
    openinference_langchain = _utils._import_openinference_langchain_or_warn()
    opentelemetry = _utils._import_opentelemetry_or_warn()
    opentelemetry_sdk_trace = _utils._import_opentelemetry_sdk_trace_or_warn()
    if all(
        (
            cloud_trace_exporter,
            cloud_trace_v2,
            openinference_langchain,
            opentelemetry,
            opentelemetry_sdk_trace,
        )
    ):
        import google.auth

        credentials, _ = google.auth.default()
        span_exporter = cloud_trace_exporter.CloudTraceSpanExporter(
            project_id=project_id,
            client=cloud_trace_v2.TraceServiceClient(
                credentials=credentials.with_quota_project(project_id),
            ),
        )
        span_processor: SpanProcessor = (
            opentelemetry_sdk_trace.export.SimpleSpanProcessor(
                span_exporter=span_exporter,
            )
        )
        tracer_provider: TracerProvider = opentelemetry.trace.get_tracer_provider()
        # Get the appropriate tracer provider:
        # 1. If _TRACER_PROVIDER is already set, use that.
        # 2. Otherwise, if the OTEL_PYTHON_TRACER_PROVIDER environment
        # variable is set, use that.
        # 3. As a final fallback, use _PROXY_TRACER_PROVIDER.
        # If none of the above is set, we log a warning, and
        # create a tracer provider.
        if not tracer_provider:
            from google.cloud.aiplatform import base

            base.Logger(__name__).warning(
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
        # When creating multiple LangchainAgents,
        # we need to keep the instrumentation up-to-date.
        # We deliberately override the instrument each time,
        # so that if different agents end up using different
        # instrumentations, we guarantee that the user is always
        # working with the most recent agent's instrumentation.
        instrumentor = openinference_langchain.LangChainInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        instrumentor.instrument()
        return instrumentor
    else:
        from google.cloud.aiplatform import base

        _LOGGER = base.Logger(__name__)
        _LOGGER.warning(
            "enable_tracing=True but proceeding with tracing disabled "
            "because not all packages for tracing have been installed"
        )
        return None


def _validate_callable_parameters_are_annotated(callable: Callable):
    """Validates that the parameters of the callable have type annotations.

    This ensures that they can be used for constructing LangChain tools that are
    usable with Gemini function calling.

    Args:
        callable (Callable): The callable to validate.

    Raises:
        TypeError: If any parameter is not annotated.
    """
    import inspect

    parameters = dict(inspect.signature(callable).parameters)
    for name, parameter in parameters.items():
        if parameter.annotation == inspect.Parameter.empty:
            raise TypeError(
                f"Callable={callable.__name__} has untyped input_arg={name}. "
                f"Please specify a type when defining it, e.g. `{name}: str`."
            )


def _validate_tools(tools: Sequence["_ToolLike"]):
    """Validates that the tools are usable for tool calling.

    Args:
        tools (Sequence[_ToolLike]): The tools to validate.

    Raises:
        TypeError: If any tool is a callable with untyped parameters.
    """
    for tool in tools:
        if isinstance(tool, Callable):
            _validate_callable_parameters_are_annotated(tool)


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


class LanggraphAgent:
    """A LangGraph Agent."""

    agent_framework = "langgraph"

    def __init__(
        self,
        model: str,
        *,
        tools: Optional[Sequence["_ToolLike"]] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        model_tool_kwargs: Optional[Mapping[str, Any]] = None,
        model_builder: Optional[Callable[..., "BaseLanguageModel"]] = None,
        runnable_kwargs: Optional[Mapping[str, Any]] = None,
        runnable_builder: Optional[Callable[..., Any]] = None,
        checkpointer_kwargs: Optional[Mapping[str, Any]] = None,
        checkpointer_builder: Optional[Callable[..., "BaseCheckpointSaver"]] = None,
        enable_tracing: bool = False,
        instrumentor_builder: Optional[Callable[..., Any]] = None,
    ):
        """Initializes the LangGraph Agent.

        Under-the-hood, assuming .set_up() is called, this will correspond to
        ```python
        model = model_builder(model_name=model, model_kwargs=model_kwargs)
        runnable = runnable_builder(
            model=model,
            tools=tools,
            model_tool_kwargs=model_tool_kwargs,
            runnable_kwargs=runnable_kwargs,
        )
        ```

        When everything is based on their default values, this corresponds to
        ```python
        # model_builder
        from langchain_google_vertexai import ChatVertexAI
        llm = ChatVertexAI(model_name=model, **model_kwargs)

        # runnable_builder
        from langgraph.prebuilt import create_react_agent
        llm_with_tools = llm.bind_tools(tools=tools, **model_tool_kwargs)
        runnable = create_react_agent(
            llm_with_tools,
            tools=tools,
            **runnable_kwargs,
        )
        ```

        By default, no checkpointer is used (i.e. there is no state history). To
        enable checkpointing, provide a `checkpointer_builder` function that
        returns a checkpointer instance.

        **Example using Spanner:**
        ```python
        def checkpointer_builder(instance_id, database_id, project_id, **kwargs):
            from langchain_google_spanner import SpannerCheckpointSaver

            checkpointer = SpannerCheckpointSaver(instance_id, database_id, project_id)
            with checkpointer.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS checkpoints")
                cur.execute("DROP TABLE IF EXISTS checkpoint_writes")
            checkpointer.setup()

            return checkpointer
        ```

        **Example using an in-memory checkpointer:**
        ```python
        def checkpointer_builder(**kwargs):
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        ```

        The `checkpointer_builder` function will be called with any keyword
        arguments passed to the agent's constructor.  Ensure your
        `checkpointer_builder` function accepts `**kwargs` to handle these
        arguments, even if unused.

        Args:
            model (str):
                Optional. The name of the model (e.g. "gemini-1.0-pro").
            tools (Sequence[langchain_core.tools.BaseTool, Callable]):
                Optional. The tools for the agent to be able to use. All input
                callables (e.g. function or class method) will be converted
                to a langchain.tools.base.StructuredTool. Defaults to None.
            model_kwargs (Mapping[str, Any]):
                Optional. Additional keyword arguments for the constructor of
                chat_models.ChatVertexAI. An example would be
                ```
                {
                    # temperature (float): Sampling temperature, it controls the
                    # degree of randomness in token selection.
                    "temperature": 0.28,
                    # max_output_tokens (int): Token limit determines the
                    # maximum amount of text output from one prompt.
                    "max_output_tokens": 1000,
                    # top_p (float): Tokens are selected from most probable to
                    # least, until the sum of their probabilities equals the
                    # top_p value.
                    "top_p": 0.95,
                    # top_k (int): How the model selects tokens for output, the
                    # next token is selected from among the top_k most probable
                    # tokens.
                    "top_k": 40,
                }
                ```
            model_tool_kwargs (Mapping[str, Any]):
                Optional. Additional keyword arguments when binding tools to the
                model using `model.bind_tools()`.
            model_builder (Callable[..., BaseLanguageModel]):
                Optional. Callable that returns a new language model. Defaults
                to a a callable that returns ChatVertexAI based on `model`,
                `model_kwargs` and the parameters in `vertexai.init`.
            runnable_kwargs (Mapping[str, Any]):
                Optional. Additional keyword arguments for the constructor of
                langchain.runnables.history.RunnableWithMessageHistory if
                chat_history is specified. If chat_history is None, this will be
                ignored.
            runnable_builder (Callable[..., RunnableSerializable]):
                Optional. Callable that returns a new runnable. This can be used
                for customizing the orchestration logic of the Agent based on
                the model returned by `model_builder` and the rest of the input
                arguments.
            checkpointer_kwargs (Mapping[str, Any]):
                Optional. Additional keyword arguments for the constructor of
                the checkpointer returned by `checkpointer_builder`.
            checkpointer_builder (Callable[..., "BaseCheckpointSaver"]):
                Optional. Callable that returns a checkpointer. This can be used
                for defining the checkpointer of the Agent. Defaults to None.
            enable_tracing (bool):
                Optional. Whether to enable tracing in Cloud Trace. Defaults to
                False.
            instrumentor_builder (Callable[..., Any]):
                Optional. Callable that returns a new instrumentor. This can be
                used for customizing the instrumentation logic of the Agent.
                If not provided, a default instrumentor builder will be used.
                This parameter is ignored if `enable_tracing` is False.

        Raises:
            TypeError: If there is an invalid tool (e.g. function with an input
            that did not specify its type).
        """
        from google.cloud.aiplatform import initializer

        self._tmpl_attrs: dict[str, Any] = {
            "project": initializer.global_config.project,
            "location": initializer.global_config.location,
            "tools": [],
            "model_name": model,
            "model_kwargs": model_kwargs,
            "model_tool_kwargs": model_tool_kwargs,
            "runnable_kwargs": runnable_kwargs,
            "checkpointer_kwargs": checkpointer_kwargs,
            "model": None,
            "model_builder": model_builder,
            "runnable": None,
            "runnable_builder": runnable_builder,
            "checkpointer": None,
            "checkpointer_builder": checkpointer_builder,
            "enable_tracing": enable_tracing,
            "instrumentor": None,
            "instrumentor_builder": instrumentor_builder,
        }
        if tools:
            # We validate tools at initialization for actionable feedback before
            # they are deployed.
            _validate_tools(tools)
            self._tmpl_attrs["tools"] = tools

    def set_up(self):
        """Sets up the agent for execution of queries at runtime.

        It initializes the model, binds the model with tools, and connects it
        with the prompt template and output parser.

        This method should not be called for an object that being passed to
        the ReasoningEngine service for deployment, as it initializes clients
        that can not be serialized.
        """
        if self._tmpl_attrs.get("enable_tracing"):
            instrumentor_builder = (
                self._tmpl_attrs.get("instrumentor_builder")
                or _default_instrumentor_builder
            )
            self._tmpl_attrs["instrumentor"] = instrumentor_builder(
                project_id=self._tmpl_attrs.get("project")
            )
        model_builder = self._tmpl_attrs.get("model_builder") or _default_model_builder
        self._tmpl_attrs["model"] = model_builder(
            model_name=self._tmpl_attrs.get("model_name"),
            model_kwargs=self._tmpl_attrs.get("model_kwargs"),
            project=self._tmpl_attrs.get("project"),
            location=self._tmpl_attrs.get("location"),
        )
        checkpointer_builder = self._tmpl_attrs.get("checkpointer_builder")
        if checkpointer_builder:
            checkpointer_kwargs = self._tmpl_attrs.get("checkpointer_kwargs") or {}
            self._tmpl_attrs["checkpointer"] = checkpointer_builder(
                **checkpointer_kwargs
            )
        runnable_builder = (
            self._tmpl_attrs.get("runnable_builder") or _default_runnable_builder
        )
        self._tmpl_attrs["runnable"] = runnable_builder(
            model=self._tmpl_attrs.get("model"),
            tools=self._tmpl_attrs.get("tools"),
            checkpointer=self._tmpl_attrs.get("checkpointer"),
            model_tool_kwargs=self._tmpl_attrs.get("model_tool_kwargs"),
            runnable_kwargs=self._tmpl_attrs.get("runnable_kwargs"),
        )

    def clone(self) -> "LanggraphAgent":
        """Returns a clone of the LanggraphAgent."""
        import copy

        return LanggraphAgent(
            model=self._tmpl_attrs.get("model_name"),
            tools=copy.deepcopy(self._tmpl_attrs.get("tools")),
            model_kwargs=copy.deepcopy(self._tmpl_attrs.get("model_kwargs")),
            model_tool_kwargs=copy.deepcopy(self._tmpl_attrs.get("model_tool_kwargs")),
            runnable_kwargs=copy.deepcopy(self._tmpl_attrs.get("runnable_kwargs")),
            checkpointer_kwargs=copy.deepcopy(
                self._tmpl_attrs.get("checkpointer_kwargs")
            ),
            model_builder=self._tmpl_attrs.get("model_builder"),
            runnable_builder=self._tmpl_attrs.get("runnable_builder"),
            checkpointer_builder=self._tmpl_attrs.get("checkpointer_builder"),
            enable_tracing=self._tmpl_attrs.get("enable_tracing"),
            instrumentor_builder=self._tmpl_attrs.get("instrumentor_builder"),
        )

    def query(
        self,
        *,
        input: Union[str, Mapping[str, Any]],
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Queries the Agent with the given input and config.

        Args:
            input (Union[str, Mapping[str, Any]]):
                Required. The input to be passed to the Agent.
            config (langchain_core.runnables.RunnableConfig):
                Optional. The config (if any) to be used for invoking the Agent.
            **kwargs:
                Optional. Any additional keyword arguments to be passed to the
                `.invoke()` method of the corresponding AgentExecutor.

        Returns:
            The output of querying the Agent with the given input and config.
        """
        from langchain.load import dump as langchain_load_dump

        if isinstance(input, str):
            input = {"input": input}
        if not self._tmpl_attrs.get("runnable"):
            self.set_up()
        return langchain_load_dump.dumpd(
            self._tmpl_attrs.get("runnable").invoke(
                input=input, config=config, **kwargs
            )
        )

    def stream_query(
        self,
        *,
        input: Union[str, Mapping[str, Any]],
        config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Iterable[Any]:
        """Stream queries the Agent with the given input and config.

        Args:
            input (Union[str, Mapping[str, Any]]):
                Required. The input to be passed to the Agent.
            config (langchain_core.runnables.RunnableConfig):
                Optional. The config (if any) to be used for invoking the Agent.
            **kwargs:
                Optional. Any additional keyword arguments to be passed to the
                `.invoke()` method of the corresponding AgentExecutor.

        Yields:
            The output of querying the Agent with the given input and config.
        """
        from langchain.load import dump as langchain_load_dump

        if isinstance(input, str):
            input = {"input": input}
        if not self._tmpl_attrs.get("runnable"):
            self.set_up()
        for chunk in self._tmpl_attrs.get("runnable").stream(
            input=input,
            config=config,
            **kwargs,
        ):
            yield langchain_load_dump.dumpd(chunk)

    def get_state_history(
        self,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Gets the state history of the Agent.

        Args:
            config (Optional[RunnableConfig]):
                Optional. The config for invoking the Agent.
            **kwargs:
                Optional. Additional keyword arguments for the `.invoke()` method.

        Yields:
            Dict[str, Any]: The state history of the Agent.
        """
        if not self._tmpl_attrs.get("runnable"):
            self.set_up()
        for state_snapshot in self._tmpl_attrs.get("runnable").get_state_history(
            config=config,
            **kwargs,
        ):
            yield state_snapshot._asdict()

    def get_state(
        self,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Gets the current state of the Agent.

        Args:
            config (Optional[RunnableConfig]):
                Optional. The config for invoking the Agent.
            **kwargs:
                Optional. Additional keyword arguments for the `.invoke()` method.

        Returns:
            Dict[str, Any]: The current state of the Agent.
        """
        if not self._tmpl_attrs.get("runnable"):
            self.set_up()
        return (
            self._tmpl_attrs.get("runnable")
            .get_state(config=config, **kwargs)
            ._asdict()
        )

    def update_state(
        self,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Updates the state of the Agent.

        Args:
            config (Optional[RunnableConfig]):
                Optional. The config for invoking the Agent.
            **kwargs:
                Optional. Additional keyword arguments for the `.invoke()` method.

        Returns:
            Dict[str, Any]: The updated state of the Agent.
        """
        if not self._tmpl_attrs.get("runnable"):
            self.set_up()
        return self._tmpl_attrs.get("runnable").update_state(config=config, **kwargs)

    def register_operations(self) -> Mapping[str, Sequence[str]]:
        """Registers the operations of the Agent.

        This mapping defines how different operation modes (e.g., "", "stream")
        are implemented by specific methods of the Agent.  The "default" mode,
        represented by the empty string ``, is associated with the `query` API,
        while the "stream" mode is associated with the `stream_query` API.

        Returns:
            Mapping[str, Sequence[str]]: A mapping of operation modes to a list
            of method names that implement those operation modes.
        """
        return {
            "": ["query", "get_state", "update_state"],
            "stream": ["stream_query", "get_state_history"],
        }
