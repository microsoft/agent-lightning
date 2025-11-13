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
        from langchain_core import runnables
        from langchain_core import tools as lc_tools
        from langchain_core.language_models import base as lc_language_models

        BaseTool = lc_tools.BaseTool
        BaseLanguageModel = lc_language_models.BaseLanguageModel
        GetSessionHistoryCallable = runnables.history.GetSessionHistoryCallable
        RunnableConfig = runnables.RunnableConfig
        RunnableSerializable = runnables.RunnableSerializable
    except ImportError:
        BaseTool = Any
        BaseLanguageModel = Any
        GetSessionHistoryCallable = Any
        RunnableConfig = Any
        RunnableSerializable = Any

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


def _default_runnable_kwargs(has_history: bool) -> Mapping[str, Any]:
    # https://github.com/langchain-ai/langchain/blob/5784dfed001730530637793bea1795d9d5a7c244/libs/core/langchain_core/runnables/history.py#L237-L241
    runnable_kwargs = {
        # input_messages_key (str): Must be specified if the underlying
        # agent accepts a dict as input.
        "input_messages_key": "input",
        # output_messages_key (str): Must be specified if the underlying
        # agent returns a dict as output.
        "output_messages_key": "output",
    }
    if has_history:
        # history_messages_key (str): Must be specified if the underlying
        # agent accepts a dict as input and a separate key for historical
        # messages.
        runnable_kwargs["history_messages_key"] = "history"
    return runnable_kwargs


def _default_output_parser():
    try:
        from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
    except (ModuleNotFoundError, ImportError):
        # Fallback to an older version if needed.
        from langchain.agents.output_parsers.openai_tools import (
            OpenAIToolsAgentOutputParser as ToolsAgentOutputParser,
        )

    return ToolsAgentOutputParser()


def _default_model_builder(
    model_name: str,
    *,
    project: str,
    location: str,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> "BaseLanguageModel":
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
    system_instruction: Optional[str] = None,
    tools: Optional[Sequence["_ToolLike"]] = None,
    prompt: Optional["RunnableSerializable"] = None,
    output_parser: Optional["RunnableSerializable"] = None,
    chat_history: Optional["GetSessionHistoryCallable"] = None,
    model_tool_kwargs: Optional[Mapping[str, Any]] = None,
    agent_executor_kwargs: Optional[Mapping[str, Any]] = None,
    runnable_kwargs: Optional[Mapping[str, Any]] = None,
) -> "RunnableSerializable":
    from langchain_core import tools as lc_tools
    from langchain.agents import AgentExecutor
    from langchain.tools.base import StructuredTool

    # The prompt template and runnable_kwargs needs to be customized depending
    # on whether the user intends for the agent to have history. The way the
    # user would reflect that is by setting chat_history (which defaults to
    # None).
    has_history: bool = chat_history is not None
    prompt = prompt or _default_prompt(
        has_history=has_history,
        system_instruction=system_instruction,
    )
    output_parser = output_parser or _default_output_parser()
    model_tool_kwargs = model_tool_kwargs or {}
    agent_executor_kwargs = agent_executor_kwargs or {}
    runnable_kwargs = runnable_kwargs or _default_runnable_kwargs(has_history)
    if tools:
        model = model.bind_tools(tools=tools, **model_tool_kwargs)
    else:
        tools = []
    agent_executor = AgentExecutor(
        agent=prompt | model | output_parser,
        tools=[
            (
                tool
                if isinstance(tool, lc_tools.BaseTool)
                else StructuredTool.from_function(tool)
            )
            for tool in tools
            if isinstance(tool, (Callable, lc_tools.BaseTool))
        ],
        **agent_executor_kwargs,
    )
    if has_history:
        from langchain_core.runnables.history import RunnableWithMessageHistory

        return RunnableWithMessageHistory(
            runnable=agent_executor,
            get_session_history=chat_history,
            **runnable_kwargs,
        )
    return agent_executor


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


def _default_prompt(
    has_history: bool,
    system_instruction: Optional[str] = None,
) -> "RunnableSerializable":
    from langchain_core import prompts

    try:
        from langchain.agents.format_scratchpad.tools import format_to_tool_messages
    except (ModuleNotFoundError, ImportError):
        # Fallback to an older version if needed.
        from langchain.agents.format_scratchpad.openai_tools import (
            format_to_openai_tool_messages as format_to_tool_messages,
        )

    system_instructions = []
    if system_instruction:
        system_instructions = [("system", system_instruction)]

    if has_history:
        return {
            "history": lambda x: x["history"],
            "input": lambda x: x["input"],
            "agent_scratchpad": (
                lambda x: format_to_tool_messages(x["intermediate_steps"])
            ),
        } | prompts.ChatPromptTemplate.from_messages(
            system_instructions
            + [
                prompts.MessagesPlaceholder(variable_name="history"),
                ("user", "{input}"),
                prompts.MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    else:
        return {
            "input": lambda x: x["input"],
            "agent_scratchpad": (
                lambda x: format_to_tool_messages(x["intermediate_steps"])
            ),
        } | prompts.ChatPromptTemplate.from_messages(
            system_instructions
            + [
                ("user", "{input}"),
                prompts.MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )


def _validate_callable_parameters_are_annotated(callable: Callable):
    """Validates that the parameters of the callable have type annotations.

    This ensures that they can be used for constructing LangChain tools that are
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


def _validate_tools(tools: Sequence["_ToolLike"]):
    """Validates that the tools are usable for tool calling."""
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


class LangchainAgent:
    """A Langchain Agent.

    See https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/develop
    for details.
    """

    agent_framework = "langchain"

    def __init__(
        self,
        model: str,
        *,
        system_instruction: Optional[str] = None,
        prompt: Optional["RunnableSerializable"] = None,
        tools: Optional[Sequence["_ToolLike"]] = None,
        output_parser: Optional["RunnableSerializable"] = None,
        chat_history: Optional["GetSessionHistoryCallable"] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        model_tool_kwargs: Optional[Mapping[str, Any]] = None,
        agent_executor_kwargs: Optional[Mapping[str, Any]] = None,
        runnable_kwargs: Optional[Mapping[str, Any]] = None,
        model_builder: Optional[Callable] = None,
        runnable_builder: Optional[Callable] = None,
        enable_tracing: bool = False,
        instrumentor_builder: Optional[Callable[..., Any]] = None,
    ):
        """Initializes the LangchainAgent.

        Under-the-hood, assuming .set_up() is called, this will correspond to

        ```
        model = model_builder(model_name=model, model_kwargs=model_kwargs)
        runnable = runnable_builder(
            prompt=prompt,
            model=model,
            tools=tools,
            output_parser=output_parser,
            chat_history=chat_history,
            agent_executor_kwargs=agent_executor_kwargs,
            runnable_kwargs=runnable_kwargs,
        )
        ```

        When everything is based on their default values, this corresponds to
        ```
        # model_builder
        from langchain_google_vertexai import ChatVertexAI
        llm = ChatVertexAI(model_name=model, **model_kwargs)

        # runnable_builder
        from langchain import agents
        from langchain_core.runnables.history import RunnableWithMessageHistory
        llm_with_tools = llm.bind_tools(tools=tools, **model_tool_kwargs)
        agent_executor = agents.AgentExecutor(
            agent=prompt | llm_with_tools | output_parser,
            tools=tools,
            **agent_executor_kwargs,
        )
        runnable = RunnableWithMessageHistory(
            runnable=agent_executor,
            get_session_history=chat_history,
            **runnable_kwargs,
        )
        ```

        Args:
            model (str):
                Optional. The name of the model (e.g. "gemini-1.0-pro").
            system_instruction (str):
                Optional. The system instruction to use for the agent. This
                argument should not be specified if `prompt` is specified.
            prompt (langchain_core.runnables.RunnableSerializable):
                Optional. The prompt template for the model. Defaults to a
                ChatPromptTemplate.
            tools (Sequence[langchain_core.tools.BaseTool, Callable]):
                Optional. The tools for the agent to be able to use. All input
                callables (e.g. function or class method) will be converted
                to a langchain.tools.base.StructuredTool. Defaults to None.
            output_parser (langchain_core.runnables.RunnableSerializable):
                Optional. The output parser for the model. Defaults to an
                output parser that works with Gemini function-calling.
            chat_history (langchain_core.runnables.history.GetSessionHistoryCallable):
                Optional. Callable that returns a new BaseChatMessageHistory.
                Defaults to None, i.e. chat_history is not preserved.
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
            agent_executor_kwargs (Mapping[str, Any]):
                Optional. Additional keyword arguments for the constructor of
                langchain.agents.AgentExecutor. An example would be
                ```
                {
                    # Whether to return the agent's trajectory of intermediate
                    # steps at the end in addition to the final output.
                    "return_intermediate_steps": False,
                    # The maximum number of steps to take before ending the
                    # execution loop.
                    "max_iterations": 15,
                    # The method to use for early stopping if the agent never
                    # returns `AgentFinish`. Either 'force' or 'generate'.
                    "early_stopping_method": "force",
                    # How to handle errors raised by the agent's output parser.
                    # Defaults to `False`, which raises the error.
                    "handle_parsing_errors": False,
                }
                ```
            runnable_kwargs (Mapping[str, Any]):
                Optional. Additional keyword arguments for the constructor of
                langchain.runnables.history.RunnableWithMessageHistory if
                chat_history is specified. If chat_history is None, this will be
                ignored.
            model_builder (Callable):
                Optional. Callable that returns a new language model. Defaults
                to a a callable that returns ChatVertexAI based on `model`,
                `model_kwargs` and the parameters in `vertexai.init`.
            runnable_builder (Callable):
                Optional. Callable that returns a new runnable. This can be used
                for customizing the orchestration logic of the Agent based on
                the model returned by `model_builder` and the rest of the input
                arguments.
            enable_tracing (bool):
                Optional. Whether to enable tracing in Cloud Trace. Defaults to
                False.
            instrumentor_builder (Callable[..., Any]):
                Optional. Callable that returns a new instrumentor. This can be
                used for customizing the instrumentation logic of the Agent.
                If not provided, a default instrumentor builder will be used.
                This parameter is ignored if `enable_tracing` is False.

        Raises:
            ValueError: If both `prompt` and `system_instruction` are specified.
            TypeError: If there is an invalid tool (e.g. function with an input
            that did not specify its type).
        """
        from google.cloud.aiplatform import initializer

        self._tmpl_attrs: dict[str, Any] = {
            "project": initializer.global_config.project,
            "location": initializer.global_config.location,
            "tools": [],
            "model_name": model,
            "system_instruction": system_instruction,
            "prompt": prompt,
            "output_parser": output_parser,
            "chat_history": chat_history,
            "model_kwargs": model_kwargs,
            "model_tool_kwargs": model_tool_kwargs,
            "agent_executor_kwargs": agent_executor_kwargs,
            "runnable_kwargs": runnable_kwargs,
            "model_builder": model_builder,
            "runnable_builder": runnable_builder,
            "enable_tracing": enable_tracing,
            "model": None,
            "runnable": None,
            "instrumentor": None,
            "instrumentor_builder": instrumentor_builder,
        }
        if tools:
            # We validate tools at initialization for actionable feedback before
            # they are deployed.
            _validate_tools(tools)
            self._tmpl_attrs["tools"] = tools
        if prompt and system_instruction:
            raise ValueError(
                "Only one of `prompt` or `system_instruction` should be specified. "
                "Consider incorporating the system instruction into the prompt "
                "rather than passing it separately as an argument."
            )

    def set_up(self):
        """Sets up the agent for execution of queries at runtime.

        It initializes the model, binds the model with tools, and connects it
        with the prompt template and output parser.

        This method should not be called for an object being passed to the
        service for deployment, as it might initialize clients that can not be
        serialized.
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
        runnable_builder = (
            self._tmpl_attrs.get("runnable_builder") or _default_runnable_builder
        )
        self._tmpl_attrs["runnable"] = runnable_builder(
            prompt=self._tmpl_attrs.get("prompt"),
            model=self._tmpl_attrs.get("model"),
            tools=self._tmpl_attrs.get("tools"),
            system_instruction=self._tmpl_attrs.get("system_instruction"),
            output_parser=self._tmpl_attrs.get("output_parser"),
            chat_history=self._tmpl_attrs.get("chat_history"),
            model_tool_kwargs=self._tmpl_attrs.get("model_tool_kwargs"),
            agent_executor_kwargs=self._tmpl_attrs.get("agent_executor_kwargs"),
            runnable_kwargs=self._tmpl_attrs.get("runnable_kwargs"),
        )

    def clone(self) -> "LangchainAgent":
        """Returns a clone of the LangchainAgent."""
        import copy

        return LangchainAgent(
            model=self._tmpl_attrs.get("model_name"),
            system_instruction=self._tmpl_attrs.get("system_instruction"),
            prompt=copy.deepcopy(self._tmpl_attrs.get("prompt")),
            tools=copy.deepcopy(self._tmpl_attrs.get("tools")),
            output_parser=copy.deepcopy(self._tmpl_attrs.get("output_parser")),
            chat_history=copy.deepcopy(self._tmpl_attrs.get("chat_history")),
            model_kwargs=copy.deepcopy(self._tmpl_attrs.get("model_kwargs")),
            model_tool_kwargs=copy.deepcopy(self._tmpl_attrs.get("model_tool_kwargs")),
            agent_executor_kwargs=copy.deepcopy(
                self._tmpl_attrs.get("agent_executor_kwargs")
            ),
            runnable_kwargs=copy.deepcopy(self._tmpl_attrs.get("runnable_kwargs")),
            model_builder=self._tmpl_attrs.get("model_builder"),
            runnable_builder=self._tmpl_attrs.get("runnable_builder"),
            enable_tracing=self._tmpl_attrs.get("enable_tracing"),
            instrumentor_builder=self._tmpl_attrs.get("instrumentor_builder"),
        )

    def query(
        self,
        *,
        input: Union[str, Mapping[str, Any]],
        config: Optional["RunnableConfig"] = None,
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
        config: Optional["RunnableConfig"] = None,
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
