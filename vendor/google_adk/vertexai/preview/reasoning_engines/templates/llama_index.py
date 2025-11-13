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
        from llama_index.core.base.query_pipeline import query
        from llama_index.core.llms import function_calling
        from llama_index.core import query_pipeline

        FunctionCallingLLM = function_calling.FunctionCallingLLM
        QueryComponent = query.QUERY_COMPONENT_TYPE
        QueryPipeline = query_pipeline.QueryPipeline
    except ImportError:
        FunctionCallingLLM = Any
        QueryComponent = Any
        QueryPipeline = Any

    try:
        from opentelemetry.sdk import trace

        TracerProvider = trace.TracerProvider
        SpanProcessor = trace.SpanProcessor
        SynchronousMultiSpanProcessor = trace.SynchronousMultiSpanProcessor
    except ImportError:
        TracerProvider = Any
        SpanProcessor = Any
        SynchronousMultiSpanProcessor = Any


def _default_model_builder(
    model_name: str,
    *,
    project: str,
    location: str,
    model_kwargs: Optional[Mapping[str, Any]] = None,
) -> "FunctionCallingLLM":
    """Creates a default model builder for LlamaIndex."""
    import vertexai
    from google.cloud.aiplatform import initializer
    from llama_index.llms import google_genai

    model_kwargs = model_kwargs or {}
    model = google_genai.GoogleGenAI(
        model=model_name,
        vertexai_config={"project": project, "location": location},
        **model_kwargs,
    )
    current_project = initializer.global_config.project
    current_location = initializer.global_config.location
    vertexai.init(project=current_project, location=current_location)
    return model


def _default_runnable_builder(
    model: "FunctionCallingLLM",
    *,
    system_instruction: Optional[str] = None,
    prompt: Optional["QueryComponent"] = None,
    retriever: Optional["QueryComponent"] = None,
    response_synthesizer: Optional["QueryComponent"] = None,
    runnable_kwargs: Optional[Mapping[str, Any]] = None,
) -> "QueryPipeline":
    """Creates a default runnable builder for LlamaIndex."""
    try:
        from llama_index.core.query_pipeline import QueryPipeline
    except ImportError:
        raise ImportError(
            "Please call 'pip install google-cloud-aiplatform[llama_index]'."
        )

    prompt = prompt or _default_prompt(
        system_instruction=system_instruction,
    )
    pipeline = QueryPipeline(**runnable_kwargs)
    pipeline_modules = {
        "prompt": prompt,
        "model": model,
    }
    if retriever:
        pipeline_modules["retriever"] = retriever
    if response_synthesizer:
        pipeline_modules["response_synthesizer"] = response_synthesizer

    pipeline.add_modules(pipeline_modules)
    pipeline.add_link("prompt", "model")
    if "retriever" in pipeline_modules:
        pipeline.add_link("model", "retriever")
    if "response_synthesizer" in pipeline_modules:
        pipeline.add_link("model", "response_synthesizer", dest_key="query_str")
        if "retriever" in pipeline_modules:
            pipeline.add_link("retriever", "response_synthesizer", dest_key="nodes")

    return pipeline


def _default_prompt(
    system_instruction: Optional[str] = None,
) -> "QueryComponent":
    """Creates a default prompt template for LlamaIndex.

    Handles both system instruction and user input.

    Args:
        system_instruction (str, optional):  The system instruction to use.

    Returns:
        QueryComponent:  The LlamaIndex QueryComponent.
    """
    try:
        from llama_index.core import prompts
        from llama_index.core.base.llms import types
    except ImportError:
        raise ImportError(
            "Please call 'pip install google-cloud-aiplatform[llama_index]'."
        )

    # Define a prompt template
    message_templates = []
    if system_instruction:
        message_templates.append(
            types.ChatMessage(role=types.MessageRole.SYSTEM, content=system_instruction)
        )
    # Add user input message
    message_templates.append(
        types.ChatMessage(role=types.MessageRole.USER, content="{input}")
    )

    # Create the prompt template
    return prompts.ChatPromptTemplate(message_templates=message_templates)


def _override_active_span_processor(
    tracer_provider: "TracerProvider",
    active_span_processor: "SynchronousMultiSpanProcessor",
):
    """Overrides the active span processor.

    When working with multiple LlamaIndexQueryPipelineAgents in the same
    environment, it's crucial to manage trace exports carefully.
    Each agent needs its own span processor tied to a unique project ID.
    While we add a new span processor for each agent, this can lead to
    unexpected behavior.
    For instance, with two agents linked to different projects, traces from the
    second agent might be sent to both projects.
    To prevent this and guarantee traces go to the correct project, we overwrite
    the active span processor whenever a new LlamaIndexQueryPipelineAgent is
    created.

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


class LlamaIndexQueryPipelineAgent:
    """A LlamaIndex Query Pipeline Agent.

    This agent uses a query pipeline for LLAIndex, including prompt, model,
    retrieval and summarization steps. More details can be found in
    https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/.
    """

    agent_framework = "llama-index"

    def __init__(
        self,
        model: str,
        *,
        system_instruction: Optional[str] = None,
        prompt: Optional["QueryComponent"] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        model_builder: Optional[Callable[..., "FunctionCallingLLM"]] = None,
        retriever_kwargs: Optional[Mapping[str, Any]] = None,
        retriever_builder: Optional[Callable[..., "QueryComponent"]] = None,
        response_synthesizer_kwargs: Optional[Mapping[str, Any]] = None,
        response_synthesizer_builder: Optional[Callable[..., "QueryComponent"]] = None,
        runnable_kwargs: Optional[Mapping[str, Any]] = None,
        runnable_builder: Optional[Callable[..., "QueryPipeline"]] = None,
        enable_tracing: bool = False,
    ):
        """Initializes the LlamaIndexQueryPipelineAgent.

        Under-the-hood, assuming .set_up() is called, this will correspond to
        ```python
        # model_builder
        model = model_builder(model_name, project, location, model_kwargs)

        # runnable_builder
        runnable = runnable_builder(
            prompt=prompt,
            model=model,
            retriever=retriever_builder(model, retriever_kwargs),
            response_synthesizer=response_synthesizer_builder(
                model, response_synthesizer_kwargs
            ),
            runnable_kwargs=runnable_kwargs,
        )
        ```

        When everything is based on their default values, this corresponds to a
        query pipeline `Prompt - Model`:
        ```python
        # Default Model Builder
        model = google_genai.GoogleGenAI(
            model=model_name,
            vertexai_config={
                "project": initializer.global_config.project,
                "location": initializer.global_config.location,
            },
        )

        # Default Prompt Builder
        prompt = prompts.ChatPromptTemplate(
            message_templates=[
                types.ChatMessage(
                    role=types.MessageRole.USER,
                    content="{input}",
                ),
            ],
        )

        # Default Runnable Builder
        runnable = QueryPipeline(
            modules = {
                "prompt": prompt,
                "model": model,
            },
        )
        pipeline.add_link("prompt", "model")
        ```

        When `system_instruction` is specified, the prompt will be updated to
        include the system instruction.
        ```python
        # Updated Prompt Builder
        prompt = prompts.ChatPromptTemplate(
            message_templates=[
                types.ChatMessage(
                    role=types.MessageRole.SYSTEM,
                    content=system_instruction,
                ),
                types.ChatMessage(
                    role=types.MessageRole.USER,
                    content="{input}",
                ),
            ],
        )
        ```

        When all inputs are specified, this corresponds to a query pipeline
        `Prompt - Model - Retriever - Summarizer`:
        ```python
        runnable = QueryPipeline(
            modules = {
                "prompt": prompt,
                "model": model,
                "retriever": retriever_builder(retriever_kwargs),
                "response_synthesizer": response_synthesizer_builder(
                    response_synthesizer_kwargs
                ),
            },
        )
        pipeline.add_link("prompt", "model")
        pipeline.add_link("model", "retriever")
        pipeline.add_link("model", "response_synthesizer", dest_key="query_str")
        pipeline.add_link("retriever", "response_synthesizer", dest_key="nodes")
        ```

        Args:
            model (str):
                The name of the model (e.g. "gemini-1.0-pro").
            system_instruction (str):
                Optional. The system instruction to use for the agent.
            prompt (llama_index.core.base.query_pipeline.query.QUERY_COMPONENT_TYPE):
                Optional.  The prompt template for the model.
            model_kwargs (Mapping[str, Any]):
                Optional. Keyword arguments for the model constructor of the
                google_genai.GoogleGenAI. An example of a model_kwargs is:
                ```python
                {
                    # api_key (string): The API key for the GoogleGenAI model.
                    # The API can also be fetched from the GOOGLE_API_KEY
                    # environment variable. If `vertexai_config` is provided,
                    # the API key is ignored.
                    "api_key": "your_api_key",
                    # temperature (float): Sampling temperature, it controls the
                    # degree of randomness in token selection. If not provided,
                    # the default temperature is 0.1.
                    "temperature": 0.1,
                    # context_window (int): The context window of the model.
                    # If not provided, the default context window is 200000.
                    "context_window": 200000,
                    # max_tokens (int): Token limit determines the maximum
                    # amount of text output from one prompt. If not provided,
                    # the default max_tokens is 256.
                    "max_tokens": 256,
                    # is_function_calling_model (bool): Whether the model is a
                    # function calling model. If not provided, the default
                    # is_function_calling_model is True.
                    "is_function_calling_model": True,
                }
                ```
            model_builder (Callable):
                Optional. Callable that returns a language model.
            retriever_kwargs (Mapping[str, Any]):
                Optional. Keyword arguments for the retriever constructor.
            retriever_builder (Callable):
                Optional. Callable that returns a retriever object.
            response_synthesizer_kwargs (Mapping[str, Any]):
                Optional. Keyword arguments for the response synthesizer constructor.
            response_synthesizer_builder (Callable):
                Optional. Callable that returns a response_synthesizer object.
            runnable_kwargs (Mapping[str, Any]):
                Optional. Keyword arguments for the runnable constructor.
            runnable_builder (Callable):
                Optional. Callable that returns a runnable (query pipeline).
            enable_tracing (bool):
                Optional. Whether to enable tracing. Defaults to False.
        """
        from google.cloud.aiplatform import initializer

        self._project = initializer.global_config.project
        self._location = initializer.global_config.location
        self._model_name = model
        self._system_instruction = system_instruction
        self._prompt = prompt

        self._model = None
        self._model_kwargs = model_kwargs or {}
        self._model_builder = model_builder

        self._retriever = None
        self._retriever_kwargs = retriever_kwargs or {}
        self._retriever_builder = retriever_builder

        self._response_synthesizer = None
        self._response_synthesizer_kwargs = response_synthesizer_kwargs or {}
        self._response_synthesizer_builder = response_synthesizer_builder

        self._runnable = None
        self._runnable_kwargs = runnable_kwargs or {}
        self._runnable_builder = runnable_builder

        self._instrumentor = None
        self._enable_tracing = enable_tracing

    def set_up(self):
        """Sets up the agent for execution of queries at runtime.

        It initializes the model, connects it with the prompt template,
        retriever and response_synthesizer.

        This method should not be called for an object that being passed to
        the ReasoningEngine service for deployment, as it initializes clients
        that can not be serialized.
        """
        if self._enable_tracing:
            from vertexai.reasoning_engines import _utils

            cloud_trace_exporter = _utils._import_cloud_trace_exporter_or_warn()
            cloud_trace_v2 = _utils._import_cloud_trace_v2_or_warn()
            openinference_llama_index = (
                _utils._import_openinference_llama_index_or_warn()
            )
            opentelemetry = _utils._import_opentelemetry_or_warn()
            opentelemetry_sdk_trace = _utils._import_opentelemetry_sdk_trace_or_warn()
            if all(
                (
                    cloud_trace_exporter,
                    cloud_trace_v2,
                    openinference_llama_index,
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
                # When creating multiple LlamaIndexQueryPipelineAgents,
                # we need to keep the instrumentation up-to-date.
                # We deliberately override the instrument each time,
                # so that if different agents end up using different
                # instrumentations, we guarantee that the user is always
                # working with the most recent agent's instrumentation.
                self._instrumentor = openinference_llama_index.LlamaIndexInstrumentor()
                if self._instrumentor.is_instrumented_by_opentelemetry:
                    self._instrumentor.uninstrument()
                self._instrumentor.instrument()
            else:
                from google.cloud.aiplatform import base

                _LOGGER = base.Logger(__name__)
                _LOGGER.warning(
                    "enable_tracing=True but proceeding with tracing disabled "
                    "because not all packages for tracing have been installed"
                )

        model_builder = self._model_builder or _default_model_builder
        self._model = model_builder(
            model_name=self._model_name,
            model_kwargs=self._model_kwargs,
            project=self._project,
            location=self._location,
        )

        if self._retriever_builder:
            self._retriever = self._retriever_builder(
                model=self._model,
                retriever_kwargs=self._retriever_kwargs,
            )

        if self._response_synthesizer_builder:
            self._response_synthesizer = self._response_synthesizer_builder(
                model=self._model,
                response_synthesizer_kwargs=self._response_synthesizer_kwargs,
            )

        runnable_builder = self._runnable_builder or _default_runnable_builder
        self._runnable = runnable_builder(
            prompt=self._prompt,
            model=self._model,
            system_instruction=self._system_instruction,
            retriever=self._retriever,
            response_synthesizer=self._response_synthesizer,
            runnable_kwargs=self._runnable_kwargs,
        )

    def clone(self) -> "LlamaIndexQueryPipelineAgent":
        """Returns a clone of the LlamaIndexQueryPipelineAgent."""
        import copy

        return LlamaIndexQueryPipelineAgent(
            model=self._model_name,
            system_instruction=self._system_instruction,
            prompt=copy.deepcopy(self._prompt),
            model_kwargs=copy.deepcopy(self._model_kwargs),
            model_builder=self._model_builder,
            retriever_kwargs=copy.deepcopy(self._retriever_kwargs),
            retriever_builder=self._retriever_builder,
            response_synthesizer_kwargs=copy.deepcopy(
                self._response_synthesizer_kwargs
            ),
            response_synthesizer_builder=self._response_synthesizer_builder,
            runnable_kwargs=copy.deepcopy(self._runnable_kwargs),
            runnable_builder=self._runnable_builder,
            enable_tracing=self._enable_tracing,
        )

    def query(
        self,
        input: Union[str, Mapping[str, Any]],
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]]:
        """Queries the Agent with the given input and config.

        Args:
            input (Union[str, Mapping[str, Any]]):
                Required. The input to be passed to the Agent.
            **kwargs:
                Optional. Any additional keyword arguments to be passed to the
                `.invoke()` method of the corresponding AgentExecutor.

        Returns:
            The output of querying the Agent with the given input and config.
        """
        from vertexai.reasoning_engines import _utils

        if isinstance(input, str):
            input = {"input": input}

        if not self._runnable:
            self.set_up()

        if kwargs.get("batch"):
            nest_asyncio = _utils._import_nest_asyncio_or_warn()
            nest_asyncio.apply()

        return _utils.to_json_serializable_llama_index_object(
            self._runnable.run(**input, **kwargs)
        )
