# Internal API References

!!! danger

    The following APIs should be used with extra caution because they are very likely to change in the future.

## Algorithms and Adapters

::: agentlightning.adapter.messages.OpenAIMessages

::: agentlightning.adapter.triplet.TraceTree

::: agentlightning.adapter.triplet.Transition

::: agentlightning.adapter.triplet.RewardMatchPolicy

::: agentlightning.algorithm.decorator.FunctionalAlgorithm

## LitAgent

::: agentlightning.litagent.decorator.FunctionalLitAgent

::: agentlightning.litagent.decorator.llm_rollout

::: agentlightning.litagent.decorator.prompt_rollout

## LLM Proxy

::: agentlightning.llm_proxy.ModelConfig

::: agentlightning.llm_proxy.LightningSpanExporter

::: agentlightning.llm_proxy.LightningOpenTelemetry

::: agentlightning.llm_proxy.AddReturnTokenIds

::: agentlightning.llm_proxy.StreamConversionMiddleware

::: agentlightning.llm_proxy.MessageInspectionMiddleware

::: agentlightning.llm_proxy.RolloutAttemptMiddleware

## Store

::: agentlightning.store.base.UNSET

::: agentlightning.store.utils.rollout_status_from_attempt

::: agentlightning.store.utils.scan_unhealthy_rollouts

## Tracing and OpenTelemetry

::: agentlightning.tracer.otel.LightningSpanProcessor

## Utilities

::: agentlightning.utils.server_launcher.PythonServerLauncher

::: agentlightning.utils.server_launcher.PythonServerLauncherArgs

::: agentlightning.utils.server_launcher.LaunchMode

::: agentlightning.utils.otel.full_qualified_name

::: agentlightning.utils.otel.get_tracer_provider

::: agentlightning.utils.otel.get_tracer

::: agentlightning.utils.otel.make_tag_attributes

::: agentlightning.utils.otel.extract_tags_from_attributes

::: agentlightning.utils.otel.make_link_attributes

::: agentlightning.utils.otel.query_linked_spans

::: agentlightning.utils.otel.extract_links_from_attributes

::: agentlightning.utils.otel.filter_attributes

::: agentlightning.utils.otel.filter_and_unflatten_attributes

::: agentlightning.utils.otel.flatten_attributes

::: agentlightning.utils.otel.unflatten_attributes

::: agentlightning.utils.otlp.handle_otlp_export

::: agentlightning.utils.otlp.spans_from_proto

## Deprecated APIs

::: agentlightning.emitter.reward.reward

::: agentlightning.server.AgentLightningServer

::: agentlightning.server.ServerDataStore

::: agentlightning.client.AgentLightningClient

::: agentlightning.client.DevTaskLoader

::: agentlightning.Task

::: agentlightning.TaskInput

::: agentlightning.TaskIfAny

::: agentlightning.RolloutRawResultLegacy

::: agentlightning.RolloutLegacy
