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

::: agentlightning.emitter.annotation.OperationContext

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
