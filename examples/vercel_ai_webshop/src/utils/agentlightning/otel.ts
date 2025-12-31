/**
 * Agent Lightning OpenTelemetry Tracer Factory
 *
 * Creates tracers that embed rollout_id and attempt_id in Resource attributes,
 * following Agent Lightning's conventions for span correlation.
 */

import {
  diag,
  DiagConsoleLogger,
  DiagLogLevel,
  type Tracer,
} from '@opentelemetry/api';
import { Resource } from '@opentelemetry/resources';
import {
  BasicTracerProvider,
  SimpleSpanProcessor,
  type SpanProcessor,
  type Span as SdkSpan,
  type ReadableSpan,
} from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { ATTR_SERVICE_NAME } from '@opentelemetry/semantic-conventions';

export interface TracerConfig {
  otlpEndpoint: string;
  serviceName: string;
  rolloutId: string;
  attemptId: string;
}

/**
 * Custom SpanProcessor that stamps rollout metadata and sequence_id on spans.
 * Agent Lightning expects sequence_id for ordering within an attempt.
 */
class RolloutSpanProcessor implements SpanProcessor {
  private seq = 0;

  constructor(
    private delegate: SpanProcessor,
    private rolloutId: string,
    private attemptId: string
  ) {}

  onStart(span: SdkSpan): void {
    // Stamp rollout metadata on every span
    span.setAttribute('agentlightning.rollout_id', this.rolloutId);
    span.setAttribute('agentlightning.attempt_id', this.attemptId);
    span.setAttribute('agentlightning.sequence_id', this.seq++);
    this.delegate.onStart(span);
  }

  onEnd(span: ReadableSpan): void {
    this.delegate.onEnd(span);
  }

  shutdown(): Promise<void> {
    return this.delegate.shutdown();
  }

  forceFlush(): Promise<void> {
    return this.delegate.forceFlush();
  }
}

/**
 * Create a per-rollout tracer whose Resource contains rollout_id/attempt_id.
 * This mirrors Agent Lightning's OTel tracer approach.
 */
export function createRolloutTracer(config: TracerConfig): {
  tracer: Tracer;
  provider: BasicTracerProvider;
} {
  // Enable OTel diagnostics in debug mode
  if (process.env.OTEL_DIAG_LOG_LEVEL) {
    const level = process.env.OTEL_DIAG_LOG_LEVEL as keyof typeof DiagLogLevel;
    if (DiagLogLevel[level] !== undefined) {
      diag.setLogger(new DiagConsoleLogger(), DiagLogLevel[level]);
    }
  }

  // Create Resource with rollout metadata (Agent Lightning convention)
  const resource = new Resource({
    [ATTR_SERVICE_NAME]: config.serviceName,
    'agentlightning.rollout_id': config.rolloutId,
    'agentlightning.attempt_id': config.attemptId,
  });

  const provider = new BasicTracerProvider({ resource });

  // Configure OTLP exporter to send to Agent Lightning Store
  const exporter = new OTLPTraceExporter({
    url: config.otlpEndpoint,
  });

  const simpleProcessor = new SimpleSpanProcessor(exporter);
  provider.addSpanProcessor(
    new RolloutSpanProcessor(simpleProcessor, config.rolloutId, config.attemptId)
  );

  const tracer = provider.getTracer(config.serviceName);
  return { tracer, provider };
}

/**
 * Create AI SDK telemetry settings with custom tracer.
 * This object can be passed to AI SDK's experimental_telemetry option.
 */
export function makeAiSdkTelemetry(tracer: Tracer) {
  return {
    isEnabled: true,
    tracer,
    recordInputs: true,
    recordOutputs: true,
  };
}

/**
 * Get OTLP endpoint from environment variables.
 * If AGENT_LIGHTNING_OTLP_ENDPOINT is not set, derives it from AGENT_LIGHTNING_STORE_URL.
 */
export function getOtlpEndpoint(): string | null {
  const explicit = process.env.AGENT_LIGHTNING_OTLP_ENDPOINT;
  if (explicit) return explicit;

  const storeUrl = process.env.AGENT_LIGHTNING_STORE_URL;
  if (storeUrl) return `${storeUrl.replace(/\/+$/, '')}/v1/traces`;

  return null;
}
