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
  SpanKind,
  type Tracer,
  type Context,
} from '@opentelemetry/api';
import { get_encoding, type Tiktoken } from '@dqbd/tiktoken';
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

  onStart(span: SdkSpan, parentContext: Context): void {
    // Stamp rollout metadata on every span
    span.setAttribute('agentlightning.rollout_id', this.rolloutId);
    span.setAttribute('agentlightning.attempt_id', this.attemptId);
    span.setAttribute('agentlightning.span_sequence_id', this.seq++);
    this.delegate.onStart(span, parentContext);
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
  // Enable OTel diagnostics in debug mode (DEBUG_OTLP=1 or OTEL_DIAG_LOG_LEVEL=DEBUG)
  if (process.env.DEBUG_OTLP || process.env.OTEL_DIAG_LOG_LEVEL) {
    const level = (process.env.OTEL_DIAG_LOG_LEVEL as keyof typeof DiagLogLevel) || 'DEBUG';
    if (DiagLogLevel[level] !== undefined) {
      diag.setLogger(new DiagConsoleLogger(), DiagLogLevel[level]);
      console.log(`[OTLP DEBUG] Enabled with level=${level}`);
    }
  }

  console.log(`[OTLP] Creating tracer for rollout=${config.rolloutId.slice(0, 12)}..., endpoint=${config.otlpEndpoint}`);

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

/**
 * Emit a reward span that can be recognized by Agent Lightning daemon.
 * Uses the AgentOps format: {"type": "reward", "value": <float>}
 *
 * This span will be matched by the daemon's get_reward_value() function
 * when extracting final_reward for training.
 */
export function emitReward(tracer: Tracer, reward: number): void {
  const span = tracer.startSpan('reward', { kind: SpanKind.INTERNAL });
  span.setAttribute(
    'agentops.task.output',
    JSON.stringify({ type: 'reward', value: reward })
  );
  span.end();
}

// Lazy-loaded tokenizer instance
let _encoder: Tiktoken | null = null;

/**
 * Get or create the tiktoken encoder.
 * Uses cl100k_base encoding (GPT-4/3.5 compatible).
 */
function getEncoder(): Tiktoken {
  if (!_encoder) {
    _encoder = get_encoding('cl100k_base');
  }
  return _encoder;
}

/**
 * Tokenize a text string into token IDs.
 *
 * Uses tiktoken's cl100k_base encoding which is compatible with GPT-4 and GPT-3.5.
 * Note: For Qwen models, the actual tokenizer vocabulary differs, but this
 * provides a reasonable approximation for training flow validation.
 *
 * @param text - The text to tokenize
 * @returns Array of token IDs
 */
export function tokenize(text: string): number[] {
  const encoder = getEncoder();
  return Array.from(encoder.encode(text));
}

/**
 * Emit an LLM call span with token IDs for training.
 *
 * This creates a span that the TracerTraceToTriplet adapter can convert
 * into a Triplet with proper token IDs for training batches.
 *
 * @param tracer - OpenTelemetry tracer
 * @param promptText - The prompt text sent to the LLM
 * @param responseText - The response text from the LLM
 * @param modelId - Optional model ID
 */
export function emitLlmCallSpan(
  tracer: Tracer,
  promptText: string,
  responseText: string,
  modelId?: string
): void {
  const span = tracer.startSpan('ai.generateText', { kind: SpanKind.INTERNAL });

  // Tokenize prompt and response
  const promptTokenIds = tokenize(promptText);
  const responseTokenIds = tokenize(responseText);

  // Set token ID attributes that TracerTraceToTriplet looks for
  span.setAttribute('prompt_token_ids', promptTokenIds);
  span.setAttribute('response_token_ids', responseTokenIds);

  // Also set raw content for reference
  span.setAttribute('gen_ai.prompt.0.role', 'user');
  span.setAttribute('gen_ai.prompt.0.content', promptText);
  span.setAttribute('gen_ai.completion.0.role', 'assistant');
  span.setAttribute('gen_ai.completion.0.content', responseText);

  if (modelId) {
    span.setAttribute('gen_ai.request.model', modelId);
  }

  span.end();
}
