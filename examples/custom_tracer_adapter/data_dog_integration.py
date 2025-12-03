# Copyright (c) Microsoft. All rights reserved.

"""Example: DataDog integration using custom tracer pattern.

This shows how to use the minimal example's tracer pattern to integrate
with a real observability platform (DataDog).

Install: pip install ddtrace
"""

from typing import Any, Optional

try:
    from ddtrace import tracer as dd_tracer
except ImportError:
    dd_tracer = None  # type: ignore

import agentlightning as agl


class DataDogTracer(agl.Tracer):
    """Tracer that sends spans to DataDog APM.
    
    Usage:
        tracer = DataDogTracer(service="my-agent")
        trainer = agl.Trainer(..., tracer=tracer)
    """

    def __init__(self, service: str = "agent-lightning"):
        if dd_tracer is None:
            raise ImportError("Install ddtrace: pip install ddtrace")
        self.service = service
        self.current_span: Optional[Any] = None

    def start_span(self, name: str, **attributes) -> Any:
        """Start a DataDog span."""
        self.current_span = dd_tracer.trace(
            name=name,
            service=self.service,
            resource=attributes.get("resource", name),
        )
        
        # Add custom tags
        for key, value in attributes.items():
            self.current_span.set_tag(key, value)
        
        return self.current_span

    def end_span(self, span: Any) -> None:
        """Finalize the DataDog span."""
        if span:
            span.finish()

    def add_event(self, name: str, **attributes) -> None:
        """Add event as DataDog span tags."""
        if self.current_span:
            # Events become span tags in DataDog
            self.current_span.set_tag(f"event.{name}", attributes)
            
            # Special handling for errors
            if name == "error":
                self.current_span.set_tag("error", True)
                self.current_span.set_tag("error.msg", attributes.get("message", ""))


class PrometheusMetricsAdapter(agl.Adapter):
    
    """Adapter that exports metrics to Prometheus.
    
    Usage:
        adapter = PrometheusMetricsAdapter()
        trainer = agl.Trainer(..., adapter=adapter)
    """

    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram
            self.reward_dist = Histogram('agent_reward', 'Agent reward distribution')
            self.correct_counter = Counter('agent_correct', 'Correct answers')
            self.error_counter = Counter('agent_errors', 'Agent errors')
        except ImportError:
            raise ImportError("Install prometheus_client: pip install prometheus_client")

    def extract(self, trace: Any) -> Optional[agl.Triplet]:
        """Extract triplet and export metrics."""


        # Assume trace has standard structure
        prompt = getattr(trace, 'prompt', '')
        response = getattr(trace, 'response', '')
        reward = getattr(trace, 'reward', 0.0)
        
        # Export to Prometheus
        self.reward_dist.observe(reward)
        if reward > 0.5:
            self.correct_counter.inc()
        if reward < 0:
            self.error_counter.inc()
        
        return agl.Triplet(prompt=prompt, response=response, reward=reward)


# Example usage

if __name__ == "__main__":

    print("DataDog Integration Example")

    print("\n To use DataDog tracing in your agent:")
    print("""
    from extensions.datadog_integration import DataDogTracer
    
    tracer = DataDogTracer(service="my-math-agent")
    trainer = agl.Trainer(
        algorithm=algorithm,
        tracer=tracer,  # Use DataDog tracer
        n_runners=10
    )
    trainer.fit(agent, train_data)
    
    """)
    print("\nSpans will appear in your DataDog APM dashboard.")

    print("=" * 60)