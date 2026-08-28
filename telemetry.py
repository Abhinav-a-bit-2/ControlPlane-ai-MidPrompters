import os
import contextlib
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

# Initialize tracer
resource = Resource.create({SERVICE_NAME: "control-plane-rag"})
provider = TracerProvider(resource=resource)
# Export to Phoenix OTLP gRPC endpoint running in docker
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

@contextlib.contextmanager
def trace_span(name: str, attributes: dict = None, span_kind: trace.SpanKind = trace.SpanKind.INTERNAL):
    """
    Helper to quickly trace steps.
    """
    with tracer.start_as_current_span(name, kind=span_kind) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span

def set_llm_attributes(span, model_name: str, prompt_tokens: int, completion_tokens: int):
    """
    Sets standard OpenInference attributes so Phoenix recognizes this as an LLM call
    and automatically calculates the total cost.
    """
    span.set_attribute("openinference.span.kind", "LLM")
    span.set_attribute("llm.model_name", model_name)
    if prompt_tokens > 0:
        span.set_attribute("llm.token_count.prompt", prompt_tokens)
    if completion_tokens > 0:
        span.set_attribute("llm.token_count.completion", completion_tokens)


def init_telemetry():
    """
    Called on app startup. Also auto-instruments libraries if needed.
    """
    # Note: openinference-instrumentation-groq / chromadb can be used 
    # to auto-instrument those specific libraries for Arize Phoenix
    # For now we rely on our manual spans and any future auto-instrumentation
    pass
