"""OpenTelemetry tracing setup (optional)."""
import os
from typing import Optional

# OpenTelemetry is optional - only import if enabled
OTEL_ENABLED = os.getenv("ENABLE_OTEL", "false").lower() == "true"

if OTEL_ENABLED:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        OTEL_ENABLED = False
        print("Warning: OpenTelemetry packages not installed. Tracing disabled.")


def setup_tracing(service_name: str = "llm-eval-harness", endpoint: Optional[str] = None):
    """
    Setup OpenTelemetry tracing.

    Args:
        service_name: Name of the service
        endpoint: OTLP endpoint URL (default: from OTEL_ENDPOINT env var)

    Returns:
        Tracer instance or None if OTEL disabled
    """
    if not OTEL_ENABLED:
        return None

    # Create resource
    resource = Resource(attributes={
        "service.name": service_name
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add span processor based on configuration
    endpoint = endpoint or os.getenv("OTEL_ENDPOINT")

    if endpoint:
        # Use OTLP exporter for production
        exporter = OTLPSpanExporter(endpoint=endpoint)
    else:
        # Use console exporter for development
        exporter = ConsoleSpanExporter()

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Return tracer
    return trace.get_tracer(__name__)


def get_tracer():
    """
    Get the current tracer.

    Returns:
        Tracer instance or None if OTEL disabled
    """
    if not OTEL_ENABLED:
        return None

    return trace.get_tracer(__name__)
