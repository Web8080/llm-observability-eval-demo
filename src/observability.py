"""
Observability: LangSmith tracing (primary) or optional OpenTelemetry OTLP export.
Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT so every chain and LLM call
is traced; you get trace IDs, latency, token usage, and prompt/completion visibility in LangSmith.
If LangSmith is not used, OTLP endpoint can be set for environment-based trace export.
"""

import os

from .config import LANGCHAIN_PROJECT, LANGCHAIN_TRACING_V2, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME


def configure_tracing():
    """Apply env-based tracing. LangSmith is used when LANGCHAIN_TRACING_V2 is true."""
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT or "llm-observability-demo")
    if LANGCHAIN_TRACING_V2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        return
    if OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_SERVICE_NAME:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider()
            exporter = OTLPSpanExporter(
                endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=OTEL_EXPORTER_OTLP_ENDPOINT.strip().startswith("http://"),
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
        except Exception:
            pass
