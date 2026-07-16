"""
OpenTelemetry configuration and SDK setup.

Configures tracing, metrics, and logging with proper resource attributes,
semantic conventions, and sampling strategies.
"""
import logging
import os
import platform
from pathlib import Path
from typing import Optional
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ALWAYS_ON, ALWAYS_OFF
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

logger = logging.getLogger(__name__)


def get_resource_attributes() -> Resource:
    """
    Create resource attributes with semantic conventions.
    
    Returns:
        Resource with proper service metadata
    """
    return Resource.create({
        SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "indian-budget-rag"),
        "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
        "deployment.environment": os.getenv("OTEL_ENVIRONMENT", "development"),
        "host.name": platform.node(),
        "python.version": platform.python_version(),
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": "1.22.0",
    })


def get_sampler():
    """
    Get sampling strategy based on environment.
    
    Returns:
        Sampler instance
    """
    environment = os.getenv("OTEL_ENVIRONMENT", "development")
    sampling_ratio = float(os.getenv("OTEL_SAMPLING_RATIO", "1.0"))
    
    if environment == "development":
        logger.info("Using ALWAYS_ON sampler for development")
        return ALWAYS_ON
    else:
        logger.info(f"Using TraceIdRatioBased sampler with ratio: {sampling_ratio}")
        return TraceIdRatioBased(sampling_ratio)


def configure_telemetry(app=None, environment: str = "development"):
    """
    Configure OpenTelemetry SDK with tracing, metrics, and logging.
    
    Args:
        app: Optional FastAPI app for instrumentation
        environment: Environment (development, production)
    """
    logger.info(f"Configuring OpenTelemetry for environment: {environment}")
    
    # Set environment variable
    os.environ["OTEL_ENVIRONMENT"] = environment
    
    # Create resource
    resource = get_resource_attributes()
    
    # Configure Tracing
    configure_tracing(resource, environment)
    
    # Configure Metrics
    configure_metrics(resource, environment)
    
    # Configure Logging
    configure_logging()
    
    # Instrument FastAPI if provided
    if app:
        instrument_fastapi(app)
    
    # Instrument HTTP clients
    instrument_http_clients()
    
    logger.info("OpenTelemetry configuration complete")


def configure_tracing(resource: Resource, environment: str):
    """
    Configure tracing with appropriate exporters.
    
    Args:
        resource: Resource attributes
        environment: Environment name
    """
    # Create tracer provider
    tracer_provider = TracerProvider(
        resource=resource,
        sampler=get_sampler()
    )
    
    # Use file-based exporter for development to see traces locally
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    import sys
    import os
    
    # Create traces directory if it doesn't exist
    traces_dir = Path("traces")
    traces_dir.mkdir(exist_ok=True)
    
    # Create a custom file exporter
    class FileSpanExporter:
        def __init__(self, file_path):
            self.file_path = file_path
            
        def export(self, spans):
            with open(self.file_path, 'a') as f:
                for span in spans:
                    f.write(f"Span: {span.name}\n")
                    f.write(f"  Trace ID: {span.context.trace_id}\n")
                    f.write(f"  Span ID: {span.context.span_id}\n")
                    f.write(f"  Parent ID: {span.parent}\n")
                    f.write(f"  Start: {span.start_time}\n")
                    f.write(f"  End: {span.end_time}\n")
                    f.write(f"  Attributes: {span.attributes}\n")
                    f.write(f"  Events: {span.events}\n")
                    f.write(f"  Status: {span.status}\n")
                    f.write("-" * 80 + "\n")
            return 0
        
        def shutdown(self):
            pass
    
    file_exporter = FileSpanExporter(traces_dir / "spans.log")
    tracer_provider.add_span_processor(SimpleSpanProcessor(file_exporter))
    logger.info(f"Using file span exporter: {traces_dir / 'spans.log'}")
    
    # Also add OTLP exporter if Docker is available
    try:
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"Also using OTLP span exporter: {otlp_endpoint}")
    except Exception as e:
        logger.warning(f"Could not configure OTLP exporter: {e}")
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Initialize TelemetryManager
    from .manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()
    telemetry_manager.initialize(
        tracer=trace.get_tracer(__name__),
        meter=None,  # Will be set in configure_metrics
        logger_provider=None
    )


def configure_metrics(resource: Resource, environment: str):
    """
    Configure metrics with appropriate exporters.
    
    Args:
        resource: Resource attributes
        environment: Environment name
    """
    # Skip metrics export when using Jaeger (Jaeger doesn't support metrics)
    # Jaeger is primarily a tracing backend, not a metrics backend
    # For metrics, you would typically use Prometheus + Grafana
    logger.info("Skipping metrics export (Jaeger doesn't support metrics - use Prometheus for metrics)")
    
    # Create a no-op meter provider for now
    meter_provider = MeterProvider(resource=resource)
    
    # Set global meter provider
    metrics.set_meter_provider(meter_provider)
    
    # Update TelemetryManager with meter
    from .manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()
    telemetry_manager._meter = metrics.get_meter(__name__)


def configure_logging():
    """
    Configure logging instrumentation for log-trace correlation.
    """
    try:
        LoggingInstrumentor().instrument(
            set_logging_format=True,
            log_level=logging.INFO
        )
        logger.info("Logging instrumentation configured")
    except Exception as e:
        logger.error(f"Failed to configure logging instrumentation: {e}")


def instrument_fastapi(app):
    """
    Instrument FastAPI with auto-instrumentation.
    
    Args:
        app: FastAPI application
    """
    try:
        # Exclude health endpoints from tracing
        excluded_urls = [
            "/health",
            "/health/live",
            "/health/ready",
            "/api/v1/health",
            "/api/v1/health/live",
            "/api/v1/health/ready"
        ]
        
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            excluded_urls=",".join(excluded_urls)
        )
        logger.info("FastAPI instrumentation configured (health endpoints excluded)")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


def instrument_http_clients():
    """
    Instrument HTTP clients for external service calls.
    """
    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX client instrumentation configured")
    except Exception as e:
        logger.error(f"Failed to instrument HTTPX: {e}")


def shutdown_telemetry():
    """
    Shutdown OpenTelemetry SDK gracefully.
    """
    try:
        trace.get_tracer_provider().shutdown()
        metrics.get_meter_provider().shutdown()
        logger.info("OpenTelemetry SDK shutdown complete")
    except Exception as e:
        logger.error(f"Error during telemetry shutdown: {e}")
