"""
TelemetryManager abstraction layer for OpenTelemetry observability.

This provides a clean abstraction that keeps business logic decoupled from
OpenTelemetry implementation details.
"""
import logging
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class TelemetryManager:
    """
    Abstraction layer for OpenTelemetry operations.
    
    Provides a simple interface for tracing, metrics, and events without
    exposing OpenTelemetry implementation details to business logic.
    """
    
    def __init__(self):
        """Initialize TelemetryManager."""
        self._tracer = None
        self._meter = None
        self._logger = None
        self._initialized = False
        self._metrics = {}
        
    def initialize(self, tracer, meter, logger_provider=None):
        """
        Initialize with OpenTelemetry components.
        
        Args:
            tracer: OpenTelemetry tracer
            meter: OpenTelemetry meter
            logger_provider: Optional logger provider for log correlation
        """
        self._tracer = tracer
        self._meter = meter
        self._logger = logger_provider
        self._initialized = True
        logger.info("TelemetryManager initialized")
        
    def is_initialized(self) -> bool:
        """Check if telemetry manager is initialized."""
        return self._initialized
    
    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: str = "internal"
    ) -> ContextManager:
        """
        Start a new span with context manager.
        
        Args:
            name: Span name
            attributes: Span attributes
            kind: Span kind (internal, server, client, producer, consumer)
            
        Yields:
            Span object for adding events and attributes
        """
        if not self._initialized:
            logger.warning(f"Telemetry not initialized, skipping span: {name}")
            yield None
            return
            
        span = None
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode
            
            # Map kind string to SpanKind
            kind_map = {
                "internal": trace.SpanKind.INTERNAL,
                "server": trace.SpanKind.SERVER,
                "client": trace.SpanKind.CLIENT,
                "producer": trace.SpanKind.PRODUCER,
                "consumer": trace.SpanKind.CONSUMER
            }
            span_kind = kind_map.get(kind, trace.SpanKind.INTERNAL)
            
            span = self._tracer.start_span(name, kind=span_kind)
            if attributes:
                span.set_attributes(attributes)
            
            # Make span the current span
            token = trace.use_span(span, end_on_exit=False)
            token.__enter__()
            
            yield span
            
        except Exception as e:
            logger.error(f"Error in span {name}: {e}")
            if span:
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            if span:
                token.__exit__(None, None, None)
                span.end()
    
    def record_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Record an event on the current span.
        
        Args:
            name: Event name
            attributes: Event attributes
        """
        if not self._initialized:
            logger.warning(f"Telemetry not initialized, skipping event: {name}")
            return
            
        try:
            from opentelemetry import trace
            current_span = trace.get_current_span()
            
            if current_span and current_span.is_recording():
                current_span.add_event(name, attributes or {})
        except Exception as e:
            logger.error(f"Error recording event {name}: {e}")
    
    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Record an exception on the current span.
        
        Args:
            exception: Exception to record
            attributes: Additional attributes
        """
        if not self._initialized:
            logger.warning(f"Telemetry not initialized, skipping exception recording")
            return
            
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode
            
            current_span = trace.get_current_span()
            
            if current_span and current_span.is_recording():
                current_span.record_exception(exception)
                current_span.set_status(Status(StatusCode.ERROR, str(exception)))
                
                if attributes:
                    current_span.set_attributes(attributes)
        except Exception as e:
            logger.error(f"Error recording exception: {e}")
    
    def increment_counter(
        self,
        name: str,
        value: int = 1,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            attributes: Metric attributes
        """
        if not self._initialized:
            return
            
        try:
            if name not in self._metrics:
                from opentelemetry.metrics import Counter
                self._metrics[name] = self._meter.create_counter(
                    name,
                    description=f"Counter metric for {name}"
                )
            
            counter = self._metrics[name]
            counter.add(value, attributes or {})
        except Exception as e:
            logger.error(f"Error incrementing counter {name}: {e}")
    
    def record_histogram(
        self,
        name: str,
        value: float,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Record a histogram metric.
        
        Args:
            name: Metric name
            value: Value to record
            attributes: Metric attributes
        """
        if not self._initialized:
            return
            
        try:
            if name not in self._metrics:
                from opentelemetry.metrics import Histogram
                self._metrics[name] = self._meter.create_histogram(
                    name,
                    description=f"Histogram metric for {name}"
                )
            
            histogram = self._metrics[name]
            histogram.record(value, attributes or {})
        except Exception as e:
            logger.error(f"Error recording histogram {name}: {e}")
    
    def record_gauge(
        self,
        name: str,
        value: float,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Record a gauge metric.
        
        Args:
            name: Metric name
            value: Value to record
            attributes: Metric attributes
        """
        if not self._initialized:
            return
            
        try:
            if name not in self._metrics:
                from opentelemetry.metrics import ObservableGauge
                # For gauges, we typically use callbacks
                # This is a simplified version
                self._metrics[name] = self._meter.create_observable_gauge(
                    name,
                    [lambda x: value],
                    description=f"Gauge metric for {name}"
                )
        except Exception as e:
            logger.error(f"Error recording gauge {name}: {e}")
    
    def set_span_attribute(self, key: str, value: Any):
        """
        Set an attribute on the current span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self._initialized:
            return
            
        try:
            from opentelemetry import trace
            current_span = trace.get_current_span()
            
            if current_span and current_span.is_recording():
                current_span.set_attribute(key, value)
        except Exception as e:
            logger.error(f"Error setting span attribute {key}: {e}")
    
    def get_trace_id(self) -> Optional[str]:
        """
        Get current trace ID.
        
        Returns:
            Trace ID as hex string or None
        """
        if not self._initialized:
            return None
            
        try:
            from opentelemetry import trace
            current_span = trace.get_current_span()
            
            if current_span:
                from opentelemetry.trace import format_trace_id
                return format_trace_id(current_span.get_span_context().trace_id)
        except Exception as e:
            logger.error(f"Error getting trace ID: {e}")
        
        return None
    
    def get_span_id(self) -> Optional[str]:
        """
        Get current span ID.
        
        Returns:
            Span ID as hex string or None
        """
        if not self._initialized:
            return None
            
        try:
            from opentelemetry import trace
            current_span = trace.get_current_span()
            
            if current_span:
                from opentelemetry.trace import format_span_id
                return format_span_id(current_span.get_span_context().span_id)
        except Exception as e:
            logger.error(f"Error getting span ID: {e}")
        
        return None


# Global telemetry manager instance
_telemetry_manager = TelemetryManager()


def get_telemetry_manager() -> TelemetryManager:
    """
    Get the global telemetry manager instance.
    
    Returns:
        TelemetryManager instance
    """
    return _telemetry_manager
