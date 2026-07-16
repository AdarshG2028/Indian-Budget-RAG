"""
Telemetry module for OpenTelemetry observability.
"""
from .manager import TelemetryManager
from .config import configure_telemetry

__all__ = ["TelemetryManager", "configure_telemetry"]
