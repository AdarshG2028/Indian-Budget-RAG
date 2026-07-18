"""
Shared telemetry bootstrap for business-logic modules.

Modules in this codebase are imported under two different sys.path layouts:
as ``src.<pkg>.<mod>`` when the repo root is on sys.path (pytest, uvicorn)
and as ``<pkg>.<mod>`` when only ``src/`` is on sys.path (the CLIs). The
telemetry manager must resolve under both, otherwise instrumentation
silently disappears depending on how the process was launched.
"""
import logging

logger = logging.getLogger(__name__)

try:
    from src.api.telemetry.manager import get_telemetry_manager
except ImportError:
    try:
        from api.telemetry.manager import get_telemetry_manager
    except ImportError:
        get_telemetry_manager = None
        logger.warning("TelemetryManager not available - observability disabled")


def get_telemetry():
    """Return the global TelemetryManager, or None if unavailable."""
    if get_telemetry_manager is None:
        return None
    return get_telemetry_manager()
