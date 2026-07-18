"""
Rate limiting middleware to protect expensive endpoints (LLM calls via Groq).

Storage is pluggable via :class:`RateLimitStore` (in-memory by default);
see rate_limit_store.py for the algorithm rationale and how to swap in
a Redis backend.
"""
import hashlib
import logging
from dataclasses import dataclass
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Optional

from ..models import ErrorResponse, ErrorCode
from ..telemetry.manager import get_telemetry_manager
from ..telemetry.metrics import MetricNames
from .rate_limit_store import RateLimitStore, InMemoryRateLimiter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitRule:
    """
    A rate limit applied to paths starting with ``path_prefix``.

    Each rule has its own quota bucket per client, so hitting the limit
    on one endpoint does not consume quota for another.
    """
    path_prefix: str
    max_requests: int
    window_seconds: int


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to rate limit requests using a sliding window.

    Only paths starting with the configured prefixes are limited, so cheap
    endpoints (health checks, docs) are unaffected. Limits are tracked
    per client.

    Streaming (SSE) awareness: a request is counted once, at admission
    time — before the response body is produced. A long-lived streaming
    connection therefore consumes exactly one unit of quota, no matter
    how long it stays open or how many events it emits.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 10,
        window_seconds: int = 60,
        limited_path_prefixes: tuple[str, ...] = ("/rag/",),
        rules: Optional[dict[str, dict[str, int]]] = None,
        trust_forwarded_for: bool = False,
        api_key_header: Optional[str] = None,
        store: Optional[RateLimitStore] = None
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: ASGI application
            max_requests: Maximum requests allowed per window
            window_seconds: Sliding window size in seconds
            limited_path_prefixes: Path prefixes that trigger rate limiting
                with the default limit
            rules: Per-endpoint overrides mapping a path prefix to
                ``{"requests": N, "window": seconds}``, e.g.
                ``{"/rag/query/stream": {"requests": 5, "window": 60}}``.
                The longest matching prefix wins; paths matching only
                ``limited_path_prefixes`` use the default limit.
            trust_forwarded_for: Use the X-Forwarded-For header to identify
                clients. Enable ONLY when deployed behind a trusted reverse
                proxy that sets the header; otherwise clients can spoof it.
            api_key_header: Header carrying the client's API key; when
                present, its (hashed) value identifies the client instead
                of the IP address
            store: Rate limit state backend (defaults to in-memory;
                pass a Redis-backed implementation for multi-worker
                deployments)
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.limited_path_prefixes = limited_path_prefixes
        self.trust_forwarded_for = trust_forwarded_for
        self.api_key_header = api_key_header
        self.store = store if store is not None else InMemoryRateLimiter()
        # Longest prefix first so the most specific rule wins
        self._rules = sorted(
            (
                RateLimitRule(
                    path_prefix=prefix,
                    max_requests=cfg["requests"],
                    window_seconds=cfg["window"]
                )
                for prefix, cfg in (rules or {}).items()
            ),
            key=lambda r: len(r.path_prefix),
            reverse=True
        )
        self._default_rule = RateLimitRule(
            path_prefix="",
            max_requests=max_requests,
            window_seconds=window_seconds
        )

    def _resolve_rule(self, path: str) -> Optional[RateLimitRule]:
        """
        Find the rate limit rule applying to a path, if any.

        Returns the most specific per-endpoint rule, the default rule
        for other limited paths, or None when the path is not limited.
        """
        for rule in self._rules:
            if path.startswith(rule.path_prefix):
                return rule
        if any(path.startswith(p) for p in self.limited_path_prefixes):
            return self._default_rule
        return None

    def _get_client_key(self, request: Request) -> str:
        """
        Identify the client for rate limiting purposes.

        Priority order (most to least specific):
        1. API key from the configured header (hashed, so raw secrets
           never appear in logs, metrics, or store keys)
        2. Authenticated user ID, read from ``request.state.user_id``
           (set by future auth middleware — no changes needed here)
        3. Client IP: trusted X-Forwarded-For entry if enabled,
           otherwise the direct socket address
        """
        if self.api_key_header:
            api_key = request.headers.get(self.api_key_header)
            if api_key:
                digest = hashlib.sha256(api_key.encode()).hexdigest()[:16]
                return f"key:{digest}"
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        if self.trust_forwarded_for:
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                # First entry is the original client; later entries are proxies
                return f"ip:{forwarded_for.split(',')[0].strip()}"
        if request.client:
            return f"ip:{request.client.host}"
        return "ip:unknown"

    async def dispatch(self, request: Request, call_next):
        """
        Enforce the rate limit before processing the request.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response, or a 429 error response if the limit is exceeded
        """
        rule = self._resolve_rule(request.url.path)
        if rule is None:
            return await call_next(request)

        client_key = self._get_client_key(request)
        # Separate bucket per rule, so per-endpoint quotas are independent
        bucket = f"{client_key}:{rule.path_prefix or 'default'}"
        decision = await self.store.hit(
            key=bucket,
            max_requests=rule.max_requests,
            window_seconds=rule.window_seconds
        )

        # Rule prefix (not client) as attribute: bounded cardinality
        telemetry = get_telemetry_manager()
        metric_attributes = {"http.route": rule.path_prefix or "default"}
        telemetry.increment_counter(
            MetricNames.RATE_LIMIT_ALLOWED if decision.allowed
            else MetricNames.RATE_LIMIT_BLOCKED,
            attributes=metric_attributes
        )
        telemetry.record_histogram(
            MetricNames.RATE_LIMIT_TRACKED_CLIENTS,
            await self.store.active_clients()
        )

        if not decision.allowed:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.warning(
                f"Rate limit exceeded - client: {client_key}, "
                f"path: {request.url.path}, request_id: {request_id}, "
                f"retry_after: {decision.retry_after_seconds}s",
                extra={
                    "rate_limit.client": client_key,
                    "rate_limit.path": request.url.path,
                    "rate_limit.request_id": request_id,
                    "rate_limit.retry_after_seconds": decision.retry_after_seconds,
                    "rate_limit.limit": decision.limit,
                    "rate_limit.window_seconds": rule.window_seconds
                }
            )
            error_response = ErrorResponse(
                request_id=request_id,
                error_code=ErrorCode.RATE_LIMITED,
                message=(
                    f"Too many requests. Limit is {decision.limit} "
                    f"per {rule.window_seconds} seconds."
                ),
                details={
                    "retry_after_seconds": decision.retry_after_seconds,
                    "limit": decision.limit,
                    "window_seconds": rule.window_seconds
                }
            )
            return JSONResponse(
                status_code=429,
                content=error_response.model_dump(mode='json'),
                headers={
                    # Retry-After: delta-seconds per RFC 7231.
                    # X-RateLimit-Reset: delta-seconds until quota next
                    # replenishes, per the IETF ratelimit-headers draft
                    # (deliberately not epoch, immune to clock skew).
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(decision.reset_after_seconds)
                }
            )

        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        response.headers["X-RateLimit-Reset"] = str(decision.reset_after_seconds)
        return response
