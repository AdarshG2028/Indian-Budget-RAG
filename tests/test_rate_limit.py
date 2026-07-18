"""
Tests for the rate limiting middleware and its storage backends.

Uses a minimal FastAPI app so no Qdrant/Groq services are needed.
"""
import asyncio
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.rate_limit_store import (
    RateLimitStore,
    RateLimitDecision,
    InMemoryRateLimiter
)


class UserIDMiddleware(BaseHTTPMiddleware):
    """Test stand-in for auth middleware: sets user_id from a header."""

    async def dispatch(self, request, call_next):
        user_id = request.headers.get("X-Test-User")
        if user_id:
            request.state.user_id = user_id
        return await call_next(request)


def build_app(
    max_requests: int = 3,
    window_seconds: int = 60,
    rules: dict | None = None,
    trust_forwarded_for: bool = False,
    api_key_header: str | None = None,
    store: RateLimitStore | None = None,
    with_user_middleware: bool = False
) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=max_requests,
        window_seconds=window_seconds,
        limited_path_prefixes=("/rag/",),
        rules=rules,
        trust_forwarded_for=trust_forwarded_for,
        api_key_header=api_key_header,
        store=store
    )
    if with_user_middleware:
        # Added after so it runs before the rate limiter
        app.add_middleware(UserIDMiddleware)

    @app.post("/rag/query")
    async def rag_query():
        return {"answer": "ok"}

    @app.post("/rag/query/stream")
    async def rag_query_stream():
        return {"answer": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# Original middleware behavior
# ---------------------------------------------------------------------------

def test_requests_under_limit_succeed():
    client = TestClient(build_app(max_requests=3))
    for _ in range(3):
        response = client.post("/rag/query")
        assert response.status_code == 200


def test_requests_over_limit_get_429():
    client = TestClient(build_app(max_requests=3))
    for _ in range(3):
        client.post("/rag/query")

    response = client.post("/rag/query")
    assert response.status_code == 429
    body = response.json()
    assert body["error_code"] == "RATE_LIMITED"
    assert body["details"]["limit"] == 3
    assert "Retry-After" in response.headers


def test_rate_limit_headers_present():
    client = TestClient(build_app(max_requests=3))
    response = client.post("/rag/query")
    assert response.headers["X-RateLimit-Limit"] == "3"
    assert response.headers["X-RateLimit-Remaining"] == "2"


def test_unlimited_paths_not_rate_limited():
    client = TestClient(build_app(max_requests=1))
    client.post("/rag/query")
    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" not in response.headers


def test_window_expiry_resets_limit(monkeypatch):
    import src.api.middleware.rate_limit_store as store_module

    fake_now = [1000.0]
    monkeypatch.setattr(
        store_module.time, "monotonic", lambda: fake_now[0]
    )

    client = TestClient(build_app(max_requests=1, window_seconds=60))
    assert client.post("/rag/query").status_code == 200
    assert client.post("/rag/query").status_code == 429

    fake_now[0] += 61
    assert client.post("/rag/query").status_code == 200


# ---------------------------------------------------------------------------
# Storage backend
# ---------------------------------------------------------------------------

def test_store_hit_decisions():
    async def run():
        store = InMemoryRateLimiter()
        first = await store.hit("k", 2, 60)
        second = await store.hit("k", 2, 60)
        third = await store.hit("k", 2, 60)
        return first, second, third

    first, second, third = asyncio.run(run())
    assert first.allowed and first.remaining == 1
    assert second.allowed and second.remaining == 0
    assert not third.allowed
    assert third.remaining == 0
    assert third.retry_after_seconds >= 1
    assert third.reset_after_seconds >= 1


def test_store_keys_are_independent():
    async def run():
        store = InMemoryRateLimiter()
        await store.hit("a", 1, 60)
        blocked = await store.hit("a", 1, 60)
        other = await store.hit("b", 1, 60)
        count = await store.active_clients()
        return blocked, other, count

    blocked, other, count = asyncio.run(run())
    assert not blocked.allowed
    assert other.allowed
    assert count == 2


def test_custom_store_is_used():
    class DenyAllStore(RateLimitStore):
        async def hit(self, key, max_requests, window_seconds):
            return RateLimitDecision(
                allowed=False, limit=max_requests, remaining=0,
                retry_after_seconds=7, reset_after_seconds=7
            )

        async def active_clients(self):
            return 0

    client = TestClient(build_app(store=DenyAllStore()))
    response = client.post("/rag/query")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "7"


# ---------------------------------------------------------------------------
# Per-endpoint rules
# ---------------------------------------------------------------------------

def test_per_endpoint_rule_overrides_default():
    rules = {"/rag/query/stream": {"requests": 1, "window": 60}}
    client = TestClient(build_app(max_requests=3, rules=rules))

    assert client.post("/rag/query/stream").status_code == 200
    assert client.post("/rag/query/stream").status_code == 429
    # Default-limited endpoint has its own bucket and higher limit
    for _ in range(3):
        assert client.post("/rag/query").status_code == 200


def test_rule_buckets_are_independent():
    rules = {"/rag/query/stream": {"requests": 5, "window": 60}}
    client = TestClient(build_app(max_requests=1, rules=rules))

    assert client.post("/rag/query").status_code == 200
    assert client.post("/rag/query").status_code == 429
    # Exhausting the default bucket does not consume the rule's bucket
    assert client.post("/rag/query/stream").status_code == 200


# ---------------------------------------------------------------------------
# Client key generation
# ---------------------------------------------------------------------------

def test_api_keys_get_separate_buckets():
    client = TestClient(
        build_app(max_requests=1, api_key_header="X-API-Key")
    )
    assert client.post(
        "/rag/query", headers={"X-API-Key": "key-a"}
    ).status_code == 200
    assert client.post(
        "/rag/query", headers={"X-API-Key": "key-a"}
    ).status_code == 429
    # A different API key from the same IP is a different client
    assert client.post(
        "/rag/query", headers={"X-API-Key": "key-b"}
    ).status_code == 200


def test_user_id_gets_separate_bucket():
    client = TestClient(
        build_app(max_requests=1, with_user_middleware=True)
    )
    assert client.post(
        "/rag/query", headers={"X-Test-User": "alice"}
    ).status_code == 200
    assert client.post(
        "/rag/query", headers={"X-Test-User": "alice"}
    ).status_code == 429
    assert client.post(
        "/rag/query", headers={"X-Test-User": "bob"}
    ).status_code == 200


def test_forwarded_for_used_only_when_trusted():
    trusted = TestClient(build_app(max_requests=1, trust_forwarded_for=True))
    assert trusted.post(
        "/rag/query", headers={"X-Forwarded-For": "1.1.1.1"}
    ).status_code == 200
    assert trusted.post(
        "/rag/query", headers={"X-Forwarded-For": "1.1.1.1"}
    ).status_code == 429
    assert trusted.post(
        "/rag/query", headers={"X-Forwarded-For": "2.2.2.2"}
    ).status_code == 200

    untrusted = TestClient(build_app(max_requests=1))
    assert untrusted.post(
        "/rag/query", headers={"X-Forwarded-For": "1.1.1.1"}
    ).status_code == 200
    # Spoofed header is ignored; still the same client (test socket IP)
    assert untrusted.post(
        "/rag/query", headers={"X-Forwarded-For": "2.2.2.2"}
    ).status_code == 429


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def test_reset_header_on_success_and_429():
    client = TestClient(build_app(max_requests=1, window_seconds=60))

    ok = client.post("/rag/query")
    assert 1 <= int(ok.headers["X-RateLimit-Reset"]) <= 60

    blocked = client.post("/rag/query")
    assert blocked.status_code == 429
    reset = int(blocked.headers["X-RateLimit-Reset"])
    assert 1 <= reset <= 60
    assert reset == int(blocked.headers["Retry-After"])


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

class TelemetryStub:
    def __init__(self):
        self.counters = []
        self.histograms = []

    def increment_counter(self, name, value=1, attributes=None):
        self.counters.append((name, attributes))

    def record_histogram(self, name, value, attributes=None):
        self.histograms.append((name, value))


def test_metrics_recorded(monkeypatch):
    import src.api.middleware.rate_limit as rate_limit_module
    from src.api.telemetry.metrics import MetricNames

    stub = TelemetryStub()
    monkeypatch.setattr(
        rate_limit_module, "get_telemetry_manager", lambda: stub
    )

    client = TestClient(build_app(max_requests=1))
    client.post("/rag/query")
    client.post("/rag/query")
    client.get("/health")

    counter_names = [name for name, _ in stub.counters]
    assert counter_names == [
        MetricNames.RATE_LIMIT_ALLOWED,
        MetricNames.RATE_LIMIT_BLOCKED
    ]
    tracked = [
        value for name, value in stub.histograms
        if name == MetricNames.RATE_LIMIT_TRACKED_CLIENTS
    ]
    assert tracked == [1, 1]
