


# Fixes for Version 2

Pre-deploy review findings. Two issues were fixed directly (see git history)
rather than deferred:
- Docker image bloat: unnecessary `data/` copy + missing `.dockerignore`.
- Full GPU torch pulled in for CPU-only inference: `torch` now resolves to
  the CPU-only wheel on Linux via `[tool.uv.sources]` in `pyproject.toml`,
  dropping the entire `nvidia-*`/`cuda-toolkit`/`triton` stack (~2.5GB) from
  the lockfile. Required adding `torch` as an explicit direct dependency —
  `tool.uv.sources` overrides don't apply to a package that's only pulled in
  transitively (via `sentence-transformers`), only to ones uv sees as part
  of the resolved requirement graph directly.

Everything below is deferred to v2.

## 1. No authentication by default

`src/api/config.py:39` — `require_auth: bool = False`, `api_keys: list[str] = []`.

As shipped, anyone who finds the deployed URL can call `/rag/query` (which
spends Groq API quota/cost) with zero auth. The only protection is a shared
rate limit of 10 requests/60s (`config.py:47-49`).

**Fix:** set `REQUIRE_AUTH=true` and populate `API_KEYS` via env before any
public deployment. Confirm the auth check is actually wired into the router
dependencies (`src/api/dependencies.py`) if not already.

## 2. CORS wide open

`src/api/config.py:33` — `cors_origins: list[str] = ["*"]`.

Any website's JS can call the API cross-origin. Fine for local dev; once a
real frontend exists, restrict `CORS_ORIGINS` to that frontend's domain(s)
via env rather than the wildcard default.

## 3. Error responses leak exception internals

`src/api/middleware/error_handler.py:65` — on any unhandled 500, the JSON
response body includes `details={"exception": str(e)}`, sending raw Python
exception text to the client. It's already logged server-side correctly
(lines 68-73); the client-facing copy should be dropped or replaced with a
generic message.

## 4. Unbounded local trace file in production

`src/api/telemetry/config.py` — the file span exporter writes to
`traces/spans.log` unconditionally, regardless of environment. On a
long-running production container this grows without bound. Either gate it
on `environment == "development"` or add rotation/size limits.

## 5. In-memory rate limiter

`src/api/middleware/rate_limit_store.py` — `InMemoryRateLimiter` is
per-process. Fine for a single instance; if this ever runs multiple
replicas/workers, limits won't be shared across them, and a restart resets
all counters. Swap to a shared backend (Redis) if scaling beyond one
instance.

## 6. Container runs as root

Neither `Dockerfile` nor `Dockerfile.dev` declares a `USER`, so the app runs
as root inside the container. Standard hardening: add a non-root user and
`USER` directive. Not urgent, but cheap to fix alongside the others.
