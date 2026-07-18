"""
Storage backends for rate limiting.

Algorithm choice
----------------
A sliding window (per-client deque of request timestamps) was chosen over
a fixed window or token bucket because:

- Fixed windows allow bursts of up to 2x the limit at window boundaries
  (e.g. 10 requests at 11:59:59 and 10 more at 12:00:00). Each request
  here triggers a paid Groq API call, so boundary bursts matter.
- The sliding window is exact: at any instant, at most ``max_requests``
  requests are admitted in any trailing ``window_seconds`` interval.
- With one deque per client the cost is O(1) amortized per request
  (each timestamp is appended once and popped once).

Cleanup
-------
Client entries live in a dict keyed by client identity. Expired
timestamps are pruned lazily on each hit for that client; entries for
clients that went idle (no timestamps inside the largest window seen)
are dropped in a periodic sweep so memory does not grow with the number
of distinct clients ever seen.

Swapping in Redis later
-----------------------
Implement :class:`RateLimitStore` with the same ``hit`` semantics —
atomically record-and-check, returning a :class:`RateLimitDecision`.
In Redis this maps to a sorted set per key (ZREMRANGEBYSCORE to expire,
ZCARD to count, ZADD to record) executed as a Lua script for atomicity,
plus a TTL on the key so idle clients expire server-side (no sweep
needed). Then pass the instance to ``RateLimitMiddleware(store=...)``;
the middleware needs no other changes. ``hit`` is async precisely so a
networked backend can await I/O.
"""
import abc
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    """
    Outcome of recording one request attempt against a rate limit.

    Attributes:
        allowed: Whether the request is admitted
        limit: Maximum requests allowed in the window
        remaining: Requests left in the current window (0 when blocked)
        retry_after_seconds: Whole seconds until a blocked client may
            retry (0 when allowed)
        reset_after_seconds: Whole seconds until quota next replenishes,
            i.e. the oldest tracked request leaves the window
    """
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int
    reset_after_seconds: int


class RateLimitStore(abc.ABC):
    """
    Interface for rate limit state storage.

    Implementations must make ``hit`` atomic per key: check the current
    count and record the request in one step, so two concurrent requests
    cannot both be admitted into the last remaining slot.
    """

    @abc.abstractmethod
    async def hit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> RateLimitDecision:
        """
        Record a request attempt for ``key`` and decide whether to admit it.

        Args:
            key: Bucket identity (client + endpoint group)
            max_requests: Maximum requests allowed per window
            window_seconds: Sliding window size in seconds

        Returns:
            Decision including remaining quota and retry/reset timings
        """

    @abc.abstractmethod
    async def active_clients(self) -> int:
        """Number of buckets currently tracked (for observability)."""


class InMemoryRateLimiter(RateLimitStore):
    """
    Sliding-window rate limiter backed by process-local memory.

    Safe without locks: FastAPI middleware dispatch runs on the event
    loop and ``hit`` contains no awaits between reading and writing the
    deque, so each call is effectively atomic within one process.

    Note: state is per-process. If running multiple workers or replicas,
    each worker enforces the limit independently (use a Redis-backed
    store for a shared budget — see module docstring).
    """

    def __init__(self):
        self._request_times: dict[str, deque[float]] = defaultdict(deque)
        self._last_cleanup = time.monotonic()
        # Largest window seen; used as both sweep interval and staleness
        # horizon so no live bucket is ever dropped.
        self._max_window_seconds = 60

    async def hit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> RateLimitDecision:
        now = time.monotonic()
        self._max_window_seconds = max(self._max_window_seconds, window_seconds)
        self._cleanup_stale_clients(now)

        timestamps = self._request_times[key]
        window_start = now - window_seconds

        # Drop timestamps outside the sliding window
        while timestamps and timestamps[0] <= window_start:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            seconds_until_slot = timestamps[0] + window_seconds - now
            wait = max(1, math.ceil(seconds_until_slot))
            return RateLimitDecision(
                allowed=False,
                limit=max_requests,
                remaining=0,
                retry_after_seconds=wait,
                reset_after_seconds=wait
            )

        timestamps.append(now)
        return RateLimitDecision(
            allowed=True,
            limit=max_requests,
            remaining=max(0, max_requests - len(timestamps)),
            retry_after_seconds=0,
            reset_after_seconds=max(
                1, math.ceil(timestamps[0] + window_seconds - now)
            )
        )

    async def active_clients(self) -> int:
        return len(self._request_times)

    def _cleanup_stale_clients(self, now: float) -> None:
        """Drop buckets with no requests inside the largest window."""
        if now - self._last_cleanup < self._max_window_seconds:
            return
        horizon = now - self._max_window_seconds
        stale = [
            key for key, times in self._request_times.items()
            if not times or times[-1] <= horizon
        ]
        for key in stale:
            del self._request_times[key]
        self._last_cleanup = now
