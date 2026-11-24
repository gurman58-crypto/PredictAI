"""Client identification (document ownership) and rate limiting.

Ownership model: the React app creates a random anonymous client id once per browser and sends it
in the ``X-Client-Id`` header.  Every document query is filtered by that id, so one client can never
read, change or delete another client's documents (they get 404, which does not reveal whether the
document exists).  This is ownership scoping, NOT user authentication - see documentation/SECURITY.md.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, Request

from app.errors import ApiError

CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def optional_client_id(x_client_id: str | None = Header(default=None)) -> str | None:
    if x_client_id is None:
        return None
    if not CLIENT_ID_RE.match(x_client_id):
        raise ApiError(400, "invalid_client_id", "X-Client-Id must be 16-64 characters: letters, digits, '-' or '_'.")
    return x_client_id


def require_client_id(x_client_id: str | None = Header(default=None)) -> str:
    client_id = optional_client_id(x_client_id)
    if client_id is None:
        raise ApiError(401, "missing_client_id", "Send your anonymous client id in the X-Client-Id header.")
    return client_id


class RateLimiter:
    """Sliding-window limiter kept in memory (per process)."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Return ``None`` if allowed, otherwise the number of seconds to wait."""
        if self.limit <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            if len(self._hits) > 10_000:  # opportunistic clean-up of idle keys
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                    del self._hits[k]
            return None


def enforce_rate_limit(request: Request) -> None:
    limiter: RateLimiter = request.app.state.rate_limiter
    key = request.client.host if request.client else "unknown"
    wait = limiter.check(key)
    if wait is not None:
        raise ApiError(
            429,
            "rate_limited",
            "Too many requests. Please slow down.",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )
