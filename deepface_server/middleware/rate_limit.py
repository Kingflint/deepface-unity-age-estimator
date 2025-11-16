"""In-memory token bucket rate limiter, keyed by client IP + API key."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Callable

from flask import Flask, request

from ..config import Settings
from ..errors import RateLimited


class TokenBucket:
    def __init__(self, capacity: int, refill_per_second: float, now: Callable[[], float] = time.monotonic):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_per_second = refill_per_second
        self._now = now
        self._last = now()
        self._lock = threading.Lock()

    def take(self, amount: float = 1.0) -> bool:
        with self._lock:
            now = self._now()
            elapsed = now - self._last
            self._last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False


class RateLimiter:
    def __init__(self, capacity: int, refill_per_second: float):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._buckets: dict[str, TokenBucket] = defaultdict(self._new_bucket)
        self._lock = threading.Lock()

    def _new_bucket(self) -> TokenBucket:
        return TokenBucket(self.capacity, self.refill_per_second)

    def hit(self, key: str) -> bool:
        with self._lock:
            bucket = self._buckets[key]
        return bucket.take()


def _client_key() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() or request.remote_addr or "unknown"
    api_key = request.headers.get("X-API-Key", "anon")
    return f"{ip}:{api_key}"


def register_rate_limit(app: Flask, settings: Settings) -> None:
    if settings.rate_limit_per_minute <= 0:
        return
    refill = settings.rate_limit_per_minute / 60.0
    limiter = RateLimiter(settings.rate_limit_per_minute, refill)
    app.extensions["rate_limiter"] = limiter

    @app.before_request
    def _enforce():
        if request.path == "/" or request.path == "/healthz":
            return None
        if not limiter.hit(_client_key()):
            raise RateLimited("Rate limit exceeded")
        return None
