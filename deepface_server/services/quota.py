"""Sliding-window quota tracking.

Each tenant gets a configurable budget per window (e.g. 1000 calls per
hour). The implementation uses an in-memory ring of timestamps, which
is fine for a single-process worker; the persistence layer pushes the
state to Redis when running in distributed mode.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict


class QuotaError(ValueError):
    pass


class QuotaExceeded(Exception):
    """Raised when a request would exceed the tenant's allowance."""

    def __init__(self, tenant: str, retry_after: float) -> None:
        super().__init__(f"quota exceeded for {tenant}; retry in {retry_after:.1f}s")
        self.tenant = tenant
        self.retry_after = retry_after


@dataclass
class QuotaWindow:
    limit: int
    window_seconds: float
    calls: Deque[datetime] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise QuotaError("limit must be positive")
        if self.window_seconds <= 0:
            raise QuotaError("window_seconds must be positive")

    def _trim(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()

    def record(self, now: datetime) -> None:
        self._trim(now)
        self.calls.append(now)

    def remaining(self, now: datetime) -> int:
        self._trim(now)
        return max(0, self.limit - len(self.calls))

    def retry_after(self, now: datetime) -> float:
        self._trim(now)
        if len(self.calls) < self.limit:
            return 0.0
        oldest = self.calls[0]
        delta = (oldest + timedelta(seconds=self.window_seconds)) - now
        return max(0.0, delta.total_seconds())


class QuotaTracker:
    """Holds per-tenant quota windows."""

    def __init__(self, default_limit: int, default_window_seconds: float) -> None:
        self._default_limit = default_limit
        self._default_window = default_window_seconds
        self._windows: Dict[str, QuotaWindow] = {}
        self._overrides: Dict[str, tuple] = {}

    def set_override(self, tenant: str, *, limit: int, window_seconds: float) -> None:
        self._overrides[tenant] = (limit, window_seconds)
        self._windows.pop(tenant, None)

    def clear_override(self, tenant: str) -> None:
        self._overrides.pop(tenant, None)
        self._windows.pop(tenant, None)

    def _window(self, tenant: str) -> QuotaWindow:
        win = self._windows.get(tenant)
        if win is not None:
            return win
        if tenant in self._overrides:
            limit, secs = self._overrides[tenant]
        else:
            limit, secs = self._default_limit, self._default_window
        win = QuotaWindow(limit=limit, window_seconds=secs)
        self._windows[tenant] = win
        return win

    def consume(self, tenant: str, *, now: "datetime | None" = None) -> int:
        """Record a call and return remaining quota.

        Raises :class:`QuotaExceeded` if the tenant has no remaining
        budget in the current window.
        """
        moment = now or datetime.now(timezone.utc)
        win = self._window(tenant)
        if win.remaining(moment) <= 0:
            raise QuotaExceeded(tenant, win.retry_after(moment))
        win.record(moment)
        return win.remaining(moment)

    def remaining(self, tenant: str, *, now: "datetime | None" = None) -> int:
        moment = now or datetime.now(timezone.utc)
        return self._window(tenant).remaining(moment)

    def reset(self, tenant: str) -> None:
        self._windows.pop(tenant, None)


__all__ = ["QuotaError", "QuotaExceeded", "QuotaTracker", "QuotaWindow"]
