from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from deepface_server.services import quota as q


def test_consume_decrements():
    tracker = q.QuotaTracker(default_limit=3, default_window_seconds=60)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert tracker.consume("t1", now=now) == 2
    assert tracker.consume("t1", now=now) == 1
    assert tracker.consume("t1", now=now) == 0


def test_quota_exceeded_raised():
    tracker = q.QuotaTracker(default_limit=2, default_window_seconds=60)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    tracker.consume("t1", now=now)
    tracker.consume("t1", now=now)
    with pytest.raises(q.QuotaExceeded):
        tracker.consume("t1", now=now)


def test_quota_recovers_after_window():
    tracker = q.QuotaTracker(default_limit=2, default_window_seconds=60)
    base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    tracker.consume("t1", now=base)
    tracker.consume("t1", now=base)
    later = base + timedelta(seconds=120)
    assert tracker.consume("t1", now=later) >= 0


def test_remaining():
    tracker = q.QuotaTracker(default_limit=5, default_window_seconds=60)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert tracker.remaining("t1", now=now) == 5
    tracker.consume("t1", now=now)
    assert tracker.remaining("t1", now=now) == 4


def test_per_tenant_isolation():
    tracker = q.QuotaTracker(default_limit=1, default_window_seconds=60)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    tracker.consume("a", now=now)
    # b gets its own bucket
    assert tracker.consume("b", now=now) == 0


def test_set_override():
    tracker = q.QuotaTracker(default_limit=2, default_window_seconds=60)
    tracker.set_override("vip", limit=10, window_seconds=60)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    for _ in range(10):
        tracker.consume("vip", now=now)
    assert tracker.remaining("vip", now=now) == 0


def test_clear_override_reverts_to_default():
    tracker = q.QuotaTracker(default_limit=2, default_window_seconds=60)
    tracker.set_override("vip", limit=10, window_seconds=60)
    tracker.clear_override("vip")
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert tracker.remaining("vip", now=now) == 2


def test_reset():
    tracker = q.QuotaTracker(default_limit=2, default_window_seconds=60)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    tracker.consume("t1", now=now)
    tracker.reset("t1")
    assert tracker.remaining("t1", now=now) == 2


def test_invalid_window():
    with pytest.raises(q.QuotaError):
        q.QuotaWindow(limit=0, window_seconds=60)
    with pytest.raises(q.QuotaError):
        q.QuotaWindow(limit=1, window_seconds=0)


def test_retry_after_when_full():
    tracker = q.QuotaTracker(default_limit=1, default_window_seconds=60)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    tracker.consume("t1", now=now)
    try:
        tracker.consume("t1", now=now)
    except q.QuotaExceeded as exc:
        assert exc.retry_after > 0
        assert exc.tenant == "t1"
    else:
        pytest.fail("expected QuotaExceeded")
