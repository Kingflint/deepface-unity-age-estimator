from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from deepface_server.utils import time_utils as tu


def test_utcnow_is_aware():
    assert tu.utcnow().tzinfo is not None


def test_parse_iso_z():
    dt = tu.parse_iso8601("2024-01-02T03:04:05Z")
    assert dt.tzinfo is not None
    assert dt.year == 2024 and dt.month == 1


def test_parse_iso_offset():
    dt = tu.parse_iso8601("2024-01-02T03:04:05+02:00")
    assert dt.hour == 1  # converted to UTC


def test_parse_iso_naive_treated_utc():
    dt = tu.parse_iso8601("2024-01-02T03:04:05")
    assert dt.tzinfo is not None


def test_parse_iso_invalid():
    with pytest.raises(ValueError):
        tu.parse_iso8601("not-a-time")


def test_parse_iso_empty():
    with pytest.raises(ValueError):
        tu.parse_iso8601("")


def test_to_iso_round_trip():
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    s = tu.to_iso8601(dt)
    assert s.endswith("Z")
    dt2 = tu.parse_iso8601(s)
    assert dt2 == dt


def test_parse_duration():
    assert tu.parse_duration("1h30m") == timedelta(hours=1, minutes=30)
    assert tu.parse_duration("2d") == timedelta(days=2)
    assert tu.parse_duration("1w") == timedelta(weeks=1)


def test_parse_duration_invalid():
    with pytest.raises(ValueError):
        tu.parse_duration("xyz")


def test_format_duration():
    assert tu.format_duration(timedelta(hours=1, minutes=30)) == "1h30m"
    assert tu.format_duration(timedelta(0)) == "0s"


def test_is_within_retention():
    now = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
    ts = now - timedelta(hours=1)
    assert tu.is_within_retention(ts, timedelta(hours=2), now=now)
    assert not tu.is_within_retention(ts, timedelta(minutes=10), now=now)


def test_expired_items():
    now = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
    items = [
        ("a", now - timedelta(days=10)),
        ("b", now - timedelta(hours=1)),
    ]
    assert tu.expired_items(items, timedelta(days=1), now=now) == ["a"]


def test_humanise_relative_seconds():
    now = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
    assert "second" in tu.humanise_relative(now - timedelta(seconds=5), now=now)


def test_humanise_relative_hours():
    now = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
    assert "hour" in tu.humanise_relative(now - timedelta(hours=3), now=now)


def test_humanise_relative_future():
    now = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
    out = tu.humanise_relative(now + timedelta(hours=2), now=now)
    assert "from now" in out


def test_truncate_helpers():
    dt = datetime(2024, 6, 1, 12, 34, 56, 789, tzinfo=timezone.utc)
    assert tu.truncate_to_minute(dt).second == 0
    assert tu.truncate_to_hour(dt).minute == 0
    assert tu.truncate_to_day(dt).hour == 0
