"""Time helpers: ISO parsing, durations, retention windows."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple


_DURATION_RE = re.compile(
    r"^\s*(?:(?P<weeks>\d+)w)?\s*(?:(?P<days>\d+)d)?\s*(?:(?P<hours>\d+)h)?"
    r"\s*(?:(?P<minutes>\d+)m)?\s*(?:(?P<seconds>\d+)s)?\s*$",
    re.IGNORECASE,
)


def utcnow() -> datetime:
    """Timezone-aware ``datetime.utcnow`` replacement (UTC)."""
    return datetime.now(timezone.utc)


def parse_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware ``datetime``.

    Accepts both ``Z`` and explicit ``+HH:MM`` offsets. Naive timestamps
    are interpreted as UTC.
    """
    if not value:
        raise ValueError("empty timestamp")
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso8601(dt: datetime) -> str:
    """Format ``dt`` as an ISO-8601 string with millisecond precision."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{dt.microsecond // 1000:03d}Z"
    )


def parse_duration(text: str) -> timedelta:
    """Parse a compact duration like ``2h30m`` or ``1d12h`` into ``timedelta``.

    Supported units: ``w``, ``d``, ``h``, ``m``, ``s``. Order is flexible
    but each unit may appear at most once.
    """
    if text is None:
        raise ValueError("duration is None")
    m = _DURATION_RE.match(text)
    if not m or not any(m.groupdict().values()):
        raise ValueError(f"invalid duration: {text!r}")
    parts = {k: int(v) for k, v in m.groupdict().items() if v}
    weeks = parts.pop("weeks", 0)
    return timedelta(weeks=weeks, **parts)


def format_duration(delta: timedelta) -> str:
    """Human-friendly duration string like ``"3d4h12m"``."""
    total = int(delta.total_seconds())
    if total == 0:
        return "0s"
    sign = "-" if total < 0 else ""
    total = abs(total)
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds and not days:
        parts.append(f"{seconds}s")
    return sign + "".join(parts) if parts else f"{sign}0s"


def is_within_retention(
    timestamp: datetime,
    retention: timedelta,
    *,
    now: Optional[datetime] = None,
) -> bool:
    """Return ``True`` when ``timestamp`` is younger than ``retention``."""
    reference = now or utcnow()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return reference - timestamp <= retention


def expired_items(
    items: Iterable[Tuple[str, datetime]],
    retention: timedelta,
    *,
    now: Optional[datetime] = None,
) -> List[str]:
    """Return identifiers of ``items`` whose timestamp exceeds ``retention``."""
    reference = now or utcnow()
    expired: list[str] = []
    for ident, ts in items:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if reference - ts > retention:
            expired.append(ident)
    return expired


def humanise_relative(timestamp: datetime, *, now: Optional[datetime] = None) -> str:
    """Convert a timestamp into ``"3 hours ago"`` style English text."""
    reference = now or utcnow()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    delta = reference - timestamp
    seconds = int(delta.total_seconds())
    future = seconds < 0
    seconds = abs(seconds)

    if seconds < 60:
        unit, value = "second", seconds or 1
    elif seconds < 3600:
        unit, value = "minute", seconds // 60
    elif seconds < 86_400:
        unit, value = "hour", seconds // 3600
    elif seconds < 604_800:
        unit, value = "day", seconds // 86_400
    elif seconds < 2_592_000:
        unit, value = "week", seconds // 604_800
    elif seconds < 31_536_000:
        unit, value = "month", seconds // 2_592_000
    else:
        unit, value = "year", seconds // 31_536_000

    plural = "" if value == 1 else "s"
    suffix = "from now" if future else "ago"
    return f"{value} {unit}{plural} {suffix}"


def truncate_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def truncate_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def truncate_to_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


__all__ = [
    "expired_items",
    "format_duration",
    "humanise_relative",
    "is_within_retention",
    "parse_duration",
    "parse_iso8601",
    "to_iso8601",
    "truncate_to_day",
    "truncate_to_hour",
    "truncate_to_minute",
    "utcnow",
]
