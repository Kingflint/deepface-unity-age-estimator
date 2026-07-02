"""Cron-lite scheduler used for periodic background work.

We support a small but useful subset of cron syntax:

- five fields: ``minute hour day-of-month month day-of-week``
- ``*`` wildcards
- comma-separated lists ``1,5,30``
- ranges ``1-5``
- step values ``*/5`` and ``2-30/3``

Everything is computed in UTC.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Sequence


class CronError(ValueError):
    pass


_FIELD_RANGES = (
    (0, 59),  # minute
    (0, 23),  # hour
    (1, 31),  # day of month
    (1, 12),  # month
    (0, 6),  # day of week (0 = Mon for our scheduler)
)


def _parse_field(expr: str, lo: int, hi: int) -> List[int]:
    out: set[int] = set()
    for part in expr.split(","):
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)
            if step <= 0:
                raise CronError("step must be positive")
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if start < lo or end > hi or start > end:
            raise CronError(f"value out of range: {part}")
        out.update(range(start, end + 1, step))
    return sorted(out)


@dataclass(frozen=True)
class CronExpression:
    minutes: Sequence[int]
    hours: Sequence[int]
    days_of_month: Sequence[int]
    months: Sequence[int]
    days_of_week: Sequence[int]

    @classmethod
    def parse(cls, expr: str) -> "CronExpression":
        parts = expr.split()
        if len(parts) != 5:
            raise CronError("expected 5 fields")
        fields = [_parse_field(p, lo, hi) for p, (lo, hi) in zip(parts, _FIELD_RANGES)]
        return cls(*fields)  # type: ignore[arg-type]

    def matches(self, when: datetime) -> bool:
        when = when.astimezone(timezone.utc)
        # Map weekday: Python Monday=0..Sunday=6 — same as our convention.
        return (
            when.minute in self.minutes
            and when.hour in self.hours
            and when.day in self.days_of_month
            and when.month in self.months
            and when.weekday() in self.days_of_week
        )

    def next_run_after(self, after: datetime, *, max_iterations: int = 60 * 24 * 366) -> datetime:
        """Return the next datetime strictly after ``after`` matching this cron."""
        candidate = after.astimezone(timezone.utc).replace(second=0, microsecond=0)
        candidate += timedelta(minutes=1)
        for _ in range(max_iterations):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise CronError("no match found within search window")


@dataclass(frozen=True)
class ScheduledTask:
    name: str
    cron: CronExpression

    def next_run(self, now: datetime) -> datetime:
        return self.cron.next_run_after(now)


class Scheduler:
    """Holds named tasks and reports the next ones due."""

    def __init__(self) -> None:
        self._tasks: List[ScheduledTask] = []

    def add(self, name: str, cron_expr: str) -> ScheduledTask:
        for existing in self._tasks:
            if existing.name == name:
                raise CronError(f"duplicate task: {name}")
        task = ScheduledTask(name=name, cron=CronExpression.parse(cron_expr))
        self._tasks.append(task)
        return task

    def remove(self, name: str) -> bool:
        for i, task in enumerate(self._tasks):
            if task.name == name:
                del self._tasks[i]
                return True
        return False

    def upcoming(self, now: datetime, limit: int = 10) -> List[tuple]:
        out = [(task.next_run(now), task.name) for task in self._tasks]
        out.sort(key=lambda kv: kv[0])
        return out[:limit]

    def __len__(self) -> int:
        return len(self._tasks)


__all__ = [
    "CronError",
    "CronExpression",
    "ScheduledTask",
    "Scheduler",
]
