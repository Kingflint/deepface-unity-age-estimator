from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from deepface_server.jobs import scheduler as s


def test_parse_field_wildcard():
    cron = s.CronExpression.parse("* * * * *")
    assert len(cron.minutes) == 60
    assert len(cron.hours) == 24


def test_parse_field_list():
    cron = s.CronExpression.parse("0,15,30 * * * *")
    assert list(cron.minutes) == [0, 15, 30]


def test_parse_field_range():
    cron = s.CronExpression.parse("1-5 * * * *")
    assert list(cron.minutes) == [1, 2, 3, 4, 5]


def test_parse_field_step():
    cron = s.CronExpression.parse("*/15 * * * *")
    assert list(cron.minutes) == [0, 15, 30, 45]


def test_parse_invalid_count():
    with pytest.raises(s.CronError):
        s.CronExpression.parse("* * *")


def test_parse_out_of_range():
    with pytest.raises(s.CronError):
        s.CronExpression.parse("60 * * * *")


def test_parse_step_invalid():
    with pytest.raises(s.CronError):
        s.CronExpression.parse("*/0 * * * *")


def test_matches_specific_time():
    cron = s.CronExpression.parse("30 14 1 6 *")
    when = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
    assert cron.matches(when)


def test_matches_negative():
    cron = s.CronExpression.parse("30 14 1 6 *")
    when = datetime(2024, 6, 1, 14, 31, tzinfo=timezone.utc)
    assert not cron.matches(when)


def test_next_run_after_advances_one_minute():
    cron = s.CronExpression.parse("* * * * *")
    after = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    nxt = cron.next_run_after(after)
    assert nxt == after + timedelta(minutes=1)


def test_next_run_specific():
    cron = s.CronExpression.parse("0 9 * * *")
    after = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    nxt = cron.next_run_after(after)
    assert nxt.hour == 9 and nxt.minute == 0


def test_scheduler_add_and_remove():
    sched = s.Scheduler()
    sched.add("daily", "0 0 * * *")
    assert len(sched) == 1
    assert sched.remove("daily")
    assert len(sched) == 0
    assert not sched.remove("missing")


def test_scheduler_duplicate():
    sched = s.Scheduler()
    sched.add("daily", "0 0 * * *")
    with pytest.raises(s.CronError):
        sched.add("daily", "0 1 * * *")


def test_scheduler_upcoming_sorted():
    sched = s.Scheduler()
    sched.add("hourly", "0 * * * *")
    sched.add("daily", "0 0 * * *")
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    upcoming = sched.upcoming(now, limit=2)
    assert upcoming[0][0] <= upcoming[1][0]
