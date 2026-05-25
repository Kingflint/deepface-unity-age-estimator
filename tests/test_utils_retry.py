from __future__ import annotations

import pytest

from deepface_server.utils import retry as r


def test_compute_backoff_no_jitter_growth():
    a = r.compute_backoff(1, base=1.0, factor=2.0, jitter=False)
    b = r.compute_backoff(2, base=1.0, factor=2.0, jitter=False)
    c = r.compute_backoff(3, base=1.0, factor=2.0, jitter=False)
    assert a == 1.0 and b == 2.0 and c == 4.0


def test_compute_backoff_capped():
    delay = r.compute_backoff(20, base=1.0, factor=2.0, cap=5.0, jitter=False)
    assert delay == 5.0


def test_compute_backoff_invalid_attempt():
    with pytest.raises(ValueError):
        r.compute_backoff(0)


def test_retry_succeeds_on_third():
    calls = {"n": 0}

    @r.retry(attempts=3, base_delay=0, jitter=False, sleep=lambda _x: None)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("nope")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_exhausts():
    @r.retry(attempts=3, base_delay=0, jitter=False, sleep=lambda _x: None)
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(r.RetryError) as exc_info:
        always_fails()
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, ValueError)


def test_retry_specific_exceptions_only():
    @r.retry(attempts=3, exceptions=(KeyError,), base_delay=0, jitter=False, sleep=lambda _x: None)
    def fails_with_value():
        raise ValueError("not retried")

    with pytest.raises(ValueError):
        fails_with_value()


def test_retry_invalid_attempts():
    with pytest.raises(ValueError):
        r.retry(attempts=0)


def test_retry_call_succeeds():
    state = {"n": 0}

    def f(x):
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError()
        return x * 2

    out = r.retry_call(f, 5, attempts=3, base_delay=0, jitter=False, sleep=lambda _x: None)
    assert out == 10


def test_on_retry_hook_called():
    calls = []

    def hook(attempt, exc):
        calls.append((attempt, type(exc).__name__))

    @r.retry(
        attempts=3,
        base_delay=0,
        jitter=False,
        sleep=lambda _x: None,
        on_retry=hook,
    )
    def flaky():
        raise RuntimeError("x")

    with pytest.raises(r.RetryError):
        flaky()
    assert len(calls) == 2  # called between failures, not after final
