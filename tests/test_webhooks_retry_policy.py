from __future__ import annotations

import random

import pytest

from deepface_server.webhooks import retry_policy as rp


def test_default_policy_valid():
    p = rp.RetryPolicy()
    assert p.max_attempts >= 1


def test_delay_grows_exponentially():
    p = rp.RetryPolicy(base_delay=1.0, factor=2.0, jitter=0.0, max_delay=100.0)
    assert p.delay_for(1) == 1.0
    assert p.delay_for(2) == 2.0
    assert p.delay_for(3) == 4.0


def test_delay_capped():
    p = rp.RetryPolicy(base_delay=1.0, factor=10.0, max_delay=5.0, jitter=0.0)
    assert p.delay_for(5) == 5.0


def test_delay_invalid_attempt():
    p = rp.RetryPolicy()
    with pytest.raises(rp.RetryPolicyError):
        p.delay_for(0)


def test_jitter_within_bounds():
    p = rp.RetryPolicy(base_delay=10.0, factor=1.0, max_delay=10.0, jitter=0.1)
    rand = random.Random(0)
    for _ in range(20):
        d = p.delay_for(1, rand=rand)
        assert 9.0 <= d <= 11.0


def test_schedule_length_matches_attempts():
    p = rp.RetryPolicy(max_attempts=5, jitter=0.0)
    sched = p.schedule()
    assert len(sched) == 5


def test_is_terminal():
    p = rp.RetryPolicy(max_attempts=3)
    assert not p.is_terminal(1)
    assert p.is_terminal(3)


def test_invalid_policy_validation():
    with pytest.raises(rp.RetryPolicyError):
        rp.RetryPolicy(max_attempts=0)
    with pytest.raises(rp.RetryPolicyError):
        rp.RetryPolicy(base_delay=0)
    with pytest.raises(rp.RetryPolicyError):
        rp.RetryPolicy(base_delay=10, max_delay=1)
    with pytest.raises(rp.RetryPolicyError):
        rp.RetryPolicy(factor=0.5)
    with pytest.raises(rp.RetryPolicyError):
        rp.RetryPolicy(jitter=2.0)


def test_presets_are_valid():
    assert rp.AGGRESSIVE_POLICY.max_attempts == 8
    assert rp.GENTLE_POLICY.jitter == 0.0
    assert rp.DEFAULT_POLICY.max_attempts == 5
