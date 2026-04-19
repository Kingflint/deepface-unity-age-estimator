import json

import pytest

from deepface_server import create_app
from deepface_server.config import Settings
from deepface_server.middleware.rate_limit import RateLimiter, TokenBucket


def test_token_bucket_drains_then_refills():
    now = [0.0]
    bucket = TokenBucket(capacity=2, refill_per_second=1.0, now=lambda: now[0])
    assert bucket.take() is True
    assert bucket.take() is True
    assert bucket.take() is False
    now[0] += 1.0
    assert bucket.take() is True


def test_rate_limiter_isolates_keys():
    now = [0.0]
    limiter = RateLimiter(capacity=1, refill_per_second=0.0)
    # Patch every bucket's clock so the test is deterministic.
    original = limiter._new_bucket

    def fake_bucket():
        return TokenBucket(1, 0.0, now=lambda: now[0])

    limiter._new_bucket = fake_bucket  # type: ignore[assignment]
    assert original  # silence unused
    assert limiter.hit("a") is True
    assert limiter.hit("a") is False
    assert limiter.hit("b") is True


@pytest.fixture
def limited_client(settings):
    s = Settings(**{**settings.__dict__, "rate_limit_per_minute": 2})
    return create_app(s).test_client()


def test_rate_limit_returns_429_after_burst(limited_client):
    payload = json.dumps({"image": ""})
    headers = {"Content-Type": "application/json"}

    statuses = [
        limited_client.post("/analyze", data=payload, headers=headers).status_code
        for _ in range(5)
    ]
    assert 429 in statuses, statuses
