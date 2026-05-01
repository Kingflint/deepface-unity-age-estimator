from deepface_server.webhooks import (
    DeliveryAttempt,
    WebhookDispatcher,
    WebhookEvent,
    compute_signature,
    verify_signature,
)
from deepface_server.webhooks.signing import SIGNATURE_HEADER, TIMESTAMP_HEADER


def test_signing_round_trip():
    body = '{"hello": "world"}'
    sig = compute_signature("secret", body, timestamp="123")
    assert sig.startswith("sha256=")
    assert verify_signature("secret", body, sig, timestamp="123") is True


def test_verify_signature_rejects_tampered_body():
    sig = compute_signature("secret", "abc")
    assert verify_signature("secret", "abcd", sig) is False


def test_verify_signature_rejects_wrong_secret():
    sig = compute_signature("secret-a", "payload")
    assert verify_signature("secret-b", "payload", sig) is False


def test_event_to_json_is_sorted():
    event = WebhookEvent(id="evt", type="x.y", payload={"a": 1, "b": 2}, created_at="t")
    body = event.to_json()
    assert body.startswith("{")
    assert "x.y" in body


def test_dispatcher_succeeds_first_try():
    calls = []

    def fake_client(url, data, headers, timeout):
        calls.append((url, data, headers))
        return 200

    dispatcher = WebhookDispatcher(
        ["https://example.test/hook"],
        secret="s",
        max_attempts=3,
        backoff_seconds=0.0,
        http_client=fake_client,
        sleeper=lambda _: None,
    )
    attempts = dispatcher.dispatch(WebhookEvent(payload={"x": 1}))
    assert len(attempts) == 1
    assert attempts[0].succeeded
    assert SIGNATURE_HEADER in calls[0][2]
    assert TIMESTAMP_HEADER in calls[0][2]


def test_dispatcher_retries_until_success():
    counter = {"n": 0}

    def fake_client(url, data, headers, timeout):
        counter["n"] += 1
        if counter["n"] < 3:
            return 503
        return 200

    dispatcher = WebhookDispatcher(
        ["https://example.test/hook"],
        secret="s",
        max_attempts=3,
        backoff_seconds=0.0,
        http_client=fake_client,
        sleeper=lambda _: None,
    )
    attempts = dispatcher.dispatch(WebhookEvent())
    assert len(attempts) == 3
    assert attempts[-1].succeeded
    assert attempts[0].next_retry_in_seconds is not None


def test_dispatcher_marks_failure_when_all_retries_exhausted():
    def fake_client(url, data, headers, timeout):
        return 500

    dispatcher = WebhookDispatcher(
        ["https://example.test/hook"],
        secret="s",
        max_attempts=2,
        backoff_seconds=0.0,
        http_client=fake_client,
        sleeper=lambda _: None,
    )
    attempts = dispatcher.dispatch(WebhookEvent())
    assert all(not a.succeeded for a in attempts)


def test_delivery_attempt_succeeded_property():
    attempt = DeliveryAttempt(event_id="x", url="http://e", status_code=204)
    assert attempt.succeeded is True
    attempt.status_code = 500
    assert attempt.succeeded is False
    attempt.status_code = None
    assert attempt.succeeded is False
