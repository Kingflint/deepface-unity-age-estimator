import time

import pytest

from deepface_server.security import (
    SignatureMismatch,
    TokenIssuer,
    sign_token,
    verify_token,
)


def test_sign_and_verify_round_trip():
    token = sign_token({"sub": "user-1"}, "secret")
    payload = verify_token(token, "secret")
    assert payload["sub"] == "user-1"


def test_verify_rejects_wrong_secret():
    token = sign_token({"sub": "x"}, "secret-a")
    with pytest.raises(SignatureMismatch):
        verify_token(token, "secret-b")


def test_verify_rejects_tampered_payload():
    token = sign_token({"sub": "x"}, "secret")
    head, sig = token.split(".")
    tampered = head + "abc" + "." + sig
    with pytest.raises(SignatureMismatch):
        verify_token(tampered, "secret")


def test_verify_rejects_missing_signature_segment():
    with pytest.raises(SignatureMismatch):
        verify_token("only-body", "secret")


def test_token_issuer_round_trip():
    issuer = TokenIssuer("my-secret", default_ttl_seconds=60)
    token = issuer.issue("user-42", scope="read", extra={"team": "a"})
    payload = issuer.verify(token)
    assert payload.sub == "user-42"
    assert payload.scope == "read"
    assert payload.extra and payload.extra["team"] == "a"


def test_token_issuer_rejects_expired():
    issuer = TokenIssuer("secret", default_ttl_seconds=-10)
    token = issuer.issue("user")
    with pytest.raises(SignatureMismatch):
        issuer.verify(token)


def test_token_issuer_requires_secret():
    with pytest.raises(ValueError):
        TokenIssuer("", default_ttl_seconds=60)


def test_token_issuer_overrides_ttl():
    issuer = TokenIssuer("secret", default_ttl_seconds=60)
    token = issuer.issue("u", ttl_seconds=120)
    payload = issuer.verify(token)
    assert payload.expires_at - payload.issued_at == 120


def test_token_issuer_rejects_missing_fields():
    token = sign_token({"only": "data"}, "secret")
    issuer = TokenIssuer("secret")
    with pytest.raises(SignatureMismatch):
        issuer.verify(token)


def test_token_payload_uses_now_for_iat():
    issuer = TokenIssuer("secret", default_ttl_seconds=10)
    token = issuer.issue("u")
    payload = issuer.verify(token)
    assert abs(payload.issued_at - int(time.time())) < 5
