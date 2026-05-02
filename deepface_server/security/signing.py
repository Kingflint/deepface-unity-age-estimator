"""Lightweight token signing using HMAC-SHA256 + base64url."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any


class SignatureMismatch(Exception):
    """Raised when a signature does not match the payload."""


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_token(payload: dict[str, Any], secret: str) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return f"{_b64url_encode(body)}.{_b64url_encode(digest)}"


def verify_token(token: str, secret: str) -> dict[str, Any]:
    if "." not in token:
        raise SignatureMismatch("token missing signature segment")
    body_b64, sig_b64 = token.split(".", 1)
    try:
        body = _b64url_decode(body_b64)
        signature = _b64url_decode(sig_b64)
    except (ValueError, base64.binascii.Error) as exc:
        raise SignatureMismatch("invalid base64") from exc
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, signature):
        raise SignatureMismatch("signature mismatch")
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SignatureMismatch("body is not valid JSON") from exc
