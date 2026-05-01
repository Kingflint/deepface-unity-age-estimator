"""HMAC-SHA256 signing utilities for outbound webhook payloads."""
from __future__ import annotations

import hashlib
import hmac


SIGNATURE_HEADER = "X-Signature-256"
TIMESTAMP_HEADER = "X-Signature-Timestamp"


def compute_signature(secret: str, payload: str | bytes, timestamp: str = "") -> str:
    if isinstance(payload, str):
        payload_bytes = payload.encode("utf-8")
    else:
        payload_bytes = payload
    if timestamp:
        message = timestamp.encode("utf-8") + b"." + payload_bytes
    else:
        message = payload_bytes
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(
    secret: str, payload: str | bytes, signature: str, timestamp: str = ""
) -> bool:
    expected = compute_signature(secret, payload, timestamp)
    return hmac.compare_digest(expected, signature or "")
