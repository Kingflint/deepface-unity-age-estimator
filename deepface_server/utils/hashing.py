"""Content hashing helpers for deduplication and cache keys."""
from __future__ import annotations

import hashlib
import hmac
from typing import Iterable, Mapping


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def short_hash(data: bytes, *, length: int = 12) -> str:
    """Return a truncated SHA-256 digest for compact cache keys."""
    if not 4 <= length <= 64:
        raise ValueError("length must be between 4 and 64")
    return sha256_hex(data)[:length]


def stable_dict_hash(value: Mapping) -> str:
    """Hash a mapping in a key-order-independent, deterministic way."""
    parts = sorted(
        (str(k), _normalise_value(v)) for k, v in value.items()
    )
    serialised = "\u001f".join(f"{k}={v}" for k, v in parts).encode("utf-8")
    return sha256_hex(serialised)


def _normalise_value(value) -> str:
    if isinstance(value, Mapping):
        return stable_dict_hash(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_normalise_value(v) for v in value) + "]"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def hmac_sign(key: str, data: bytes, *, length: int = 32) -> str:
    """Return a hex HMAC-SHA256 signature, optionally truncated."""
    sig = hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    if length < 8:
        raise ValueError("length must be >= 8")
    return sig[:length]


def hmac_verify(key: str, data: bytes, signature: str) -> bool:
    """Constant-time comparison of an HMAC signature."""
    expected = hmac_sign(key, data, length=len(signature))
    return hmac.compare_digest(expected, signature)


def chained_hash(items: Iterable[bytes]) -> str:
    """Hash a sequence of byte chunks, fold-left style.

    Equivalent to hashing the concatenation but avoids holding the whole
    buffer in memory.
    """
    h = hashlib.sha256()
    for chunk in items:
        h.update(chunk)
    return h.hexdigest()


__all__ = [
    "chained_hash",
    "hmac_sign",
    "hmac_verify",
    "sha256_hex",
    "short_hash",
    "stable_dict_hash",
]
