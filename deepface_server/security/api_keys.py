"""API key generation, hashing, and parsing.

Keys have the format ``df_<env>_<random>`` and are stored as a salted
SHA-256 digest. The plaintext key is shown to the user once at creation
time. Verification is constant time.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from typing import Optional

DEFAULT_PREFIX = "df"
ALLOWED_ENVS = ("live", "test", "dev")
RANDOM_BYTES = 24  # 192 bits of entropy → 32 base32 characters
HASH_ITERATIONS = 1


class APIKeyError(ValueError):
    """Raised when a key is malformed."""


@dataclass(frozen=True)
class APIKey:
    """A parsed API key.

    Attributes:
        prefix: Vendor prefix, defaults to ``df``.
        environment: One of ``ALLOWED_ENVS``.
        secret: The random portion of the key (base32, no padding).
    """

    prefix: str
    environment: str
    secret: str

    @property
    def public_id(self) -> str:
        """A short non-secret identifier safe to log (first 6 chars of secret)."""
        return f"{self.prefix}_{self.environment}_{self.secret[:6]}"

    def __str__(self) -> str:
        return f"{self.prefix}_{self.environment}_{self.secret}"


def _b32(data: bytes) -> str:
    import base64

    return base64.b32encode(data).decode("ascii").rstrip("=").lower()


def generate_api_key(environment: str = "live", prefix: str = DEFAULT_PREFIX) -> APIKey:
    """Generate a fresh API key with cryptographically random bytes."""
    if environment not in ALLOWED_ENVS:
        raise APIKeyError(f"environment must be one of {ALLOWED_ENVS!r}")
    if not prefix.isalnum():
        raise APIKeyError("prefix must be alphanumeric")
    secret = _b32(secrets.token_bytes(RANDOM_BYTES))
    return APIKey(prefix=prefix, environment=environment, secret=secret)


def parse_api_key(key: str) -> APIKey:
    """Parse a string back into an :class:`APIKey`."""
    if not key or not isinstance(key, str):
        raise APIKeyError("empty key")
    parts = key.strip().split("_")
    if len(parts) != 3:
        raise APIKeyError("expected format <prefix>_<env>_<secret>")
    prefix, env, secret = parts
    if not prefix.isalnum():
        raise APIKeyError("invalid prefix")
    if env not in ALLOWED_ENVS:
        raise APIKeyError("invalid environment")
    if len(secret) < 16 or not secret.isalnum():
        raise APIKeyError("invalid secret")
    return APIKey(prefix=prefix, environment=env, secret=secret)


def hash_api_key(key: str, *, salt: Optional[bytes] = None) -> str:
    """Return a salted SHA-256 hex digest of an API key.

    The salt is stored alongside the digest in the format
    ``<hex_salt>$<hex_digest>``. Pass ``salt`` for deterministic
    hashing in tests.
    """
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.sha256(salt + key.encode("utf-8")).hexdigest()
    return f"{salt.hex()}${digest}"


def verify_api_key(key: str, stored: str) -> bool:
    """Constant-time verification of a key against its stored hash."""
    try:
        hex_salt, expected_digest = stored.split("$", 1)
        salt = bytes.fromhex(hex_salt)
    except ValueError:
        return False
    candidate = hashlib.sha256(salt + key.encode("utf-8")).hexdigest()
    return hmac.compare_digest(candidate, expected_digest)


def is_test_key(key: APIKey) -> bool:
    return key.environment in ("test", "dev")


__all__ = [
    "ALLOWED_ENVS",
    "APIKey",
    "APIKeyError",
    "DEFAULT_PREFIX",
    "generate_api_key",
    "hash_api_key",
    "is_test_key",
    "parse_api_key",
    "verify_api_key",
]
