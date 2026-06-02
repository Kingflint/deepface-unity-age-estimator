"""Password hashing using PBKDF2-HMAC-SHA256 with per-password salt.

Stored hashes use the format ``pbkdf2$<iterations>$<salt_b64>$<hash_b64>``.
This lets us bump iteration counts without breaking existing rows: the
verifier reads the iteration count straight out of the stored value.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from typing import Tuple


_DEFAULT_ITERATIONS = 240_000
_HASH_SCHEME = "pbkdf2"
_SALT_BYTES = 16
_HASH_BYTES = 32


class PasswordPolicyError(ValueError):
    """Raised when a password does not meet the configured policy."""


def hash_password(
    password: str,
    *,
    iterations: int = _DEFAULT_ITERATIONS,
    salt: bytes | None = None,
) -> str:
    """Return a self-describing PBKDF2 hash for ``password``."""
    if not isinstance(password, str) or password == "":
        raise PasswordPolicyError("password must be a non-empty string")
    if iterations < 10_000:
        raise ValueError("iterations must be at least 10,000")
    if salt is None:
        salt = os.urandom(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations, dklen=_HASH_BYTES
    )
    return _format(_HASH_SCHEME, iterations, salt, derived)


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check that ``password`` matches ``encoded``."""
    try:
        scheme, iterations, salt, expected = _parse(encoded)
    except ValueError:
        return False
    if scheme != _HASH_SCHEME:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations, dklen=len(expected)
    )
    return hmac.compare_digest(candidate, expected)


def needs_rehash(encoded: str, *, iterations: int = _DEFAULT_ITERATIONS) -> bool:
    """True when the stored hash uses fewer iterations than the current policy."""
    try:
        _, current_iterations, _, _ = _parse(encoded)
    except ValueError:
        return True
    return current_iterations < iterations


def enforce_policy(
    password: str,
    *,
    min_length: int = 12,
    require_digit: bool = True,
    require_symbol: bool = True,
    require_mixed_case: bool = True,
) -> None:
    """Raise :class:`PasswordPolicyError` when the password is too weak."""
    if not isinstance(password, str):
        raise PasswordPolicyError("password must be a string")
    if len(password) < min_length:
        raise PasswordPolicyError(
            f"password must be at least {min_length} characters"
        )
    if require_digit and not any(ch.isdigit() for ch in password):
        raise PasswordPolicyError("password must contain a digit")
    if require_symbol and password.isalnum():
        raise PasswordPolicyError("password must contain a symbol")
    if require_mixed_case:
        if password.lower() == password or password.upper() == password:
            raise PasswordPolicyError("password must contain mixed case")


def generate_strong_password(length: int = 20) -> str:
    """Produce a cryptographically random password meeting the default policy."""
    if length < 12:
        raise ValueError("length must be >= 12")
    alphabet = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*-_=+"
    )
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(length))
        try:
            enforce_policy(candidate)
        except PasswordPolicyError:
            continue
        return candidate


def _format(scheme: str, iterations: int, salt: bytes, derived: bytes) -> str:
    return "{}${}${}${}".format(
        scheme,
        iterations,
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(derived).decode("ascii").rstrip("="),
    )


def _parse(encoded: str) -> Tuple[str, int, bytes, bytes]:
    if not isinstance(encoded, str) or encoded.count("$") != 3:
        raise ValueError("malformed hash")
    scheme, iter_str, salt_b64, hash_b64 = encoded.split("$")
    iterations = int(iter_str)
    salt = _b64decode(salt_b64)
    derived = _b64decode(hash_b64)
    return scheme, iterations, salt, derived


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


__all__ = [
    "PasswordPolicyError",
    "enforce_policy",
    "generate_strong_password",
    "hash_password",
    "needs_rehash",
    "verify_password",
]
