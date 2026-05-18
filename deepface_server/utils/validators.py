"""Validators for emails, URLs, IP addresses and other simple inputs.

These run before any heavyweight processing so we never feed garbage to
DeepFace or to the persistence layer.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Iterable
from urllib.parse import urlparse


_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$"
)
_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)*"
    r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)$"
)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_email(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) > 254:
        return False
    return bool(_EMAIL_RE.match(value.strip()))


def is_url(value: str, *, schemes: Iterable[str] = ("http", "https")) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return False
    if parsed.scheme.lower() not in {s.lower() for s in schemes}:
        return False
    return bool(parsed.netloc)


def is_hostname(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return bool(_HOSTNAME_RE.match(value.strip()))


def is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(str(value).strip())
    except (ValueError, TypeError):
        return False
    return True


def is_private_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(str(value).strip()).is_private
    except (ValueError, TypeError):
        return False


def is_uuid(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_UUID_RE.match(value.strip()))


def is_safe_filename(value: str) -> bool:
    """True when ``value`` cannot escape its containing directory."""
    if not isinstance(value, str) or not value:
        return False
    forbidden = {"..", "/", "\\"}
    if any(token in value for token in forbidden):
        return False
    return value not in {".", ""} and not value.startswith(".")


def is_port(value) -> bool:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return False
    return 1 <= port <= 65535


def validate_image_dimensions(width: int, height: int, *, max_dim: int) -> None:
    """Raise ``ValueError`` if dimensions are missing or too large."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if width > max_dim or height > max_dim:
        raise ValueError(
            f"image dimensions {width}x{height} exceed limit {max_dim}"
        )


def validate_age(value: int) -> int:
    """Coerce an estimated age to a sane integer in [0, 120]."""
    age = int(value)
    if age < 0:
        return 0
    if age > 120:
        return 120
    return age


def normalise_email(value: str) -> str:
    """Lowercase the domain and trim whitespace; raise on invalid input."""
    if not is_email(value):
        raise ValueError(f"invalid email: {value!r}")
    local, _, domain = value.strip().partition("@")
    return f"{local}@{domain.lower()}"


__all__ = [
    "is_email",
    "is_hostname",
    "is_ip_address",
    "is_port",
    "is_private_ip",
    "is_safe_filename",
    "is_url",
    "is_uuid",
    "normalise_email",
    "validate_age",
    "validate_image_dimensions",
]
