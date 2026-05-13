"""Text helpers: sanitisation, slug, redaction, truncation."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


_SLUG_INVALID = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")


def slugify(value: str, *, max_length: int = 80, separator: str = "-") -> str:
    """Convert ``value`` to a URL-safe slug.

    - Unicode is normalised (NFKD) and ASCII-only characters are kept.
    - Whitespace and punctuation collapse to ``separator``.
    - Result is trimmed to ``max_length`` and stripped of leading/trailing
      separators.
    """
    if value is None:
        return ""
    normalised = unicodedata.normalize("NFKD", str(value))
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii").lower()
    cleaned = _SLUG_INVALID.sub(separator, ascii_only).strip(separator)
    if max_length > 0 and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(separator)
    return cleaned


def truncate(value: str, max_length: int, *, ellipsis: str = "…") -> str:
    """Return ``value`` shortened to ``max_length`` runes.

    The ellipsis (default ``…``) replaces the last character when truncation
    is required so the visible width remains <= ``max_length``.
    """
    if max_length <= 0:
        return ""
    if value is None:
        return ""
    s = str(value)
    if len(s) <= max_length:
        return s
    if len(ellipsis) >= max_length:
        return ellipsis[:max_length]
    return s[: max_length - len(ellipsis)].rstrip() + ellipsis


def normalise_whitespace(value: str) -> str:
    """Collapse runs of whitespace and trim ends."""
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def strip_control_chars(value: str) -> str:
    """Remove ASCII control characters (except common whitespace)."""
    if value is None:
        return ""
    return _CONTROL_CHARS.sub("", str(value))


def redact_emails(value: str, *, replacement: str = "[redacted-email]") -> str:
    if value is None:
        return ""
    return _EMAIL_RE.sub(replacement, str(value))


def redact_phones(value: str, *, replacement: str = "[redacted-phone]") -> str:
    if value is None:
        return ""
    return _PHONE_RE.sub(replacement, str(value))


def redact_pii(value: str) -> str:
    """Apply email + phone redaction in one pass."""
    return redact_phones(redact_emails(value))


def to_snake_case(value: str) -> str:
    """``CamelCase`` / ``camelCase`` -> ``snake_case``."""
    if value is None:
        return ""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(value))
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.replace("-", "_").replace(" ", "_").lower()


def to_camel_case(value: str) -> str:
    """``snake_case`` -> ``camelCase`` (preserves the first segment)."""
    if value is None:
        return ""
    parts = re.split(r"[_\s-]+", str(value))
    parts = [p for p in parts if p]
    if not parts:
        return ""
    head = parts[0].lower()
    tail = [p.title() for p in parts[1:]]
    return head + "".join(tail)


def deduplicate_lines(lines: Iterable[str]) -> List[str]:
    """Return ``lines`` with consecutive duplicates collapsed."""
    out: List[str] = []
    prev: str | None = None
    for line in lines:
        if line != prev:
            out.append(line)
        prev = line
    return out


def split_into_chunks(value: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not value:
        return []
    return [value[i : i + chunk_size] for i in range(0, len(value), chunk_size)]


__all__ = [
    "deduplicate_lines",
    "normalise_whitespace",
    "redact_emails",
    "redact_phones",
    "redact_pii",
    "slugify",
    "split_into_chunks",
    "strip_control_chars",
    "to_camel_case",
    "to_snake_case",
    "truncate",
]
