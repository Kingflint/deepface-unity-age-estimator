"""Pagination helpers using opaque base64 cursors.

Cursors carry the offset and an HMAC tag derived from the secret so they
cannot be forged or used across services with different secrets.
"""
from __future__ import annotations

import base64
import hmac
import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Page:
    """A page of results with a forward cursor."""

    items: tuple
    next_cursor: Optional[str]
    total: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "items": list(self.items),
            "next_cursor": self.next_cursor,
            "total": self.total,
        }


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()[:16]


def encode_cursor(offset: int, secret: str = "") -> str:
    """Build an opaque, HMAC-signed cursor token for ``offset``."""
    if offset < 0:
        raise ValueError("offset must be >= 0")
    payload = json.dumps({"o": offset}, separators=(",", ":")).encode("utf-8")
    sig = _sign(payload, secret)
    return _b64encode(payload) + "." + sig


def decode_cursor(cursor: str, secret: str = "") -> int:
    """Validate and decode a cursor; raises ``ValueError`` if tampered with."""
    if not cursor:
        raise ValueError("empty cursor")
    if "." not in cursor:
        raise ValueError("malformed cursor")
    body, sig = cursor.rsplit(".", 1)
    payload = _b64decode(body)
    if not hmac.compare_digest(sig, _sign(payload, secret)):
        raise ValueError("cursor signature mismatch")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cursor payload corrupt") from exc
    offset = int(decoded.get("o", 0))
    if offset < 0:
        raise ValueError("cursor offset negative")
    return offset


def paginate(
    items: Iterable,
    *,
    limit: int,
    cursor: Optional[str] = None,
    secret: str = "",
    total: Optional[int] = None,
) -> Page:
    """Slice ``items`` into a page of size ``limit``.

    ``items`` should be a list-like with stable ordering. The cursor is
    treated as a numeric offset; tamper-resistant via HMAC.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if limit > 1000:
        raise ValueError("limit cannot exceed 1000")
    materialised: List = list(items)
    start = 0
    if cursor:
        start = decode_cursor(cursor, secret)
    end = start + limit
    page_items = tuple(materialised[start:end])
    next_cursor = encode_cursor(end, secret) if end < len(materialised) else None
    return Page(items=page_items, next_cursor=next_cursor, total=total or len(materialised))


def split_into_pages(
    items: Iterable,
    *,
    page_size: int,
) -> List[Tuple[int, list]]:
    """Pre-compute every (page_index, items) tuple. Useful for batch jobs."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    materialised = list(items)
    out: List[Tuple[int, list]] = []
    for idx, start in enumerate(range(0, len(materialised), page_size)):
        out.append((idx, materialised[start : start + page_size]))
    return out


__all__ = [
    "Page",
    "decode_cursor",
    "encode_cursor",
    "paginate",
    "split_into_pages",
]
