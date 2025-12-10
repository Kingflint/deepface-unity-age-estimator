"""Helpers for working with base64 payloads and data URIs."""
from __future__ import annotations

DATA_URI_PREFIX = "data:"


def strip_data_uri(value: str) -> str:
    """If the input is a ``data:image/...;base64,XXXX`` URI return the XXXX part."""
    if not value.startswith(DATA_URI_PREFIX):
        return value
    comma = value.find(",")
    if comma == -1:
        return value
    return value[comma + 1 :]


def looks_like_base64(value: str) -> bool:
    if not value:
        return False
    cleaned = strip_data_uri(value).strip()
    if not cleaned:
        return False
    if len(cleaned) % 4 != 0:
        return False
    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    return all(ch in allowed for ch in cleaned)
