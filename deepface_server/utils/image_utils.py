"""Pure-Python image helpers (no OpenCV dependency)."""
from __future__ import annotations

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WEBP_MAGIC = b"RIFF"
WEBP_TAIL = b"WEBP"


def detect_format(data: bytes) -> str:
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(WEBP_MAGIC) and data[8:12] == WEBP_TAIL:
        return "webp"
    return "unknown"


def estimate_dimensions(data: bytes) -> tuple[int, int] | None:
    """Best-effort dimension extraction from PNG headers (used for early reject)."""
    if data.startswith(PNG_MAGIC) and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height
    return None
