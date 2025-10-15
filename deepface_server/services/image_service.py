"""Image decoding helpers."""
from __future__ import annotations

import base64
import binascii
import hashlib

from ..errors import ImageDecodeError


class ImageService:
    def __init__(self, max_bytes: int = 5 * 1024 * 1024):
        self.max_bytes = max_bytes

    def decode_b64(self, encoded: str) -> bytes:
        if not encoded:
            raise ImageDecodeError("image payload is empty")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageDecodeError(f"could not decode base64 image: {exc}") from exc
        if len(raw) > self.max_bytes:
            raise ImageDecodeError(f"image is too large: {len(raw)} bytes")
        return raw

    def fingerprint(self, raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()