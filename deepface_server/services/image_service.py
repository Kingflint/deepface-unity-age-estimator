"""Image decoding + size enforcement, kept independent from Flask."""
from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any

from ..errors import ImageDecodeError, ImageTooLarge


class ImageService:
    def __init__(self, max_bytes: int = 5 * 1024 * 1024, max_dimension: int = 4096):
        self.max_bytes = max_bytes
        self.max_dimension = max_dimension

    def decode_b64(self, encoded: str) -> bytes:
        if not isinstance(encoded, str) or not encoded:
            raise ImageDecodeError("image payload is empty")
        # Reject payloads that are obviously too large before allocating numpy buffers.
        # Each base64 character represents 6 bits, so estimate raw size as len * 0.75.
        estimated = int(len(encoded) * 0.75)
        if estimated > self.max_bytes:
            raise ImageTooLarge(
                f"image is too large: {estimated} bytes (max {self.max_bytes})"
            )
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageDecodeError(f"could not decode base64 image: {exc}") from exc
        if len(raw) > self.max_bytes:
            raise ImageTooLarge(f"image is too large: {len(raw)} bytes (max {self.max_bytes})")
        if not raw:
            raise ImageDecodeError("image payload is empty")
        return raw

    def fingerprint(self, raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def to_ndarray(self, raw: bytes) -> Any:
        """Decode bytes into a BGR ndarray via OpenCV. Imported lazily."""
        try:
            import cv2  # noqa: WPS433
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImageDecodeError(f"OpenCV is not available: {exc}") from exc

        np_array = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageDecodeError("OpenCV could not decode the image")

        height, width = img.shape[:2]
        if max(height, width) > self.max_dimension:
            raise ImageTooLarge(
                f"image dimension {max(height, width)} exceeds limit {self.max_dimension}"
            )
        return img
