"""Tiny benchmark for the cache and image service (no network calls)."""
from __future__ import annotations

import base64
import io
import time

from deepface_server.services.cache import LRUCache
from deepface_server.services.image_service import ImageService


def _png_bytes(size: int = 64) -> bytes:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        raise SystemExit("Pillow is required for benchmarks")
    img = Image.new("RGB", (size, size), (128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    raw = _png_bytes()
    encoded = base64.b64encode(raw).decode("ascii")

    image_service = ImageService()
    cache = LRUCache(max_entries=128)

    iterations = 5_000
    start = time.perf_counter()
    for i in range(iterations):
        decoded = image_service.decode_b64(encoded)
        fp = image_service.fingerprint(decoded)
        if cache.get(fp) is None:
            cache.set(fp, {"i": i})
    elapsed = time.perf_counter() - start
    rps = iterations / elapsed
    print(f"ImageService.decode + fingerprint: {iterations} ops in {elapsed:.3f}s ({rps:,.0f}/s)")
    print("cache stats:", cache.stats())


if __name__ == "__main__":
    main()
