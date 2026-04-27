"""EXIF orientation handling."""
from __future__ import annotations

from enum import IntEnum
from typing import Any


class EXIFOrientation(IntEnum):
    NORMAL = 1
    MIRROR_HORIZONTAL = 2
    ROTATE_180 = 3
    MIRROR_VERTICAL = 4
    MIRROR_HORIZONTAL_ROTATE_270 = 5
    ROTATE_90 = 6
    MIRROR_HORIZONTAL_ROTATE_90 = 7
    ROTATE_270 = 8


_ORIENTATION_TAG = 0x0112


def read_orientation(image: Any) -> EXIFOrientation:
    """Best-effort EXIF orientation extraction.

    Falls back to :data:`EXIFOrientation.NORMAL` whenever the metadata is
    missing, the image is a numpy array (no EXIF), or PIL is not installed.
    """
    try:
        from PIL import Image  # type: ignore

        if isinstance(image, Image.Image):
            exif = image.getexif() if hasattr(image, "getexif") else {}
            return EXIFOrientation(exif.get(_ORIENTATION_TAG, EXIFOrientation.NORMAL))
    except Exception:
        return EXIFOrientation.NORMAL
    return EXIFOrientation.NORMAL


def apply_orientation(image: Any, orientation: EXIFOrientation) -> Any:
    if orientation == EXIFOrientation.NORMAL:
        return image
    try:
        from PIL import Image  # type: ignore

        if not isinstance(image, Image.Image):
            return image
        if orientation == EXIFOrientation.MIRROR_HORIZONTAL:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        if orientation == EXIFOrientation.ROTATE_180:
            return image.rotate(180, expand=True)
        if orientation == EXIFOrientation.MIRROR_VERTICAL:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        if orientation == EXIFOrientation.ROTATE_90:
            return image.rotate(-90, expand=True)
        if orientation == EXIFOrientation.ROTATE_270:
            return image.rotate(-270, expand=True)
        if orientation == EXIFOrientation.MIRROR_HORIZONTAL_ROTATE_90:
            return image.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True)
        if orientation == EXIFOrientation.MIRROR_HORIZONTAL_ROTATE_270:
            return image.transpose(Image.FLIP_LEFT_RIGHT).rotate(-270, expand=True)
    except Exception:
        return image
    return image
