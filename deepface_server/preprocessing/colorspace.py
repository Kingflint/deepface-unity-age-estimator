"""Color-space conversions with safe fallbacks."""
from __future__ import annotations

from typing import Any


def bgr_to_rgb(image: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr[..., ::-1].copy()
    except Exception:
        return image
    return image


def rgb_to_bgr(image: Any) -> Any:
    return bgr_to_rgb(image)


def ensure_rgb(image: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(image)
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            return arr[..., :3]
    except Exception:
        return image
    return image


def rgb_to_grayscale(image: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(image, dtype=float)
        if arr.ndim < 3:
            return arr
        weights = np.array([0.299, 0.587, 0.114])
        return (arr[..., :3] @ weights).astype(arr.dtype if arr.dtype.kind != "f" else float)
    except Exception:
        return image
