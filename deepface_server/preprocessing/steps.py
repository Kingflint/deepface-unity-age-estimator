"""Concrete preprocessing steps. All operate on numpy arrays when available
and otherwise fall back to byte-passthrough so unit tests can run without
heavy image libraries installed.
"""
from __future__ import annotations

from typing import Any

from .colorspace import bgr_to_rgb, rgb_to_grayscale
from .exif import apply_orientation, read_orientation
from .pipeline import PreprocessStep


class Resize(PreprocessStep):
    name = "resize"

    def __init__(self, target_size: tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.params = {"target_size": target_size}

    def apply(self, image: Any, metadata: dict) -> Any:
        try:
            import cv2  # type: ignore

            return cv2.resize(image, self.target_size)
        except Exception:
            metadata["resize_skipped"] = True
            return image


class Normalize(PreprocessStep):
    name = "normalize"

    def __init__(self, mean: float = 0.0, std: float = 255.0):
        self.mean = mean
        self.std = std
        self.params = {"mean": mean, "std": std}

    def apply(self, image: Any, metadata: dict) -> Any:
        try:
            import numpy as np  # type: ignore

            arr = np.asarray(image, dtype=np.float32)
            return (arr - self.mean) / max(self.std, 1e-6)
        except Exception:
            metadata["normalize_skipped"] = True
            return image


class GrayscaleConvert(PreprocessStep):
    name = "grayscale"

    def __init__(self, keep_channels: bool = False):
        self.keep_channels = keep_channels
        self.params = {"keep_channels": keep_channels}

    def apply(self, image: Any, metadata: dict) -> Any:
        gray = rgb_to_grayscale(image)
        if not self.keep_channels:
            return gray
        try:
            import numpy as np  # type: ignore

            return np.stack([gray, gray, gray], axis=-1)
        except Exception:
            return gray


class EXIFRotate(PreprocessStep):
    name = "exif_rotate"

    def apply(self, image: Any, metadata: dict) -> Any:
        try:
            orientation = read_orientation(image)
            metadata["exif_orientation"] = orientation
            return apply_orientation(image, orientation)
        except Exception:
            return image


class Clahe(PreprocessStep):
    """Contrast limited adaptive histogram equalization."""

    name = "clahe"

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.params = {"clip_limit": clip_limit, "tile_grid_size": tile_grid_size}

    def apply(self, image: Any, metadata: dict) -> Any:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            arr = np.asarray(image)
            if arr.ndim == 3:
                lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
                l_channel = lab[:, :, 0]
                clahe = cv2.createCLAHE(self.clip_limit, self.tile_grid_size)
                lab[:, :, 0] = clahe.apply(l_channel)
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            clahe = cv2.createCLAHE(self.clip_limit, self.tile_grid_size)
            return clahe.apply(arr)
        except Exception:
            metadata["clahe_skipped"] = True
            return image


class FaceCropPlaceholder(PreprocessStep):
    """Light-weight crop using the largest detected square region.

    The actual face detection happens inside the analyzer; this step is here so
    the preprocessing pipeline can be configured to crop to a centered square
    of given fraction before handing off.
    """

    name = "face_crop"

    def __init__(self, fraction: float = 0.9):
        self.fraction = max(0.1, min(1.0, fraction))
        self.params = {"fraction": self.fraction}

    def apply(self, image: Any, metadata: dict) -> Any:
        try:
            import numpy as np  # type: ignore

            arr = np.asarray(image)
            if arr.ndim < 2:
                return image
            h, w = arr.shape[:2]
            side = int(min(h, w) * self.fraction)
            y0 = max(0, (h - side) // 2)
            x0 = max(0, (w - side) // 2)
            return arr[y0 : y0 + side, x0 : x0 + side]
        except Exception:
            metadata["face_crop_skipped"] = True
            return image
