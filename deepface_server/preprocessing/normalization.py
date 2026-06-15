"""Numeric normalisation helpers used before feeding tensors to models."""
from __future__ import annotations

from math import sqrt
from typing import List, Sequence, Tuple

Number = float
Vector = Sequence[Number]


class NormalizationError(ValueError):
    """Raised on invalid input."""


def min_max(values: Vector, *, target_min: float = 0.0, target_max: float = 1.0) -> List[float]:
    """Linearly scale values into ``[target_min, target_max]``.

    Constant input is mapped to the midpoint of the target range to
    avoid divide-by-zero.
    """
    if target_max <= target_min:
        raise NormalizationError("target_max must be > target_min")
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0:
        mid = (target_min + target_max) / 2.0
        return [mid for _ in values]
    out_span = target_max - target_min
    return [target_min + (v - lo) * out_span / span for v in values]


def z_score(values: Vector) -> List[float]:
    """Subtract the mean and divide by the standard deviation."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = sqrt(var)
    if sd == 0:
        return [0.0 for _ in values]
    return [(v - mean) / sd for v in values]


def unit_vector(values: Vector) -> List[float]:
    """Scale a vector to unit L2 norm."""
    n = sum(v * v for v in values)
    if n == 0:
        return [0.0 for _ in values]
    norm = sqrt(n)
    return [v / norm for v in values]


def softmax(values: Vector) -> List[float]:
    """Stable softmax: subtracts the max before exponentiating."""
    if not values:
        return []
    from math import exp

    m = max(values)
    exps = [exp(v - m) for v in values]
    s = sum(exps)
    if s == 0:
        return [0.0 for _ in values]
    return [e / s for e in exps]


def clip(values: Vector, lo: float, hi: float) -> List[float]:
    if hi < lo:
        raise NormalizationError("hi must be >= lo")
    return [min(hi, max(lo, v)) for v in values]


def standardise_rgb(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Scale RGB byte values into ``[0.0, 1.0]``."""
    r, g, b = rgb
    return (r / 255.0, g / 255.0, b / 255.0)


def imagenet_normalise(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Apply ImageNet mean/std normalisation to a unit RGB pixel."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return tuple((v - m) / s for v, m, s in zip(rgb, mean, std))  # type: ignore[return-value]


__all__ = [
    "NormalizationError",
    "clip",
    "imagenet_normalise",
    "min_max",
    "softmax",
    "standardise_rgb",
    "unit_vector",
    "z_score",
]
