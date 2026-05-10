"""Color space conversions and utilities.

Pure-Python helpers for working with RGB/HSV/HSL/HEX representations.
Used by image preprocessing, frontend overlays and the diagnostics
endpoint (``/admin/colors``) when reviewing colour distributions in
batches of input frames.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


RGB = Tuple[int, int, int]
HSV = Tuple[float, float, float]
HSL = Tuple[float, float, float]


@dataclass(frozen=True)
class Color:
    """An immutable colour with multiple representations."""

    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        for name, value in (("r", self.r), ("g", self.g), ("b", self.b)):
            if not 0 <= value <= 255:
                raise ValueError(f"channel {name}={value} outside 0..255")

    @classmethod
    def from_hex(cls, value: str) -> "Color":
        v = value.strip().lstrip("#")
        if len(v) == 3:
            v = "".join(ch * 2 for ch in v)
        if len(v) != 6:
            raise ValueError(f"invalid hex colour: {value!r}")
        try:
            r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
        except ValueError as exc:  # pragma: no cover - re-raised
            raise ValueError(f"invalid hex colour: {value!r}") from exc
        return cls(r, g, b)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float) -> "Color":
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp01(s), _clamp01(v))
        return cls(round(r * 255), round(g * 255), round(b * 255))

    def to_rgb(self) -> RGB:
        return (self.r, self.g, self.b)

    def to_hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(self.r, self.g, self.b)

    def to_hsv(self) -> HSV:
        h, s, v = colorsys.rgb_to_hsv(self.r / 255, self.g / 255, self.b / 255)
        return (h, s, v)

    def to_hsl(self) -> HSL:
        h, l, s = colorsys.rgb_to_hls(self.r / 255, self.g / 255, self.b / 255)
        return (h, s, l)

    def relative_luminance(self) -> float:
        """WCAG relative luminance, used for contrast checks."""

        def _ch(c: float) -> float:
            c = c / 255
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * _ch(self.r) + 0.7152 * _ch(self.g) + 0.0722 * _ch(self.b)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def contrast_ratio(a: Color, b: Color) -> float:
    """WCAG 2.1 contrast ratio between two colours (1.0 .. 21.0)."""
    l1, l2 = sorted([a.relative_luminance(), b.relative_luminance()], reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def euclidean_distance(a: Color, b: Color) -> float:
    """Plain RGB euclidean distance (max ~441.67)."""
    return ((a.r - b.r) ** 2 + (a.g - b.g) ** 2 + (a.b - b.b) ** 2) ** 0.5


def perceptual_distance(a: Color, b: Color) -> float:
    """Approximate perceptual distance using the redmean weighting.

    See https://www.compuphase.com/cmetric.htm. Faster than full CIE Lab
    while remaining a good visual proxy for small palettes.
    """
    rmean = (a.r + b.r) / 2
    dr = a.r - b.r
    dg = a.g - b.g
    db = a.b - b.b
    return (
        (2 + rmean / 256) * dr * dr
        + 4 * dg * dg
        + (2 + (255 - rmean) / 256) * db * db
    ) ** 0.5


def nearest_color(target: Color, palette: Sequence[Color]) -> Color:
    """Return the closest colour in ``palette`` by perceptual distance."""
    if not palette:
        raise ValueError("palette is empty")
    return min(palette, key=lambda c: perceptual_distance(target, c))


def average_color(colors: Iterable[Color]) -> Color:
    """Channel-wise mean of a non-empty iterable of colours."""
    items = list(colors)
    if not items:
        raise ValueError("colors is empty")
    n = len(items)
    return Color(
        round(sum(c.r for c in items) / n),
        round(sum(c.g for c in items) / n),
        round(sum(c.b for c in items) / n),
    )


def quantize(color: Color, levels: int = 8) -> Color:
    """Reduce each channel to ``levels`` discrete steps.

    Useful for histogram bucketing and palette extraction.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    step = 256 // levels
    return Color(
        min(255, (color.r // step) * step),
        min(255, (color.g // step) * step),
        min(255, (color.b // step) * step),
    )


def dominant_palette(colors: Iterable[Color], k: int = 5, levels: int = 8) -> List[Color]:
    """Return the ``k`` most common quantised colours.

    A lightweight stand-in for k-means clustering: we quantise each input
    colour, count occurrences, and emit the top ``k`` bucket centroids.
    Sufficient for diagnostic palettes and never needs SciPy.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    counts: dict = {}
    for c in colors:
        bucket = quantize(c, levels=levels)
        counts[bucket.to_rgb()] = counts.get(bucket.to_rgb(), 0) + 1
    if not counts:
        return []
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [Color(*rgb) for rgb, _ in ordered[:k]]


__all__ = [
    "Color",
    "average_color",
    "contrast_ratio",
    "dominant_palette",
    "euclidean_distance",
    "nearest_color",
    "perceptual_distance",
    "quantize",
]
