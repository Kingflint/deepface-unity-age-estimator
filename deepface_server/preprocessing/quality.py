"""Quality scoring for input images.

The functions here operate on raw 2D pixel grids (lists of lists of
integers in the range 0-255). They are deliberately framework-free so
they can be used in tests without numpy or Pillow.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Sequence, Tuple

Pixel = Sequence[int]
Grid = Sequence[Sequence[int]]


class QualityError(ValueError):
    """Raised when an input grid is malformed."""


def _validate(grid: Grid) -> Tuple[int, int]:
    rows = len(grid)
    if rows == 0:
        raise QualityError("empty grid")
    cols = len(grid[0])
    if cols == 0:
        raise QualityError("empty row")
    for row in grid:
        if len(row) != cols:
            raise QualityError("rows must be the same length")
    return rows, cols


def brightness(grid: Grid) -> float:
    """Average luminance (0-255)."""
    rows, cols = _validate(grid)
    total = 0
    for row in grid:
        total += sum(row)
    return total / (rows * cols)


def contrast(grid: Grid) -> float:
    """Population standard deviation of luminance (0-127.5 typical max)."""
    rows, cols = _validate(grid)
    n = rows * cols
    mean = brightness(grid)
    sq = 0.0
    for row in grid:
        for v in row:
            d = v - mean
            sq += d * d
    return sqrt(sq / n)


def dynamic_range(grid: Grid) -> int:
    """``max - min`` value across the grid."""
    _validate(grid)
    lo = 255
    hi = 0
    for row in grid:
        for v in row:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
    return hi - lo


def sharpness(grid: Grid) -> float:
    """A simple Laplacian-variance-style sharpness score.

    The kernel used is::

        0  1  0
        1 -4  1
        0  1  0

    which approximates the Laplacian. Higher values mean sharper.
    """
    rows, cols = _validate(grid)
    if rows < 3 or cols < 3:
        return 0.0
    values: List[float] = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            v = (
                grid[r - 1][c]
                + grid[r + 1][c]
                + grid[r][c - 1]
                + grid[r][c + 1]
                - 4 * grid[r][c]
            )
            values.append(float(v))
    n = len(values)
    mean = sum(values) / n
    sq = sum((v - mean) ** 2 for v in values)
    return sq / n


def is_blurry(grid: Grid, threshold: float = 50.0) -> bool:
    """True when the sharpness is below ``threshold``."""
    return sharpness(grid) < threshold


def is_overexposed(grid: Grid, white_ratio: float = 0.5, white_value: int = 245) -> bool:
    """True when too much of the image is near-white."""
    rows, cols = _validate(grid)
    near_white = sum(1 for row in grid for v in row if v >= white_value)
    return near_white / (rows * cols) >= white_ratio


def is_underexposed(grid: Grid, black_ratio: float = 0.5, black_value: int = 10) -> bool:
    """True when too much of the image is near-black."""
    rows, cols = _validate(grid)
    near_black = sum(1 for row in grid for v in row if v <= black_value)
    return near_black / (rows * cols) >= black_ratio


@dataclass(frozen=True)
class QualityReport:
    brightness: float
    contrast: float
    sharpness: float
    dynamic_range: int
    blurry: bool
    overexposed: bool
    underexposed: bool

    @property
    def acceptable(self) -> bool:
        return not (self.blurry or self.overexposed or self.underexposed)

    def issues(self) -> List[str]:
        out: List[str] = []
        if self.blurry:
            out.append("blurry")
        if self.overexposed:
            out.append("overexposed")
        if self.underexposed:
            out.append("underexposed")
        return out


def evaluate(grid: Grid, *, sharpness_threshold: float = 50.0) -> QualityReport:
    """Compute a full :class:`QualityReport` for an image grid."""
    return QualityReport(
        brightness=brightness(grid),
        contrast=contrast(grid),
        sharpness=sharpness(grid),
        dynamic_range=dynamic_range(grid),
        blurry=is_blurry(grid, sharpness_threshold),
        overexposed=is_overexposed(grid),
        underexposed=is_underexposed(grid),
    )


__all__ = [
    "Grid",
    "Pixel",
    "QualityError",
    "QualityReport",
    "brightness",
    "contrast",
    "dynamic_range",
    "evaluate",
    "is_blurry",
    "is_overexposed",
    "is_underexposed",
    "sharpness",
]
