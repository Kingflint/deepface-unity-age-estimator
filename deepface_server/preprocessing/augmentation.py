"""Lightweight image augmentation primitives.

Operate on lists of lists of pixel values (or RGB tuples). Pure-Python
implementations chosen for testability — production paths use OpenCV.
"""
from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")
Grid = Sequence[Sequence[T]]


class AugmentationError(ValueError):
    pass


def flip_horizontal(grid: Grid[T]) -> List[List[T]]:
    """Mirror the grid left-to-right."""
    return [list(reversed(row)) for row in grid]


def flip_vertical(grid: Grid[T]) -> List[List[T]]:
    """Mirror the grid top-to-bottom."""
    return [list(row) for row in reversed(grid)]


def rotate_90(grid: Grid[T]) -> List[List[T]]:
    """Rotate clockwise by 90 degrees."""
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0])
    return [[grid[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]


def rotate_180(grid: Grid[T]) -> List[List[T]]:
    return flip_vertical(flip_horizontal(grid))


def rotate_270(grid: Grid[T]) -> List[List[T]]:
    return rotate_90(rotate_180(grid))


def crop(grid: Grid[T], *, top: int, left: int, height: int, width: int) -> List[List[T]]:
    """Return a sub-region of ``grid``."""
    if top < 0 or left < 0 or height <= 0 or width <= 0:
        raise AugmentationError("crop bounds must be non-negative and positive size")
    rows = len(grid)
    if rows == 0:
        raise AugmentationError("empty grid")
    cols = len(grid[0])
    if top + height > rows or left + width > cols:
        raise AugmentationError("crop exceeds grid bounds")
    return [list(grid[r][left : left + width]) for r in range(top, top + height)]


def pad(grid: Grid[T], *, top: int, bottom: int, left: int, right: int, fill: T) -> List[List[T]]:
    """Add ``fill`` borders around the grid."""
    if min(top, bottom, left, right) < 0:
        raise AugmentationError("padding must be non-negative")
    if not grid:
        raise AugmentationError("empty grid")
    cols = len(grid[0])
    new_cols = cols + left + right
    horizontal = [fill] * new_cols
    out: List[List[T]] = [list(horizontal) for _ in range(top)]
    for row in grid:
        out.append([fill] * left + list(row) + [fill] * right)
    out.extend(list(horizontal) for _ in range(bottom))
    return out


def center_crop(grid: Grid[T], *, height: int, width: int) -> List[List[T]]:
    rows = len(grid)
    if rows == 0:
        raise AugmentationError("empty grid")
    cols = len(grid[0])
    if height > rows or width > cols:
        raise AugmentationError("crop larger than grid")
    top = (rows - height) // 2
    left = (cols - width) // 2
    return crop(grid, top=top, left=left, height=height, width=width)


def resize_nearest(grid: Grid[T], *, height: int, width: int) -> List[List[T]]:
    """Nearest-neighbour resize."""
    if height <= 0 or width <= 0:
        raise AugmentationError("output dimensions must be positive")
    rows = len(grid)
    if rows == 0:
        raise AugmentationError("empty grid")
    cols = len(grid[0])
    out: List[List[T]] = []
    for r in range(height):
        src_r = min(rows - 1, int(r * rows / height))
        new_row: List[T] = []
        for c in range(width):
            src_c = min(cols - 1, int(c * cols / width))
            new_row.append(grid[src_r][src_c])
        out.append(new_row)
    return out


__all__ = [
    "AugmentationError",
    "center_crop",
    "crop",
    "flip_horizontal",
    "flip_vertical",
    "pad",
    "resize_nearest",
    "rotate_90",
    "rotate_180",
    "rotate_270",
]
