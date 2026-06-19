from __future__ import annotations

import pytest

from deepface_server.preprocessing import augmentation as a


def test_flip_horizontal():
    grid = [[1, 2, 3], [4, 5, 6]]
    assert a.flip_horizontal(grid) == [[3, 2, 1], [6, 5, 4]]


def test_flip_vertical():
    grid = [[1, 2], [3, 4]]
    assert a.flip_vertical(grid) == [[3, 4], [1, 2]]


def test_rotate_90():
    grid = [[1, 2], [3, 4]]
    assert a.rotate_90(grid) == [[3, 1], [4, 2]]


def test_rotate_180_is_double_flip():
    grid = [[1, 2, 3], [4, 5, 6]]
    assert a.rotate_180(grid) == [[6, 5, 4], [3, 2, 1]]


def test_rotate_270_returns_to_start_after_two():
    grid = [[1, 2], [3, 4]]
    assert a.rotate_90(a.rotate_270(grid)) == grid


def test_crop_basic():
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert a.crop(grid, top=0, left=1, height=2, width=2) == [[2, 3], [5, 6]]


def test_crop_out_of_bounds():
    grid = [[1, 2], [3, 4]]
    with pytest.raises(a.AugmentationError):
        a.crop(grid, top=0, left=0, height=5, width=5)


def test_crop_negative_rejected():
    grid = [[1, 2], [3, 4]]
    with pytest.raises(a.AugmentationError):
        a.crop(grid, top=-1, left=0, height=1, width=1)


def test_pad_adds_borders():
    grid = [[1, 2], [3, 4]]
    out = a.pad(grid, top=1, bottom=0, left=1, right=0, fill=0)
    assert out == [[0, 0, 0], [0, 1, 2], [0, 3, 4]]


def test_pad_negative_rejected():
    with pytest.raises(a.AugmentationError):
        a.pad([[1]], top=-1, bottom=0, left=0, right=0, fill=0)


def test_center_crop():
    grid = [[i for i in range(5)] for _ in range(5)]
    out = a.center_crop(grid, height=3, width=3)
    assert len(out) == 3 and len(out[0]) == 3
    assert out[0] == [1, 2, 3]


def test_center_crop_too_large():
    with pytest.raises(a.AugmentationError):
        a.center_crop([[1]], height=5, width=5)


def test_resize_nearest_upscale():
    grid = [[1, 2], [3, 4]]
    out = a.resize_nearest(grid, height=4, width=4)
    assert len(out) == 4 and len(out[0]) == 4
    assert out[0][0] == 1


def test_resize_nearest_invalid():
    with pytest.raises(a.AugmentationError):
        a.resize_nearest([[1]], height=0, width=1)


def test_resize_nearest_empty():
    with pytest.raises(a.AugmentationError):
        a.resize_nearest([], height=2, width=2)
