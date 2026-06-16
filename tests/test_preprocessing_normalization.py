from __future__ import annotations

import pytest

from deepface_server.preprocessing import normalization as n


def test_min_max_basic():
    out = n.min_max([0, 5, 10])
    assert out == [0.0, 0.5, 1.0]


def test_min_max_constant():
    out = n.min_max([7, 7, 7])
    assert out == [0.5, 0.5, 0.5]


def test_min_max_custom_range():
    out = n.min_max([0, 10], target_min=-1.0, target_max=1.0)
    assert out == [-1.0, 1.0]


def test_min_max_invalid_target_range():
    with pytest.raises(n.NormalizationError):
        n.min_max([1, 2], target_min=1.0, target_max=0.0)


def test_min_max_empty():
    assert n.min_max([]) == []


def test_z_score_zero_mean():
    out = n.z_score([1, 2, 3, 4, 5])
    assert sum(out) == pytest.approx(0.0, abs=1e-9)


def test_z_score_constant():
    assert n.z_score([5, 5, 5]) == [0.0, 0.0, 0.0]


def test_unit_vector_norm_one():
    out = n.unit_vector([3.0, 4.0])
    assert sum(v * v for v in out) == pytest.approx(1.0)


def test_unit_vector_zero():
    assert n.unit_vector([0, 0]) == [0.0, 0.0]


def test_softmax_sums_to_one():
    out = n.softmax([1.0, 2.0, 3.0])
    assert sum(out) == pytest.approx(1.0)


def test_softmax_empty():
    assert n.softmax([]) == []


def test_softmax_stable_with_large():
    out = n.softmax([1000.0, 1001.0])
    assert sum(out) == pytest.approx(1.0)


def test_clip_basic():
    assert n.clip([-1, 0.5, 2], 0.0, 1.0) == [0.0, 0.5, 1.0]


def test_clip_invalid():
    with pytest.raises(n.NormalizationError):
        n.clip([1], 1.0, 0.0)


def test_standardise_rgb():
    assert n.standardise_rgb((255, 0, 128)) == pytest.approx((1.0, 0.0, 128 / 255))


def test_imagenet_normalise_zero_mean():
    out = n.imagenet_normalise((0.485, 0.456, 0.406))
    for v in out:
        assert abs(v) < 1e-9
