from __future__ import annotations

import pytest

from deepface_server.preprocessing import quality as q


def _grid(value=128, rows=10, cols=10):
    return [[value] * cols for _ in range(rows)]


def test_brightness_uniform():
    assert q.brightness(_grid(100)) == 100.0


def test_brightness_empty_rejected():
    with pytest.raises(q.QualityError):
        q.brightness([])


def test_brightness_inconsistent_rows():
    with pytest.raises(q.QualityError):
        q.brightness([[1, 2], [3]])


def test_contrast_uniform_zero():
    assert q.contrast(_grid(100)) == 0.0


def test_contrast_nonzero():
    grid = [[0, 255, 0], [255, 0, 255], [0, 255, 0]]
    assert q.contrast(grid) > 0


def test_dynamic_range():
    grid = [[10, 50], [200, 250]]
    assert q.dynamic_range(grid) == 240


def test_sharpness_uniform_is_zero():
    assert q.sharpness(_grid(50)) == 0.0


def test_sharpness_high_for_edges():
    grid = [[0] * 5 for _ in range(5)]
    grid[2][2] = 255
    assert q.sharpness(grid) > 0


def test_sharpness_small_grid():
    assert q.sharpness([[1, 2], [3, 4]]) == 0.0


def test_is_blurry_threshold():
    assert q.is_blurry(_grid(50), threshold=10)


def test_is_overexposed():
    grid = [[250] * 4 for _ in range(4)]
    assert q.is_overexposed(grid)


def test_is_underexposed():
    grid = [[5] * 4 for _ in range(4)]
    assert q.is_underexposed(grid)


def test_evaluate_returns_report():
    grid = [[i * 5 + j for j in range(10)] for i in range(10)]
    report = q.evaluate(grid)
    assert isinstance(report, q.QualityReport)
    assert report.brightness > 0


def test_quality_report_acceptable_and_issues():
    grid = _grid(250)
    report = q.evaluate(grid)
    assert report.overexposed
    assert "overexposed" in report.issues()
    assert not report.acceptable
