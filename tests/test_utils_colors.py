"""Tests for utility colour helpers."""
from __future__ import annotations

import pytest

from deepface_server.utils import colors as cm


def test_color_round_trip_hex():
    color = cm.Color.from_hex("#1A2B3C")
    assert color.to_hex() == "#1a2b3c"
    assert color.to_rgb() == (26, 43, 60)


def test_color_from_hex_short():
    color = cm.Color.from_hex("#abc")
    assert color.to_rgb() == (0xAA, 0xBB, 0xCC)


def test_color_invalid_hex():
    with pytest.raises(ValueError):
        cm.Color.from_hex("not-a-color")


def test_color_channel_validation():
    with pytest.raises(ValueError):
        cm.Color(300, 0, 0)


def test_hsv_round_trip():
    color = cm.Color(120, 200, 80)
    h, s, v = color.to_hsv()
    back = cm.Color.from_hsv(h, s, v)
    for a, b in zip(color.to_rgb(), back.to_rgb()):
        assert abs(a - b) <= 2


def test_contrast_ratio_white_black():
    white = cm.Color(255, 255, 255)
    black = cm.Color(0, 0, 0)
    assert round(cm.contrast_ratio(white, black), 1) == 21.0


def test_contrast_ratio_same_color_is_one():
    c = cm.Color(120, 130, 140)
    assert cm.contrast_ratio(c, c) == pytest.approx(1.0)


def test_perceptual_distance_zero_for_same():
    c = cm.Color(120, 130, 140)
    assert cm.perceptual_distance(c, c) == 0.0


def test_perceptual_distance_positive():
    a = cm.Color(0, 0, 0)
    b = cm.Color(255, 255, 255)
    assert cm.perceptual_distance(a, b) > 0


def test_euclidean_distance_zero_for_same():
    c = cm.Color(10, 20, 30)
    assert cm.euclidean_distance(c, c) == 0.0


def test_nearest_color_picks_closest():
    palette = [cm.Color(255, 0, 0), cm.Color(0, 255, 0), cm.Color(0, 0, 255)]
    target = cm.Color(0, 250, 10)
    assert cm.nearest_color(target, palette).to_rgb() == (0, 255, 0)


def test_nearest_color_empty_palette():
    with pytest.raises(ValueError):
        cm.nearest_color(cm.Color(0, 0, 0), [])


def test_average_color():
    colors = [cm.Color(0, 0, 0), cm.Color(255, 255, 255)]
    avg = cm.average_color(colors)
    for v in avg.to_rgb():
        assert v in (127, 128)


def test_average_color_empty():
    with pytest.raises(ValueError):
        cm.average_color([])


def test_quantize_rounds_components():
    c = cm.Color(100, 150, 200)
    q = cm.quantize(c, levels=4)
    step = 256 // 4
    for v in q.to_rgb():
        assert v % step == 0


def test_quantize_invalid_levels():
    with pytest.raises(ValueError):
        cm.quantize(cm.Color(0, 0, 0), levels=1)


def test_dominant_palette_returns_top_buckets():
    colors = [cm.Color(255, 0, 0)] * 5 + [cm.Color(0, 255, 0)] * 3 + [cm.Color(0, 0, 255)] * 2
    palette = cm.dominant_palette(colors, k=3, levels=8)
    assert len(palette) == 3


def test_dominant_palette_empty():
    assert cm.dominant_palette([], k=3) == []


def test_dominant_palette_invalid_k():
    with pytest.raises(ValueError):
        cm.dominant_palette([cm.Color(0, 0, 0)], k=0)
