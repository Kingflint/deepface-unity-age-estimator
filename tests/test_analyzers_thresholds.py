from __future__ import annotations

import pytest

from deepface_server.analyzers import thresholds as th


def test_threshold_passes():
    t = th.Threshold("low", 0.0)
    assert t.passes(0.5)
    assert not t.passes(-0.1)


def test_thresholdset_classify():
    s = th.default_confidence_thresholds()
    assert s.classify(0.95) == "very_high"
    assert s.classify(0.80) == "high"
    assert s.classify(0.6) == "moderate"
    assert s.classify(0.0) == "low"


def test_thresholdset_unsorted_rejected():
    with pytest.raises(th.ThresholdError):
        th.ThresholdSet([th.Threshold("a", 0.5), th.Threshold("b", 0.1)])


def test_thresholdset_duplicate_name():
    with pytest.raises(th.ThresholdError):
        th.ThresholdSet([th.Threshold("a", 0.0), th.Threshold("a", 0.5)])


def test_thresholdset_get():
    s = th.default_confidence_thresholds()
    assert s.get("high").minimum == 0.75
    with pytest.raises(KeyError):
        s.get("missing")


def test_thresholdset_names():
    s = th.default_confidence_thresholds()
    assert s.names() == ["low", "moderate", "high", "very_high"]


def test_from_mapping():
    s = th.from_mapping([("a", 0.0), ("b", 0.5)])
    assert s.classify(0.6) == "b"


def test_below_lowest_returns_none_when_lowest_is_strict():
    # default has "low" at 0.0, so any negative score has no match
    s = th.default_confidence_thresholds()
    assert s.classify(-0.01) is None
