from __future__ import annotations

import pytest

from deepface_server.analyzers import aggregation as a


def test_weighted_average_basic():
    samples = [
        a.WeightedSample(label="x", score=1.0, weight=1.0),
        a.WeightedSample(label="x", score=3.0, weight=1.0),
    ]
    out = a.weighted_average(samples)
    assert out["x"] == 2.0


def test_weighted_average_with_weights():
    samples = [
        a.WeightedSample(label="x", score=1.0, weight=1.0),
        a.WeightedSample(label="x", score=3.0, weight=3.0),
    ]
    out = a.weighted_average(samples)
    assert out["x"] == 2.5


def test_weighted_average_zero_weight_skipped():
    samples = [a.WeightedSample(label="x", score=1.0, weight=0.0)]
    assert a.weighted_average(samples) == {}


def test_majority_vote_simple():
    assert a.majority_vote(["a", "b", "a", "c"]) == "a"


def test_majority_vote_tie_first_wins():
    assert a.majority_vote(["a", "b"]) == "a"


def test_majority_vote_empty_raises():
    with pytest.raises(a.AggregationError):
        a.majority_vote([])


def test_majority_vote_with_weights():
    out = a.majority_vote(["a", "b"], weights=[1.0, 5.0])
    assert out == "b"


def test_majority_vote_weight_mismatch():
    with pytest.raises(a.AggregationError):
        a.majority_vote(["a", "b"], weights=[1.0])


def test_top_k_orders_descending():
    out = a.top_k({"a": 1.0, "b": 3.0, "c": 2.0}, k=2)
    assert out == [("b", 3.0), ("c", 2.0)]


def test_top_k_invalid():
    with pytest.raises(a.AggregationError):
        a.top_k({}, k=0)


def test_consensus_label_pass():
    candidates = [["a"], ["a"], ["b"]]
    assert a.consensus_label(candidates, min_agreement=0.6) == "a"


def test_consensus_label_fail():
    candidates = [["a"], ["b"], ["c"]]
    assert a.consensus_label(candidates, min_agreement=0.6) is None


def test_consensus_label_empty():
    assert a.consensus_label([]) is None


def test_consensus_label_invalid_threshold():
    with pytest.raises(a.AggregationError):
        a.consensus_label([["a"]], min_agreement=0)


def test_trimmed_mean_basic():
    assert a.trimmed_mean([1, 2, 3, 4, 100], trim=0.2) == pytest.approx(3.0)


def test_trimmed_mean_zero_trim():
    assert a.trimmed_mean([1, 2, 3], trim=0.0) == 2.0


def test_trimmed_mean_invalid():
    with pytest.raises(a.AggregationError):
        a.trimmed_mean([], trim=0.1)
    with pytest.raises(a.AggregationError):
        a.trimmed_mean([1, 2], trim=0.5)


def test_median_odd():
    assert a.median([3, 1, 2]) == 2


def test_median_even():
    assert a.median([1, 2, 3, 4]) == 2.5


def test_median_empty():
    with pytest.raises(a.AggregationError):
        a.median([])
