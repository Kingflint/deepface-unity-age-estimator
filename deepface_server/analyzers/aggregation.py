"""Aggregate per-frame analyzer outputs into a final decision."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class AggregationError(ValueError):
    pass


@dataclass(frozen=True)
class WeightedSample:
    label: str
    score: float
    weight: float = 1.0


def weighted_average(samples: Iterable[WeightedSample]) -> Dict[str, float]:
    """Aggregate ``label -> score`` using weighted averaging.

    Returns a dict mapping each label to its weighted-average score.
    """
    totals: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    for sample in samples:
        totals[sample.label] = totals.get(sample.label, 0.0) + sample.score * sample.weight
        weights[sample.label] = weights.get(sample.label, 0.0) + sample.weight
    out: Dict[str, float] = {}
    for label, total in totals.items():
        w = weights[label]
        if w == 0:
            continue
        out[label] = total / w
    return out


def majority_vote(labels: Sequence[str], *, weights: Optional[Sequence[float]] = None) -> str:
    """Return the most common label, breaking ties by first occurrence.

    When ``weights`` is provided the vote is weighted.
    """
    if not labels:
        raise AggregationError("no labels provided")
    if weights is None:
        counts = Counter(labels)
        # preserve insertion order on ties
        best: Tuple[int, int] = (-1, len(labels))
        winner = labels[0]
        for idx, label in enumerate(labels):
            count = counts[label]
            if count > best[0] or (count == best[0] and idx < best[1]):
                winner = label
                best = (count, idx)
        return winner

    if len(weights) != len(labels):
        raise AggregationError("weights and labels must be the same length")
    totals: Dict[str, float] = {}
    first_idx: Dict[str, int] = {}
    for idx, (label, weight) in enumerate(zip(labels, weights)):
        totals[label] = totals.get(label, 0.0) + weight
        first_idx.setdefault(label, idx)
    return max(totals.items(), key=lambda kv: (kv[1], -first_idx[kv[0]]))[0]


def top_k(scores: Mapping[str, float], k: int) -> List[Tuple[str, float]]:
    """Return the top ``k`` ``(label, score)`` pairs sorted by score desc."""
    if k <= 0:
        raise AggregationError("k must be positive")
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


def consensus_label(
    candidate_labels: Sequence[Sequence[str]],
    *,
    min_agreement: float = 0.5,
) -> Optional[str]:
    """Return the most-agreed-on label across multiple analyzers.

    Each analyzer contributes its top label in ``candidate_labels``.
    The result is the label that appears in at least ``min_agreement``
    fraction of analyzers' top picks, or ``None`` if no label clears
    the bar.
    """
    if not candidate_labels:
        return None
    if not 0.0 < min_agreement <= 1.0:
        raise AggregationError("min_agreement must be in (0, 1]")
    top_picks = [labels[0] for labels in candidate_labels if labels]
    if not top_picks:
        return None
    counts = Counter(top_picks)
    label, count = counts.most_common(1)[0]
    if count / len(top_picks) >= min_agreement:
        return label
    return None


def trimmed_mean(values: Sequence[float], *, trim: float = 0.1) -> float:
    """Mean of ``values`` after dropping ``trim`` fraction at each tail."""
    if not values:
        raise AggregationError("empty values")
    if not 0.0 <= trim < 0.5:
        raise AggregationError("trim must be in [0, 0.5)")
    sorted_values = sorted(values)
    n = len(sorted_values)
    cut = int(n * trim)
    trimmed = sorted_values[cut : n - cut] if n - 2 * cut > 0 else sorted_values
    return sum(trimmed) / len(trimmed)


def median(values: Sequence[float]) -> float:
    if not values:
        raise AggregationError("empty values")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


__all__ = [
    "AggregationError",
    "WeightedSample",
    "consensus_label",
    "majority_vote",
    "median",
    "top_k",
    "trimmed_mean",
    "weighted_average",
]
