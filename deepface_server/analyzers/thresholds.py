"""Confidence threshold definitions and lookups.

Different downstream consumers care about different cutoffs (UI display,
auto-publish, escalation). We model them as ordered, named bands so we
can map any score into a category and so the configuration can be
serialised to JSON.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


class ThresholdError(ValueError):
    pass


@dataclass(frozen=True)
class Threshold:
    name: str
    minimum: float

    def passes(self, score: float) -> bool:
        return score >= self.minimum


@dataclass(frozen=True)
class ThresholdSet:
    """An ordered collection of thresholds, lowest to highest."""

    thresholds: Sequence[Threshold]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        last = float("-inf")
        for t in self.thresholds:
            if t.name in seen:
                raise ThresholdError(f"duplicate threshold name: {t.name}")
            seen.add(t.name)
            if t.minimum < last:
                raise ThresholdError("thresholds must be sorted ascending by minimum")
            last = t.minimum

    def classify(self, score: float) -> Optional[str]:
        """Return the highest-named threshold a score satisfies, else None."""
        match: Optional[str] = None
        for t in self.thresholds:
            if t.passes(score):
                match = t.name
            else:
                break
        return match

    def names(self) -> List[str]:
        return [t.name for t in self.thresholds]

    def get(self, name: str) -> Threshold:
        for t in self.thresholds:
            if t.name == name:
                return t
        raise KeyError(name)


def default_confidence_thresholds() -> ThresholdSet:
    """Sensible defaults shared across age/gender/emotion analyzers."""
    return ThresholdSet(
        [
            Threshold("low", 0.0),
            Threshold("moderate", 0.5),
            Threshold("high", 0.75),
            Threshold("very_high", 0.9),
        ]
    )


def from_mapping(values: Iterable[tuple]) -> ThresholdSet:
    """Build a :class:`ThresholdSet` from ``(name, minimum)`` pairs."""
    return ThresholdSet([Threshold(name, float(m)) for name, m in values])


__all__ = [
    "Threshold",
    "ThresholdError",
    "ThresholdSet",
    "default_confidence_thresholds",
    "from_mapping",
]
