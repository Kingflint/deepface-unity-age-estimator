"""Common types and the Analyzer protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass
class FaceRegion:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class AnalysisOutcome:
    """Normalized output from an analyzer.

    All optional confidences are in the range ``[0.0, 1.0]``.
    """

    age: float | None = None
    age_confidence: float | None = None
    dominant_emotion: str | None = None
    emotion_scores: dict[str, float] = field(default_factory=dict)
    dominant_gender: str | None = None
    gender_scores: dict[str, float] = field(default_factory=dict)
    region: FaceRegion = field(default_factory=FaceRegion)
    backend: str = "unknown"
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["region"] = self.region.to_dict()
        return d

    def merged_with(self, other: "AnalysisOutcome", weight_self: float = 0.5) -> "AnalysisOutcome":
        weight_other = 1.0 - weight_self
        merged = AnalysisOutcome(
            age=_weighted_mean(self.age, other.age, weight_self, weight_other),
            age_confidence=_weighted_mean(
                self.age_confidence, other.age_confidence, weight_self, weight_other
            ),
            backend=f"{self.backend}+{other.backend}",
        )
        emotion = _merge_scores(self.emotion_scores, other.emotion_scores, weight_self)
        gender = _merge_scores(self.gender_scores, other.gender_scores, weight_self)
        merged.emotion_scores = emotion
        merged.gender_scores = gender
        if emotion:
            merged.dominant_emotion = max(emotion, key=emotion.get)
        if gender:
            merged.dominant_gender = max(gender, key=gender.get)
        return merged


def _weighted_mean(a: float | None, b: float | None, wa: float, wb: float) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a * wa + b * wb


def _merge_scores(
    a: dict[str, float], b: dict[str, float], weight_a: float
) -> dict[str, float]:
    weight_b = 1.0 - weight_a
    keys = set(a) | set(b)
    return {key: a.get(key, 0.0) * weight_a + b.get(key, 0.0) * weight_b for key in keys}


class Analyzer(ABC):
    """Strategy interface for image analyzers."""

    name: str = "base"

    @abstractmethod
    def analyze(self, image: bytes, actions: Sequence[str]) -> AnalysisOutcome:
        """Analyze ``image`` and return a normalized :class:`AnalysisOutcome`."""

    def supports(self, action: str) -> bool:
        return action in {"age", "emotion", "gender"}

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "actions": ["age", "emotion", "gender"]}
