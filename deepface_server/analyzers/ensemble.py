"""Ensemble analyzer running several backends and merging their outcomes."""
from __future__ import annotations

from typing import Iterable, Sequence

from .base import AnalysisOutcome, Analyzer


class EnsembleAnalyzer(Analyzer):
    name = "ensemble"

    def __init__(self, members: Iterable[Analyzer], weights: Iterable[float] | None = None):
        self.members = list(members)
        if not self.members:
            raise ValueError("EnsembleAnalyzer needs at least one member")
        if weights is None:
            equal = 1.0 / len(self.members)
            self.weights = [equal] * len(self.members)
        else:
            weights_list = list(weights)
            if len(weights_list) != len(self.members):
                raise ValueError("weights count must match members count")
            total = sum(weights_list)
            if total <= 0:
                raise ValueError("weights must sum to a positive number")
            self.weights = [w / total for w in weights_list]

    def analyze(self, image: bytes, actions: Sequence[str]) -> AnalysisOutcome:
        outcomes = [member.analyze(image, actions) for member in self.members]
        return self._merge(outcomes)

    def _merge(self, outcomes: list[AnalysisOutcome]) -> AnalysisOutcome:
        first = outcomes[0]
        if len(outcomes) == 1:
            return first
        running = first
        accumulated = self.weights[0]
        for outcome, weight in zip(outcomes[1:], self.weights[1:]):
            new_total = accumulated + weight
            self_share = accumulated / new_total
            running = running.merged_with(outcome, weight_self=self_share)
            accumulated = new_total
        running.backend = "+".join(member.name for member in self.members)
        return running

    def describe(self) -> dict:
        return {
            "name": self.name,
            "members": [member.describe() for member in self.members],
            "weights": self.weights,
        }
