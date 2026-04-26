"""Registry mapping analyzer names to factories."""
from __future__ import annotations

from typing import Callable, Iterable

from .base import Analyzer
from .deepface_backend import DeepFaceAnalyzer
from .ensemble import EnsembleAnalyzer
from .mock_backend import DeterministicMockAnalyzer


class AnalyzerRegistry:
    def __init__(self):
        self._factories: dict[str, Callable[[], Analyzer]] = {}

    def register(self, name: str, factory: Callable[[], Analyzer]) -> None:
        self._factories[name] = factory

    def names(self) -> list[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> Analyzer:
        if name not in self._factories:
            raise KeyError(f"unknown analyzer: {name}")
        return self._factories[name]()

    def create_ensemble(
        self, names: Iterable[str], weights: Iterable[float] | None = None
    ) -> EnsembleAnalyzer:
        members = [self.create(name) for name in names]
        return EnsembleAnalyzer(members, weights=weights)


def build_default_registry() -> AnalyzerRegistry:
    registry = AnalyzerRegistry()
    registry.register("deepface", DeepFaceAnalyzer)
    registry.register("mock", DeterministicMockAnalyzer)
    return registry
