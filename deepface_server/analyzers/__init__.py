"""Pluggable image analyzer backends."""
from .base import AnalysisOutcome, Analyzer, FaceRegion
from .deepface_backend import DeepFaceAnalyzer
from .ensemble import EnsembleAnalyzer
from .mock_backend import DeterministicMockAnalyzer
from .registry import AnalyzerRegistry, build_default_registry

__all__ = [
    "AnalysisOutcome",
    "Analyzer",
    "AnalyzerRegistry",
    "DeepFaceAnalyzer",
    "DeterministicMockAnalyzer",
    "EnsembleAnalyzer",
    "FaceRegion",
    "build_default_registry",
]
