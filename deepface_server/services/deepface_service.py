"""Wrap DeepFace.analyze with input validation and graceful fallbacks."""
from __future__ import annotations

from typing import Any, Iterable

from ..errors import AnalyzerError


class DeepFaceService:
    """Thin adapter around the DeepFace library.

    The constructor takes only configuration; the library itself is
    imported lazily inside :meth:`analyze` so the package can be
    imported (and unit tested) in environments without the DeepFace
    model weights.
    """

    def __init__(
        self,
        actions: Iterable[str] = ("emotion", "age", "gender"),
        enforce_detection: bool = False,
        detector_backend: str = "opencv",
    ):
        self.actions = tuple(actions)
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend

    def analyze(self, image: Any) -> Any:
        try:
            from deepface import DeepFace
        except ImportError as exc:  # pragma: no cover - exercised at runtime only
            raise AnalyzerError(f"deepface is not available: {exc}") from exc

        try:
            return DeepFace.analyze(
                image,
                actions=list(self.actions),
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend,
            )
        except Exception as exc:  # noqa: BLE001 - DeepFace raises plain Exceptions
            raise AnalyzerError(str(exc)) from exc

    def describe(self) -> dict:
        return {
            "actions": list(self.actions),
            "enforce_detection": self.enforce_detection,
            "detector_backend": self.detector_backend,
        }
