"""Wrapper that delegates to the upstream :mod:`deepface` library."""
from __future__ import annotations

from typing import Any, Sequence

from .base import AnalysisOutcome, Analyzer, FaceRegion


class DeepFaceAnalyzer(Analyzer):
    name = "deepface"

    def __init__(self, enforce_detection: bool = False, detector_backend: str = "opencv"):
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend

    def analyze(self, image: bytes, actions: Sequence[str]) -> AnalysisOutcome:
        from deepface import DeepFace  # type: ignore

        try:
            import numpy as np  # type: ignore
            import cv2  # type: ignore

            arr = np.frombuffer(image, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            decoded = image

        result = DeepFace.analyze(
            decoded,
            actions=list(actions),
            enforce_detection=self.enforce_detection,
            detector_backend=self.detector_backend,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        return self._normalize(result)

    def _normalize(self, data: dict[str, Any]) -> AnalysisOutcome:
        region = data.get("region") or {}
        face = FaceRegion(
            x=int(region.get("x", 0)),
            y=int(region.get("y", 0)),
            w=int(region.get("w", 0)),
            h=int(region.get("h", 0)),
        )
        emotion_scores = {
            k: float(v) / 100.0 if v > 1.0 else float(v)
            for k, v in (data.get("emotion") or {}).items()
        }
        gender_scores = {
            k: float(v) / 100.0 if v > 1.0 else float(v)
            for k, v in (data.get("gender") or {}).items()
        }
        return AnalysisOutcome(
            age=_safe_float(data.get("age")),
            dominant_emotion=data.get("dominant_emotion"),
            emotion_scores=emotion_scores,
            dominant_gender=data.get("dominant_gender"),
            gender_scores=gender_scores,
            region=face,
            backend=self.name,
            raw=data,
        )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
