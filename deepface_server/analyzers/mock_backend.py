"""Deterministic mock analyzer used in tests and offline demos."""
from __future__ import annotations

import hashlib
from typing import Sequence

from .base import AnalysisOutcome, Analyzer, FaceRegion


class DeterministicMockAnalyzer(Analyzer):
    name = "mock"

    EMOTIONS = ("neutral", "happy", "sad", "surprise", "angry", "fear", "disgust")
    GENDERS = ("Man", "Woman")

    def __init__(self, seed_salt: str = ""):
        self.seed_salt = seed_salt

    def analyze(self, image: bytes, actions: Sequence[str]) -> AnalysisOutcome:
        digest = hashlib.sha256(image + self.seed_salt.encode()).digest()

        age_byte = digest[0]
        emotion_idx = digest[1] % len(self.EMOTIONS)
        gender_idx = digest[2] % len(self.GENDERS)

        emotion_scores = {label: 0.0 for label in self.EMOTIONS}
        emotion_scores[self.EMOTIONS[emotion_idx]] = 0.6 + (digest[3] / 255.0) * 0.3
        remaining = 1.0 - sum(emotion_scores.values())
        if remaining > 0:
            for label in self.EMOTIONS:
                if label != self.EMOTIONS[emotion_idx]:
                    emotion_scores[label] = remaining / (len(self.EMOTIONS) - 1)

        gender_scores = {
            self.GENDERS[gender_idx]: 0.7 + (digest[4] / 255.0) * 0.25,
            self.GENDERS[1 - gender_idx]: 0.0,
        }
        gender_scores[self.GENDERS[1 - gender_idx]] = 1.0 - gender_scores[self.GENDERS[gender_idx]]

        return AnalysisOutcome(
            age=15.0 + (age_byte / 255.0) * 60.0,
            age_confidence=0.5 + (digest[5] / 255.0) * 0.4,
            dominant_emotion=self.EMOTIONS[emotion_idx],
            emotion_scores=emotion_scores,
            dominant_gender=self.GENDERS[gender_idx],
            gender_scores=gender_scores,
            region=FaceRegion(x=0, y=0, w=128, h=128),
            backend=self.name,
            raw={"mock": True, "digest": digest.hex()[:16]},
        )
