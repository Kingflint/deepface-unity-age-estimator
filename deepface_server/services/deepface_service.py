"""Adapter around DeepFace.analyze."""
from __future__ import annotations


class DeepFaceService:
    def __init__(self, actions=("emotion", "age", "gender"), enforce_detection=False):
        self.actions = tuple(actions)
        self.enforce_detection = enforce_detection

    def analyze(self, image):
        from deepface import DeepFace
        return DeepFace.analyze(
            image,
            actions=list(self.actions),
            enforce_detection=self.enforce_detection,
        )

    def describe(self):
        return {"actions": list(self.actions), "enforce_detection": self.enforce_detection}