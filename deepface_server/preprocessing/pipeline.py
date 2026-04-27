"""Composable preprocessing pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class StepResult:
    image: Any
    metadata: dict = field(default_factory=dict)


class PreprocessStep(ABC):
    """A single image transformation step."""

    name: str = "step"

    @abstractmethod
    def apply(self, image: Any, metadata: dict) -> Any:
        """Return the transformed image."""

    def describe(self) -> dict:
        return {"name": self.name, "params": getattr(self, "params", {})}


class Pipeline:
    """Sequentially applies a list of :class:`PreprocessStep`."""

    def __init__(self, steps: Iterable[PreprocessStep] | None = None):
        self.steps: list[PreprocessStep] = list(steps or [])

    def add(self, step: PreprocessStep) -> "Pipeline":
        self.steps.append(step)
        return self

    def remove(self, name: str) -> bool:
        for i, step in enumerate(self.steps):
            if step.name == name:
                del self.steps[i]
                return True
        return False

    def run(self, image: Any) -> StepResult:
        metadata: dict = {"applied": []}
        for step in self.steps:
            image = step.apply(image, metadata)
            metadata["applied"].append(step.name)
        return StepResult(image=image, metadata=metadata)

    def describe(self) -> list[dict]:
        return [step.describe() for step in self.steps]

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


def build_default_pipeline(target_size: tuple[int, int] = (224, 224)) -> Pipeline:
    """Construct the standard preprocessing pipeline used before analysis."""
    from .steps import EXIFRotate, Normalize, Resize

    return Pipeline([EXIFRotate(), Resize(target_size), Normalize()])
