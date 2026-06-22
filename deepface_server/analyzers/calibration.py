"""Confidence calibration for analyzer outputs.

Real models often return over-confident probabilities. We expose two
common calibration techniques:

- **Platt scaling** — fit ``sigmoid(a*x + b)``.
- **Temperature scaling** — divide logits by ``T``.

Both implementations use plain Python and tiny gradient descent so they
remain dependency-free for unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import List, Sequence


class CalibrationError(ValueError):
    pass


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return log(p / (1.0 - p))


@dataclass(frozen=True)
class PlattModel:
    a: float
    b: float

    def predict(self, score: float) -> float:
        return _sigmoid(self.a * score + self.b)


def fit_platt(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    iterations: int = 200,
    learning_rate: float = 0.05,
) -> PlattModel:
    """Fit a Platt scaling model via simple gradient descent.

    ``labels`` must be 0/1. The optimiser is intentionally minimal but
    sufficient for the small batches we calibrate with in production.
    """
    if len(scores) != len(labels):
        raise CalibrationError("scores and labels must have the same length")
    if not scores:
        raise CalibrationError("empty input")
    if any(label not in (0, 1) for label in labels):
        raise CalibrationError("labels must be 0 or 1")

    a, b = 1.0, 0.0
    n = len(scores)
    for _ in range(iterations):
        grad_a = 0.0
        grad_b = 0.0
        for s, y in zip(scores, labels):
            p = _sigmoid(a * s + b)
            err = p - y
            grad_a += err * s
            grad_b += err
        a -= learning_rate * grad_a / n
        b -= learning_rate * grad_b / n
    return PlattModel(a=a, b=b)


@dataclass(frozen=True)
class TemperatureModel:
    temperature: float

    def predict(self, logits: Sequence[float]) -> List[float]:
        if self.temperature <= 0:
            raise CalibrationError("temperature must be > 0")
        scaled = [v / self.temperature for v in logits]
        m = max(scaled)
        exps = [exp(v - m) for v in scaled]
        s = sum(exps)
        return [e / s for e in exps]


def fit_temperature(
    logits_list: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    iterations: int = 100,
    learning_rate: float = 0.05,
) -> TemperatureModel:
    """Fit a single scalar temperature on a validation set."""
    if len(logits_list) != len(labels):
        raise CalibrationError("mismatched lengths")
    if not logits_list:
        raise CalibrationError("empty input")

    t = 1.0
    for _ in range(iterations):
        grad = 0.0
        for logits, y in zip(logits_list, labels):
            scaled = [v / t for v in logits]
            m = max(scaled)
            exps = [exp(v - m) for v in scaled]
            s = sum(exps)
            probs = [e / s for e in exps]
            # gradient of NLL w.r.t. t (negative of expectation minus chosen)
            mean_logit = sum(p * lv for p, lv in zip(probs, logits))
            chosen = logits[y] if 0 <= y < len(logits) else mean_logit
            grad += (mean_logit - chosen) / (t * t)
        t -= learning_rate * grad / len(logits_list)
        if t < 1e-3:
            t = 1e-3
    return TemperatureModel(temperature=t)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> float:
    """Expected calibration error (ECE) over equal-width bins."""
    if len(probabilities) != len(labels):
        raise CalibrationError("mismatched lengths")
    if bins <= 0:
        raise CalibrationError("bins must be positive")
    if not probabilities:
        return 0.0
    bucket_correct = [0.0] * bins
    bucket_conf = [0.0] * bins
    bucket_count = [0] * bins
    for p, y in zip(probabilities, labels):
        idx = min(bins - 1, int(p * bins))
        bucket_count[idx] += 1
        bucket_conf[idx] += p
        bucket_correct[idx] += 1.0 if y == 1 else 0.0
    n = len(probabilities)
    ece = 0.0
    for i in range(bins):
        if bucket_count[i] == 0:
            continue
        acc = bucket_correct[i] / bucket_count[i]
        conf = bucket_conf[i] / bucket_count[i]
        ece += (bucket_count[i] / n) * abs(acc - conf)
    return ece


__all__ = [
    "CalibrationError",
    "PlattModel",
    "TemperatureModel",
    "expected_calibration_error",
    "fit_platt",
    "fit_temperature",
]
