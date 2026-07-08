"""Retry policies for outbound webhook deliveries."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


class RetryPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter.

    Delay for attempt ``n`` (1-indexed) is computed as::

        min(max_delay, base_delay * factor ** (n - 1))

    Optionally a uniform jitter is added in ``[-jitter, +jitter]``
    multiplied by the computed delay.
    """

    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    factor: float = 2.0
    jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise RetryPolicyError("max_attempts must be >= 1")
        if self.base_delay <= 0:
            raise RetryPolicyError("base_delay must be > 0")
        if self.max_delay < self.base_delay:
            raise RetryPolicyError("max_delay must be >= base_delay")
        if self.factor < 1:
            raise RetryPolicyError("factor must be >= 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise RetryPolicyError("jitter must be in [0, 1]")

    def delay_for(self, attempt: int, *, rand: "random.Random | None" = None) -> float:
        if attempt < 1:
            raise RetryPolicyError("attempt must be >= 1")
        delay = min(self.max_delay, self.base_delay * (self.factor ** (attempt - 1)))
        if self.jitter:
            r = (rand or random).uniform(-self.jitter, self.jitter)
            delay = max(0.0, delay * (1.0 + r))
        return delay

    def schedule(self, *, rand: "random.Random | None" = None) -> List[float]:
        return [self.delay_for(i, rand=rand) for i in range(1, self.max_attempts + 1)]

    def is_terminal(self, attempt: int) -> bool:
        return attempt >= self.max_attempts


# A few well-known presets used by different webhook tiers.
DEFAULT_POLICY = RetryPolicy()
AGGRESSIVE_POLICY = RetryPolicy(max_attempts=8, base_delay=0.5, max_delay=120.0, factor=2.0)
GENTLE_POLICY = RetryPolicy(max_attempts=3, base_delay=5.0, max_delay=30.0, factor=2.0, jitter=0.0)


__all__ = [
    "AGGRESSIVE_POLICY",
    "DEFAULT_POLICY",
    "GENTLE_POLICY",
    "RetryPolicy",
    "RetryPolicyError",
]
