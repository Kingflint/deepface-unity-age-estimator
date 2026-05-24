"""Generic retry decorator with exponential backoff."""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Optional, Tuple, Type


_LOG = logging.getLogger(__name__)


class RetryError(Exception):
    """Raised when all retries are exhausted."""

    def __init__(self, attempts: int, last_exception: BaseException):
        super().__init__(f"failed after {attempts} attempts: {last_exception!r}")
        self.attempts = attempts
        self.last_exception = last_exception


def compute_backoff(
    attempt: int,
    *,
    base: float = 0.5,
    factor: float = 2.0,
    cap: float = 30.0,
    jitter: bool = True,
) -> float:
    """Return the delay (in seconds) before retry ``attempt`` (1-indexed)."""
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    delay = min(cap, base * (factor ** (attempt - 1)))
    if jitter:
        delay = random.uniform(0, delay)
    return max(0.0, delay)


def retry(
    *,
    attempts: int = 3,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    base_delay: float = 0.5,
    factor: float = 2.0,
    cap: float = 30.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
):
    """Decorator that retries a callable on failure.

    Parameters
    ----------
    attempts:
        Total number of attempts (must be >= 1). 1 means no retry.
    exceptions:
        Tuple of exception types that trigger a retry.
    base_delay, factor, cap:
        Exponential backoff parameters.
    jitter:
        When True, the actual delay is uniform in ``[0, computed]``.
    on_retry:
        Optional hook invoked as ``on_retry(attempt, exc)`` after a failure
        but before sleeping. Useful for metrics emission.
    sleep:
        Override the sleep function (used by tests).
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == attempts:
                        break
                    if on_retry is not None:
                        try:
                            on_retry(attempt, exc)
                        except Exception:  # pragma: no cover - hook errors
                            _LOG.exception("on_retry hook failed")
                    delay = compute_backoff(
                        attempt,
                        base=base_delay,
                        factor=factor,
                        cap=cap,
                        jitter=jitter,
                    )
                    sleep(delay)
            assert last_exc is not None
            raise RetryError(attempts, last_exc) from last_exc

        return wrapper

    return decorator


def retry_call(
    func: Callable,
    *args,
    attempts: int = 3,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    base_delay: float = 0.5,
    factor: float = 2.0,
    cap: float = 30.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
):
    """Imperative form of :func:`retry` for one-off calls."""
    decorated = retry(
        attempts=attempts,
        exceptions=exceptions,
        base_delay=base_delay,
        factor=factor,
        cap=cap,
        jitter=jitter,
        on_retry=on_retry,
        sleep=sleep,
    )(func)
    return decorated(*args, **kwargs)


__all__ = ["RetryError", "compute_backoff", "retry", "retry_call"]
