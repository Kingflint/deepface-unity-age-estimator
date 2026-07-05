"""A FIFO-stable priority queue used by the job manager."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class PriorityQueueEmpty(Exception):
    pass


@dataclass(order=True)
class _Entry(Generic[T]):
    priority: int
    sequence: int
    payload: T = field(compare=False)


class PriorityQueue(Generic[T]):
    """A min-heap priority queue with stable ordering.

    Lower priority values are returned first. Items pushed with the
    same priority are returned in FIFO order.
    """

    def __init__(self) -> None:
        self._heap: List[_Entry[T]] = []
        self._counter = itertools.count()
        self._removed: set[int] = set()

    def push(self, payload: T, priority: int = 0) -> int:
        """Push ``payload`` and return a token that can be used to cancel."""
        seq = next(self._counter)
        heapq.heappush(self._heap, _Entry(priority=priority, sequence=seq, payload=payload))
        return seq

    def pop(self) -> T:
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.sequence in self._removed:
                self._removed.discard(entry.sequence)
                continue
            return entry.payload
        raise PriorityQueueEmpty("queue is empty")

    def peek(self) -> T:
        while self._heap:
            entry = self._heap[0]
            if entry.sequence in self._removed:
                heapq.heappop(self._heap)
                self._removed.discard(entry.sequence)
                continue
            return entry.payload
        raise PriorityQueueEmpty("queue is empty")

    def cancel(self, token: int) -> bool:
        """Cancel a queued item by its token."""
        for entry in self._heap:
            if entry.sequence == token:
                self._removed.add(token)
                return True
        return False

    def __len__(self) -> int:
        return sum(1 for entry in self._heap if entry.sequence not in self._removed)

    def __bool__(self) -> bool:
        return len(self) > 0

    def drain(self) -> Iterator[T]:
        while True:
            try:
                yield self.pop()
            except PriorityQueueEmpty:
                return

    def snapshot(self) -> List[T]:
        """Return remaining payloads in priority order without consuming."""
        live = [e for e in self._heap if e.sequence not in self._removed]
        live.sort()
        return [e.payload for e in live]


__all__ = ["PriorityQueue", "PriorityQueueEmpty"]
