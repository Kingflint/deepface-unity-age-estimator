"""Tiny in-process metrics collector for /metrics."""
from __future__ import annotations

import threading
from collections import defaultdict


class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, int], int] = defaultdict(int)
        self._sum_ms: dict[str, float] = defaultdict(float)
        self._n: dict[str, int] = defaultdict(int)

    def record(self, path: str, status_code: int, elapsed_ms: float) -> None:
        with self._lock:
            self._counts[(path, status_code)] += 1
            self._sum_ms[path] += elapsed_ms
            self._n[path] += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg = {p: round(self._sum_ms[p] / self._n[p], 3) for p in self._n}
            counts = {f"{p} {s}": c for (p, s), c in self._counts.items()}
            total = sum(self._n.values())
            return {
                "total_requests": total,
                "counts_by_path_status": counts,
                "avg_ms_by_path": avg,
            }

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._sum_ms.clear()
            self._n.clear()
