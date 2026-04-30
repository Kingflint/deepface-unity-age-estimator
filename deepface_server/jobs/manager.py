"""High level API around :class:`JobQueue` and :class:`JobWorker`."""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from .models import Job, JobStatus
from .queue import JobQueue
from .worker import JobWorker


class JobManager:
    """Owns a queue and a pool of worker threads."""

    def __init__(self, handler: Callable[[Job], Any], worker_count: int = 1):
        if worker_count < 0:
            raise ValueError("worker_count must be >= 0")
        self.queue = JobQueue()
        self.handler = handler
        self.workers: list[JobWorker] = []
        self._lock = threading.Lock()
        self._worker_count = worker_count

    def start(self) -> None:
        with self._lock:
            if self.workers:
                return
            for i in range(self._worker_count):
                worker = JobWorker(
                    self.queue, self.handler, name=f"JobWorker-{i}"
                )
                worker.start()
                self.workers.append(worker)

    def stop(self, timeout: Optional[float] = 1.0) -> None:
        with self._lock:
            for worker in self.workers:
                worker.stop()
            for worker in self.workers:
                worker.join(timeout=timeout)
            self.workers.clear()

    def submit(self, kind: str, payload: dict[str, Any] | None = None) -> Job:
        job = Job(kind=kind, payload=payload or {})
        return self.queue.submit(job)

    def submit_sync(self, kind: str, payload: dict[str, Any] | None = None) -> Job:
        """Submit + run synchronously without spawning a worker thread."""
        job = Job(kind=kind, payload=payload or {})
        self.queue.update(job)
        worker = JobWorker(self.queue, self.handler, name="JobWorker-sync")
        return worker.process(job)

    def get(self, job_id: str) -> Optional[Job]:
        return self.queue.get(job_id)

    def cancel(self, job_id: str) -> bool:
        return self.queue.cancel(job_id)

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self.queue),
            "pending_in_queue": self.queue.pending(),
            "succeeded": len(self.queue.by_status(JobStatus.SUCCEEDED)),
            "failed": len(self.queue.by_status(JobStatus.FAILED)),
            "cancelled": len(self.queue.by_status(JobStatus.CANCELLED)),
            "running": len(self.queue.by_status(JobStatus.RUNNING)),
        }
