"""Thread-safe job queue."""
from __future__ import annotations

import queue
import threading
from typing import Optional

from .models import Job, JobStatus


class JobQueue:
    """A simple FIFO queue with an in-memory snapshot of every job ever seen."""

    def __init__(self):
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job)
        return job

    def take(self, timeout: Optional[float] = None) -> Optional[Job]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        try:
            self._queue.task_done()
        except ValueError:
            pass

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def by_status(self, status: JobStatus) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status == status]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status.is_terminal():
                return False
            job.status = JobStatus.CANCELLED
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def pending(self) -> int:
        return self._queue.qsize()
