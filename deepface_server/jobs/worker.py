"""Worker thread that drains a :class:`JobQueue`."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Callable

from .models import Job, JobStatus
from .queue import JobQueue

logger = logging.getLogger("deepface_server.jobs.worker")


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class JobWorker(threading.Thread):
    """Drains jobs from a queue and executes them via a handler function."""

    def __init__(
        self,
        job_queue: JobQueue,
        handler: Callable[[Job], object],
        *,
        name: str = "JobWorker",
        poll_timeout: float = 0.5,
    ):
        super().__init__(name=name, daemon=True)
        self.job_queue = job_queue
        self.handler = handler
        self.poll_timeout = poll_timeout
        self._stop_event = threading.Event()

    def run(self) -> None:  # pragma: no cover - thread loop
        while not self._stop_event.is_set():
            job = self.job_queue.take(timeout=self.poll_timeout)
            if job is None:
                continue
            try:
                self.process(job)
            finally:
                self.job_queue.task_done()

    def process(self, job: Job) -> Job:
        if job.status == JobStatus.CANCELLED:
            return job
        job.status = JobStatus.RUNNING
        job.started_at = _utcnow()
        self.job_queue.update(job)
        try:
            job.result = self.handler(job)
            job.status = JobStatus.SUCCEEDED
        except Exception as exc:  # pragma: no cover - exercised in tests
            logger.exception("job %s failed", job.id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
        job.finished_at = _utcnow()
        self.job_queue.update(job)
        return job

    def stop(self) -> None:
        self._stop_event.set()
