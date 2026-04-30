from deepface_server.jobs import Job, JobManager, JobQueue, JobStatus, JobWorker


def test_job_status_terminality():
    assert JobStatus.SUCCEEDED.is_terminal()
    assert JobStatus.FAILED.is_terminal()
    assert JobStatus.CANCELLED.is_terminal()
    assert not JobStatus.PENDING.is_terminal()
    assert not JobStatus.RUNNING.is_terminal()


def test_queue_submit_take_round_trip():
    queue = JobQueue()
    job = Job(kind="analyze", payload={"x": 1})
    queue.submit(job)
    taken = queue.take(timeout=0.1)
    assert taken is job
    assert queue.get(job.id) is job


def test_queue_cancel_pending():
    queue = JobQueue()
    job = Job()
    queue.submit(job)
    assert queue.cancel(job.id) is True
    assert queue.get(job.id).status == JobStatus.CANCELLED


def test_queue_cancel_terminal_returns_false():
    queue = JobQueue()
    job = Job()
    queue.submit(job)
    job.status = JobStatus.SUCCEEDED
    queue.update(job)
    assert queue.cancel(job.id) is False


def test_worker_processes_job_synchronously():
    queue = JobQueue()

    def handler(job: Job) -> dict:
        return {"echo": job.payload}

    worker = JobWorker(queue, handler)
    job = Job(payload={"hello": "world"})
    queue.submit(job)
    queue.take(timeout=0.1)
    processed = worker.process(job)
    assert processed.status == JobStatus.SUCCEEDED
    assert processed.result == {"echo": {"hello": "world"}}


def test_worker_records_failure():
    queue = JobQueue()

    def handler(job: Job):
        raise ValueError("nope")

    worker = JobWorker(queue, handler)
    job = Job()
    queue.submit(job)
    queue.take(timeout=0.1)
    processed = worker.process(job)
    assert processed.status == JobStatus.FAILED
    assert "nope" in processed.error


def test_manager_submit_sync_runs_handler():
    manager = JobManager(handler=lambda job: {"ok": True}, worker_count=0)
    result = manager.submit_sync("analyze", {"image": "abc"})
    assert result.status == JobStatus.SUCCEEDED
    assert result.result == {"ok": True}


def test_manager_stats_shape():
    manager = JobManager(handler=lambda j: None, worker_count=0)
    manager.submit("analyze")
    stats = manager.stats()
    assert {"total", "pending_in_queue", "succeeded", "failed", "cancelled", "running"}.issubset(stats)


def test_manager_negative_worker_count_rejected():
    import pytest

    with pytest.raises(ValueError):
        JobManager(handler=lambda j: None, worker_count=-1)


def test_job_to_dict_contains_status_string():
    job = Job(kind="x")
    data = job.to_dict()
    assert data["status"] == "pending"
    assert data["kind"] == "x"
