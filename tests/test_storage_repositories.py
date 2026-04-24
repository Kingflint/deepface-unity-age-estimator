import pytest

from deepface_server.storage.connection import ConnectionFactory
from deepface_server.storage.migrations import migrate
from deepface_server.storage.models import AnalysisRecord, BatchRecord, JobRecord
from deepface_server.storage.repository import (
    AnalysisRepository,
    BatchRepository,
    JobRepository,
)


@pytest.fixture
def factory():
    f = ConnectionFactory(":memory:")
    migrate(f)
    yield f
    f.close()


def _record(**overrides) -> AnalysisRecord:
    base = dict(
        request_id="req-1",
        fingerprint="fp-1",
        actions="age,emotion",
        age=32.0,
        dominant_emotion="neutral",
        dominant_gender="Man",
        raw_result='{"age": 32.0}',
        duration_ms=12.5,
    )
    base.update(overrides)
    return AnalysisRecord(**base)


def test_save_and_get_round_trip(factory):
    repo = AnalysisRepository(factory)
    saved = repo.save(_record())
    assert saved.id is not None
    fetched = repo.get(saved.id)
    assert fetched is not None
    assert fetched.fingerprint == "fp-1"


def test_by_fingerprint_filters_correctly(factory):
    repo = AnalysisRepository(factory)
    repo.save(_record(fingerprint="alpha"))
    repo.save(_record(fingerprint="alpha"))
    repo.save(_record(fingerprint="beta"))
    assert len(repo.by_fingerprint("alpha")) == 2
    assert len(repo.by_fingerprint("beta")) == 1


def test_count_and_pagination(factory):
    repo = AnalysisRepository(factory)
    for i in range(5):
        repo.save(_record(request_id=f"req-{i}"))
    assert repo.count() == 5
    page = repo.list(limit=2, offset=1)
    assert len(page) == 2


def test_bulk_save(factory):
    repo = AnalysisRepository(factory)
    inserted = repo.bulk_save([_record() for _ in range(4)])
    assert inserted == 4
    assert repo.count() == 4


def test_delete_older_than(factory):
    repo = AnalysisRepository(factory)
    repo.save(_record(created_at="2020-01-01T00:00:00Z"))
    repo.save(_record(created_at="2099-01-01T00:00:00Z"))
    deleted = repo.delete_older_than("2025-01-01T00:00:00Z")
    assert deleted == 1
    assert repo.count() == 1


def test_batch_repository_aggregate(factory):
    repo = BatchRepository(factory)
    repo.save(BatchRecord(request_id="r1", item_count=3, success_count=2, failure_count=1, duration_ms=120.0))
    repo.save(BatchRecord(request_id="r2", item_count=4, success_count=4, failure_count=0, duration_ms=200.0))
    summary = repo.aggregate()
    assert summary["batches"] == 2
    assert summary["total_items"] == 7
    assert summary["successes"] == 6
    assert summary["failures"] == 1


def test_job_repository_lifecycle(factory):
    repo = JobRepository(factory)
    repo.save(JobRecord(id="j1", status="pending", payload="{}"))
    assert repo.get("j1") is not None
    assert repo.by_status("pending")[0].id == "j1"
    repo.update_status("j1", "running", started_at="2026-04-30T00:00:00Z")
    assert repo.get("j1").status == "running"
