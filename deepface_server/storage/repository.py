"""Repository classes wrapping SQL access for each model."""
from __future__ import annotations

from typing import Iterable, Optional

from .connection import ConnectionFactory
from .models import AnalysisRecord, BatchRecord, JobRecord


class AnalysisRepository:
    def __init__(self, factory: ConnectionFactory):
        self.factory = factory

    def save(self, record: AnalysisRecord) -> AnalysisRecord:
        conn = self.factory.get()
        cur = conn.execute(
            """
            INSERT INTO analysis_records
            (request_id, fingerprint, actions, age, dominant_emotion,
             dominant_gender, raw_result, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.request_id,
                record.fingerprint,
                record.actions,
                record.age,
                record.dominant_emotion,
                record.dominant_gender,
                record.raw_result,
                record.duration_ms,
                record.created_at,
            ),
        )
        record.id = cur.lastrowid
        return record

    def get(self, record_id: int) -> Optional[AnalysisRecord]:
        cur = self.factory.get().execute(
            "SELECT * FROM analysis_records WHERE id = ?", (record_id,)
        )
        row = cur.fetchone()
        return AnalysisRecord.from_row(row) if row else None

    def by_fingerprint(self, fingerprint: str, limit: int = 10) -> list[AnalysisRecord]:
        cur = self.factory.get().execute(
            """
            SELECT * FROM analysis_records
            WHERE fingerprint = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (fingerprint, limit),
        )
        return [AnalysisRecord.from_row(row) for row in cur.fetchall()]

    def list(self, limit: int = 50, offset: int = 0) -> list[AnalysisRecord]:
        cur = self.factory.get().execute(
            "SELECT * FROM analysis_records ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [AnalysisRecord.from_row(row) for row in cur.fetchall()]

    def count(self) -> int:
        cur = self.factory.get().execute("SELECT COUNT(*) FROM analysis_records")
        return int(cur.fetchone()[0])

    def delete_older_than(self, iso_timestamp: str) -> int:
        cur = self.factory.get().execute(
            "DELETE FROM analysis_records WHERE created_at < ?",
            (iso_timestamp,),
        )
        return cur.rowcount or 0

    def bulk_save(self, records: Iterable[AnalysisRecord]) -> int:
        records = list(records)
        if not records:
            return 0
        with self.factory.transaction() as conn:
            for record in records:
                cur = conn.execute(
                    """
                    INSERT INTO analysis_records
                    (request_id, fingerprint, actions, age, dominant_emotion,
                     dominant_gender, raw_result, duration_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.request_id,
                        record.fingerprint,
                        record.actions,
                        record.age,
                        record.dominant_emotion,
                        record.dominant_gender,
                        record.raw_result,
                        record.duration_ms,
                        record.created_at,
                    ),
                )
                record.id = cur.lastrowid
        return len(records)


class BatchRepository:
    def __init__(self, factory: ConnectionFactory):
        self.factory = factory

    def save(self, record: BatchRecord) -> BatchRecord:
        conn = self.factory.get()
        cur = conn.execute(
            """
            INSERT INTO batch_records
            (request_id, item_count, success_count, failure_count,
             duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.request_id,
                record.item_count,
                record.success_count,
                record.failure_count,
                record.duration_ms,
                record.created_at,
            ),
        )
        record.id = cur.lastrowid
        return record

    def list(self, limit: int = 50) -> list[BatchRecord]:
        cur = self.factory.get().execute(
            "SELECT * FROM batch_records ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [BatchRecord.from_row(row) for row in cur.fetchall()]

    def aggregate(self) -> dict:
        cur = self.factory.get().execute(
            """
            SELECT COUNT(*) AS batches,
                   COALESCE(SUM(item_count), 0) AS total_items,
                   COALESCE(SUM(success_count), 0) AS successes,
                   COALESCE(SUM(failure_count), 0) AS failures,
                   COALESCE(AVG(duration_ms), 0) AS avg_duration_ms
            FROM batch_records
            """
        )
        row = cur.fetchone()
        return {k: row[k] for k in row.keys()}


class JobRepository:
    def __init__(self, factory: ConnectionFactory):
        self.factory = factory

    def save(self, record: JobRecord) -> JobRecord:
        conn = self.factory.get()
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs
            (id, status, submitted_at, started_at, finished_at,
             payload, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.status,
                record.submitted_at,
                record.started_at,
                record.finished_at,
                record.payload,
                record.result,
                record.error,
            ),
        )
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        cur = self.factory.get().execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        return JobRecord.from_row(row) if row else None

    def by_status(self, status: str) -> list[JobRecord]:
        cur = self.factory.get().execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY submitted_at",
            (status,),
        )
        return [JobRecord.from_row(row) for row in cur.fetchall()]

    def update_status(self, job_id: str, status: str, **fields) -> Optional[JobRecord]:
        record = self.get(job_id)
        if record is None:
            return None
        record.status = status
        for key, value in fields.items():
            if hasattr(record, key):
                setattr(record, key, value)
        return self.save(record)
