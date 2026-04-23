"""Schema migrations for the SQLite store."""
from __future__ import annotations

from typing import Callable, Sequence

from .connection import ConnectionFactory


_MIGRATIONS: list[tuple[int, str, Callable[[object], None]]] = []


def _migration(version: int, name: str):
    def decorator(fn: Callable[[object], None]):
        _MIGRATIONS.append((version, name, fn))
        return fn

    return decorator


@_migration(1, "initial schema")
def _initial(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            actions TEXT NOT NULL,
            age REAL,
            dominant_emotion TEXT,
            dominant_gender TEXT,
            raw_result TEXT NOT NULL,
            duration_ms REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analysis_fingerprint
        ON analysis_records(fingerprint)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analysis_request_id
        ON analysis_records(request_id)
        """
    )


@_migration(2, "batch records")
def _batches(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            item_count INTEGER NOT NULL,
            success_count INTEGER NOT NULL,
            failure_count INTEGER NOT NULL,
            duration_ms REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


@_migration(3, "jobs")
def _jobs(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            payload TEXT NOT NULL,
            result TEXT,
            error TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")


def applied_versions(conn) -> list[int]:
    cur = conn.execute("SELECT version FROM schema_version ORDER BY version")
    return [row[0] for row in cur.fetchall()]


def migrate(factory: ConnectionFactory) -> Sequence[int]:
    """Apply all pending migrations and return the list of new versions applied."""
    conn = factory.get()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    already = set(applied_versions(conn))
    applied: list[int] = []
    for version, name, fn in sorted(_MIGRATIONS, key=lambda m: m[0]):
        if version in already:
            continue
        fn(conn)
        conn.execute(
            "INSERT INTO schema_version(version, name) VALUES(?, ?)",
            (version, name),
        )
        applied.append(version)
    return applied


def latest_version() -> int:
    return max(v for v, _, _ in _MIGRATIONS) if _MIGRATIONS else 0
