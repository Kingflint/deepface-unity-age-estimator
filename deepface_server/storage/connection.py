"""SQLite connection factory with thread-local caching."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ConnectionFactory:
    """Creates and caches sqlite3 connections per thread.

    Each call to :meth:`get` returns a connection scoped to the current thread.
    SQLite connections cannot safely be shared across threads with the default
    check_same_thread=True, so we keep one per thread instead of using a pool.
    """

    def __init__(self, database_path: str | Path):
        self._path = str(database_path)
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return self._path

    def get(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with self._lock:
                conn.close()
                self._local.conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.get()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


_default: ConnectionFactory | None = None
_default_lock = threading.Lock()


def get_connection(path: str | Path = ":memory:") -> ConnectionFactory:
    """Return a process-wide :class:`ConnectionFactory` for ``path``."""
    global _default
    with _default_lock:
        if _default is None or _default.path != str(path):
            if _default is not None:
                _default.close()
            _default = ConnectionFactory(path)
        return _default
