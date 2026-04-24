from deepface_server.storage.connection import ConnectionFactory
from deepface_server.storage.migrations import latest_version, migrate


def test_connection_returns_thread_local_instance():
    factory = ConnectionFactory(":memory:")
    assert factory.get() is factory.get()
    factory.close()


def test_migrate_creates_tables():
    factory = ConnectionFactory(":memory:")
    applied = migrate(factory)
    assert applied == [1, 2, 3]
    cur = factory.get().execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert {"analysis_records", "batch_records", "jobs", "schema_version"}.issubset(tables)
    factory.close()


def test_migrate_is_idempotent():
    factory = ConnectionFactory(":memory:")
    migrate(factory)
    second = migrate(factory)
    assert second == []
    assert latest_version() >= 3
    factory.close()


def test_transaction_rolls_back_on_error():
    factory = ConnectionFactory(":memory:")
    migrate(factory)
    try:
        with factory.transaction() as conn:
            conn.execute(
                "INSERT INTO schema_version(version, name) VALUES(?, ?)", (999, "boom")
            )
            raise RuntimeError("force rollback")
    except RuntimeError:
        pass
    cur = factory.get().execute("SELECT COUNT(*) FROM schema_version WHERE version=999")
    assert cur.fetchone()[0] == 0
    factory.close()
