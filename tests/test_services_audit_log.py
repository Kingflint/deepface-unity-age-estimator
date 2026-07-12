from __future__ import annotations

import json
from datetime import datetime, timezone

from deepface_server.services import audit_log as al


def test_log_writes_jsonline():
    logger = al.AuditLogger()
    logger.log(actor="alice", action="login", resource="session")
    out = logger.buffer().strip().splitlines()
    assert len(out) == 1
    record = json.loads(out[0])
    assert record["actor"] == "alice"
    assert record["action"] == "login"
    assert record["success"] is True


def test_redacts_sensitive_keys():
    logger = al.AuditLogger()
    logger.log(
        actor="bob",
        action="settings.update",
        resource="account",
        details={"password": "hunter2", "name": "Bob"},
    )
    record = json.loads(logger.buffer().strip())
    assert record["details"]["password"] == "***"
    assert record["details"]["name"] == "Bob"


def test_redacts_nested():
    logger = al.AuditLogger()
    logger.log(
        actor="bob",
        action="api.call",
        resource="account",
        details={"meta": {"api_key": "secret123", "host": "example.com"}},
    )
    record = json.loads(logger.buffer().strip())
    assert record["details"]["meta"]["api_key"] == "***"
    assert record["details"]["meta"]["host"] == "example.com"


def test_explicit_timestamp_used():
    ts = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    logger = al.AuditLogger()
    logger.log(actor="x", action="y", resource="z", timestamp=ts)
    assert "2024-06-01T12:00:00+00:00" in logger.buffer()


def test_failure_event():
    logger = al.AuditLogger()
    logger.log(actor="x", action="login", resource="session", success=False)
    record = json.loads(logger.buffer().strip())
    assert record["success"] is False


def test_events_returns_recorded():
    logger = al.AuditLogger()
    logger.log(actor="a", action="x", resource="r")
    logger.log(actor="b", action="y", resource="r")
    assert len(logger.events()) == 2


def test_to_file(tmp_path):
    path = tmp_path / "audit.log"
    logger = al.AuditLogger.to_file(str(path))
    logger.log(actor="x", action="y", resource="z")
    logger.close()
    assert path.exists()
    assert "actor" in path.read_text()
