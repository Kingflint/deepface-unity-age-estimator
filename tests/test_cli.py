import io
import json
import sys

import pytest

from deepface_server.cli import main


def _capture(monkeypatch, argv):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    rc = main(argv)
    return rc, buf.getvalue()


def test_version_command(monkeypatch):
    rc, out = _capture(monkeypatch, ["version"])
    assert rc == 0
    assert out.strip()


def test_config_command(monkeypatch):
    rc, out = _capture(monkeypatch, ["config"])
    assert rc == 0
    parsed = json.loads(out)
    assert "port" in parsed
    assert "log_level" in parsed


def test_migrate_command(monkeypatch):
    rc, out = _capture(monkeypatch, ["migrate", "--database", ":memory:"])
    assert rc == 0
    parsed = json.loads(out)
    assert parsed["database"] == ":memory:"
    assert parsed["latest_known"] >= 1


def test_openapi_command_stdout(monkeypatch):
    rc, out = _capture(monkeypatch, ["openapi", "--pretty"])
    assert rc == 0
    parsed = json.loads(out)
    assert parsed["openapi"].startswith("3.")
    assert "/analyze" in parsed["paths"]


def test_openapi_command_file(tmp_path, monkeypatch):
    target = tmp_path / "spec.json"
    rc, _ = _capture(monkeypatch, ["openapi", "--output", str(target)])
    assert rc == 0
    parsed = json.loads(target.read_text())
    assert parsed["info"]["title"]


def test_smoke_command(monkeypatch):
    rc, _ = _capture(monkeypatch, ["smoke"])
    assert rc == 0


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["nope"])
