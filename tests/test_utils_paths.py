from __future__ import annotations

from pathlib import Path

import pytest

from deepface_server.utils import paths as ph


def test_safe_join_basic(tmp_path: Path):
    out = ph.safe_join(tmp_path, "a", "b.txt")
    assert str(out).startswith(str(tmp_path.resolve()))


def test_safe_join_traversal_blocked(tmp_path: Path):
    with pytest.raises(ph.PathSandboxError):
        ph.safe_join(tmp_path, "..", "etc", "passwd")


def test_safe_join_absolute_blocked(tmp_path: Path):
    with pytest.raises(ph.PathSandboxError):
        ph.safe_join(tmp_path, "/etc/passwd")


def test_is_within_true(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    assert ph.is_within(sub, tmp_path)


def test_is_within_false(tmp_path: Path):
    other = tmp_path.parent
    assert not ph.is_within(other, tmp_path)


def test_ensure_directory_creates(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    out = ph.ensure_directory(target)
    assert out.is_dir()


def test_iter_files_filters_extensions(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    files = list(ph.iter_files(tmp_path, extensions=("txt",)))
    assert len(files) == 1
    assert files[0].suffix == ".txt"


def test_iter_files_no_filter(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.png").write_bytes(b"\x89")
    assert len(list(ph.iter_files(tmp_path))) == 2


def test_file_size(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abcd")
    assert ph.file_size(f) == 4


def test_total_size_recursive(tmp_path: Path):
    (tmp_path / "a.txt").write_bytes(b"abc")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_bytes(b"de")
    assert ph.total_size(tmp_path) == 5
