"""Path safety helpers — guard against directory traversal and symlinks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


class PathSandboxError(Exception):
    """Raised when a path attempts to escape its sandbox."""


def safe_join(base: str | os.PathLike, *parts: str) -> Path:
    """Join ``parts`` under ``base`` and verify the result stays inside it.

    Raises :class:`PathSandboxError` for traversal (``../`` components,
    absolute paths in ``parts``, or symlink targets outside ``base``).
    """
    base_path = Path(base).resolve()
    candidate = base_path
    for part in parts:
        if not part:
            continue
        as_path = Path(part)
        if as_path.is_absolute():
            raise PathSandboxError(f"absolute component disallowed: {part!r}")
        candidate = candidate / part
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_path)
    except ValueError as exc:
        raise PathSandboxError(
            f"path {resolved} escapes sandbox {base_path}"
        ) from exc
    return resolved


def is_within(child: str | os.PathLike, parent: str | os.PathLike) -> bool:
    """Return True when ``child`` lies inside ``parent`` after resolving."""
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def ensure_directory(path: str | os.PathLike) -> Path:
    """Create ``path`` and any missing parents; return as :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_files(
    root: str | os.PathLike,
    *,
    extensions: Iterable[str] | None = None,
) -> Iterable[Path]:
    """Yield every file under ``root`` matching ``extensions`` (case-insensitive)."""
    root_path = Path(root)
    allowed = None
    if extensions is not None:
        allowed = {ext.lower().lstrip(".") for ext in extensions}
    for entry in root_path.rglob("*"):
        if not entry.is_file():
            continue
        if allowed is not None and entry.suffix.lower().lstrip(".") not in allowed:
            continue
        yield entry


def file_size(path: str | os.PathLike) -> int:
    return Path(path).stat().st_size


def total_size(root: str | os.PathLike) -> int:
    """Total bytes under ``root`` (recursive). Skips broken symlinks."""
    total = 0
    for entry in Path(root).rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


__all__ = [
    "PathSandboxError",
    "ensure_directory",
    "file_size",
    "is_within",
    "iter_files",
    "safe_join",
    "total_size",
]
