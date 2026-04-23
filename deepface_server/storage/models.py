"""Dataclass models for persisted records."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class AnalysisRecord:
    """A single image analysis result persisted in the database."""

    id: int | None = None
    request_id: str = ""
    fingerprint: str = ""
    actions: str = ""  # comma separated
    age: float | None = None
    dominant_emotion: str | None = None
    dominant_gender: str | None = None
    raw_result: str = "{}"
    duration_ms: float = 0.0
    created_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: Any) -> "AnalysisRecord":
        return cls(**{k: row[k] for k in row.keys()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def parsed_result(self) -> Any:
        try:
            return json.loads(self.raw_result)
        except json.JSONDecodeError:
            return None


@dataclass
class BatchRecord:
    id: int | None = None
    request_id: str = ""
    item_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    duration_ms: float = 0.0
    created_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: Any) -> "BatchRecord":
        return cls(**{k: row[k] for k in row.keys()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobRecord:
    id: str = ""
    status: str = "pending"
    submitted_at: str = field(default_factory=_utcnow_iso)
    started_at: str | None = None
    finished_at: str | None = None
    payload: str = "{}"
    result: str | None = None
    error: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "JobRecord":
        return cls(**{k: row[k] for k in row.keys()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
