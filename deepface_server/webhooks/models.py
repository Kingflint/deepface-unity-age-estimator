"""Models for webhook events."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class WebhookEvent:
    id: str = field(default_factory=_new_id)
    type: str = "analysis.completed"
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "type": self.type,
                "payload": self.payload,
                "created_at": self.created_at,
            },
            default=str,
            sort_keys=True,
        )


@dataclass
class DeliveryAttempt:
    event_id: str
    url: str
    status_code: int | None = None
    error: str | None = None
    duration_ms: float = 0.0
    attempt: int = 1
    next_retry_in_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def succeeded(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 300
