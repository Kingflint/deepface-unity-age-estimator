"""Append-only audit logger.

Events are serialised as JSON-Lines. Sensitive fields are redacted by
default before being written.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, TextIO


_DEFAULT_REDACTED = ("password", "secret", "token", "api_key", "authorization")


def _redact(payload: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    redacted_keys = {k.lower() for k in keys}
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if k.lower() in redacted_keys:
            out[k] = "***"
        elif isinstance(v, dict):
            out[k] = _redact(v, redacted_keys)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class AuditEvent:
    actor: str
    action: str
    resource: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self, *, redact: Iterable[str] = _DEFAULT_REDACTED) -> Dict[str, Any]:
        return {
            "ts": self.timestamp.astimezone(timezone.utc).isoformat(),
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "success": self.success,
            "details": _redact(self.details, redact),
        }


class AuditLogger:
    """Write audit events to a file or in-memory stream."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        *,
        redact: Iterable[str] = _DEFAULT_REDACTED,
    ) -> None:
        self._stream = stream if stream is not None else io.StringIO()
        self._redact = tuple(redact)
        self._owns_stream = stream is None
        self._events: List[AuditEvent] = []

    @classmethod
    def to_file(cls, path: str, **kwargs: Any) -> "AuditLogger":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return cls(stream=open(path, "a", encoding="utf-8"), **kwargs)

    def log(
        self,
        *,
        actor: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        timestamp: Optional[datetime] = None,
    ) -> AuditEvent:
        event = AuditEvent(
            actor=actor,
            action=action,
            resource=resource,
            timestamp=timestamp or datetime.now(timezone.utc),
            details=dict(details or {}),
            success=success,
        )
        line = json.dumps(event.to_dict(redact=self._redact), separators=(",", ":"), sort_keys=True)
        self._stream.write(line + "\n")
        self._stream.flush()
        self._events.append(event)
        return event

    def events(self) -> List[AuditEvent]:
        return list(self._events)

    def buffer(self) -> str:
        if isinstance(self._stream, io.StringIO):
            return self._stream.getvalue()
        raise TypeError("buffer() is only available for in-memory loggers")

    def close(self) -> None:
        if self._owns_stream:
            try:
                self._stream.close()
            except Exception:  # pragma: no cover
                pass


__all__ = ["AuditEvent", "AuditLogger"]
