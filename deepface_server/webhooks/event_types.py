"""Webhook event-type registry and filter expressions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Set


class EventTypeError(ValueError):
    pass


# Canonical event type names emitted by the service.
ANALYSIS_STARTED = "analysis.started"
ANALYSIS_COMPLETED = "analysis.completed"
ANALYSIS_FAILED = "analysis.failed"
JOB_QUEUED = "job.queued"
JOB_STARTED = "job.started"
JOB_FINISHED = "job.finished"
JOB_FAILED = "job.failed"
USER_CREATED = "user.created"
USER_DELETED = "user.deleted"
QUOTA_EXCEEDED = "quota.exceeded"

KNOWN_EVENT_TYPES: FrozenSet[str] = frozenset(
    {
        ANALYSIS_STARTED,
        ANALYSIS_COMPLETED,
        ANALYSIS_FAILED,
        JOB_QUEUED,
        JOB_STARTED,
        JOB_FINISHED,
        JOB_FAILED,
        USER_CREATED,
        USER_DELETED,
        QUOTA_EXCEEDED,
    }
)


def is_valid_event_type(name: str) -> bool:
    """Permit known events plus user extensions of the form ``ns.event``."""
    if name in KNOWN_EVENT_TYPES:
        return True
    if not name or "." not in name:
        return False
    namespace, _, action = name.partition(".")
    return all(p and p.replace("_", "").isalnum() for p in (namespace, action))


@dataclass(frozen=True)
class Subscription:
    """A subscription to one or more event types.

    Wildcards on the *action* part are supported (``analysis.*``).
    """

    id: str
    patterns: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.id:
            raise EventTypeError("subscription id required")
        for p in self.patterns:
            if p == "*":
                continue
            if "." not in p:
                raise EventTypeError(f"invalid pattern: {p}")
            ns, _, action = p.partition(".")
            if not ns or not action:
                raise EventTypeError(f"invalid pattern: {p}")

    def matches(self, event_type: str) -> bool:
        if not is_valid_event_type(event_type):
            return False
        for p in self.patterns:
            if p == "*":
                return True
            if p == event_type:
                return True
            ns, _, action = p.partition(".")
            ev_ns, _, ev_action = event_type.partition(".")
            if ns == ev_ns and action == "*":
                return True
        return False


class SubscriptionRegistry:
    def __init__(self) -> None:
        self._subs: Dict[str, Subscription] = {}

    def register(self, sub: Subscription) -> None:
        if sub.id in self._subs:
            raise EventTypeError(f"duplicate subscription: {sub.id}")
        self._subs[sub.id] = sub

    def unregister(self, sub_id: str) -> bool:
        return self._subs.pop(sub_id, None) is not None

    def matching(self, event_type: str) -> Set[str]:
        return {sub.id for sub in self._subs.values() if sub.matches(event_type)}

    def __len__(self) -> int:
        return len(self._subs)

    def all_ids(self) -> Set[str]:
        return set(self._subs)


def parse_pattern_list(raw: Iterable[str]) -> FrozenSet[str]:
    out: set[str] = set()
    for item in raw:
        item = item.strip()
        if not item:
            continue
        if item == "*":
            out.add(item)
            continue
        if "." not in item:
            raise EventTypeError(f"invalid pattern: {item}")
        out.add(item)
    return frozenset(out)


__all__ = [
    "ANALYSIS_COMPLETED",
    "ANALYSIS_FAILED",
    "ANALYSIS_STARTED",
    "EventTypeError",
    "JOB_FAILED",
    "JOB_FINISHED",
    "JOB_QUEUED",
    "JOB_STARTED",
    "KNOWN_EVENT_TYPES",
    "QUOTA_EXCEEDED",
    "Subscription",
    "SubscriptionRegistry",
    "USER_CREATED",
    "USER_DELETED",
    "is_valid_event_type",
    "parse_pattern_list",
]
