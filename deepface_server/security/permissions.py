"""Role-based access control with hierarchical permissions.

Roles are sets of permission strings. A permission can use ``*`` as a
wildcard at any segment (``analyze:*`` matches both ``analyze:read`` and
``analyze:write``). Roles can extend other roles, in which case the
permissions of the parent are merged in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Set


class PermissionDeniedError(Exception):
    """Raised by :func:`require` when a check fails."""


@dataclass(frozen=True)
class Role:
    name: str
    permissions: FrozenSet[str] = field(default_factory=frozenset)
    extends: FrozenSet[str] = field(default_factory=frozenset)


class RoleRegistry:
    """Container of roles with permission resolution."""

    def __init__(self, roles: Iterable[Role] = ()) -> None:
        self._roles: Dict[str, Role] = {}
        for role in roles:
            self.register(role)

    def register(self, role: Role) -> None:
        if role.name in self._roles:
            raise ValueError(f"duplicate role: {role.name}")
        self._roles[role.name] = role

    def get(self, name: str) -> Role:
        try:
            return self._roles[name]
        except KeyError as exc:
            raise KeyError(f"unknown role: {name}") from exc

    def names(self) -> Set[str]:
        return set(self._roles)

    def resolve(self, role_name: str) -> FrozenSet[str]:
        """Return the full set of permissions for a role (with inheritance)."""
        seen: set[str] = set()
        permissions: set[str] = set()
        self._resolve(role_name, seen, permissions)
        return frozenset(permissions)

    def _resolve(self, name: str, seen: set[str], out: set[str]) -> None:
        if name in seen:
            raise ValueError(f"role inheritance cycle through {name!r}")
        seen.add(name)
        role = self.get(name)
        out.update(role.permissions)
        for parent in role.extends:
            self._resolve(parent, seen, out)


def matches(permission: str, granted: str) -> bool:
    """Return True when ``granted`` covers ``permission``.

    Supports ``*`` wildcards segment-by-segment and a final ``*`` for
    open-ended grants. Examples:

    >>> matches("analyze:read", "analyze:read")
    True
    >>> matches("analyze:read", "analyze:*")
    True
    >>> matches("analyze:read", "*")
    True
    """
    if granted == "*":
        return True
    perm_parts = permission.split(":")
    grant_parts = granted.split(":")
    if len(grant_parts) > len(perm_parts):
        return False
    for granted_seg, perm_seg in zip(grant_parts, perm_parts):
        if granted_seg == "*":
            continue
        if granted_seg != perm_seg:
            return False
    # If the grant is shorter, the trailing must be a wildcard
    if len(grant_parts) < len(perm_parts):
        return grant_parts[-1] == "*"
    return True


def has_permission(grants: Iterable[str], permission: str) -> bool:
    """True when any grant in ``grants`` covers ``permission``."""
    return any(matches(permission, g) for g in grants)


def require(grants: Iterable[str], permission: str) -> None:
    """Raise :class:`PermissionDeniedError` when the check fails."""
    if not has_permission(grants, permission):
        raise PermissionDeniedError(f"missing permission: {permission}")


def default_registry() -> RoleRegistry:
    """The roles shipped with the service.

    - ``viewer`` — read-only on analyses and history.
    - ``analyst`` — viewer plus the ability to submit analyses and download
      reports.
    - ``operator`` — analyst plus job control.
    - ``admin`` — operator plus user management and system configuration.
    """
    return RoleRegistry(
        [
            Role(
                name="viewer",
                permissions=frozenset(
                    {
                        "analyze:read",
                        "history:read",
                        "metrics:read",
                    }
                ),
            ),
            Role(
                name="analyst",
                extends=frozenset({"viewer"}),
                permissions=frozenset(
                    {
                        "analyze:write",
                        "history:export",
                    }
                ),
            ),
            Role(
                name="operator",
                extends=frozenset({"analyst"}),
                permissions=frozenset(
                    {
                        "jobs:read",
                        "jobs:cancel",
                        "jobs:retry",
                        "webhooks:read",
                    }
                ),
            ),
            Role(
                name="admin",
                extends=frozenset({"operator"}),
                permissions=frozenset(
                    {
                        "users:*",
                        "settings:*",
                        "webhooks:*",
                    }
                ),
            ),
        ]
    )


__all__ = [
    "PermissionDeniedError",
    "Role",
    "RoleRegistry",
    "default_registry",
    "has_permission",
    "matches",
    "require",
]
