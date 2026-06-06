from __future__ import annotations

import pytest

from deepface_server.security import permissions as pm


def test_matches_exact():
    assert pm.matches("a:b", "a:b")
    assert not pm.matches("a:b", "a:c")


def test_matches_wildcard_segment():
    assert pm.matches("a:b", "a:*")
    assert not pm.matches("x:y", "a:*")


def test_matches_full_wildcard():
    assert pm.matches("a:b:c", "*")


def test_matches_short_grant_with_trailing_star():
    # Trailing * at end of grant covers deeper paths
    assert pm.matches("a:b:c", "a:*")


def test_has_permission():
    grants = ["analyze:*"]
    assert pm.has_permission(grants, "analyze:read")
    assert not pm.has_permission(grants, "users:read")


def test_require_passes_silently():
    pm.require(["a:*"], "a:b")


def test_require_raises():
    with pytest.raises(pm.PermissionDeniedError):
        pm.require(["a:b"], "users:delete")


def test_default_registry_resolves_inheritance():
    reg = pm.default_registry()
    perms = reg.resolve("admin")
    assert "analyze:read" in perms
    assert "users:*" in perms


def test_default_registry_viewer_minimal():
    reg = pm.default_registry()
    perms = reg.resolve("viewer")
    assert "analyze:read" in perms
    assert "users:*" not in perms


def test_role_registry_duplicate():
    reg = pm.RoleRegistry([pm.Role(name="x")])
    with pytest.raises(ValueError):
        reg.register(pm.Role(name="x"))


def test_role_registry_unknown():
    reg = pm.RoleRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_inheritance_cycle_detected():
    reg = pm.RoleRegistry(
        [
            pm.Role(name="a", extends=frozenset({"b"})),
            pm.Role(name="b", extends=frozenset({"a"})),
        ]
    )
    with pytest.raises(ValueError):
        reg.resolve("a")
