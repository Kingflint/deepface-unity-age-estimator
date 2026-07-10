from __future__ import annotations

import pytest

from deepface_server.webhooks import event_types as et


def test_known_event_types_includes_basics():
    assert et.ANALYSIS_COMPLETED in et.KNOWN_EVENT_TYPES
    assert et.JOB_FAILED in et.KNOWN_EVENT_TYPES


def test_is_valid_event_type_known():
    assert et.is_valid_event_type(et.ANALYSIS_STARTED)


def test_is_valid_event_type_custom():
    assert et.is_valid_event_type("billing.invoice_paid")


def test_is_valid_event_type_invalid():
    assert not et.is_valid_event_type("")
    assert not et.is_valid_event_type("nodot")
    assert not et.is_valid_event_type(".empty")
    assert not et.is_valid_event_type("ns.")


def test_subscription_matches_exact():
    sub = et.Subscription(id="s1", patterns=frozenset({"analysis.completed"}))
    assert sub.matches("analysis.completed")
    assert not sub.matches("analysis.failed")


def test_subscription_matches_wildcard_action():
    sub = et.Subscription(id="s1", patterns=frozenset({"analysis.*"}))
    assert sub.matches("analysis.completed")
    assert sub.matches("analysis.failed")
    assert not sub.matches("user.created")


def test_subscription_matches_full_wildcard():
    sub = et.Subscription(id="s1", patterns=frozenset({"*"}))
    assert sub.matches("analysis.completed")


def test_subscription_validation_invalid_pattern():
    with pytest.raises(et.EventTypeError):
        et.Subscription(id="s1", patterns=frozenset({"nodot"}))


def test_subscription_validation_empty_id():
    with pytest.raises(et.EventTypeError):
        et.Subscription(id="", patterns=frozenset({"a.b"}))


def test_registry_register_and_match():
    reg = et.SubscriptionRegistry()
    reg.register(et.Subscription(id="a", patterns=frozenset({"analysis.*"})))
    reg.register(et.Subscription(id="b", patterns=frozenset({"job.*"})))
    matched = reg.matching("analysis.completed")
    assert matched == {"a"}


def test_registry_duplicate():
    reg = et.SubscriptionRegistry()
    reg.register(et.Subscription(id="a", patterns=frozenset({"a.b"})))
    with pytest.raises(et.EventTypeError):
        reg.register(et.Subscription(id="a", patterns=frozenset({"c.d"})))


def test_registry_unregister():
    reg = et.SubscriptionRegistry()
    reg.register(et.Subscription(id="a", patterns=frozenset({"a.b"})))
    assert reg.unregister("a")
    assert not reg.unregister("a")


def test_parse_pattern_list():
    out = et.parse_pattern_list(["a.b", " ", "*"])
    assert out == frozenset({"a.b", "*"})


def test_parse_pattern_list_invalid():
    with pytest.raises(et.EventTypeError):
        et.parse_pattern_list(["nodot"])
