from __future__ import annotations

import pytest

from deepface_server.utils import validators as v


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a@b.com", True),
        ("first.last+tag@sub.example.co", True),
        ("not-an-email", False),
        ("a@b", False),
        ("", False),
        (None, False),
    ],
)
def test_is_email(value, expected):
    assert v.is_email(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://example.com", True),
        ("http://example.com/path?q=1", True),
        ("ftp://example.com", False),
        ("not a url", False),
        ("", False),
    ],
)
def test_is_url(value, expected):
    assert v.is_url(value) is expected


def test_is_url_custom_schemes():
    assert v.is_url("ftp://x.com", schemes=("ftp",))


def test_is_hostname():
    assert v.is_hostname("example.com")
    assert v.is_hostname("a-b.c")
    assert not v.is_hostname("-bad.com")
    assert not v.is_hostname("")


def test_is_ip_address():
    assert v.is_ip_address("127.0.0.1")
    assert v.is_ip_address("::1")
    assert not v.is_ip_address("999.999.999.999")


def test_is_private_ip():
    assert v.is_private_ip("10.0.0.1")
    assert v.is_private_ip("192.168.1.1")
    assert not v.is_private_ip("8.8.8.8")


def test_is_uuid():
    assert v.is_uuid("550e8400-e29b-41d4-a716-446655440000")
    assert not v.is_uuid("not-a-uuid")


def test_is_safe_filename():
    assert v.is_safe_filename("report.pdf")
    assert not v.is_safe_filename("../etc/passwd")
    assert not v.is_safe_filename("a/b")
    assert not v.is_safe_filename(".hidden")
    assert not v.is_safe_filename("")


def test_is_port():
    assert v.is_port(80)
    assert v.is_port("8080")
    assert not v.is_port(0)
    assert not v.is_port(70000)
    assert not v.is_port("xyz")


def test_validate_image_dimensions_ok():
    v.validate_image_dimensions(100, 200, max_dim=2000)


def test_validate_image_dimensions_too_small():
    with pytest.raises(ValueError):
        v.validate_image_dimensions(0, 100, max_dim=2000)


def test_validate_image_dimensions_too_large():
    with pytest.raises(ValueError):
        v.validate_image_dimensions(5000, 100, max_dim=2000)


def test_validate_age_clamps():
    assert v.validate_age(-3) == 0
    assert v.validate_age(200) == 120
    assert v.validate_age(42) == 42


def test_normalise_email_lowercases_domain():
    assert v.normalise_email("Foo@EXAMPLE.com") == "Foo@example.com"


def test_normalise_email_invalid():
    with pytest.raises(ValueError):
        v.normalise_email("nope")
