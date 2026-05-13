from __future__ import annotations

import pytest

from deepface_server.utils import text as t


def test_slugify_basic():
    assert t.slugify("Hello, World!") == "hello-world"


def test_slugify_unicode():
    assert t.slugify("Café résumé") == "cafe-resume"


def test_slugify_separator():
    assert t.slugify("Hello World", separator="_") == "hello_world"


def test_slugify_max_length():
    assert len(t.slugify("a" * 200, max_length=10)) <= 10


def test_slugify_none():
    assert t.slugify(None) == ""


def test_truncate_no_change():
    assert t.truncate("hi", 10) == "hi"


def test_truncate_with_ellipsis():
    out = t.truncate("hello world", 8)
    assert len(out) <= 8
    assert out.endswith("…")


def test_truncate_zero():
    assert t.truncate("hi", 0) == ""


def test_normalise_whitespace():
    assert t.normalise_whitespace("  a  \t b\n c ") == "a b c"


def test_strip_control_chars():
    assert t.strip_control_chars("ab\x00c\x07d") == "abcd"


def test_redact_emails():
    assert "[redacted-email]" in t.redact_emails("contact me at a@b.com")


def test_redact_phones():
    out = t.redact_phones("call +1 (555) 123-4567 now")
    assert "[redacted-phone]" in out


def test_redact_pii_combined():
    out = t.redact_pii("email a@b.com or +15551234567")
    assert "[redacted-email]" in out
    assert "[redacted-phone]" in out


def test_to_snake_case():
    assert t.to_snake_case("CamelCaseName") == "camel_case_name"
    assert t.to_snake_case("camelCase") == "camel_case"
    assert t.to_snake_case("HTTPRequest") == "http_request"


def test_to_camel_case():
    assert t.to_camel_case("snake_case_name") == "snakeCaseName"


def test_deduplicate_lines():
    assert t.deduplicate_lines(["a", "a", "b", "b", "a"]) == ["a", "b", "a"]


def test_split_into_chunks():
    assert t.split_into_chunks("abcdefg", 3) == ["abc", "def", "g"]


def test_split_into_chunks_invalid():
    with pytest.raises(ValueError):
        t.split_into_chunks("abc", 0)


def test_split_into_chunks_empty():
    assert t.split_into_chunks("", 3) == []
