from __future__ import annotations

import pytest

from deepface_server.utils import hashing as h


def test_sha256_hex_matches_known_value():
    # echo -n "hello" | sha256sum
    assert h.sha256_hex(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_short_hash_length():
    digest = h.short_hash(b"hello", length=12)
    assert len(digest) == 12


def test_short_hash_invalid_length():
    with pytest.raises(ValueError):
        h.short_hash(b"x", length=2)


def test_stable_dict_hash_order_independent():
    a = h.stable_dict_hash({"a": 1, "b": 2})
    b = h.stable_dict_hash({"b": 2, "a": 1})
    assert a == b


def test_stable_dict_hash_different_values():
    a = h.stable_dict_hash({"a": 1})
    b = h.stable_dict_hash({"a": 2})
    assert a != b


def test_stable_dict_hash_nested():
    a = h.stable_dict_hash({"a": {"x": 1, "y": 2}})
    b = h.stable_dict_hash({"a": {"y": 2, "x": 1}})
    assert a == b


def test_hmac_sign_verify():
    sig = h.hmac_sign("k", b"data", length=16)
    assert h.hmac_verify("k", b"data", sig)
    assert not h.hmac_verify("k", b"other", sig)


def test_hmac_verify_wrong_key():
    sig = h.hmac_sign("k", b"data")
    assert not h.hmac_verify("other", b"data", sig)


def test_hmac_sign_short_length_rejected():
    with pytest.raises(ValueError):
        h.hmac_sign("k", b"x", length=4)


def test_chained_hash_matches_concat():
    a = h.chained_hash([b"hello", b"world"])
    b = h.sha256_hex(b"helloworld")
    assert a == b
