from __future__ import annotations

import pytest

from deepface_server.security import password as pw


def test_hash_and_verify():
    encoded = pw.hash_password("Sup3rStrong!Pass")
    assert pw.verify_password("Sup3rStrong!Pass", encoded)
    assert not pw.verify_password("wrong", encoded)


def test_hash_uses_unique_salt():
    a = pw.hash_password("samesame12!", iterations=10_000)
    b = pw.hash_password("samesame12!", iterations=10_000)
    assert a != b


def test_hash_empty_rejected():
    with pytest.raises(pw.PasswordPolicyError):
        pw.hash_password("")


def test_verify_malformed_returns_false():
    assert not pw.verify_password("x", "garbage")


def test_needs_rehash_old_iterations():
    encoded = pw.hash_password("Strong!Pass1", iterations=10_000)
    assert pw.needs_rehash(encoded, iterations=240_000)


def test_needs_rehash_current():
    encoded = pw.hash_password("Strong!Pass1", iterations=240_000)
    assert not pw.needs_rehash(encoded, iterations=240_000)


def test_needs_rehash_invalid_returns_true():
    assert pw.needs_rehash("garbage")


def test_enforce_policy_short():
    with pytest.raises(pw.PasswordPolicyError):
        pw.enforce_policy("short1!A")


def test_enforce_policy_no_digit():
    with pytest.raises(pw.PasswordPolicyError):
        pw.enforce_policy("AbcdefghIJK!@#")


def test_enforce_policy_no_symbol():
    with pytest.raises(pw.PasswordPolicyError):
        pw.enforce_policy("Abcdefgh1234")


def test_enforce_policy_no_mixed_case():
    with pytest.raises(pw.PasswordPolicyError):
        pw.enforce_policy("abcdefgh1234!")


def test_enforce_policy_passes():
    pw.enforce_policy("AbcdEfgh1234!")


def test_generate_strong_password_length():
    p = pw.generate_strong_password(20)
    assert len(p) == 20
    pw.enforce_policy(p)


def test_generate_password_too_short():
    with pytest.raises(ValueError):
        pw.generate_strong_password(8)
