from __future__ import annotations

import pytest

from deepface_server.security import api_keys as ak


def test_generate_returns_parts():
    key = ak.generate_api_key()
    assert key.environment == "live"
    assert len(key.secret) >= 16


def test_str_round_trip():
    key = ak.generate_api_key("test")
    parsed = ak.parse_api_key(str(key))
    assert parsed == key


def test_parse_invalid_format():
    with pytest.raises(ak.APIKeyError):
        ak.parse_api_key("only_two")


def test_parse_invalid_env():
    with pytest.raises(ak.APIKeyError):
        ak.parse_api_key("df_prod_abcdefghijklmnop")


def test_parse_invalid_secret_length():
    with pytest.raises(ak.APIKeyError):
        ak.parse_api_key("df_live_short")


def test_parse_empty():
    with pytest.raises(ak.APIKeyError):
        ak.parse_api_key("")


def test_hash_and_verify():
    key = str(ak.generate_api_key())
    stored = ak.hash_api_key(key)
    assert ak.verify_api_key(key, stored)
    assert not ak.verify_api_key("df_live_otherother", stored)


def test_hash_with_explicit_salt_deterministic():
    salt = b"\x00" * 16
    h1 = ak.hash_api_key("df_test_aaaaaaaaaaaaaaaa", salt=salt)
    h2 = ak.hash_api_key("df_test_aaaaaaaaaaaaaaaa", salt=salt)
    assert h1 == h2


def test_verify_malformed_stored():
    assert not ak.verify_api_key("any", "garbage")


def test_invalid_environment_rejected():
    with pytest.raises(ak.APIKeyError):
        ak.generate_api_key("staging")


def test_public_id_is_short():
    key = ak.generate_api_key("dev")
    assert key.public_id.startswith("df_dev_")
    assert len(key.public_id) < len(str(key))


def test_is_test_key():
    assert ak.is_test_key(ak.generate_api_key("test"))
    assert ak.is_test_key(ak.generate_api_key("dev"))
    assert not ak.is_test_key(ak.generate_api_key("live"))
