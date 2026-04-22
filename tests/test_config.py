from deepface_server.config import Settings, load_settings


def test_load_settings_uses_defaults_when_env_empty():
    s = load_settings(env={})
    assert s.port == 5000
    assert s.log_level == "INFO"
    assert s.deepface_actions == ("emotion", "age", "gender")
    assert s.enable_cache is True
    assert s.api_keys == frozenset()


def test_load_settings_parses_csv_actions():
    s = load_settings(env={"DEEPFACE_ACTIONS": "age, gender , bogus"})
    assert s.deepface_actions == ("age", "gender")


def test_load_settings_parses_api_keys_csv():
    s = load_settings(env={"API_KEYS": "alpha, beta ,, gamma"})
    assert s.api_keys == frozenset({"alpha", "beta", "gamma"})


def test_load_settings_parses_bools_and_ints():
    s = load_settings(
        env={
            "ENABLE_CACHE": "false",
            "ENFORCE_DETECTION": "1",
            "RATE_LIMIT_PER_MINUTE": "120",
            "MAX_IMAGE_BYTES": "not-an-int",  # should fall back to default
        }
    )
    assert s.enable_cache is False
    assert s.enforce_detection is True
    assert s.rate_limit_per_minute == 120
    assert s.max_image_bytes == 5 * 1024 * 1024


def test_settings_is_immutable():
    s = Settings()
    try:
        s.port = 1234  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Settings should be frozen")
