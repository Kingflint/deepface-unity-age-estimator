"""Test fixtures: stub heavy deps and provide a fresh app per test."""
from __future__ import annotations

import sys
import types

import pytest


def _install_stubs() -> None:
    if "deepface" not in sys.modules:
        stub = types.ModuleType("deepface")

        class _DeepFace:
            calls = []

            @classmethod
            def analyze(cls, *args, **kwargs):
                cls.calls.append(kwargs)
                return [{"age": 30, "dominant_gender": "Man", "dominant_emotion": "neutral"}]

        stub.DeepFace = _DeepFace
        sys.modules["deepface"] = stub


_install_stubs()


from deepface_server import create_app  # noqa: E402
from deepface_server.config import Settings  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings(
        port=5000,
        log_level="WARNING",
        debug=False,
        max_image_bytes=64 * 1024,
        max_image_dimension=2048,
        deepface_actions=("emotion", "age", "gender"),
        enforce_detection=False,
        detector_backend="opencv",
        enable_cache=True,
        cache_max_entries=8,
        api_keys=frozenset(),
        rate_limit_per_minute=0,  # disabled by default for tests
        max_batch_size=4,
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    return app.test_client()
