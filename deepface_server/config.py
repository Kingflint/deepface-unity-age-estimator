"""Application configuration loaded from environment variables.

We intentionally do not depend on pydantic here so the package stays
lightweight. The :class:`Settings` dataclass exposes typed accessors
and a :func:`load_settings` helper that reads from ``os.environ``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: Optional[str], default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings read from the environment."""

    # Service
    port: int = 5000
    log_level: str = "INFO"
    debug: bool = False

    # Image limits
    max_image_bytes: int = 5 * 1024 * 1024
    max_image_dimension: int = 4096

    # DeepFace
    deepface_actions: tuple = ("emotion", "age", "gender")
    enforce_detection: bool = False
    detector_backend: str = "opencv"

    # Cache
    enable_cache: bool = True
    cache_max_entries: int = 256

    # Auth & rate limit
    api_keys: frozenset = field(default_factory=frozenset)
    rate_limit_per_minute: int = 60

    # Batch
    max_batch_size: int = 8


def _parse_actions(raw: Optional[str]) -> tuple:
    if not raw:
        return ("emotion", "age", "gender")
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    valid = {"emotion", "age", "gender", "race"}
    actions = tuple(p for p in parts if p in valid)
    return actions or ("emotion", "age", "gender")


def _parse_api_keys(raw: Optional[str]) -> frozenset:
    if not raw:
        return frozenset()
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def load_settings(env: Optional[dict] = None) -> Settings:
    """Build a :class:`Settings` from a mapping (defaults to ``os.environ``)."""
    env = os.environ if env is None else env
    return Settings(
        port=_int(env.get("PORT"), 5000),
        log_level=env.get("LOG_LEVEL", "INFO").upper(),
        debug=_bool(env.get("FLASK_DEBUG"), False),
        max_image_bytes=_int(env.get("MAX_IMAGE_BYTES"), 5 * 1024 * 1024),
        max_image_dimension=_int(env.get("MAX_IMAGE_DIMENSION"), 4096),
        deepface_actions=_parse_actions(env.get("DEEPFACE_ACTIONS")),
        enforce_detection=_bool(env.get("ENFORCE_DETECTION"), False),
        detector_backend=env.get("DETECTOR_BACKEND", "opencv"),
        enable_cache=_bool(env.get("ENABLE_CACHE"), True),
        cache_max_entries=_int(env.get("CACHE_MAX_ENTRIES"), 256),
        api_keys=_parse_api_keys(env.get("API_KEYS")),
        rate_limit_per_minute=_int(env.get("RATE_LIMIT_PER_MINUTE"), 60),
        max_batch_size=_int(env.get("MAX_BATCH_SIZE"), 8),
    )
