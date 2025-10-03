"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    port: int = 5000
    log_level: str = "INFO"
    debug: bool = False


def load_settings() -> Settings:
    return Settings(
        port=int(os.environ.get("PORT", "5000")),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )