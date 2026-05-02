"""Time-bound token issuer."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .signing import SignatureMismatch, sign_token, verify_token


@dataclass
class TokenPayload:
    sub: str
    issued_at: int
    expires_at: int
    scope: str = ""
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sub": self.sub,
            "iat": self.issued_at,
            "exp": self.expires_at,
            "scope": self.scope,
        }
        if self.extra:
            d.update(self.extra)
        return d


class TokenIssuer:
    def __init__(self, secret: str, default_ttl_seconds: int = 3600):
        if not secret:
            raise ValueError("secret is required")
        self.secret = secret
        self.default_ttl_seconds = default_ttl_seconds

    def issue(
        self,
        subject: str,
        scope: str = "",
        ttl_seconds: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        now = int(time.time())
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        payload = TokenPayload(
            sub=subject,
            issued_at=now,
            expires_at=now + ttl,
            scope=scope,
            extra=extra,
        ).to_dict()
        return sign_token(payload, self.secret)

    def verify(self, token: str) -> TokenPayload:
        data = verify_token(token, self.secret)
        try:
            payload = TokenPayload(
                sub=str(data["sub"]),
                issued_at=int(data["iat"]),
                expires_at=int(data["exp"]),
                scope=str(data.get("scope", "")),
                extra={k: v for k, v in data.items() if k not in {"sub", "iat", "exp", "scope"}},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SignatureMismatch("token missing required fields") from exc
        if payload.expires_at < int(time.time()):
            raise SignatureMismatch("token expired")
        return payload
