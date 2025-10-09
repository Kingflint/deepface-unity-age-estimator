"""Custom exceptions used across the service."""
from __future__ import annotations


class DeepFaceServerError(Exception):
    """Base error for all server-side problems we recognise."""

    status_code = 500
    error_code = "internal_error"

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict:
        return {"error": str(self), "code": self.error_code}


class BadRequest(DeepFaceServerError):
    status_code = 400
    error_code = "bad_request"


class ImageTooLarge(BadRequest):
    error_code = "image_too_large"


class ImageDecodeError(BadRequest):
    error_code = "image_decode_error"


class Unauthorized(DeepFaceServerError):
    status_code = 401
    error_code = "unauthorized"


class RateLimited(DeepFaceServerError):
    status_code = 429
    error_code = "rate_limited"


class AnalyzerError(DeepFaceServerError):
    error_code = "analyzer_error"
