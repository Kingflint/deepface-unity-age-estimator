"""Custom exceptions used across the service."""
from __future__ import annotations


class DeepFaceServerError(Exception):
    status_code = 500
    error_code = "internal_error"

    def to_dict(self) -> dict:
        return {"error": str(self), "code": self.error_code}


class BadRequest(DeepFaceServerError):
    status_code = 400
    error_code = "bad_request"