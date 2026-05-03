"""Static OpenAPI specification generator.

The spec is hand-curated rather than reflectively generated, so we have full
control over examples, response shapes, and error envelopes regardless of the
order in which blueprints are registered.
"""
from __future__ import annotations

from typing import Any


def _error_envelope() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["error", "code"],
        "properties": {
            "error": {"type": "string"},
            "code": {"type": "string"},
            "request_id": {"type": "string"},
            "details": {"type": "object", "additionalProperties": True},
        },
    }


def _analysis_outcome_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "age": {"type": "number", "nullable": True},
            "age_confidence": {"type": "number", "nullable": True},
            "dominant_emotion": {"type": "string", "nullable": True},
            "emotion_scores": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
            "dominant_gender": {"type": "string", "nullable": True},
            "gender_scores": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
            "region": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "w": {"type": "integer"},
                    "h": {"type": "integer"},
                },
            },
            "backend": {"type": "string"},
        },
    }


def generate_openapi_spec(version: str = "0.5.0") -> dict[str, Any]:
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "DeepFace Unity Age Estimator",
            "version": version,
            "description": (
                "REST API for image age, emotion and gender estimation, "
                "designed for a Unity client but usable from any HTTP-capable runtime."
            ),
            "license": {"name": "MIT"},
        },
        "servers": [
            {"url": "http://localhost:5000", "description": "Local development"},
        ],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
                "BearerAuth": {"type": "http", "scheme": "bearer"},
            },
            "schemas": {
                "ErrorEnvelope": _error_envelope(),
                "AnalysisOutcome": _analysis_outcome_schema(),
                "AnalyzeRequest": {
                    "type": "object",
                    "required": ["image"],
                    "properties": {
                        "image": {"type": "string", "description": "base64 encoded image"},
                        "actions": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["age", "emotion", "gender"]},
                        },
                    },
                },
                "BatchRequest": {
                    "type": "object",
                    "required": ["images"],
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "list of base64 encoded images",
                        }
                    },
                },
                "JobSubmission": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "running", "succeeded", "failed", "cancelled"],
                        },
                    },
                },
            },
        },
        "paths": {
            "/": {
                "get": {
                    "summary": "Service heartbeat",
                    "responses": {"200": {"description": "Service is up"}},
                }
            },
            "/healthz": {
                "get": {
                    "summary": "Detailed health check",
                    "responses": {"200": {"description": "Health summary"}},
                }
            },
            "/analyze": {
                "post": {
                    "summary": "Analyze a single image",
                    "security": [{"ApiKeyAuth": []}, {"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AnalyzeRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Analysis result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AnalysisOutcome"
                                    }
                                }
                            },
                        },
                        "400": {"description": "Invalid request"},
                        "401": {"description": "Authentication required"},
                        "429": {"description": "Rate limited"},
                    },
                }
            },
            "/analyze/batch": {
                "post": {
                    "summary": "Analyze a batch of images",
                    "security": [{"ApiKeyAuth": []}, {"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/BatchRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Per-item results"},
                        "400": {"description": "Invalid request"},
                    },
                }
            },
            "/jobs": {
                "post": {
                    "summary": "Submit an asynchronous analysis job",
                    "responses": {
                        "202": {
                            "description": "Job accepted",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/JobSubmission"}
                                }
                            },
                        }
                    },
                }
            },
            "/jobs/{job_id}": {
                "get": {
                    "summary": "Inspect a job",
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {"description": "Job state"},
                        "404": {"description": "Job not found"},
                    },
                }
            },
            "/history": {
                "get": {
                    "summary": "Recent analysis history",
                    "responses": {"200": {"description": "Most recent records"}},
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Lightweight Prometheus-style metrics",
                    "responses": {"200": {"description": "Metric counters"}},
                }
            },
            "/openapi.json": {
                "get": {
                    "summary": "This document",
                    "responses": {"200": {"description": "OpenAPI document"}},
                }
            },
        },
    }
