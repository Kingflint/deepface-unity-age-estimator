# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.5.0] - 2026-04-22
### Added
- Storage layer with SQLite-backed analysis, batch and job tables.
- Pluggable analyzer registry with deepface, deterministic mock and ensemble backends.
- Image preprocessing pipeline (`Resize`, `Normalize`, `EXIFRotate`, `GrayscaleConvert`, `Clahe`, `FaceCropPlaceholder`).
- Background `JobManager` with worker thread, in-memory queue and cancel support.
- Outbound webhook dispatcher with HMAC-SHA256 signing and exponential backoff.
- Token-based security helpers (`TokenIssuer`, `sign_token`, `verify_token`).
- Admin CLI: `version`, `config`, `migrate`, `openapi`, `smoke`.
- Static OpenAPI 3.0.3 generator served at `/openapi.json`.
- New blueprints: `/jobs`, `/history`, `/admin/cache`, `/admin/analyzers`, `/openapi.json`.
- Unity client expansion: `AsyncAgeEstimatorClient`, `AgeEstimatorSettings` ScriptableObject and typed result models.
- Documentation: runbook, security, performance and language examples.
- ~50 additional pytest cases covering every new module.

## [0.4.0] - 2026-03-27
### Changed
- Bumped package version metadata.

## [0.3.0] - 2026-03-03
### Changed
- Bumped package version metadata.

## [0.2.0] - 2025-10-27
### Added
- Application factory, request id and timing middleware.
- Token bucket rate limiter, optional API key authentication.
- LRU cache, MetricsCollector, batch and metrics endpoints.

## [0.1.0] - 2025-10-01
### Added
- `deepface_server` package and Settings dataclass.
- Health blueprint and base error hierarchy.

## [0.0.1] - 2025-09-10
### Added
- Initial Flask scaffold.
