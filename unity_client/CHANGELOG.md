# Unity Client Changelog

## 0.5.0 - 2026-04-30
- Added `Models/AnalysisResult.cs`, `Models/BatchResult.cs`, `Models/JobStatus.cs`.
- Added `AgeEstimatorSettings` ScriptableObject for centralized configuration.
- Added `AsyncAgeEstimatorClient` with retry/backoff and job polling helpers.

## 0.4.0 - 2026-02-04
- Expanded `AgeEstimatorClient` with API key, timeout and structured response.

## 0.3.0 - 2026-02-02
- Initial Unity client (single-image POST, MonoBehaviour wrapper).
