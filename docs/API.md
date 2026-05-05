# API

All endpoints accept and return JSON unless noted otherwise.

## `GET /`

Liveness check.

json
{ "message": "DeepFace API is running!" }


## `GET /healthz`

json
{
  "status": "ok",
  "version": "0.5.0",
  "deepface": {
    "actions": ["emotion", "age", "gender"],
    "enforce_detection": false,
    "detector_backend": "opencv"
  },
  "limits": {
    "max_image_bytes": 5242880,
    "max_image_dimension": 4096,
    "max_batch_size": 8
  }
}


## `POST /analyze`

Body:

json
{ "image": "<base64-encoded jpg/png/webp>" }


`data:image/...;base64,` prefixes are accepted and stripped client-side
or by the helper in [deepface_server/utils/base64_utils.py](../deepface_server/utils/base64_utils.py).

200 response:

json
{
  "result": [
    { "age": 27, "dominant_gender": "Woman", "dominant_emotion": "happy" }
  ],
  "cached": false
}

Error responses follow the same shape:

json
{ "error": "No image data provided", "code": "bad_request" }


| Code                  | HTTP | When                                                      |
|-----------------------|------|-----------------------------------------------------------|
| `bad_request`         | 400  | Body shape invalid or `image` missing                     |
| `image_too_large`     | 400  | Decoded image exceeds `MAX_IMAGE_BYTES` / dimension limit |
| `image_decode_error`  | 400  | Could not base64-decode or OpenCV failed                  |
| `unauthorized`        | 401  | API keys configured and `X-API-Key` missing/wrong         |
| `rate_limited`        | 429  | Token bucket exhausted                                    |
| `analyzer_error`      | 500  | DeepFace raised                                           |
| `internal_error`      | 500  | Catch-all                                                 |

## `POST /analyze/batch`

Body:

json
{ "images": ["<b64>", "<b64>", "<b64>"] }

The array length must be in `[1, MAX_BATCH_SIZE]`.

200 response:

json
{
  "results": [
    { "index": 0, "ok": true,  "cached": false, "result": [...] },
    { "index": 1, "ok": false, "error": "image is too large: 6291456 bytes (max 5242880)", "code": "image_too_large" }
  ],
  "succeeded": 1,
  "total": 2
}


A failing image inside the batch does **not** fail the whole call.

## `GET /metrics`

json
{
  "requests": {
    "total_requests": 42,
    "counts_by_path_status": { "/analyze 200": 40, "/analyze 400": 2 },
    "avg_ms_by_path": { "/analyze": 312.4 }
  },
  "cache": {
    "size": 17, "max_entries": 256, "hits": 8, "misses": 34, "hit_rate": 0.1905
  }
}


## Headers

- `X-Request-ID` — echoed back on every response (auto-generated when absent).
- `X-Response-Time-ms` — server-side wall time.
- `X-API-Key` — required when `API_KEYS` is configured.
