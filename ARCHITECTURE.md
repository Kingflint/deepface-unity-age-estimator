# Architecture

```
deepface_server/
├── __init__.py          # exposes create_app and __version__
├── app_factory.py       # builds the Flask app, wires services and blueprints
├── config.py            # Settings dataclass + load_settings() reader
├── errors.py            # DeepFaceServerError hierarchy with status codes
├── logging_config.py    # configure_logging() + get_logger()
├── schemas.py           # request payload validation
├── middleware/
│   ├── __init__.py      # register_middleware()
│   ├── auth.py          # optional X-API-Key / Bearer token auth
│   ├── rate_limit.py    # in-memory token bucket per (IP, key)
│   ├── request_id.py    # propagates / generates X-Request-ID
│   └── timing.py        # records elapsed ms into MetricsCollector
├── services/
│   ├── cache.py         # thread-safe LRU cache
│   ├── deepface_service.py  # adapter around DeepFace.analyze
│   ├── image_service.py # base64 → ndarray, with size & dimension limits
│   └── metrics.py       # in-process request/cache metrics
├── blueprints/
│   ├── health.py        # GET /, GET /healthz
│   ├── analyze.py       # POST /analyze
│   ├── batch.py         # POST /analyze/batch
│   └── metrics.py       # GET /metrics
└── utils/
    ├── base64_utils.py  # data-URI stripping, sanity checks
    └── image_utils.py   # magic-byte format detection
```

## Request lifecycle

1. `request_id` middleware assigns or echoes `X-Request-ID`.
2. `auth` middleware enforces `X-API-Key` / `Authorization: Bearer …`
   when keys are configured, allow-listing `/`, `/healthz`, `/metrics`.
3. `rate_limit` middleware decrements a token bucket keyed on
   `(client IP, API key)`.
4. The route handler validates the JSON body via `schemas`.
5. `ImageService` decodes the base64, hashes it (SHA-256) and decodes
   bytes into an OpenCV ndarray.
6. `LRUCache` short-circuits if the hash was seen recently.
7. `DeepFaceService` calls `DeepFace.analyze` with configured actions.
8. `timing` middleware records elapsed time into `MetricsCollector`
   and adds `X-Response-Time-ms` to the response.

## Why a factory?

`create_app()` lets tests instantiate isolated apps with bespoke
`Settings` (for example, no cache, tiny image limits, or a fake
`DeepFaceService`) without monkeypatching module globals.

## Why no Pydantic / Marshmallow?

The schema surface is small. Hand-rolled validators in
[deepface_server/schemas.py](deepface_server/schemas.py) keep the
import graph trivially fast and the wheel small.
