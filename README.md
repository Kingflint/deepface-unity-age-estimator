# DeepFace Unity Age Estimator

Flask backend that accepts a base64 encoded image and returns
emotion, age and gender predictions via [DeepFace](https://github.com/serengil/deepface).
Originally built for a Unity client, but the API is language-agnostic.

## Highlights

- Modular Flask app via `create_app()` (`deepface_server` package)
- Configuration through environment variables ([deepface_server/config.py](deepface_server/config.py))
- Optional API-key auth + token-bucket rate limiting
- LRU cache keyed on image SHA-256
- Single-image (`/analyze`) and batch (`/analyze/batch`) endpoints
- `/healthz` and `/metrics` for observability
- Dockerfile, gunicorn config, and Render deployment manifest
- Unity C# client example under [unity_client/](unity_client/)
- pytest suite covering schemas, services, middleware and routes

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The service listens on `http://0.0.0.0:5000` by default. See
[docs/API.md](docs/API.md) for the full HTTP contract and
[ARCHITECTURE.md](ARCHITECTURE.md) for an overview of the package
layout.

## Endpoints

| Method | Path             | Description                                   |
|--------|------------------|-----------------------------------------------|
| GET    | `/`              | Liveness check                                |
| GET    | `/healthz`       | Detailed health + service config              |
| POST   | `/analyze`       | Analyze a single base64-encoded image         |
| POST   | `/analyze/batch` | Analyze up to `MAX_BATCH_SIZE` images         |
| GET    | `/metrics`       | In-process request and cache metrics          |

## Configuration

All settings are read from environment variables. The most useful ones:

| Variable                | Default      | Notes                                |
|-------------------------|--------------|--------------------------------------|
| `PORT`                  | `5000`       | HTTP listen port                     |
| `LOG_LEVEL`             | `INFO`       | Standard logging level               |
| `MAX_IMAGE_BYTES`       | `5242880`    | 5 MiB                                |
| `MAX_IMAGE_DIMENSION`   | `4096`       | Largest accepted side in pixels      |
| `DEEPFACE_ACTIONS`      | `emotion,age,gender` | Subset of DeepFace actions    |
| `ENFORCE_DETECTION`     | `false`      | Force face detection failure on miss |
| `DETECTOR_BACKEND`      | `opencv`     | DeepFace backend                     |
| `ENABLE_CACHE`          | `true`       | Toggle the LRU cache                 |
| `CACHE_MAX_ENTRIES`     | `256`        | LRU capacity                         |
| `RATE_LIMIT_PER_MINUTE` | `60`         | Per (IP, API key) bucket             |
| `API_KEYS`              | _(empty)_    | CSV of allowed `X-API-Key` values    |
| `MAX_BATCH_SIZE`        | `8`          | Max images per `/analyze/batch`      |

When `API_KEYS` is empty (the default), authentication is disabled.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

DeepFace is heavy and pulls model weights at import time. The test
suite stubs it out via [tests/conftest.py](tests/conftest.py), so
tests run in seconds in any clean environment.

## Docker

```bash
docker build -t deepface-age-server .
docker run --rm -p 5000:5000 deepface-age-server
```

## License

[MIT](LICENSE).
