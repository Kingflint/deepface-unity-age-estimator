# Operational Runbook

This document covers day-2 operations for the DeepFace Unity Age Estimator.

## Liveness and readiness

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Lightweight heartbeat. Returns `{"message": "..."}`. |
| `GET /healthz` | Detailed health: limits, deepface config, version. |
| `GET /metrics` | In-process counters (cache hits, request latency). |
| `GET /admin/cache` | Cache statistics. |
| `DELETE /admin/cache` | Empty the cache. |

Configure your load balancer to use `/healthz` for readiness checks. Anything
non-2xx should be treated as not-ready.

## Scaling

The worker is a Flask app served by gunicorn. Recommended gunicorn worker
counts:

| Memory | Workers |
|--------|---------|
| 512 MB | 1 |
| 1 GB   | 1-2 |
| 2 GB   | 2-3 |
| 4 GB   | 3-4 |

Each worker holds a TensorFlow runtime, so a single worker easily uses 700 MB
on warmup. Don't over-provision workers for the available memory budget; OOM
kills cause cold starts that exceed any benefit from extra parallelism.

## Caching

The LRU cache key is `(fingerprint, actions)`. Image fingerprints are SHA-256
of the decoded body. Cache hits skip both the model run and the deepface
adapter overhead, returning in under 5 ms. Hit ratio over 70 % is healthy
for repeated UGC scoring; hit ratio below 30 % indicates highly unique images
where the cache adds memory pressure for little benefit. Disable via
`DEEPFACE_ENABLE_CACHE=0`.

## Rate limiting

Token-bucket rate limiter sits in
[`deepface_server/middleware/rate_limit.py`](../deepface_server/middleware/rate_limit.py).
Limits are per `(client_ip, api_key)` pair. Default: 30 requests per 60 seconds
with a burst of 10. Override via `DEEPFACE_RATE_LIMIT_PER_MINUTE` and
`DEEPFACE_RATE_LIMIT_BURST`.

## Background jobs

Submit long-running analysis via `POST /jobs`. Jobs are stored in-memory by
default. For durable storage, point `DEEPFACE_DATABASE_URL` at a SQLite file
and run:

```
python -m deepface_server.cli migrate --database /var/lib/deepface/state.db
```

This applies migrations 1, 2 and 3 (analysis, batch, jobs). The CLI is
idempotent; rerun after every deploy.

## Webhooks

To deliver completion events:

```
DEEPFACE_WEBHOOK_URLS=https://example.com/hook,https://backup.example.com/hook
DEEPFACE_WEBHOOK_SECRET=long-random-string
```

Receivers must verify `X-Signature-256` against the raw body using the same
secret. See [`deepface_server/webhooks/signing.py`](../deepface_server/webhooks/signing.py).

## Common runbook actions

- **Cache stampede after deploy.** Immediately after restart, all requests miss
  the cache and saturate the workers. Mitigate with a warmup job or staggered
  rolling deploys.
- **Memory creep.** TensorFlow occasionally retains intermediate tensors. Set
  `gunicorn --max-requests 200 --max-requests-jitter 50` to recycle workers.
- **Slow first request.** First call after import loads model weights. The
  `/healthz` endpoint lazily warms the analyzer; hit it once after deploy.
- **Run the smoke test.** `python -m deepface_server.cli smoke` exercises the
  health endpoints from inside the running process.
