# Security

This document captures the threat model for the DeepFace Unity Age Estimator
and the controls in place.

## Threat model

| Threat | Mitigation |
|--------|-----------|
| Unauthenticated access to expensive endpoints | `X-API-Key` and `Authorization: Bearer` checks via [`auth.py`](../deepface_server/middleware/auth.py) |
| Rapid-fire abuse (single client) | Token-bucket rate limiter per `(IP, API key)` |
| Excessive payloads | `max_image_bytes` and `max_image_dimension` enforced before decode |
| Malformed or non-image payloads | Pre-decode magic-byte sniff, error envelope on decode failure |
| Replay of webhook deliveries | HMAC signature plus timestamp header on every outbound webhook |
| Token forgery | HMAC-SHA256 token signing in [`security/signing.py`](../deepface_server/security/signing.py) |
| Information leakage in errors | All known errors flow through `DeepFaceServerError.to_dict()` so internal stack traces never leak |
| Logging of secrets | Logger is configured to redact `Authorization` and `X-API-Key` headers |

## Authentication

Two mechanisms are supported, and they are checked in order:

1. `X-API-Key: <key>` against the configured `DEEPFACE_API_KEYS` set.
2. `Authorization: Bearer <key>` against the same set.

If `DEEPFACE_API_KEYS` is empty, authentication is disabled and the service
accepts anonymous requests.

## Webhook signatures

Outbound webhook bodies are signed as:

```
sha256 = HMAC_SHA256(secret, timestamp + "." + raw_body)
```

The signature is sent as `X-Signature-256: sha256=<hex>` and the timestamp as
`X-Signature-Timestamp`. Receivers must verify both. A constant-time
comparison helper is available at
[`webhooks/signing.py:verify_signature`](../deepface_server/webhooks/signing.py).

## Rate limiting

Defaults are tuned for shared-tenant deployments:

- 30 requests / 60 seconds per `(IP, API key)`
- Burst capacity of 10
- 429 response with `Retry-After` header

For dedicated deployments behind a reverse proxy you may want to set
`DEEPFACE_TRUST_FORWARDED=1` so the rate limiter keys on the original client
IP rather than the proxy.

## Reporting issues

Please email `Flintcheeze@yahoo.com` with the subject `[security]`.
