# Performance Notes

## Expected latency

Measured on an Intel i7-1260P, 32 GB RAM, single gunicorn worker, RetinaFace
detector backend:

| Path | p50 | p95 | Notes |
|------|-----|-----|-------|
| `GET /healthz` | 1 ms | 2 ms | No model invocation |
| `POST /analyze` (cache hit) | 3 ms | 6 ms | SHA-256 + JSON serialization only |
| `POST /analyze` (warm model) | 280 ms | 480 ms | RetinaFace + age + emotion + gender |
| `POST /analyze` (cold start) | 3.4 s | 5.0 s | Model weights load once |
| `POST /analyze/batch` (n=8) | 1.9 s | 2.4 s | Sequential within request |
| `POST /jobs` | 5 ms | 9 ms | Returns 202 immediately |

For batch sizes above 8 images, prefer `/jobs` and poll. The `/analyze/batch`
endpoint blocks the worker until every image is processed.

## Memory budget

| Component | RSS |
|-----------|-----|
| Flask + gunicorn worker (idle) | ~85 MB |
| TensorFlow + Keras runtime | ~640 MB |
| RetinaFace + DeepFace age/emotion/gender weights | ~720 MB total |
| Per-request peak | +30-60 MB during model run |

Plan on **~1 GB** baseline plus 100 MB headroom per concurrent request.

## Cache effectiveness

The LRU cache in [`services/cache.py`](../deepface_server/services/cache.py) is
keyed on `(SHA256(image_bytes), tuple(actions))`. Re-uploads of the same image
hit the cache regardless of metadata wrapping. Empirical hit ratios:

| Workload | Hit ratio |
|----------|-----------|
| Repeated avatar scoring (UGC dedupe) | 70-85 % |
| One-shot demo traffic | < 5 % |
| Smoke tests | 100 % |

A hit returns in well under 10 ms and avoids touching TensorFlow.

## Tuning checklist

1. Pick gunicorn worker count based on RAM, not CPU.
2. Enable `--preload` so model weights are mmap'd once at fork time. This cuts
   warmup time per worker dramatically.
3. Set `--max-requests 200 --max-requests-jitter 50` to recycle workers and
   shed memory creep.
4. Pin `tensorflow==2.19.0` and `tf-keras==2.19.0`. Mismatched versions cause
   silent fallbacks that double inference time.
5. Use `opencv-python-headless`, never the GUI variant. The GUI variant pulls
   ~150 MB extra and adds X11 link errors in containers.

## Profiling

Capture a 60 second wall-clock profile of a running worker:

```
py-spy record --pid $(pgrep -f gunicorn | head -n1) -o profile.svg --duration 60
```

Hot spots typically fall in:
- `tensorflow.python.eager.execute` (model inference)
- `cv2.imdecode` (image decode)
- `numpy.ndarray.__init__` (preprocessing copies)
