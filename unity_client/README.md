# Unity client

Two MonoBehaviours that talk to the DeepFace HTTP API:

- [`AgeEstimatorClient.cs`](AgeEstimatorClient.cs) — coroutine-based
  POST `/analyze` helper. Configure `baseUrl` and (optionally)
  `apiKey` in the inspector.
- [`AgeEstimatorExample.cs`](AgeEstimatorExample.cs) — sample
  consumer that grabs frames from a `WebCamTexture` and prints the
  result to the Unity console.

## Requirements

- Unity 2021.3 LTS or newer (uses `UnityWebRequest`).
- Camera permission on the target platform.

## Usage

1. Drop both scripts onto a GameObject.
2. Set the `Base URL` on `AgeEstimatorClient` to the DeepFace API
   endpoint, e.g. `http://localhost:5000`.
3. Press play. Webcam frames are sampled every `Request Every N Frames`
   updates and POSTed as base64 PNG.

For low-latency setups, lower `requestEveryNFrames` and keep
`ENABLE_CACHE=true` server-side; identical frames will short-circuit
through the LRU cache.
