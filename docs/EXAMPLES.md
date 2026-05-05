# Examples

End-to-end snippets in five languages.

## curl

```bash
IMG=$(base64 -w0 face.jpg)
curl -s -X POST http://localhost:5000/analyze \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $DEEPFACE_KEY" \
    -d "{\"image\":\"$IMG\"}" | jq
```

## Python

```python
import base64, requests

with open("face.jpg", "rb") as fh:
    payload = {"image": base64.b64encode(fh.read()).decode()}

resp = requests.post(
    "http://localhost:5000/analyze",
    json=payload,
    headers={"X-API-Key": "..."},
    timeout=10,
)
resp.raise_for_status()
print(resp.json())
```

## JavaScript (browser)

```javascript
async function analyze(file) {
    const buffer = await file.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
    const resp = await fetch("/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json", "X-API-Key": KEY},
        body: JSON.stringify({image: b64}),
    });
    return resp.json();
}
```

## Node.js

```javascript
const fs = require("fs");
const fetch = require("node-fetch");

(async () => {
    const data = fs.readFileSync("face.jpg").toString("base64");
    const resp = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({image: data}),
    });
    console.log(await resp.json());
})();
```

## C# (Unity)

```csharp
using DeepFaceAge;
using DeepFaceAge.Models;

private IEnumerator AnalyzeFromCamera(WebCamTexture camera)
{
    var settings = Resources.Load<AgeEstimatorSettings>("AgeEstimatorSettings");
    var client = new AsyncAgeEstimatorClient(settings);
    var snapshot = new Texture2D(camera.width, camera.height);
    snapshot.SetPixels32(camera.GetPixels32());
    snapshot.Apply();
    yield return client.AnalyzeTexture(
        snapshot,
        result => Debug.Log($"age={result.age}, emotion={result.dominantEmotion}"),
        error => Debug.LogError(error)
    );
}
```

## Async batch via /jobs

```bash
JOB_ID=$(curl -s -X POST http://localhost:5000/jobs \
    -H "Content-Type: application/json" \
    -d "{\"kind\":\"analyze\",\"payload\":{\"image\":\"$IMG\"}}" | jq -r .id)

while :; do
    STATUS=$(curl -s "http://localhost:5000/jobs/$JOB_ID" | jq -r .status)
    [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]] && break
    sleep 1
done
curl -s "http://localhost:5000/jobs/$JOB_ID" | jq
```
