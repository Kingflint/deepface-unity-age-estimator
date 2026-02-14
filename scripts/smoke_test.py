"""Tiny smoke test against a running DeepFace server."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request


def _hit(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the DeepFace API.")
    parser.add_argument("--base", default="http://localhost:5000")
    parser.add_argument("--image", default=None, help="Path to JPEG/PNG to analyze.")
    args = parser.parse_args()

    status, body = _hit(f"{args.base}/healthz")
    print("healthz", status, body.get("status"))
    if status != 200:
        return 1

    if args.image:
        with open(args.image, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        status, body = _post(f"{args.base}/analyze", {"image": encoded})
        print("analyze", status)
        print(json.dumps(body, indent=2)[:500])
        return 0 if status == 200 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
