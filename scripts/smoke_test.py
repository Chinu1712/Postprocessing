"""Smoke-test a running instance.

    python scripts/smoke_test.py                       # local, http://127.0.0.1:8000
    python scripts/smoke_test.py https://xxx.onrender.com
    python scripts/smoke_test.py https://xxx.onrender.com --api-key SECRET

Uses only the standard library so it can run anywhere, including on a box that
has none of the service's dependencies installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def call(url: str, payload: object | None, api_key: str | None, timeout: float = 60.0):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("X-API-Key", api_key)

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
            return response.status, body, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}"), (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    failures = 0

    status, body, ms = call(f"{base}/health", None, args.api_key)
    ok = status == 200 and body.get("status") == "ok"
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] GET /health -> {status} in {ms:.0f}ms  {body.get('version', '')}")

    for name, endpoint, checker in (
        ("segmentation", "/postprocess/segmentation", check_segmentation),
        ("pose", "/postprocess/pose", check_pose),
    ):
        payload = json.loads((EXAMPLES / f"{name}_request.json").read_text())
        status, body, ms = call(f"{base}{endpoint}", payload, args.api_key)
        ok = status == 200 and checker(body)
        failures += not ok
        summary = (
            f"{body.get('stats', {}).get('input_count', '?')} in -> {body.get('count', '?')} out"
            if status == 200
            else body.get("detail", body)
        )
        print(f"[{'PASS' if ok else 'FAIL'}] POST {endpoint} -> {status} in {ms:.0f}ms  {summary}")
        if not ok:
            print(json.dumps(body, indent=2)[:2000])

    print("\nall checks passed" if not failures else f"\n{failures} check(s) failed")
    return 1 if failures else 0


def check_segmentation(body: dict) -> bool:
    """Two of the four example instances should survive: one is a duplicate, one is low-score."""
    if body.get("count") != 2:
        return False
    labels = {instance["label"] for instance in body["instances"]}
    return labels == {"person", "forklift"} and all(i["mask"] for i in body["instances"])


def check_pose(body: dict) -> bool:
    """The second person is a duplicate of the first and should be suppressed by OKS."""
    if body.get("count") != 1:
        return False
    instance = body["instances"][0]
    return instance["num_visible_keypoints"] == 17 and "left_knee" in instance.get("angles", {})


if __name__ == "__main__":
    sys.exit(main())
