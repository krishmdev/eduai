"""End-to-end check of bank-only mode with no network.

Starts the API server as a child process (EDUAI_BACKEND=bank, egress canary on), then drives a
practice session and an assessment over localhost HTTP, and checks that the canary inside the server
process could not reach any external host. Run it under the sandbox so the whole process tree is
covered:

    tools/offline-run make e2e-offline   # sandbox-exec profile denying outbound network
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def wait(url: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise SystemExit(f"server did not come up at {url}")


def run_session(base: str, subject: str, mode: str) -> dict:
    s = httpx.post(f"{base}/api/sessions", json={"subject": subject, "mode": mode, "length": 8}).json()
    sid = s["id"]
    n = 0
    while True:
        nxt = httpx.get(f"{base}/api/sessions/{sid}/next").json()
        if nxt["done"]:
            break
        choice = "ABCD"[n % 4]
        r = httpx.post(f"{base}/api/sessions/{sid}/answer", json={"choice": choice})
        assert r.status_code == 200, r.text
        n += 1
        assert n <= 60
    rep = httpx.get(f"{base}/api/sessions/{sid}/report").json()
    html = httpx.get(f"{base}/sessions/{sid}/report")
    assert html.status_code == 200 and "Simulated score" in html.text
    return {
        "mode": mode,
        "answered": rep["summary"]["answered"],
        "theta": round(rep["theta"], 3),
        "sd": round(rep["sd"], 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(
            os.environ,
            EDUAI_BACKEND="bank",
            EDUAI_EGRESS_CANARY="1",
            EDUAI_DB_PATH=f"{tmp}/e2e.db",
            EDUAI_DATA_DIR=f"{tmp}/nodata",
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "eduai.cli", "serve", "--port", str(args.port)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait(f"{base}/api/health")
            health = httpx.get(f"{base}/api/health").json()
            print("health:", {k: health[k] for k in ("backend", "items")})
            print("server egress canary:", health["egress_canary"])
            assert health["backend"] == "bank-only"
            assert health["egress_canary"] and health["egress_canary"]["blocked"], (
                "server process reached the internet"
            )
            assert httpx.get(f"{base}/").status_code == 200
            for mode in ("practice", "assessment"):
                print("session:", run_session(base, "BIO", mode))
        finally:
            proc.terminate()
            proc.wait(10)
    from eduai.egress import probe

    local = probe()
    print("e2e driver canary:", local)
    assert local["blocked"], "driver process reached the internet"
    print("e2e-offline: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
