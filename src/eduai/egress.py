"""Egress canary: try to reach external hosts from inside this process.

Run as a startup check in the API process (EDUAI_EGRESS_CANARY=1) and as `python -m eduai.egress`.
Under offline-run every attempt must fail; unsandboxed, the same code should connect, which is
what keeps the check from passing vacuously.
"""

from __future__ import annotations

import json
import socket
import sys

TARGETS = (("1.1.1.1", 443), ("api.openai.com", 443), ("huggingface.co", 443))


def probe(timeout: float = 3.0) -> dict:
    results = {}
    for host, port in TARGETS:
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            results[f"{host}:{port}"] = "connected"
        except OSError as exc:
            results[f"{host}:{port}"] = f"blocked ({type(exc).__name__}: {exc.strerror or exc})"
    return {"blocked": all(v.startswith("blocked") for v in results.values()), "targets": results}


if __name__ == "__main__":
    res = probe()
    print(json.dumps(res))
    want = sys.argv[1] if len(sys.argv) > 1 else "blocked"
    sys.exit(0 if res["blocked"] == (want == "blocked") else 1)
