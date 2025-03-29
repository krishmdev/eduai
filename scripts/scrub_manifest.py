"""Drop host process/container telemetry and local tool paths from run manifests.

Keeps timing, device, package and repo fields. Usage: scrub_manifest.py FILE [FILE ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DROP = {"top_cpu", "top_mem", "docker_ps"}
LOCAL = re.compile(r"/private/tmp/|/\.tools/|claude", re.I)


def scrub(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in DROP:
                continue
            if k == "compute_lease" and isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk in ("held", "holder", "started", "holder_env")}
            v = scrub(v)
            if v is not None:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [x for x in (scrub(v) for v in obj) if x is not None]
    if isinstance(obj, str) and LOCAL.search(obj):
        return None
    return obj


if __name__ == "__main__":
    for f in sys.argv[1:]:
        p = Path(f)
        p.write_text(json.dumps(scrub(json.loads(p.read_text())), indent=2) + "\n")
