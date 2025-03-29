"""Run manifests: host/workload state from the shared run_manifest tool plus repo state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from eduai.config import ROOT

SHARED_TOOL = Path(os.environ["RUN_MANIFEST_TOOL"]) if os.environ.get("RUN_MANIFEST_TOOL") else None


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_manifest(extra: dict) -> dict:
    data: dict = {}
    if SHARED_TOOL is not None and SHARED_TOOL.exists():
        args = [sys.executable, str(SHARED_TOOL), *(f"{k}={v}" for k, v in extra.items())]
        out = subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
        try:
            data = json.loads(out)
        except ValueError:
            data = {}
    else:
        import platform
        import time

        data = {
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": {"platform": platform.platform()},
        }
        data["extra"] = extra
    data["repo"] = {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }
    try:
        from importlib.metadata import version

        data["packages"] = {
            p: version(p) for p in ("numpy", "sentence-transformers", "torch", "mlx", "mlx-lm") if _has(p)
        }
    except Exception:  # noqa: BLE001 - manifest is best effort
        pass
    return data


def _has(pkg: str) -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        version(pkg)
        return True
    except PackageNotFoundError:
        return False


def write_manifest(path: Path, extra: dict) -> dict:
    data = build_manifest(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data
