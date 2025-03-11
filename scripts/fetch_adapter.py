"""Download the released LoRA adapter, verify it against adapters/MANIFEST.json, and unpack it."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(manifest_path: Path = ROOT / "adapters" / "MANIFEST.json", source: str | None = None) -> Path:
    m = json.loads(manifest_path.read_text())
    asset = m["asset"]
    dest = ROOT / m["install_to"]
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / asset["filename"]
        if source and Path(source).exists():
            shutil.copy(source, tar_path)
        else:
            with httpx.stream("GET", source or asset["url"], follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                with open(tar_path, "wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
        if sha256(tar_path) != asset["sha256"]:
            raise SystemExit(f"sha256 mismatch for {asset['filename']}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp, filter="data")
        unpacked = Path(tmp) / asset["filename"].removesuffix(".tar.gz")
        for name, info in m["files"].items():
            if sha256(unpacked / name) != info["sha256"]:
                raise SystemExit(f"sha256 mismatch for {name}")
        dest.mkdir(parents=True, exist_ok=True)
        for name in m["files"]:
            shutil.copy(unpacked / name, dest / name)
    return dest


if __name__ == "__main__":
    print(fetch(source=sys.argv[1] if len(sys.argv) > 1 else None))
