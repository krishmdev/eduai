"""Download pinned Hugging Face artifacts into .models/ and verify them against models.lock."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eduai.config import MODEL_PINS, ROOT, ModelPin, model_path, models_dir

LOCK_PATH = ROOT / "models.lock"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_snapshot(pin: ModelPin) -> dict[str, str]:
    base = model_path(pin.key)
    return {f: sha256_file(base / f) for f in pin.allow if (base / f).exists()}


def download(keys: list[str]) -> None:
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = str(models_dir())
    for key in keys:
        pin = MODEL_PINS[key]
        snapshot_download(
            pin.repo,
            revision=pin.revision,
            allow_patterns=list(pin.allow),
            cache_dir=str(models_dir() / "hub"),
        )


def read_lock() -> dict:
    if not LOCK_PATH.exists():
        return {}
    return json.loads(LOCK_PATH.read_text())


def write_lock(keys: list[str]) -> None:
    lock = read_lock()
    for key in keys:
        pin = MODEL_PINS[key]
        lock[key] = {"repo": pin.repo, "revision": pin.revision, "files": _hash_snapshot(pin)}
    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def verify(keys: list[str]) -> list[str]:
    lock = read_lock()
    problems = []
    for key in keys:
        pin = MODEL_PINS[key]
        entry = lock.get(key)
        if entry is None:
            problems.append(f"{key}: not in models.lock")
            continue
        if entry["revision"] != pin.revision:
            problems.append(f"{key}: lock revision {entry['revision']} != pinned {pin.revision}")
        base = model_path(key)
        for fname, digest in entry["files"].items():
            path = base / fname
            if not path.exists():
                problems.append(f"{key}: missing {fname}")
            elif sha256_file(path) != digest:
                problems.append(f"{key}: sha256 mismatch for {fname}")
    return problems


def available(key: str) -> bool:
    return model_path(key).exists() and not verify([key])
