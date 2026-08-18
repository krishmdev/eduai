"""Text embedders and an on-disk vector cache tagged with the embedder's identity.

Vectors from different models are never mixed: every cache file and index generation is keyed by
`embedder_id = provider/model/revision/dim/preprocessing-hash`, and loading refuses a mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from eduai.config import MODEL_PINS, ROOT, model_path

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE_DIR = ROOT / ".cache" / "embeddings"


class Embedder(Protocol):
    embedder_id: str
    dim: int

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray: ...


def _prep_hash(desc: str) -> str:
    return hashlib.sha256(desc.encode()).hexdigest()[:8]


class EmbedderError(RuntimeError):
    pass


class SentenceTransformerEmbedder:
    """bge-small or MiniLM loaded from the pinned snapshot in .models/."""

    def __init__(
        self,
        key: str = "bge-small",
        device: str | None = None,
        batch_size: int = 64,
        query_prefix: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        pin = MODEL_PINS[key]
        path = model_path(key)
        if not path.exists():
            raise EmbedderError(f"{pin.repo} not found at {path}; run `make models`")
        self.key = key
        self.query_prefix = BGE_QUERY_PREFIX if key.startswith("bge") and query_prefix else ""
        self.device = device or _default_device()
        self.model = SentenceTransformer(str(path), device=self.device, local_files_only=True)
        self.dim = int(self.model.get_embedding_dimension())
        self.batch_size = batch_size
        prep = f"normalize=1;query_prefix={self.query_prefix!r}"
        self.embedder_id = f"st/{pin.repo}/{pin.revision[:12]}/{self.dim}/{_prep_hash(prep)}"

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        inputs = [self.query_prefix + t for t in texts] if query else list(texts)
        vecs = self.model.encode(
            inputs, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. Used in tests and as a no-model fallback."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.device = "cpu"
        self.embedder_id = f"hash/bow/v1/{dim}/{_prep_hash('lower;alnum;uni+bigram')}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokens(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


def _default_device() -> str:
    import torch

    if torch.backends.mps.is_available() and os.environ.get("EDUAI_EMBED_DEVICE") != "cpu":
        return "mps"
    return "cpu"


def text_hash(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def cached_encode(
    embedder: Embedder, texts: Sequence[str], *, query: bool = False, cache_dir: Path | None = None
) -> np.ndarray:
    if cache_dir is None:
        cache_dir = Path(os.environ.get("EDUAI_CACHE_DIR", CACHE_DIR))
    # embedder_id carries the model repo and revision; the device is keyed too, because MPS results
    # under GPU contention have differed from CPU ones.
    device = getattr(embedder, "device", "cpu")
    base = f"{embedder.embedder_id}|{int(query)}|{text_hash(texts)}"
    path = cache_dir / f"{hashlib.sha256(f'{base}|device={device}'.encode()).hexdigest()[:24]}.npy"
    # Entries written before the device was keyed all came from the default MPS device: move them over.
    legacy = cache_dir / f"{hashlib.sha256(base.encode()).hexdigest()[:24]}.npy"
    if device == "mps" and not path.exists() and legacy.exists():
        os.replace(legacy, path)
    if path.exists():
        vecs = np.load(path)
        if vecs.shape == (len(texts), embedder.dim):
            return vecs
    vecs = embedder.encode(texts, query=query)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, vecs)
    os.replace(tmp, path)
    return vecs


@dataclass
class IndexGeneration:
    embedder_id: str
    ids: list[str]
    vectors: np.ndarray


class VectorIndex:
    """Directory of index generations with an atomically swapped CURRENT pointer.

    A new generation is fully written to a temp dir, renamed into place, and only then does
    CURRENT move. If the embedder fails halfway, the old generation stays live and untouched.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def _gen_dir(self, embedder_id: str, content: str) -> Path:
        tag = hashlib.sha256(f"{embedder_id}|{content}".encode()).hexdigest()[:16]
        return self.root / f"gen-{tag}"

    def build(self, embedder: Embedder, ids: Sequence[str], texts: Sequence[str]) -> IndexGeneration:
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self.root.mkdir(parents=True, exist_ok=True)
        vectors = embedder.encode(texts)
        if vectors.shape != (len(texts), embedder.dim):
            raise EmbedderError(f"embedder returned shape {vectors.shape}")
        final = self._gen_dir(embedder.embedder_id, text_hash(texts))
        tmp = Path(tempfile.mkdtemp(dir=self.root, prefix=".building-"))
        try:
            np.save(tmp / "vectors.npy", vectors.astype(np.float32))
            (tmp / "meta.json").write_text(
                json.dumps({"embedder_id": embedder.embedder_id, "ids": list(ids)})
            )
            if final.exists():
                shutil.rmtree(final)
            os.replace(tmp, final)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        pointer = self.root / "CURRENT"
        ptmp = self.root / ".CURRENT.tmp"
        ptmp.write_text(final.name)
        os.replace(ptmp, pointer)
        return IndexGeneration(embedder.embedder_id, list(ids), vectors)

    def current(self) -> IndexGeneration | None:
        pointer = self.root / "CURRENT"
        if not pointer.exists():
            return None
        gen = self.root / pointer.read_text().strip()
        meta = json.loads((gen / "meta.json").read_text())
        return IndexGeneration(meta["embedder_id"], meta["ids"], np.load(gen / "vectors.npy"))

    def load_for(self, embedder: Embedder) -> IndexGeneration:
        gen = self.current()
        if gen is None:
            raise EmbedderError(f"no index at {self.root}")
        if gen.embedder_id != embedder.embedder_id:
            raise EmbedderError(
                f"index was built with {gen.embedder_id}, refusing to query with {embedder.embedder_id}"
            )
        return gen


def get_embedder(key: str = "bge-small", **kw) -> Embedder:
    if key == "hash":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(key, **kw)
