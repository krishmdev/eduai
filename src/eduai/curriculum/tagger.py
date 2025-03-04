"""Tag an item with its most likely learning objective.

Stages: subject gate on subject centroids, bi-encoder retrieval of the top-k objectives inside the
gated subjects, optional cross-encoder rerank, then an `aligned` flag when the final score clears
tau (calibrated on the gold set as the 20th percentile of gold-LO scores).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eduai.config import ROOT, model_path
from eduai.curriculum.embedder import Embedder, cached_encode
from eduai.curriculum.taxonomy import Taxonomy

CALIBRATION_PATH = ROOT / "configs" / "tagger_calibration.json"


@dataclass
class TagResult:
    lo_id: str
    subject: str
    score: float
    top3: list[str]
    aligned: bool
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lo_id": self.lo_id,
            "subject": self.subject,
            "score": round(float(self.score), 4),
            "top3": self.top3,
            "aligned": self.aligned,
        }


class Reranker:
    def __init__(self, device: str | None = None):
        from sentence_transformers import CrossEncoder

        from eduai.curriculum.embedder import _default_device

        self.model = CrossEncoder(
            str(model_path("rerank")), device=device or _default_device(), local_files_only=True
        )

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        # Raw logits: ms-marco sigmoid scores for (question, objective) pairs sit near 0 and
        # squash the spread tau needs.
        return np.asarray(
            self.model.predict(list(pairs), batch_size=128, show_progress_bar=False), dtype=np.float32
        )


def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


class Tagger:
    def __init__(
        self,
        taxonomy: Taxonomy,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        k: int = 10,
        gate_margin: float = 0.03,
        tau: float | None = None,
        use_gate: bool = True,
    ):
        self.tax = taxonomy
        self.embedder = embedder
        self.reranker = reranker
        self.k = k
        self.gate_margin = gate_margin
        self.use_gate = use_gate
        self.lo_ids = taxonomy.lo_ids()
        self.lo_texts = [taxonomy.lo(i).embed_text() for i in self.lo_ids]
        self.lo_subject = np.array([taxonomy.subject_of(i) for i in self.lo_ids])
        self.lo_vecs = cached_encode(embedder, self.lo_texts)
        self.subjects = list(taxonomy.subjects)
        cents = np.stack([self.lo_vecs[self.lo_subject == s].mean(axis=0) for s in self.subjects])
        self.centroids = cents / np.linalg.norm(cents, axis=1, keepdims=True)
        if tau is None:
            cal = load_calibration().get(self.mode, {})
            tau = cal.get("tau")
        self.tau = tau

    @property
    def mode(self) -> str:
        parts = self.embedder.embedder_id.split("/")
        name = parts[2] if len(parts) > 2 else parts[0]
        if getattr(self.embedder, "key", "").startswith("bge") and not getattr(
            self.embedder, "query_prefix", ""
        ):
            name += " (no query prefix)"
        return f"{name}{'+rerank' if self.reranker else ''}{'' if self.use_gate else ' (no gate)'}"

    def gate(self, qvecs: np.ndarray) -> list[list[str]]:
        sims = qvecs @ self.centroids.T
        out = []
        for row in sims:
            best = row.max()
            keep = [self.subjects[j] for j in np.argsort(-row) if row[j] >= best - self.gate_margin][:2]
            out.append(keep)
        return out

    def tag_texts(self, texts: Sequence[str]) -> list[TagResult]:
        qvecs = cached_encode(self.embedder, texts, query=True)
        sims = qvecs @ self.lo_vecs.T
        gates = self.gate(qvecs) if self.use_gate else [self.subjects] * len(texts)
        candidates: list[list[int]] = []
        for i, allowed in enumerate(gates):
            mask = np.isin(self.lo_subject, allowed)
            row = np.where(mask, sims[i], -np.inf)
            candidates.append(list(np.argsort(-row)[: self.k]))
        if self.reranker is not None:
            pairs = [(texts[i], self.lo_texts[j]) for i, cand in enumerate(candidates) for j in cand]
            flat = self.reranker.score(pairs)
            final = flat.reshape(len(texts), self.k)
        else:
            final = np.stack([sims[i, cand] for i, cand in enumerate(candidates)])
        results = []
        for i, cand in enumerate(candidates):
            order = np.argsort(-final[i])
            ranked = [self.lo_ids[cand[j]] for j in order]
            score = float(final[i, order[0]])
            results.append(
                TagResult(
                    lo_id=ranked[0],
                    subject=self.tax.subject_of(ranked[0]),
                    score=score,
                    top3=ranked[:3],
                    aligned=self.tau is not None and score >= self.tau,
                    scores={self.lo_ids[cand[j]]: float(final[i, j]) for j in range(len(cand))},
                )
            )
        return results

    def score_pair(self, texts: Sequence[str], lo_ids: Sequence[str]) -> np.ndarray:
        """Score of a specific LO for each text, on the same scale as `tag_texts`."""
        if self.reranker is not None:
            return self.reranker.score(
                [(t, self.tax.lo(lo).embed_text()) for t, lo in zip(texts, lo_ids, strict=True)]
            )
        qvecs = cached_encode(self.embedder, texts, query=True)
        idx = [self.lo_ids.index(lo) for lo in lo_ids]
        return np.einsum("ij,ij->i", qvecs, self.lo_vecs[idx])


def build_tagger(
    taxonomy: Taxonomy, embed_key: str = "bge-small", rerank: bool = True, query_prefix: bool = True, **kw
) -> Tagger:
    from eduai.curriculum.embedder import get_embedder

    emb = (
        get_embedder(embed_key) if embed_key == "hash" else get_embedder(embed_key, query_prefix=query_prefix)
    )
    return Tagger(taxonomy, emb, Reranker() if rerank else None, **kw)
