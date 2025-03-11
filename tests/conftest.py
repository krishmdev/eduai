from __future__ import annotations

import json
from pathlib import Path

import pytest

from eduai.curriculum.embedder import HashingEmbedder
from eduai.curriculum.tagger import TagResult
from eduai.curriculum.taxonomy import default_taxonomy


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EDUAI_CACHE_DIR", str(tmp_path / "emb-cache"))


@pytest.fixture
def taxonomy():
    return default_taxonomy()


class KeywordTagger:
    """Deterministic stand-in for the bge+rerank tagger: best keyword overlap wins."""

    def __init__(self, taxonomy, tau: float = 1.0):
        self.tax = taxonomy
        self.tau = tau
        self.embedder = HashingEmbedder(256)
        self.mode = "keyword"

    def tag_texts(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            scored = sorted(
                ((sum(k in low for k in lo.keywords), lo.id) for lo in self.tax.objectives.values()),
                key=lambda x: (-x[0], x[1]),
            )
            top = [lo for _, lo in scored[:3]]
            score = float(scored[0][0])
            out.append(TagResult(top[0], self.tax.subject_of(top[0]), score, top, score >= self.tau))
        return out


@pytest.fixture
def keyword_tagger(taxonomy):
    return KeywordTagger(taxonomy)


PASSAGES = [
    "Photosynthesis happens in the chloroplast. Plants use light energy to make glucose from carbon dioxide and water.",
    "Mitochondria are the site of cellular respiration. Glycolysis breaks glucose down before the Krebs cycle runs.",
    "An acid donates protons in water. The pH scale measures hydrogen ion concentration from 0 to 14.",
    "Newton's second law says net force equals mass times acceleration. A larger net force gives a larger acceleration.",
    "Plate tectonics explains earthquakes and volcanoes. Plates move over the mantle at plate boundaries.",
    "Eutrophication starts with nutrient runoff. Algal bloom decay uses up dissolved oxygen and creates a dead zone.",
]
QUESTIONS = [
    (
        "Where does photosynthesis take place in plant cells?",
        "chloroplast",
        ["nucleus", "ribosome", "vacuole"],
    ),
    (
        "Which organelle is the site of cellular respiration?",
        "mitochondria",
        ["golgi", "lysosome", "ribosome"],
    ),
    ("What does the ph scale measure?", "hydrogen ion concentration", ["temperature", "mass", "volume"]),
    (
        "According to newton's second law, net force equals mass times what?",
        "acceleration",
        ["velocity", "weight", "time"],
    ),
    ("Movement at a plate boundary can cause what?", "earthquakes", ["tides", "rainbows", "eclipses"]),
    (
        "Nutrient runoff can lead to what process in lakes?",
        "eutrophication",
        ["erosion", "sublimation", "fission"],
    ),
]


def make_sciq(
    root: Path, n_train: int = 60, n_valid: int = 12, n_test: int = 12, blank: dict | None = None
) -> Path:
    """Write a tiny SciQ-shaped dataset. `blank` = number of blank-support records per split."""
    blank = blank or {"train": 7, "valid": 2, "test": 3}
    root.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        rows = []
        for i in range(n):
            k = i % len(QUESTIONS)
            q, a, ds = QUESTIONS[k]
            support = (
                ""
                if i < blank[split]
                else f"{PASSAGES[k]} Variant {split} {i} adds detail sentence number {i}."
            )
            rows.append(
                {
                    "question": f"{q} (case {split} {i})",
                    "correct_answer": a,
                    "distractor1": ds[0],
                    "distractor2": ds[1],
                    "distractor3": ds[2],
                    "support": support,
                }
            )
        (root / f"{split}.json").write_text(json.dumps(rows))
    return root
