"""LO-to-LO similarity graph within each unit; drives damped evidence sharing in BKT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eduai.config import ROOT
from eduai.curriculum.embedder import Embedder
from eduai.curriculum.taxonomy import Taxonomy
from eduai.kt.bkt import NeighborGraph

GRAPH_PATH = ROOT / "configs" / "lo_neighbors.json"


def build_graph(tax: Taxonomy, embedder: Embedder) -> dict:
    ids = tax.lo_ids()
    vecs = embedder.encode([tax.lo(i).embed_text() for i in ids])
    sims = vecs @ vecs.T
    edges: dict[str, list[list]] = {}
    for a, i in enumerate(ids):
        unit = tax.lo(i).unit_id
        row = []
        for b, j in enumerate(ids):
            if a != b and tax.lo(j).unit_id == unit:
                row.append([j, round(float(sims[a, b]), 4)])
        edges[i] = sorted(row, key=lambda x: -x[1])
    return {"embedder_id": embedder.embedder_id, "edges": edges}


def save_graph(graph: dict, path: Path = GRAPH_PATH) -> None:
    path.write_text(json.dumps(graph, indent=1) + "\n")


def load_graph(path: Path = GRAPH_PATH, damping: float = 0.3, min_sim: float = 0.8) -> NeighborGraph:
    if not path.exists():
        return NeighborGraph(damping=damping, min_sim=min_sim)
    data = json.loads(path.read_text())
    edges = {k: [(j, float(s)) for j, s in v] for k, v in data["edges"].items()}
    return NeighborGraph(edges=edges, damping=damping, min_sim=min_sim)


def similarity_stats(graph: dict) -> dict:
    vals = np.array([s for v in graph["edges"].values() for _, s in v])
    return {
        "pairs": int(len(vals)),
        "above_0.6": int((vals > 0.6).sum()),
        "above_0.8": int((vals > 0.8).sum()),
        "median": float(np.median(vals)),
    }
