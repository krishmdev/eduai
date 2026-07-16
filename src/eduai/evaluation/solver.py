"""Answer-key check with a fixed judge that is not any arm's generator.

The judge reads next-token log-probabilities for A-D. Letter-position bias is averaged out by
scoring the item under cyclic rotations of its options and mapping the scores back to the
original options. The key "agrees" when the option with the highest mean log-probability is the
keyed one.
"""

from __future__ import annotations

import re

import numpy as np

from eduai.prompts import build_solver_messages
from eduai.schema import LETTERS

_LETTER = re.compile(r"\b([ABCD])\b")


def parse_letter(text: str) -> str | None:
    m = _LETTER.search(text.strip().upper()[:20])
    return m.group(1) if m else None


def rotate(choices: dict[str, str], k: int) -> tuple[dict[str, str], dict[str, str]]:
    """Returns rotated choices and a map new_letter -> original_letter."""
    order = list(LETTERS)
    new_to_old = {order[i]: order[(i + k) % 4] for i in range(4)}
    return {new: choices[old] for new, old in new_to_old.items()}, new_to_old


class Judge:
    def __init__(self, backend, rotations: tuple[int, ...] = (0, 1, 2, 3), open_book: bool = True):
        if not hasattr(backend, "choice_logprobs"):
            raise TypeError("judge backend must expose choice_logprobs")
        self.backend = backend
        self.rotations = rotations
        self.open_book = open_book

    def option_scores(self, item: dict, passage: str | None = None) -> dict[str, float]:
        totals = dict.fromkeys(LETTERS, 0.0)
        for k in self.rotations:
            rot, new_to_old = rotate(item["choices"], k)
            msgs = build_solver_messages(
                item["stem"], rot, item.get("stimulus"), passage if self.open_book else None
            )
            lp = self.backend.choice_logprobs(msgs)
            for new, old in new_to_old.items():
                totals[old] += lp[new] / len(self.rotations)
        return totals

    def __call__(self, item: dict, passage: str | None = None) -> dict:
        texts = [v.strip().lower() for v in item["choices"].values()]
        if texts.count(item["choices"][item["answer"]].strip().lower()) > 1:
            # The key's text also appears as a distractor, so "the keyed option" is ambiguous.
            return {"majority": None, "agrees": False, "p_key": None, "duplicate_options": True}
        scores = self.option_scores(item, passage)
        vals = np.array([scores[k] for k in LETTERS])
        probs = np.exp(vals - vals.max())
        probs /= probs.sum()
        choice = LETTERS[int(np.argmax(vals))]
        return {
            "majority": choice,
            "agrees": choice == item["answer"],
            "p_key": float(probs[LETTERS.index(item["answer"])]),
        }


class GenerativeSolver:
    """Fallback for backends without logprobs (e.g. Ollama): greedy letter only."""

    def __init__(self, backend):
        self.backend = backend

    def __call__(self, item: dict, passage: str | None = None) -> dict:
        msgs = build_solver_messages(item["stem"], item["choices"], item.get("stimulus"), passage)
        ans = parse_letter(self.backend.generate(msgs, 4, 0.0))
        return {"majority": ans, "agrees": ans == item["answer"]}
