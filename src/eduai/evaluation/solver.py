"""Answer-key self-consistency: a solver answers the item without seeing the key.

With the MLX backend the solver reads the next-token log-probabilities of A-D once, takes the
argmax as the greedy answer and draws two more answers at T = 0.7 from the same distribution.
Other backends generate a short answer three times. The majority of the three must match the key.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from eduai.prompts import build_solver_messages
from eduai.schema import LETTERS

_LETTER = re.compile(r"\b([ABCD])\b")


def parse_letter(text: str) -> str | None:
    m = _LETTER.search(text.strip().upper()[:20])
    return m.group(1) if m else None


class Solver:
    def __init__(self, backend, samples: int = 2, temperature: float = 0.7, seed: int = 0):
        self.backend = backend
        self.samples = samples
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def answers(self, item: dict) -> list[str | None]:
        msgs = build_solver_messages(item["stem"], item["choices"], item.get("stimulus"))
        if hasattr(self.backend, "choice_logprobs"):
            lp = self.backend.choice_logprobs(msgs)
            vals = np.array([lp[k] for k in LETTERS])
            greedy = LETTERS[int(np.argmax(vals))]
            z = vals / self.temperature
            probs = np.exp(z - z.max())
            probs /= probs.sum()
            draws = [LETTERS[i] for i in self.rng.choice(4, size=self.samples, p=probs)]
            return [greedy, *draws]
        out = [parse_letter(self.backend.generate(msgs, 4, 0.0))]
        out += [parse_letter(self.backend.generate(msgs, 4, self.temperature)) for _ in range(self.samples)]
        return out

    def __call__(self, item: dict) -> dict:
        ans = self.answers(item)
        counts = Counter(a for a in ans if a)
        majority = counts.most_common(1)[0][0] if counts else None
        return {
            "answers": ans,
            "majority": majority,
            "agrees": majority == item["answer"],
            "greedy_correct": float(ans[0] == item["answer"]),
        }
