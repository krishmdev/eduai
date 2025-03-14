"""generate -> validate -> dedup -> accept/reject, with per-reason counters."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field

from eduai.curriculum.tagger import Tagger
from eduai.curriculum.taxonomy import Taxonomy
from eduai.generation.dedup import NoveltyIndex
from eduai.generation.generator import Generator
from eduai.generation.validate import CheckResult, validate_item
from eduai.prompts import GenerationRequest


def make_aligner(tagger: Tagger):
    def aligner(item: dict, req: GenerationRequest) -> tuple[bool, dict]:
        text = f"{item['stem']} Answer: {item['choices'][item['answer']]}"
        res = tagger.tag_texts([text])[0]
        in_top3 = req.lo_id in res.top3
        ok = in_top3 and tagger.tau is not None and res.score >= tagger.tau
        return ok, {
            "tag_score": res.score,
            "target_in_top3": float(in_top3),
            "tag_top1_match": float(res.lo_id == req.lo_id),
        }

    return aligner


@dataclass
class PipelineStats:
    attempted: int = 0
    accepted: int = 0
    rejections: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {"attempted": self.attempted, "accepted": self.accepted, "rejections": dict(self.rejections)}


class Pipeline:
    def __init__(
        self,
        generator: Generator,
        taxonomy: Taxonomy,
        *,
        solver=None,
        tagger: Tagger | None = None,
        novelty: NoveltyIndex | None = None,
    ):
        self.generator = generator
        self.tax = taxonomy
        self.solver = solver
        self.aligner = make_aligner(tagger) if tagger is not None else None
        self.novelty = novelty
        self.stats = PipelineStats()

    def run_one(self, req: GenerationRequest) -> tuple[CheckResult, dict | None, object]:
        gen = self.generator.generate(req)
        res = validate_item(
            gen.text,
            req,
            solver=self.solver,
            aligner=self.aligner,
            novelty=self.novelty.check if self.novelty else None,
        )
        self.stats.attempted += 1
        row = None
        if res.ok:
            self.stats.accepted += 1
            row = self.to_bank_row(res.item, req)
        else:
            self.stats.rejections[res.reason] += 1
        return res, row, gen

    def to_bank_row(self, item: dict, req: GenerationRequest) -> dict:
        lo = self.tax.lo(req.lo_id)
        digest = hashlib.sha1(item["stem"].encode()).hexdigest()[:12]
        wrong = {k: v for k, v in item["choices"].items() if k != item["answer"]}
        from eduai.data.difficulty import LABEL_B

        return {
            "id": f"gen-{digest}",
            "stem": item["stem"],
            "stimulus": item.get("stimulus"),
            "choices": item["choices"],
            "answer": item["answer"],
            "explanation": item["explanation"],
            "lo_id": lo.id,
            "subject": lo.subject,
            "unit_id": lo.unit_id,
            "difficulty": req.difficulty,
            "b": LABEL_B[req.difficulty],
            "source": f"generated:{self.generator.backend.name}",
            "grounded": True,
            "ungrounded": False,
            "aligned": True,
            "misconceptions": wrong,
            "target_misconception": req.target_misconception,
        }
