"""Build the SFT chat data, the item bank, and the held-out eval prompts from SciQ.

Order matters and mirrors the data card: eligibility gate -> curriculum tagging -> difficulty ->
near-duplicate grouping and split assignment -> data card (exclusions) -> quotas -> files.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eduai.curriculum.taxonomy import Taxonomy
from eduai.data import difficulty as diff
from eduai.data import explain, splits
from eduai.data.sciq import SPLITS, SciqItem, eligibility_counts, write_jsonl
from eduai.prompts import GenerationRequest, build_messages
from eduai.schema import LETTERS


@dataclass
class Quotas:
    train: int = 3000
    valid: int = 300
    test: int = 200
    eval_prompts: int = 150
    stimulus_frac: float = 0.40
    misconception_frac: float = 0.30
    avoid_frac: float = 0.20


@dataclass
class BuildResult:
    counts: dict
    card: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)


def shuffle_choices(item: SciqItem, letter: str, rng: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    """Place the correct answer at `letter` and the distractors in random order elsewhere."""
    others = [x for x in LETTERS if x != letter]
    ds = item.distractors[:]
    rng.shuffle(ds)
    choices = {letter: item.correct}
    misconceptions = {}
    for lt, d in zip(others, ds, strict=True):
        choices[lt] = d
        misconceptions[lt] = d
    return dict(sorted(choices.items())), misconceptions


def balanced_letters(n: int, rng: random.Random) -> list[str]:
    letters = [LETTERS[i % 4] for i in range(n)]
    rng.shuffle(letters)
    return letters


def stratified_sample(pool: list[int], strata: list[tuple], n: int, rng: random.Random) -> list[int]:
    """Proportional allocation over strata with largest-remainder rounding."""
    by: dict[tuple, list[int]] = defaultdict(list)
    for i, s in zip(pool, strata, strict=True):
        by[s].append(i)
    if n >= len(pool):
        out = list(pool)
        rng.shuffle(out)
        return out
    total = len(pool)
    alloc = {s: n * len(v) / total for s, v in by.items()}
    base = {s: int(a) for s, a in alloc.items()}
    rest = n - sum(base.values())
    for s in sorted(alloc, key=lambda s: -(alloc[s] - base[s]))[:rest]:
        base[s] += 1
    out: list[int] = []
    for s in sorted(by):
        members = by[s][:]
        rng.shuffle(members)
        out += members[: base[s]]
    rng.shuffle(out)
    return out


class Builder:
    def __init__(
        self,
        taxonomy: Taxonomy,
        tagger,
        embedder,
        seed: int = 7,
        quotas: Quotas | None = None,
        containment: float | None = None,
    ):
        self.containment = containment
        self.tax = taxonomy
        self.tagger = tagger
        self.embedder = embedder
        self.seed = seed
        self.q = quotas or Quotas()

    # -- stages -----------------------------------------------------------------------------------
    def tag(self, items: list[SciqItem]) -> None:
        results = self.tagger.tag_texts([it.qa_text() for it in items])
        for it, r in zip(items, results, strict=True):
            it.tags = r.to_dict()

    def score_difficulty(self, items: list[SciqItem]) -> tuple[float, float]:
        ans = self.embedder.encode([it.correct for it in items])
        dis = self.embedder.encode([d for it in items for d in it.distractors]).reshape(len(items), 3, -1)
        raw = diff.raw_scores([it.question for it in items], ans, dis)
        eligible = np.array([it.grounded for it in items])
        cuts = diff.tertile_cuts(raw[eligible] if eligible.any() else raw)
        for it, s in zip(items, raw, strict=True):
            it.tags["difficulty_score"] = round(float(s), 4)
            it.tags["difficulty"] = diff.label(float(s), cuts)
        return cuts

    def group(self, items: list[SciqItem]) -> dict:
        vecs = self.embedder.encode([it.qa_text() for it in items])
        roots, (p, q, c, n_groups, largest) = splits.group_items(items, vecs, containment=self.containment)
        final, moved = splits.assign_splits(items, roots)
        for it, r, s in zip(items, roots, final, strict=True):
            it.tags["group"] = int(r)
            it.tags["final_split"] = s
        return {
            "passage_links": p,
            "qa_links": q,
            "cosine_links": c,
            "groups": n_groups,
            "largest_group": largest,
            "reassigned": dict(moved),
            "cross_split_groups_after": splits.leakage(items, roots, final),
        }

    # -- data card (exclusions come before any quota) ----------------------------------------------
    def data_card(self, items: list[SciqItem], grouping: dict, cuts: tuple[float, float]) -> dict:
        per_split: dict[str, Counter] = {s: Counter() for s in SPLITS}
        per_subject: dict[str, Counter] = defaultdict(Counter)
        for it in items:
            subj = it.tags["subject"]
            c = per_split[it.split]
            c["total"] += 1
            per_subject[subj]["total"] += 1
            if not it.grounded:
                c["excluded_blank_support"] += 1
                per_subject[subj]["excluded_blank_support"] += 1
                continue
            if not explain.sentences(it.support):
                c["excluded_no_sentence"] += 1
                per_subject[subj]["excluded_no_sentence"] += 1
                continue
            if not it.tags["aligned"]:
                c["excluded_unaligned"] += 1
                per_subject[subj]["excluded_unaligned"] += 1
                continue
            if it.tags["final_split"] != it.split:
                c["moved_by_grouping"] += 1
                per_subject[subj]["moved_by_grouping"] += 1
            per_split[it.tags["final_split"]]["eligible"] += 1
            per_subject[subj]["eligible"] += 1
        return {
            "eligibility": eligibility_counts(items),
            "per_split": {s: dict(v) for s, v in per_split.items()},
            "per_subject": {s: dict(v) for s, v in sorted(per_subject.items())},
            "grouping": grouping,
            "difficulty_cuts": [round(x, 4) for x in cuts],
            "difficulty_by_split": {
                s: dict(
                    Counter(
                        it.tags["difficulty"] for it in items if it.grounded and it.tags["final_split"] == s
                    )
                )
                for s in SPLITS
            },
        }

    # -- conversion -------------------------------------------------------------------------------
    def request_for(
        self, it: SciqItem, fmt: str, misconception: str | None, avoid: list[str]
    ) -> GenerationRequest:
        lo = self.tax.lo(it.tags["lo_id"])
        return GenerationRequest(
            subject=lo.subject_name,
            unit=lo.unit_name,
            topic=lo.topic_name,
            lo_id=lo.id,
            lo_text=lo.text,
            difficulty=it.tags["difficulty"],
            format=fmt,
            passage=explain.trim_passage(it.support, it.question, it.correct),
            target_misconception=misconception,
            avoid=avoid,
        )

    def completion_for(self, it: SciqItem, letter: str, fmt: str, rng: random.Random) -> dict | None:
        choices, _ = shuffle_choices(it, letter, rng)
        expl = explain.explanation(it.support, it.question, it.correct, letter)
        if expl is None:
            return None
        out = {
            "stem": it.question,
            "choices": choices,
            "answer": letter,
            "explanation": expl,
            "lo_id": it.tags["lo_id"],
            "difficulty": it.tags["difficulty"],
        }
        if fmt == "stimulus":
            stim = explain.stimulus_excerpt(it.support, it.question, it.correct)
            if stim is None:
                return None
            q = it.question[0].lower() + it.question[1:] if it.question else it.question
            out["stem"] = f"Based on the excerpt, {q}"
            out["stimulus"] = stim
        return out

    def build_split(
        self, items: list[SciqItem], idx: list[int], n_target: int, split: str, rng: random.Random
    ):
        """Select n_target examples and enforce letter/format/misconception quotas exactly."""
        strata = [(items[i].tags["subject"], items[i].tags["difficulty"]) for i in idx]
        chosen = stratified_sample(idx, strata, n_target, rng)
        n = len(chosen)
        letters = balanced_letters(n, rng)
        stim_ok = [
            i
            for i in chosen
            if explain.stimulus_excerpt(items[i].support, items[i].question, items[i].correct)
        ]
        n_stim = min(round(self.q.stimulus_frac * n), len(stim_ok))
        stim_set = set(rng.sample(stim_ok, n_stim))
        n_mis = round(self.q.misconception_frac * n)
        mis_set = set(rng.sample(chosen, n_mis))
        n_avoid = round(self.q.avoid_frac * n)
        avoid_set = set(rng.sample(chosen, n_avoid))
        by_lo: dict[str, list[int]] = defaultdict(list)
        for i in idx:
            by_lo[items[i].tags["lo_id"]].append(i)

        rows, meta = [], []
        for i, letter in zip(chosen, letters, strict=True):
            it = items[i]
            fmt = "stimulus" if i in stim_set else "standard"
            comp = self.completion_for(it, letter, fmt, rng)
            if comp is None:
                continue
            mis = None
            if i in mis_set:
                mis = rng.choice(it.distractors)
            avoid = []
            if i in avoid_set:
                others = [j for j in by_lo[it.tags["lo_id"]] if j != i]
                avoid = [items[j].question for j in rng.sample(others, min(2, len(others)))]
            req = self.request_for(it, fmt, mis, avoid)
            rows.append({"messages": build_messages(req, comp)})
            meta.append(
                {
                    "id": it.id,
                    "letter": letter,
                    "format": fmt,
                    "misconception": mis is not None,
                    "subject": it.tags["subject"],
                    "difficulty": it.tags["difficulty"],
                    "lo_id": it.tags["lo_id"],
                    "group": it.tags["group"],
                }
            )
        return rows, meta

    def bank_rows(self, items: list[SciqItem], rng: random.Random) -> list[dict]:
        rows = []
        for it in items:
            letter = rng.choice(LETTERS)
            choices, mis = shuffle_choices(it, letter, rng)
            lo = self.tax.lo(it.tags["lo_id"])
            rows.append(
                {
                    "id": f"sciq-{it.id}",
                    "stem": it.question,
                    "choices": choices,
                    "answer": letter,
                    # Blank-support items stay in the bank but get no generated explanation.
                    "explanation": explain.explanation(it.support, it.question, it.correct, letter)
                    if it.grounded
                    else None,
                    "lo_id": lo.id,
                    "subject": lo.subject,
                    "unit_id": lo.unit_id,
                    "difficulty": it.tags["difficulty"],
                    "b": diff.LABEL_B[it.tags["difficulty"]],
                    "source": "sciq",
                    "grounded": it.grounded,
                    "ungrounded": not it.grounded,
                    "aligned": bool(it.tags["aligned"]),
                    "tag_score": it.tags["score"],
                    "misconceptions": mis,
                    "sciq_split": it.split,
                }
            )
        return rows

    # -- main -------------------------------------------------------------------------------------
    def run(self, items: list[SciqItem], out_dir: Path) -> BuildResult:
        rng = random.Random(self.seed)
        self.tag(items)
        cuts = self.score_difficulty(items)
        grouping = self.group(items)
        card = self.data_card(items, grouping, cuts)

        eligible = [
            i
            for i, it in enumerate(items)
            if it.grounded and explain.sentences(it.support) and it.tags["aligned"]
        ]
        pools = {s: [i for i in eligible if items[i].tags["final_split"] == s] for s in SPLITS}

        train_rows, train_meta = self.build_split(items, pools["train"], self.q.train, "train", rng)
        valid_rows, valid_meta = self.build_split(items, pools["valid"], self.q.valid, "valid", rng)
        test_pool = pools["test"][:]
        rng.shuffle(test_pool)
        eval_idx = test_pool[: self.q.eval_prompts]
        test_rows, test_meta = self.build_split(
            items, test_pool[self.q.eval_prompts :], self.q.test, "test", rng
        )

        eval_rows = self.eval_prompts(items, eval_idx, rng)
        quotas = self.quota_report(
            {"train": train_meta, "valid": valid_meta, "test": test_meta},
            {"train": self.q.train, "valid": self.q.valid, "test": self.q.test},
            len(eval_rows),
        )
        card["quotas"] = quotas

        sft_dir = out_dir / "sft"
        write_jsonl(sft_dir / "train.jsonl", train_rows)
        write_jsonl(sft_dir / "valid.jsonl", valid_rows)
        write_jsonl(sft_dir / "test.jsonl", test_rows)
        write_jsonl(out_dir / "sft_meta.jsonl", train_meta + valid_meta + test_meta)
        write_jsonl(out_dir / "eval" / "prompts.jsonl", eval_rows)
        write_jsonl(out_dir / "items_tagged.jsonl", [it.to_dict() for it in items])
        bank = self.bank_rows(items, rng)
        write_jsonl(out_dir / "bank" / "sciq_items.jsonl", bank)
        # Train-set stems are needed later to measure memorization of generated items.
        by_id = {it.id: it for it in items}
        write_jsonl(
            out_dir / "sft" / "train_stems.jsonl",
            [{"id": m["id"], "stem": by_id[m["id"]].question} for m in train_meta],
        )
        return BuildResult(
            counts=card["eligibility"],
            card=card,
            files={"sft": str(sft_dir), "bank": len(bank), "eval": len(eval_rows)},
        )

    def eval_prompts(self, items: list[SciqItem], idx: list[int], rng: random.Random) -> list[dict]:
        rows = []
        n = len(idx)
        stim = set(rng.sample(range(n), round(self.q.stimulus_frac * n)))
        mis_k = set(rng.sample(range(n), round(self.q.misconception_frac * n)))
        for k, i in enumerate(idx):
            it = items[i]
            fmt = "stimulus" if k in stim else "standard"
            mis = rng.choice(it.distractors) if k in mis_k else None
            req = self.request_for(it, fmt, mis, [])
            rows.append(
                {
                    "id": it.id,
                    "request": req.__dict__,
                    "reference": {"question": it.question, "answer": it.correct},
                    "group": it.tags["group"],
                }
            )
        return rows

    @staticmethod
    def quota_report(metas: dict[str, list[dict]], targets: dict[str, int], n_eval: int) -> dict:
        rep = {}
        for split, meta in metas.items():
            n = len(meta)
            letters = Counter(m["letter"] for m in meta)
            rep[split] = {
                "target": targets[split],
                "achieved": n,
                "shrunk": n < targets[split],
                "letters": {k: letters.get(k, 0) for k in LETTERS},
                "stimulus": sum(m["format"] == "stimulus" for m in meta),
                "misconception": sum(m["misconception"] for m in meta),
                "by_subject": dict(Counter(m["subject"] for m in meta)),
                "by_difficulty": dict(Counter(m["difficulty"] for m in meta)),
            }
        rep["eval_prompts"] = n_eval
        return rep


def render_data_card(card: dict) -> str:
    e = card["eligibility"]
    L = [
        "# Data card",
        "",
        "Source: SciQ (Welbl, Liu and Gardner, 2017), CC BY-NC 3.0. Counts below are produced by "
        "`eduai data build`; the machine-readable version is `reports/data_card.json`.",
        "",
        "## 1. Eligibility gate (before any split sizes or quotas)",
        "",
        "Records with a blank `support` passage cannot give passage-grounded explanations. They are "
        "excluded from SFT and stimulus pools and kept only in the bank, flagged `ungrounded`, with no "
        "explanation.",
        "",
        "| SciQ split | Records | With support | Blank support (excluded from SFT) |",
        "|---|---|---|---|",
    ]
    for s in SPLITS:
        L.append(f"| {s} | {e[s]['total']} | {e[s]['grounded']} | {e[s]['blank_support']} |")
    tot = {k: sum(e[s][k] for s in SPLITS) for k in ("total", "grounded", "blank_support")}
    L.append(f"| all | {tot['total']} | {tot['grounded']} | {tot['blank_support']} |")
    L += [
        "",
        "## 2. Exclusions per split",
        "",
        "Applied in order: blank support, support with no usable sentence, then curriculum alignment "
        "(tagger score below tau). "
        "`moved` counts grounded, aligned items whose near-duplicate group touched a higher-priority "
        "split (test > valid > train) and were reassigned there.",
        "",
        "| SciQ split | Total | Blank support | No usable sentence | Not aligned | Moved by grouping |",
        "|---|---|---|---|---|---|",
    ]
    for s in SPLITS:
        c = card["per_split"][s]
        L.append(
            f"| {s} | {c.get('total', 0)} | {c.get('excluded_blank_support', 0)} | "
            f"{c.get('excluded_no_sentence', 0)} | {c.get('excluded_unaligned', 0)} | "
            f"{c.get('moved_by_grouping', 0)} |"
        )
    L += [
        "",
        "## 3. Exclusions per subject (subject = tagger's top-1)",
        "",
        "| Subject | Total | Blank support | No usable sentence | Not aligned | Eligible |",
        "|---|---|---|---|---|---|",
    ]
    for s, c in card["per_subject"].items():
        L.append(
            f"| {s} | {c.get('total', 0)} | {c.get('excluded_blank_support', 0)} | "
            f"{c.get('excluded_no_sentence', 0)} | {c.get('excluded_unaligned', 0)} | {c.get('eligible', 0)} |"
        )
    g = card["grouping"]
    L += [
        "",
        "## 4. Near-duplicate grouping",
        "",
        f"Union-find over shared passage hash ({g['passage_links']} links), identical normalized "
        f"question+answer ({g['qa_links']} links) and bge cosine > {splits.NEAR_DUP_COSINE} on question+answer "
        f"({g['cosine_links']} links). {g['groups']} groups, largest {g['largest_group']} items. "
        f"Reassigned items: {g['reassigned'] or 'none'}. Groups spanning more than one split after "
        f"assignment: {g['cross_split_groups_after']}.",
        "",
        "## 5. Eligible pools after exclusions",
        "",
        "| Split | Eligible |",
        "|---|---|",
    ]
    for s in SPLITS:
        L.append(f"| {s} | {card['per_split'][s].get('eligible', 0)} |")
    lo, hi = card["difficulty_cuts"]
    L += [
        "",
        f"Difficulty tertile cut points (from the grounded pool): {lo} and {hi}.",
        "",
        "## 6. Quotas (applied last)",
        "",
        "Targets: SFT 3,000/300/200, answer letters 25% each, 40% stimulus-style, 30% with a target "
        "misconception. Anything that could not be met is shrunk and reported here, not padded.",
        "",
        "| Split | Target | Achieved | A | B | C | D | Stimulus | Misconception |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in SPLITS:
        q = card["quotas"][s]
        n = max(q["achieved"], 1)
        lt = q["letters"]
        L.append(
            f"| {s} | {q['target']} | {q['achieved']}{' (shrunk)' if q['shrunk'] else ''} | "
            f"{lt['A']} | {lt['B']} | {lt['C']} | {lt['D']} | {q['stimulus']} ({q['stimulus'] / n:.0%}) | "
            f"{q['misconception']} ({q['misconception'] / n:.0%}) |"
        )
    L += [
        "",
        f"Held-out eval prompts (from the SciQ test split, disjoint from SFT test rows): "
        f"{card['quotas']['eval_prompts']}.",
        "",
    ]
    return "\n".join(L)


def save_card(card: dict, reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_card.json").write_text(json.dumps(card, indent=2) + "\n")
    (reports / "data_card.md").write_text(render_data_card(card))
