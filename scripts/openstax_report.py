"""OpenStax OOD check: pick the v2 headline arm from the valid report, then score the run.

    python scripts/openstax_report.py headline   # prints the arm name chosen on valid
    python scripts/openstax_report.py score      # per_item, eval.json and report.md

The headline rule is step 4 of docs/v2_selection.md: whichever of finetuned-v2 (0-shot) and
finetuned-v2-2shot has the higher valid usable rate, a tie going to 0-shot. The scoring is the main
eval's (compare.score, llama-1b judge, the same usable definition); this script adds per-subject
and pooled usable rates with paired bootstrap intervals, which the main report doesn't have.
See docs/openstax_ood.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "openstax" / "prompts.jsonl"
OUT = ROOT / "reports" / "openstax_ood"
VALID = ROOT / "reports" / "valid_eval" / "eval.json"
BASE = "base-2shot"
SUBJECTS = ("BIO", "PHYS1", "CHEM")
N_BOOT = 5000


def headline(valid_json: Path = VALID) -> str:
    s = json.loads(valid_json.read_text())["summary"]
    if "finetuned-v2" not in s or "finetuned-v2-2shot" not in s:
        raise SystemExit(f"{valid_json} has no finetuned-v2 / finetuned-v2-2shot rows yet")
    return (
        "finetuned-v2-2shot"
        if s["finetuned-v2-2shot"]["usable"] > s["finetuned-v2"]["usable"]
        else "finetuned-v2"
    )


def paired(a: list[bool], b: list[bool], rng: np.random.Generator) -> dict:
    d = np.array(a, float) - np.array(b, float)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "n": len(d),
        "diff": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


def by_subject(per_item: dict[str, list[dict]], subject_of: dict[str, str], arm: str, seed: int = 0) -> dict:
    """Usable rate per subject and pooled for BASE and `arm`, with the paired bootstrap difference."""
    rng = np.random.default_rng(seed)
    a = {r["id"]: bool(r["usable"]) for r in per_item[arm]}
    b = {r["id"]: bool(r["usable"]) for r in per_item[BASE]}
    out = {}
    for subj in (*SUBJECTS, "pooled"):
        ids = [i for i in a if subj == "pooled" or subject_of[i] == subj]
        if not ids:
            continue
        out[subj] = {
            "n": len(ids),
            f"{arm}_usable": sum(a[i] for i in ids) / len(ids),
            f"{BASE}_usable": sum(b[i] for i in ids) / len(ids),
            "diff": paired([a[i] for i in ids], [b[i] for i in ids], rng),
        }
    return out


def _p(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(res: dict, card: dict, verification: dict | None = None) -> str:
    arm = res["arm"]
    s = res["summary"]
    lines = [
        "# OpenStax out-of-distribution check",
        "",
        "Secondary check, run once, pre-registered in `docs/openstax_ood.md`. 150 prompts from pinned "
        "OpenStax AP textbooks (50 each for biology, physics and chemistry; AP Environmental Science has "
        "no OpenStax book). Target objectives are the tagger's own top-1 over the passage. Judge: "
        f"{res['judge']}. Usable is the main eval's definition. Manifest: `openstax_manifest.json`.",
        "",
        f"Reference coverage: {s['reference']['n']} of {res['n_prompts']} prompts have a human-written "
        "book question from the same section; the reference row is over those prompts only.",
        "",
        "| | n | JSON | Structure | Key agreement | Aligned | Novel | Source copy | Usable |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| Book question (ceiling) | {s['reference']['n']} | | | {_p(s['reference']['key'])} | "
        f"{_p(s['reference']['aligned'])} | | | |",
    ]
    for a in (BASE, arm):
        r = s[a]
        lines.append(
            f"| {a} | {r['n']} | {_p(r['json'])} | {_p(r['structure'])} | {_p(r['key'])} | {_p(r['aligned'])} | "
            f"{_p(r['novel'])} | {_p(r['source_copy'])} | {_p(r['usable'])} |"
        )
    lines += [
        "",
        f"Usable by subject, {arm} minus {BASE}, paired bootstrap ({N_BOOT:,} resamples over prompts), 95% CI.",
        "",
        f"| Subject | n | {BASE} | {arm} | Difference | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for subj, r in res["by_subject"].items():
        d = r["diff"]
        lines.append(
            f"| {subj} | {r['n']} | {_p(r[f'{BASE}_usable'])} | {_p(r[f'{arm}_usable'])} | "
            f"{100 * d['diff']:+.1f} | [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] |"
        )
    lines += [
        "",
        f"Rejections (first failed check): {json.dumps(res['rejections'])}",
        "",
        f"Prompt build: {json.dumps(card['screen'])}",
        "",
    ]
    if verification:
        lines += [
            "## Verification (not a second run)",
            "",
            *(f"- {verification[k]}" for k in ("what", "embedding_cache", "cpu_rescore", "regeneration")),
            "",
        ]
    return "\n".join(lines)


def score(prompts_path: Path = PROMPTS, out: Path = OUT) -> dict:
    from eduai.cli import _novelty_index
    from eduai.curriculum.tagger import build_tagger
    from eduai.curriculum.taxonomy import default_taxonomy
    from eduai.data.sciq import read_jsonl
    from eduai.evaluation import compare

    prompts = read_jsonl(prompts_path)
    gen = out / "gen"
    arms = [a for a in compare.arms_present(gen) if a != BASE]
    if len(arms) != 1:
        raise SystemExit(f"expected {BASE} and one v2 arm in {gen}, found {arms}")
    res = compare.score(
        prompts,
        build_tagger(default_taxonomy()),
        _novelty_index(ROOT / "data"),
        data_dir=ROOT / "data",
        out_dir=gen,
    )
    per_item: dict[str, list[dict]] = {}
    for r in read_jsonl(gen / "per_item.jsonl"):
        per_item.setdefault(r["arm"], []).append(r)
    subject_of = {p["id"]: p["request"]["lo_id"].split(".")[0] for p in prompts}
    res["arm"] = arms[0]
    res["by_subject"] = by_subject(per_item, subject_of, arms[0])
    res["runs_of_this_check"] = 1
    card = json.loads((out / "prompts_card.json").read_text())
    (out / "eval.json").write_text(json.dumps(res, indent=2) + "\n")
    ver = out / "verification.json"
    (out / "report.md").write_text(render(res, card, json.loads(ver.read_text()) if ver.exists() else None))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["headline", "score"])
    ap.add_argument("--valid", type=Path, default=VALID)
    args = ap.parse_args()
    if args.cmd == "headline":
        print(headline(args.valid))
    else:
        score()
        print((OUT / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
