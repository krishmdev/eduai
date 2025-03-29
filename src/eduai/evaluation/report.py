"""reports/eval_report.md from reports/eval.json."""

from __future__ import annotations

LABELS = {
    "reference": "SciQ reference item (ceiling)",
    "base-0shot": "Base 3B, 0-shot",
    "base-2shot": "Base 3B, 2-shot",
    "finetuned": "Base 3B + EduAI LoRA",
}


def _p(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def render(res: dict, manifest_name: str, card: dict) -> str:
    s = res["summary"]
    n = res["n_prompts"]
    lf = card["leakage_filter"]
    L = [
        "# Generation eval: base vs few-shot vs fine-tuned",
        "",
        f"{n} held-out prompts from the SciQ test split, each with a target learning objective assigned by an "
        "independent labeling pass that never saw tagger output (`data/gold/eval_lo_labels.jsonl`; AI-labeled, "
        f"pending human spot-check). {card['off_curriculum']} candidates labeled off-curriculum were dropped. "
        f"Leakage filter against the SFT train/valid rows: screened {lf['screened']}, dropped "
        f"{lf['dropped_passage_containment']} for >= 50% 8-gram passage containment and "
        f"{lf['dropped_same_answer_qa']} for a same-answer question with Q+A cosine >= 0.88. "
        f"Manifest: `{manifest_name}`.",
        "",
        "Every percentage uses all prompts as the denominator. Checks are scored independently:",
        "",
        "- JSON: a JSON object could be extracted and parsed (after one retry at T=0.3 if the greedy output "
        "did not parse). First-try JSON is the greedy output alone.",
        "- Structure: four distinct options, no all/none of the above, key not in the stem, the requested LO id "
        "and format, and the target misconception present as a wrong option when one was requested.",
        f"- Key agreement: a fixed open-book judge ({res['judge']}, Llama 3.2 1B 4-bit, not any arm's generator) "
        "picks the keyed option from A-D log-probabilities averaged over two cyclic rotations of the options, "
        "with the source passage in its prompt.",
        "- Aligned: the target LO is in the tagger's top 3 and its own score is at least tau. The tagger is a "
        "noisy filter (about 70% of in-curriculum gold items pass), so the reference row is the ceiling.",
        "- Novel: stem cosine < 0.92 against the whole SciQ bank, excluding the prompt's own source item and "
        "its near-duplicate group. Closeness to the source is reported separately as source copy.",
        "",
        "| | JSON | First-try JSON | Schema | Structure | Key agreement | Aligned | Novel | All checks | "
        "Source copy | Memorized (train) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ref = s["reference"]
    L.append(f"| {LABELS['reference']} | | | | | {_p(ref['key'])} | {_p(ref['aligned'])} | | | | |")
    for arm in ("base-0shot", "base-2shot", "finetuned"):
        if arm not in s:
            continue
        r = s[arm]
        L.append(
            f"| {LABELS[arm]} | {_p(r['json'])} | {_p(r['first_try_json'])} | {_p(r['schema'])} | "
            f"{_p(r['structure'])} | {_p(r['key'])} | {_p(r['aligned'])} | {_p(r['novel'])} | "
            f"{_p(r['all_checks'])} | {_p(r['source_copy'])} | {_p(r['memorized'])} |"
        )
    L += [
        "",
        "Paired bootstrap (5,000 resamples over prompts), difference in rate with 95% CI. With n = "
        f"{n}, one arm's rate has a standard error around 4 points, so differences under about 10 points "
        "should be read as noise unless the interval excludes zero.",
        "",
        "| Comparison | Metric | Difference | 95% CI |",
        "|---|---|---|---|",
    ]
    for k, v in res["bootstrap"].items():
        pair, metric = k.split(" | ")
        L.append(f"| {pair} | {metric} | {v['diff']:+.1%} | [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}] |")
    L += [
        "",
        "Secondary key agreement with the base 3B as judge (self-judged for the base arms, so biased in "
        "their favor):",
        "",
        "| | Key agreement (3B judge) |",
        "|---|---|",
    ]
    for arm in ("reference", "base-0shot", "base-2shot", "finetuned"):
        if arm in s:
            L.append(f"| {LABELS[arm]} | {_p(s[arm].get('key_secondary'))} |")
    L += [
        "",
        "First failing check per item (checks in validator order):",
        "",
        "| Arm | " + " | ".join(sorted({k for v in res["rejections"].values() for k in v})) + " |",
    ]
    cols = sorted({k for v in res["rejections"].values() for k in v})
    L.append("|---|" + "---|" * len(cols))
    for arm, hist in res["rejections"].items():
        L.append(f"| {LABELS[arm]} | " + " | ".join(str(hist.get(c, 0)) for c in cols) + " |")
    if res.get("structure_problems"):
        kinds = sorted({k for v in res["structure_problems"].values() for k in v})
        L += [
            "",
            "Structure problems among schema-valid items (an item can have several):",
            "",
            "| Arm | " + " | ".join(kinds) + " |",
            "|---|" + "---|" * len(kinds),
        ]
        for arm, c in res["structure_problems"].items():
            L.append(f"| {LABELS[arm]} | " + " | ".join(str(c.get(k, 0)) for k in kinds) + " |")
    L += [
        "",
        "Generation speed (greedy, one request at a time, MLX on the M1 Pro, under the compute lease):",
        "",
        "| Arm | Mean generation tok/s | Mean seconds per item | Peak memory (GB) |",
        "|---|---|---|---|",
    ]
    for arm, t in res["timing"].items():
        tps = "n/a" if t["mean_generation_tps"] is None else f"{t['mean_generation_tps']:.1f}"
        sec = "n/a" if t["mean_seconds_per_item"] is None else f"{t['mean_seconds_per_item']:.1f}"
        mem = "n/a" if t["peak_memory_gb"] is None else f"{t['peak_memory_gb']:.2f}"
        L.append(f"| {LABELS[arm]} | {tps} | {sec} | {mem} |")
    L.append("")
    return "\n".join(L)
