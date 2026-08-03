"""reports/eval_report.md from reports/eval.json."""

from __future__ import annotations

LABELS = {
    "reference": "SciQ reference item (ceiling)",
    "base-0shot": "Base 3B, 0-shot",
    "base-2shot": "Base 3B, 2-shot",
    "finetuned": "Base 3B + EduAI LoRA v1",
    "finetuned-v2": "Base 3B + EduAI LoRA v2",
    "finetuned-v2-2shot": "Base 3B + EduAI LoRA v2, 2-shot",
}


def label(arm: str) -> str:
    return LABELS.get(arm, arm)


def _p(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def render(res: dict, manifest_name: str, card: dict) -> str:
    s = res["summary"]
    n = res["n_prompts"]
    lf = card["leakage_filter"]
    arms = [a for a in s if a != "reference"]
    if card.get("split") == "valid":
        intro = [
            "# Model-selection eval on the valid split",
            "",
            f"The report covers {n} prompts from groups assigned to the valid split that are not SFT valid rows. "
            "It is used for choosing training settings and adapters; the test prompts are not. Target "
            "objectives are the tagger's own labels (the test prompts use an independent labeling pass), so "
            "alignment here is easier than on the test set. The leakage filter against SFT train rows "
            f"screened {lf['screened']} prompts and dropped {lf['dropped_passage_containment']} for >= 50% "
            f"8-gram passage containment and {lf['dropped_same_answer_qa']} for a same-answer question with "
            f"Q+A cosine >= 0.88. Manifest: `{manifest_name}`.",
        ]
    else:
        intro = [
            "# Generation eval: base vs few-shot vs fine-tuned",
            "",
            f"The report covers {n} prompts from groups assigned to the held-out test split, including "
            "SciQ rows originally labeled train or valid. An independent labeling pass assigned each "
            "prompt a target learning objective without seeing tagger output "
            "(`data/gold/eval_lo_labels.jsonl`; AI-labeled, pending human spot-check). "
            f"The eval dropped {card['off_curriculum']} off-curriculum candidates. The leakage filter "
            f"against SFT train/valid rows screened {lf['screened']} prompts and dropped "
            f"{lf['dropped_passage_containment']} for >= 50% 8-gram passage containment and "
            f"{lf['dropped_same_answer_qa']} for a same-answer question with Q+A cosine >= 0.88. "
            f"Manifest: `{manifest_name}`.",
        ]
    L = [
        *intro,
        "",
        "Every percentage uses all prompts as the denominator. Checks are scored independently:",
        "",
        "- JSON: a JSON object could be extracted and parsed (after one retry at T=0.3 if the greedy output "
        "did not parse). First-try JSON is the greedy output alone.",
        "- Structure: four distinct options, no all/none of the above, key not in the stem, the requested LO id "
        "and format, and the target misconception present as a wrong option when one was requested.",
        f"- Key agreement: a fixed open-book judge ({res['judge']}, Llama 3.2 1B 4-bit, not any arm's generator) "
        "picks the keyed option from A-D log-probabilities averaged over all four cyclic rotations of the options "
        "(so every option is scored in every position), "
        "with the source passage in its prompt.",
        "- Aligned: the target LO is in the tagger's top 3 and its own score is at least tau. The tagger is a "
        "noisy filter (about 70% of in-curriculum gold items pass), so the reference row is the ceiling.",
        "- Novel: stem cosine < 0.92 against the whole SciQ bank, excluding the prompt's own source item and "
        "its near-duplicate group. Closeness to the source is reported separately as source copy.",
        "- Usable: all checks and not a copy of the source question. This is the bank-promotion criterion.",
        "- Memorized: stem cosine >= 0.92 with an SFT training stem. Reported only; it doesn't reject items.",
        "",
        "- Key agreement (valid): the same judge result over schema-valid items only, which removes the "
        "effect of JSON failures. Items whose key text also appears as a distractor never count as agreeing.",
        "",
        "| | JSON | First-try JSON | Schema | Structure | Key agreement | Key agreement (valid) | Aligned | "
        "Novel | All checks | Source copy | Usable | Memorized (train) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ref = s["reference"]
    L.append(
        f"| {LABELS['reference']} | | | | | {_p(ref['key'])} | {_p(ref['key'])} | {_p(ref['aligned'])} | | | | | |"
    )
    for arm in arms:
        r = s[arm]
        L.append(
            f"| {label(arm)} | {_p(r['json'])} | {_p(r['first_try_json'])} | {_p(r['schema'])} | "
            f"{_p(r['structure'])} | {_p(r['key'])} | {_p(r['key_on_valid'])} | {_p(r['aligned'])} | {_p(r['novel'])} | "
            f"{_p(r['all_checks'])} | {_p(r['source_copy'])} | {_p(r['usable'])} | {_p(r['memorized'])} |"
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
    if any(s[a].get("key_secondary") is not None for a in ("reference", *arms)):
        L += [
            "",
            "Secondary key agreement with the base 3B as judge (self-judged for the base arms, so biased in "
            "their favor):",
            "",
            "| | Key agreement (3B judge) |",
            "|---|---|",
        ]
        for arm in ("reference", *arms):
            L.append(f"| {label(arm)} | {_p(s[arm].get('key_secondary'))} |")
    L += [
        "",
        "First failing check per item (checks in validator order):",
        "",
        "| Arm | " + " | ".join(sorted({k for v in res["rejections"].values() for k in v})) + " |",
    ]
    cols = sorted({k for v in res["rejections"].values() for k in v})
    L.append("|---|" + "---|" * len(cols))
    for arm, hist in res["rejections"].items():
        L.append(f"| {label(arm)} | " + " | ".join(str(hist.get(c, 0)) for c in cols) + " |")
    L += [
        "",
        "Answer-key letter distribution (schema-valid items) and judge key agreement by keyed letter. A "
        "skewed key position is a generation defect in its own right; four-rotation judging removes the "
        "judge's own position bias from the comparison.",
        "",
        "| Arm | A | B | C | D | Agree when key=A | B | C | D |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("reference", *arms):
        kl, ka = s[arm]["key_letters"], s[arm]["key_agreement_by_letter"]
        L.append(
            f"| {label(arm)} | "
            + " | ".join(str(kl[x]) for x in "ABCD")
            + " | "
            + " | ".join(_p(ka[x]) for x in "ABCD")
            + " |"
        )
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
            L.append(f"| {label(arm)} | " + " | ".join(str(c.get(k, 0)) for k in kinds) + " |")
    if card.get("split") == "valid":
        load = (
            "Generation speed (greedy, one request at a time, MLX on the M1 Pro). Generation held the "
            "compute lease, so no other training or benchmark job ran at the same time, but the machine "
            "wasn't idle; host load for each run is in the `*_manifest.json` files next to this report. "
            "Treat these speeds, and the gap between the adapter and base arms, as rough."
        )
    else:
        load = (
            "Generation speed (greedy, one request at a time, MLX on the M1 Pro). No other training or benchmark "
            "job ran at the same time, but the machine wasn't idle: `eval_manifest.json` records a load average "
            "of 7 to 9.5, 6 running containers and about 12 GB of swap in use. Treat these speeds, and the gap "
            "between the adapter and base arms, as rough."
        )
    L += [
        "",
        load,
        "",
        "| Arm | Mean generation tok/s | Mean seconds per item | Peak memory (GB) |",
        "|---|---|---|---|",
    ]
    for arm, t in res["timing"].items():
        tps = "n/a" if t["mean_generation_tps"] is None else f"{t['mean_generation_tps']:.1f}"
        sec = "n/a" if t["mean_seconds_per_item"] is None else f"{t['mean_seconds_per_item']:.1f}"
        mem = "n/a" if t["peak_memory_gb"] is None else f"{t['peak_memory_gb']:.2f}"
        L.append(f"| {label(arm)} | {tps} | {sec} | {mem} |")
    L.append("")
    return "\n".join(L)
