"""Write the README "Generation eval" section from reports/eval.json (between the eval markers)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- eval:start -->", "<!-- eval:end -->"
ARMS = (
    ("Base 3B, 0-shot", "base-0shot"),
    ("Base 3B, 2-shot", "base-2shot"),
    ("Base 3B + EduAI LoRA v1", "finetuned"),
    ("Base 3B + EduAI LoRA v2", "finetuned-v2"),
    ("Base 3B + EduAI LoRA v2, 2-shot", "finetuned-v2-2shot"),
)


def pct(x: float) -> str:
    return f"{x:.1%}"


def ci(v: dict) -> str:
    return f"{v['diff'] * 100:+.1f} points (95% CI {v['ci95'][0] * 100:+.1f} to {v['ci95'][1] * 100:+.1f})"


def render(r: dict, card: dict) -> str:
    s, b, t = r["summary"], r["bootstrap"], r["timing"]
    lf = card["leakage_filter"]
    ft = s["finetuned"]
    sp = r["structure_problems"]["finetuned"]
    arms = [(label, a) for label, a in ARMS if a in s]
    L = [
        f"There are {r['n_prompts']} prompts from groups assigned to the held-out test split "
        "(including SciQ rows originally labeled train or valid), and every arm gets the same "
        "prompts ([reports/eval_report.md](reports/eval_report.md), raw generations in `reports/eval/`). "
        "This section is written by `scripts/readme_eval.py`.",
        "",
        "- **Target objectives are independent.** Each prompt's target LO comes from an independent labeling "
        f"pass, not from the tagger. {card['off_curriculum']} off-curriculum candidates were dropped.",
        "- **Leaky prompts are filtered out.** A prompt was dropped if its passage shares 50% or more of its "
        "8-grams with a training passage, or if a training item has the same answer and a question-plus-answer "
        f"cosine of 0.88 or more. That removed {lf['dropped_passage_containment'] + lf['dropped_same_answer_qa']} "
        f"of {lf['screened']} screened candidates.",
        "- **The judge is fixed, open-book, and not one of the generators.** Llama 3.2 1B sees the passage and "
        "scores the options by log-probability, averaged over all four rotations of the options.",
        "- **Novelty excludes the prompt's own source item and its group.** Copies of the source question are "
        "counted separately. **Usable** means passing all checks and not being a source copy; it is the "
        "criterion for promoting an item into the bank.",
        "- **Memorization is reported only.** It counts items within 0.92 cosine of an SFT training stem, and "
        "it does not reject anything.",
        "",
        "| Arm | Schema valid | Structure | Key agreement (schema-valid items) | Aligned | Novel | All checks | Source copy | Usable | Gen tok/s |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| SciQ reference item (ceiling) | | | {pct(s['reference']['key'])} | {pct(s['reference']['aligned'])} | | | | | |",
    ]
    for label, a in arms:
        x = s[a]
        L.append(
            f"| {label} | {pct(x['schema'])} | {pct(x['structure'])} | {pct(x['key_on_valid'])} | {pct(x['aligned'])} | "
            f"{pct(x['novel'])} | {pct(x['all_checks'])} | {pct(x['source_copy'])} | {pct(x['usable'])} | "
            f"{t[a]['mean_generation_tps']:.1f} |"
        )
    kl = ft["key_letters"]
    n_valid = sum(kl.values())
    L += [
        "",
        "The v1 fine-tuned model does not generate better questions overall. The paired bootstrap over prompts "
        "gives these differences:",
        "",
        f"- v1 fine-tuned vs 2-shot on usable items: {ci(b['finetuned - base-2shot | usable'])}.",
        f"- v1 fine-tuned vs 0-shot on usable items: {ci(b['finetuned - base-0shot | usable'])}.",
        f"- 2-shot vs 0-shot on usable items: {ci(b['base-2shot - base-0shot | usable'])}.",
        "",
        "v1 fine-tuning fixed the output format:",
        "",
        f"- It produced schema-valid JSON on {pct(ft['schema'])} of prompts, against {pct(s['base-2shot']['schema'])} "
        "for 2-shot.",
        "- It almost never drops the requested misconception.",
        "",
        f"Key agreement on schema-valid items is {pct(ft['key_on_valid'])} for the fine-tune, "
        f"{pct(s['base-2shot']['key_on_valid'])} for 2-shot and {pct(s['base-0shot']['key_on_valid'])} for 0-shot. "
        "An item whose key text is repeated among its options counts as not agreeing, and many of the "
        "fine-tune's items have repeated options.",
        "",
        "It also learned the wrong things from its targets, which were the SciQ source questions:",
        "",
        f"- It copies the source question {pct(ft['source_copy'])} of the time.",
        f"- It writes near-identical options on {sp.get('near-identical options', 0)} of {r['n_prompts']} items, "
        f"including {sp.get('all four options identical', 0)} where all four options are the same string.",
        f"- It puts the key at A on {kl['A']} of {n_valid} valid items, even though the SFT answer "
        "letters were exactly 25% each.",
        "",
        "Before items enter the bank, their options are reshuffled and the key is remapped. "
        f"Alignment is at the reference ceiling ({pct(s['reference']['aligned'])}) for 0-shot and fine-tuned; "
        f"2-shot is lower at {pct(s['base-2shot']['aligned'])}. Generation with the unfused adapter ran at "
        f"{t['finetuned']['mean_generation_tps']:.0f} tok/s, against {t['base-0shot']['mean_generation_tps']:.0f} "
        "for the base model, on a machine with other background load (see the eval manifest), so treat the "
        "speeds as rough.",
        "",
        "In short, the v1 LoRA fine-tune of Llama 3.2 3B taught format reliability, but the 3,000 SciQ-derived "
        "targets also taught copying and a key-position bias.",
    ]
    if "finetuned-v2" in s:
        L += ["", *v2_section(r)]
    return "\n".join(L)


def v2_section(r: dict) -> list[str]:
    s, b = r["summary"], r["bootstrap"]
    v2, v2s, b2, v1 = s["finetuned-v2"], s["finetuned-v2-2shot"], s["base-2shot"], s["finetuned"]
    sp = r["structure_problems"]["finetuned-v2"]
    head = b["finetuned-v2 - base-2shot | usable"]
    won = head["ci95"][0] > 0
    sens = r["v2_sensitivity"]
    runs = r["test_runs"]["started"]
    kl = v2["key_letters"]
    return [
        "### v2 adapter",
        "",
        "The v2 LoRA was trained on items the base 3B wrote itself: samples drawn with the eval's two fixed "
        "examples, kept only when they passed the checks above with the base 3B as the key judge, one per "
        "training prompt (1,606 train rows; [reports/rft_card.json](reports/rft_card.json)). The checkpoint, and "
        "the choice between v2 with and without the two examples, were made on the valid split and written down "
        "before the test run ([docs/v2_selection.md](docs/v2_selection.md)). "
        f"The v2 arms were run on the test set {'once' if runs == 1 else f'{runs} times'}. The base and v1 rows above "
        "are the committed ones from before v2.",
        "",
        f"- Pre-registered headline, v2 0-shot vs 2-shot base on usable items: {ci(head)}. "
        + (
            "The interval excludes zero, so v2 beat 2-shot prompting."
            if won
            else "The interval includes zero, so v2 did not beat 2-shot prompting of the base model."
        ),
        f"- Without the {len(sens['excluded'])} test prompts whose passages are also v2 training passages "
        f"({', '.join(sens['excluded'])}): {ci(sens['bootstrap']['finetuned-v2 - base-2shot | usable'])}.",
        f"- v2 with the two fixed examples ({pct(v2s['usable'])} usable) vs 2-shot base: {ci(b['finetuned-v2-2shot - base-2shot | usable'])}. "
        "This arm scored higher than v2 0-shot on test but lower on valid, where the choice was made, so it "
        "isn't the headline.",
        f"- v2 vs v1 on usable items: {ci(b['finetuned-v2 - finetuned | usable'])}.",
        "",
        f"Compared with v1, v2 copies the source question less ({pct(v2['source_copy'])} against "
        f"{pct(v1['source_copy'])}), puts the key at A less often ({kl['A']} of {sum(kl.values())} valid items, against "
        f"{v1['key_letters']['A']}), "
        f"and writes near-identical options on {sp.get('near-identical options', 0)} items instead of "
        f"{r['structure_problems']['finetuned'].get('near-identical options', 0)}. It keeps v1's schema-valid "
        f"rate ({pct(v2['schema'])}). Its key agreement on schema-valid items ({pct(v2['key_on_valid'])}) is about "
        f"the same as 2-shot base ({pct(b2['key_on_valid'])}). Its most common structure problem is a stem that "
        f"gives away the answer ({sp.get('stem gives away the answer', 0)} items, against "
        f"{r['structure_problems']['base-2shot'].get('stem gives away the answer', 0)} for 2-shot base).",
        "",
        "Usable overstates v2 more than the other arms. v2's targets were picked with these same checks and "
        "with the base 3B as key judge, and the 3B agrees closely with the 1B eval judge. The manual blind "
        "audit planned in the protocol was not done. Instead, two LLM judges from other model families "
        "audited a sample of usable test items blind (see the blind audit below). No person has checked "
        "these items.",
    ]


def main() -> None:
    r = json.loads((ROOT / "reports" / "eval.json").read_text())
    card = json.loads((ROOT / "reports" / "eval_prompts_card.json").read_text())
    readme = (ROOT / "README.md").read_text()
    a, b = readme.index(START) + len(START), readme.index(END)
    (ROOT / "README.md").write_text(readme[:a] + "\n" + render(r, card) + "\n" + readme[b:])


if __name__ == "__main__":
    main()
