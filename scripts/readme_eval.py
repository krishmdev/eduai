"""Write the README "Generation eval" section from reports/eval.json (between the eval markers)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- eval:start -->", "<!-- eval:end -->"
ARMS = (
    ("Base 3B, 0-shot", "base-0shot"),
    ("Base 3B, 2-shot", "base-2shot"),
    ("Base 3B + EduAI LoRA", "finetuned"),
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
    L = [
        f"There are {r['n_prompts']} prompts from groups assigned to the held-out test split "
        "(including SciQ rows originally labeled train or valid), and all three arms get the same "
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
        "| Arm | Schema valid | Structure | Key agreement | Aligned | Novel | All checks | Source copy | Usable | Gen tok/s |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| SciQ reference item (ceiling) | | | {pct(s['reference']['key'])} | {pct(s['reference']['aligned'])} | | | | | |",
    ]
    for label, a in ARMS:
        x = s[a]
        L.append(
            f"| {label} | {pct(x['schema'])} | {pct(x['structure'])} | {pct(x['key'])} | {pct(x['aligned'])} | "
            f"{pct(x['novel'])} | {pct(x['all_checks'])} | {pct(x['source_copy'])} | {pct(x['usable'])} | "
            f"{t[a]['mean_generation_tps']:.1f} |"
        )
    kl = ft["key_letters"]
    n_valid = sum(kl.values())
    L += [
        "",
        "The fine-tuned model does not generate better questions overall. The paired bootstrap over prompts "
        "gives these differences:",
        "",
        f"- Fine-tuned vs 2-shot on usable items: {ci(b['finetuned - base-2shot | usable'])}.",
        f"- Fine-tuned vs 0-shot on usable items: {ci(b['finetuned - base-0shot | usable'])}.",
        f"- 2-shot vs 0-shot on usable items: {ci(b['base-2shot - base-0shot | usable'])}.",
        "",
        "Fine-tuning fixed the output format:",
        "",
        f"- It produced schema-valid JSON on {pct(ft['schema'])} of prompts, against {pct(s['base-2shot']['schema'])} "
        "for 2-shot.",
        "- It almost never drops the requested misconception.",
        f"- Its keys agree with the judge more often: {ci(b['finetuned - base-2shot | key'])} vs 2-shot.",
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
        "for the base model.",
        "",
        "In short, the LoRA fine-tune taught format reliability, but the 3,000 SciQ-derived targets also taught "
        "copying and a key-position bias. With these data, 2-shot prompting of the base model produces the "
        "most usable items.",
    ]
    return "\n".join(L)


def main() -> None:
    r = json.loads((ROOT / "reports" / "eval.json").read_text())
    card = json.loads((ROOT / "reports" / "eval_prompts_card.json").read_text())
    readme = (ROOT / "README.md").read_text()
    a, b = readme.index(START) + len(START), readme.index(END)
    (ROOT / "README.md").write_text(readme[:a] + "\n" + render(r, card) + "\n" + readme[b:])


if __name__ == "__main__":
    main()
