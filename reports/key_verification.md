# Post-hoc: answer-key verification (exploratory)

Pre-registered in `docs/key_verification.md` after the test audit, so this is post-hoc and exploratory. The registered eval result in `reports/eval_report.md` is unchanged.

Verifier: `qwen3.5-9b` (revision 938d891, thinking off, open-book, 4 rotations). An item is verified when the verifier's top option is the key and p(key) > 0.5. Verified usable = usable (the eval's criterion) and verified.

## Valid split

| Arm | n | Usable | Verified usable | Usable items that survive | Verified, of schema-valid |
|---|---|---|---|---|---|
| Base 3B, 2-shot | 150 | 32.0% (48) | 28.0% (42) | 87.5% | 76.0% of 125 |
| Base 3B + EduAI LoRA v2 | 150 | 38.0% (57) | 30.7% (46) | 80.7% | 67.8% of 149 |
| Base 3B + EduAI LoRA v2, 2-shot | 150 | 34.0% (51) | 28.7% (43) | 84.3% | 73.5% of 147 |

Paired bootstrap over prompts, verified usable (descriptive):

- finetuned-v2 - base-2shot: +2.7 points (95% CI -6.7 to +12.0)
- finetuned-v2-2shot - base-2shot: +0.7 points (95% CI -8.7 to +10.0)

## Test split

| Arm | n | Usable | Verified usable | Usable items that survive | Verified, of schema-valid |
|---|---|---|---|---|---|
| Base 3B, 2-shot | 150 | 34.7% (52) | 28.7% (43) | 82.7% | 81.0% of 121 |
| Base 3B + EduAI LoRA v2 | 150 | 32.0% (48) | 26.7% (40) | 83.3% | 68.0% of 150 |
| Base 3B + EduAI LoRA v2, 2-shot | 150 | 37.3% (56) | 34.0% (51) | 91.1% | 80.5% of 149 |

Paired bootstrap over prompts, verified usable (descriptive):

- finetuned-v2 - base-2shot: -2.0 points (95% CI -12.0 to +8.0)
- finetuned-v2-2shot - base-2shot: +5.3 points (95% CI -3.3 to +14.7)

## Independent check: Gemma 4 12B audit subset (test)

Gemma 4 12B (thinking on) judged 39 usable test items blind (items without a parsed verdict left out: 1). Its key verdicts, split by whether the item passes verification:

| Verification | Items | Key judged correct | 95% CI |
|---|---|---|---|
| Pass | 35 | 80.0% (28) | 64% to 90% |
| Fail | 4 | 50.0% (2) | 15% to 85% |
| All audited | 39 | 76.9% (30) | 62% to 87% |

Pass minus fail: +30.0 points (Fisher exact two-sided p = 0.223).

## Note: Qwen3.5 9B audit (same family as the verifier, circular)

On the 80 items of the Qwen audit, key judged correct was 69.2% of 65 verified items and 6.7% of 15 unverified ones. The verifier and this auditor are the same model family, so this isn't evidence that verification works.

The auditors are language models, not people; their verdicts are a second machine opinion.
