# Generation eval: base vs few-shot vs fine-tuned

The report covers 150 prompts from groups assigned to the held-out test split, including SciQ rows originally labeled train or valid. An independent labeling pass assigned each prompt a target learning objective without seeing tagger output (`data/gold/eval_lo_labels.jsonl`; AI-labeled, pending human spot-check). The eval dropped 59 off-curriculum candidates. The leakage filter against SFT train/valid rows screened 244 prompts and dropped 5 for >= 50% 8-gram passage containment and 9 for a same-answer question with Q+A cosine >= 0.88. Manifest: `eval_manifest.json`.

Every percentage uses all prompts as the denominator. Checks are scored independently:

- JSON: a JSON object could be extracted and parsed (after one retry at T=0.3 if the greedy output did not parse). First-try JSON is the greedy output alone.
- Structure: four distinct options, no all/none of the above, key not in the stem, the requested LO id and format, and the target misconception present as a wrong option when one was requested.
- Key agreement: a fixed open-book judge (llama-1b, Llama 3.2 1B 4-bit, not any arm's generator) picks the keyed option from A-D log-probabilities averaged over all four cyclic rotations of the options (so every option is scored in every position), with the source passage in its prompt.
- Aligned: the target LO is in the tagger's top 3 and its own score is at least tau. The tagger is a noisy filter (about 70% of in-curriculum gold items pass), so the reference row is the ceiling.
- Novel: stem cosine < 0.92 against the whole SciQ bank, excluding the prompt's own source item and its near-duplicate group. Closeness to the source is reported separately as source copy.
- Usable: all checks and not a copy of the source question. This is the bank-promotion criterion.
- Memorized: stem cosine >= 0.92 with an SFT training stem. Reported only; it doesn't reject items.

| | JSON | First-try JSON | Schema | Structure | Key agreement | Aligned | Novel | All checks | Source copy | Usable | Memorized (train) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SciQ reference item (ceiling) | | | | | 93.3% | 70.7% | | | | | |
| Base 3B, 0-shot | 100.0% | 99.3% | 93.3% | 40.7% | 70.7% | 74.7% | 92.7% | 26.7% | 4.7% | 23.3% | 0.7% |
| Base 3B, 2-shot | 100.0% | 98.0% | 80.7% | 60.7% | 60.7% | 64.0% | 80.0% | 36.7% | 2.7% | 34.7% | 0.0% |
| Base 3B + EduAI LoRA | 100.0% | 100.0% | 100.0% | 54.0% | 77.3% | 72.0% | 96.0% | 30.7% | 25.3% | 21.3% | 1.3% |

Paired bootstrap (5,000 resamples over prompts), difference in rate with 95% CI. With n = 150, one arm's rate has a standard error around 4 points, so differences under about 10 points should be read as noise unless the interval excludes zero.

| Comparison | Metric | Difference | 95% CI |
|---|---|---|---|
| finetuned - base-0shot | usable | -2.0% | [-12.0%, +7.3%] |
| finetuned - base-0shot | all_checks | +4.0% | [-5.3%, +14.0%] |
| finetuned - base-0shot | aligned | -2.7% | [-10.7%, +4.7%] |
| finetuned - base-0shot | key | +6.7% | [-3.3%, +16.7%] |
| finetuned - base-0shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned - base-2shot | usable | -13.3% | [-23.3%, -3.3%] |
| finetuned - base-2shot | all_checks | -6.0% | [-16.7%, +4.0%] |
| finetuned - base-2shot | aligned | +8.0% | [-0.7%, +17.3%] |
| finetuned - base-2shot | key | +16.7% | [+6.7%, +26.7%] |
| finetuned - base-2shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned - reference | aligned | +1.3% | [-5.3%, +8.0%] |
| finetuned - reference | key | -16.0% | [-23.3%, -8.7%] |
| base-2shot - base-0shot | usable | +11.3% | [+1.3%, +20.7%] |
| base-2shot - base-0shot | all_checks | +10.0% | [+0.7%, +19.3%] |
| base-2shot - base-0shot | aligned | -10.7% | [-18.7%, -2.7%] |
| base-2shot - base-0shot | key | -10.0% | [-20.0%, +0.0%] |
| base-2shot - base-0shot | json | +0.0% | [+0.0%, +0.0%] |

Secondary key agreement with the base 3B as judge (self-judged for the base arms, so biased in their favor):

| | Key agreement (3B judge) |
|---|---|
| SciQ reference item (ceiling) | 96.7% |
| Base 3B, 0-shot | 76.7% |
| Base 3B, 2-shot | 65.3% |
| Base 3B + EduAI LoRA | 84.0% |

First failing check per item (checks in validator order):

| Arm | accepted | duplicate | key_disagreement | not_aligned | schema | structure |
|---|---|---|---|---|---|---|
| Base 3B, 0-shot | 40 | 0 | 11 | 10 | 10 | 79 |
| Base 3B, 2-shot | 55 | 1 | 18 | 17 | 29 | 30 |
| Base 3B + EduAI LoRA | 46 | 3 | 11 | 21 | 0 | 69 |

Answer-key letter distribution (schema-valid items) and judge key agreement by keyed letter. A skewed key position is a generation defect in its own right; four-rotation judging removes the judge's own position bias from the comparison.

| Arm | A | B | C | D | Agree when key=A | B | C | D |
|---|---|---|---|---|---|---|---|---|
| SciQ reference item (ceiling) | 32 | 37 | 31 | 50 | 87.5% | 94.6% | 93.5% | 96.0% |
| Base 3B, 0-shot | 15 | 73 | 38 | 14 | 53.3% | 75.3% | 81.6% | 85.7% |
| Base 3B, 2-shot | 19 | 57 | 22 | 23 | 73.7% | 71.9% | 100.0% | 60.9% |
| Base 3B + EduAI LoRA | 104 | 28 | 9 | 9 | 79.8% | 75.0% | 88.9% | 44.4% |

Structure problems among schema-valid items (an item can have several):

| Arm | all four options identical | near-identical options | stem gives away the answer | stimulus format without a stimulus | target misconception missing | wrong lo_id |
|---|---|---|---|---|---|---|
| Base 3B, 0-shot | 0 | 52 | 1 | 0 | 32 | 13 |
| Base 3B, 2-shot | 0 | 11 | 11 | 1 | 11 | 0 |
| Base 3B + EduAI LoRA | 9 | 63 | 12 | 0 | 1 | 0 |

Generation speed (greedy, one request at a time, MLX on the M1 Pro, nothing else heavy running):

| Arm | Mean generation tok/s | Mean seconds per item | Peak memory (GB) |
|---|---|---|---|
| Base 3B, 0-shot | 80.3 | 3.0 | 2.50 |
| Base 3B, 2-shot | 76.1 | 3.8 | 2.74 |
| Base 3B + EduAI LoRA | 48.0 | 3.6 | 3.05 |
