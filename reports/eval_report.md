# Generation eval: base vs few-shot vs fine-tuned

The report covers 150 prompts from groups assigned to the held-out test split, including SciQ rows originally labeled train or valid. An independent labeling pass assigned each prompt a target learning objective without seeing tagger output (`data/gold/eval_lo_labels.jsonl`; AI-labeled, pending human spot-check). The eval dropped 59 off-curriculum candidates. The leakage filter against SFT train/valid rows screened 244 prompts and dropped 5 for >= 50% 8-gram passage containment and 9 for a same-answer question with Q+A cosine >= 0.88. Manifest: `eval_manifest.json`.

Test runs for v2: 1. Every run of scripts/run_eval.sh on the test prompts is logged in `reports/eval/test_runs.jsonl` before it starts; the v1 arms were run before that log existed.

Every percentage uses all prompts as the denominator. Checks are scored independently:

- JSON: a JSON object could be extracted and parsed (after one retry at T=0.3 if the greedy output did not parse). First-try JSON is the greedy output alone.
- Structure: four distinct options, no all/none of the above, key not in the stem, the requested LO id and format, and the target misconception present as a wrong option when one was requested.
- Key agreement: a fixed open-book judge (llama-1b, Llama 3.2 1B 4-bit, not any arm's generator) picks the keyed option from A-D log-probabilities averaged over all four cyclic rotations of the options (so every option is scored in every position), with the source passage in its prompt.
- Aligned: the target LO is in the tagger's top 3 and its own score is at least tau. The tagger is a noisy filter (about 70% of in-curriculum gold items pass), so the reference row is the ceiling.
- Novel: stem cosine < 0.92 against the whole SciQ bank, excluding the prompt's own source item and its near-duplicate group. Closeness to the source is reported separately as source copy.
- Usable: all checks and not a copy of the source question. This is the bank-promotion criterion.
- Memorized: stem cosine >= 0.92 with an SFT training stem. Reported only; it doesn't reject items.
- v2 caveat: the v2 adapters' training targets are base 3B samples kept only when they passed these same checks (structure, tagger alignment, novelty, not a source copy), with the base 3B as the key judge. Usable is partly what v2 was trained to pass, so it overstates v2 more than the other arms. The only planned check on that is an automated audit by two other LLM judges (Qwen3.5-9B and Gemma 4 12B); no person has rated these items. Memorized is measured against the v1 SFT train stems, which the v2 targets were screened against at build time.

- Key agreement (valid): the same judge result over schema-valid items only, which removes the effect of JSON failures. Items whose key text also appears as a distractor never count as agreeing.

| | JSON | First-try JSON | Schema | Structure | Key agreement | Key agreement (valid) | Aligned | Novel | All checks | Source copy | Usable | Memorized (train) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SciQ reference item (ceiling) | | | | | 93.3% | 93.3% | 70.7% | | | | | |
| Base 3B, 0-shot | 100.0% | 99.3% | 93.3% | 40.7% | 70.7% | 75.7% | 74.7% | 92.7% | 26.7% | 4.7% | 23.3% | 0.7% |
| Base 3B, 2-shot | 100.0% | 98.0% | 80.7% | 60.7% | 60.7% | 75.2% | 64.0% | 80.0% | 36.7% | 2.7% | 34.7% | 0.0% |
| Base 3B + EduAI LoRA v1 | 100.0% | 100.0% | 100.0% | 54.0% | 66.0% | 66.0% | 72.0% | 96.0% | 30.7% | 25.3% | 21.3% | 1.3% |
| Base 3B + EduAI LoRA v2 | 100.0% | 100.0% | 100.0% | 66.0% | 75.3% | 75.3% | 79.3% | 99.3% | 38.0% | 8.7% | 32.0% | 0.0% |
| Base 3B + EduAI LoRA v2, 2-shot | 99.3% | 99.3% | 99.3% | 73.3% | 76.7% | 77.2% | 78.0% | 99.3% | 44.0% | 9.3% | 37.3% | 0.0% |

Paired bootstrap (5,000 resamples over prompts), difference in rate with 95% CI. With n = 150, one arm's rate has a standard error around 4 points, so differences under about 10 points should be read as noise unless the interval excludes zero.

The key and aligned differences count every prompt, and an item that isn't schema-valid counts as neither agreeing nor aligned. So a difference against an arm with many schema failures (base-0shot, 93.3% schema-valid; base-2shot, 80.7% schema-valid) partly reflects those failures rather than the keys or the objectives. The key agreement (valid) column above compares keys on schema-valid items only.

| Comparison | Metric | Difference | 95% CI |
|---|---|---|---|
| finetuned - base-0shot | usable | -2.0% | [-12.0%, +7.3%] |
| finetuned - base-0shot | all_checks | +4.0% | [-5.3%, +14.0%] |
| finetuned - base-0shot | aligned | -2.7% | [-10.7%, +4.7%] |
| finetuned - base-0shot | key | -4.7% | [-15.3%, +6.7%] |
| finetuned - base-0shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned - base-2shot | usable | -13.3% | [-23.3%, -3.3%] |
| finetuned - base-2shot | all_checks | -6.0% | [-16.7%, +4.0%] |
| finetuned - base-2shot | aligned | +8.0% | [-0.7%, +17.3%] |
| finetuned - base-2shot | key | +5.3% | [-6.0%, +16.7%] |
| finetuned - base-2shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned - reference | aligned | +1.3% | [-5.3%, +8.0%] |
| finetuned - reference | key | -27.3% | [-36.0%, -19.3%] |
| base-2shot - base-0shot | usable | +11.3% | [+1.3%, +20.7%] |
| base-2shot - base-0shot | all_checks | +10.0% | [+0.7%, +19.3%] |
| base-2shot - base-0shot | aligned | -10.7% | [-18.7%, -2.7%] |
| base-2shot - base-0shot | key | -10.0% | [-20.0%, +0.0%] |
| base-2shot - base-0shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned-v2 - base-0shot | usable | +8.7% | [-1.3%, +18.7%] |
| finetuned-v2 - base-0shot | all_checks | +11.3% | [+1.3%, +20.7%] |
| finetuned-v2 - base-0shot | aligned | +4.7% | [-2.7%, +12.0%] |
| finetuned-v2 - base-0shot | key | +4.7% | [-4.7%, +14.0%] |
| finetuned-v2 - base-0shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned-v2 - base-2shot | usable | -2.7% | [-13.3%, +8.0%] |
| finetuned-v2 - base-2shot | all_checks | +1.3% | [-9.3%, +12.0%] |
| finetuned-v2 - base-2shot | aligned | +15.3% | [+6.7%, +24.0%] |
| finetuned-v2 - base-2shot | key | +14.7% | [+4.7%, +24.7%] |
| finetuned-v2 - base-2shot | json | +0.0% | [+0.0%, +0.0%] |
| finetuned-v2 - finetuned | usable | +10.7% | [+1.3%, +20.0%] |
| finetuned-v2 - finetuned | all_checks | +7.3% | [-2.7%, +17.3%] |
| finetuned-v2 - finetuned | aligned | +7.3% | [+0.0%, +14.0%] |
| finetuned-v2 - finetuned | key | +9.3% | [+0.0%, +18.7%] |
| finetuned-v2 - finetuned | json | +0.0% | [+0.0%, +0.0%] |
| finetuned-v2 - reference | aligned | +8.7% | [+0.7%, +17.3%] |
| finetuned-v2 - reference | key | -18.0% | [-25.3%, -10.0%] |
| finetuned-v2-2shot - base-2shot | usable | +2.7% | [-6.7%, +12.0%] |
| finetuned-v2-2shot - base-2shot | all_checks | +7.3% | [-2.0%, +16.0%] |
| finetuned-v2-2shot - base-2shot | aligned | +14.0% | [+6.7%, +21.3%] |
| finetuned-v2-2shot - base-2shot | key | +16.0% | [+6.7%, +25.3%] |
| finetuned-v2-2shot - base-2shot | json | -0.7% | [-2.0%, +0.0%] |
| finetuned-v2-2shot - finetuned-v2 | usable | +5.3% | [-4.7%, +16.0%] |
| finetuned-v2-2shot - finetuned-v2 | all_checks | +6.0% | [-4.0%, +16.0%] |
| finetuned-v2-2shot - finetuned-v2 | aligned | -1.3% | [-8.0%, +5.3%] |
| finetuned-v2-2shot - finetuned-v2 | key | +1.3% | [-8.7%, +11.3%] |
| finetuned-v2-2shot - finetuned-v2 | json | -0.7% | [-2.0%, +0.0%] |

Sensitivity (pre-registered): the same usable differences on the 148 prompts left after dropping test-00916, train-02955, whose passages are also v2 training passages.

| Comparison | Difference | 95% CI |
|---|---|---|
| finetuned-v2 - base-2shot | -4.1% | [-14.9%, +6.8%] |
| finetuned-v2-2shot - base-2shot | +2.7% | [-6.8%, +12.2%] |

Exploratory, not used to judge v2: usable rate split by whether the tagger aligns the SciQ reference item with the target objective (reference aligned / not aligned; n = 106 / 44).

| Arm | Reference aligned | Reference not aligned |
|---|---|---|
| Base 3B, 0-shot | 26.4% | 15.9% |
| Base 3B, 2-shot | 39.6% | 22.7% |
| Base 3B + EduAI LoRA v1 | 21.7% | 20.5% |
| Base 3B + EduAI LoRA v2 | 33.0% | 29.5% |
| Base 3B + EduAI LoRA v2, 2-shot | 40.6% | 29.5% |
| finetuned-v2 - base-2shot | -6.6% [-18.9%, +6.6%] | +6.8% [-11.4%, +27.3%] |
| finetuned-v2-2shot - base-2shot | +0.9% [-10.4%, +12.3%] | +6.8% [-11.4%, +25.0%] |

Secondary key agreement with the base 3B as judge (self-judged for the base arms, so biased in their favor; the 3B also picked the v2 training data, so it favors v2 too):

| | Key agreement (3B judge) |
|---|---|
| SciQ reference item (ceiling) | 96.7% |
| Base 3B, 0-shot | 75.3% |
| Base 3B, 2-shot | 65.3% |
| Base 3B + EduAI LoRA v1 | 71.3% |
| Base 3B + EduAI LoRA v2 | 78.0% |
| Base 3B + EduAI LoRA v2, 2-shot | 82.7% |

First failing check per item (checks in validator order):

| Arm | accepted | bad_json | duplicate | key_disagreement | not_aligned | schema | structure |
|---|---|---|---|---|---|---|---|
| Base 3B, 0-shot | 40 | 0 | 0 | 11 | 10 | 10 | 79 |
| Base 3B, 2-shot | 55 | 0 | 1 | 18 | 17 | 29 | 30 |
| Base 3B + EduAI LoRA v1 | 46 | 0 | 3 | 11 | 21 | 0 | 69 |
| Base 3B + EduAI LoRA v2 | 57 | 0 | 0 | 23 | 19 | 0 | 51 |
| Base 3B + EduAI LoRA v2, 2-shot | 66 | 1 | 0 | 27 | 17 | 0 | 39 |

Answer-key letter distribution (schema-valid items) and judge key agreement by keyed letter. A skewed key position is a generation defect in its own right; four-rotation judging removes the judge's own position bias from the comparison.

| Arm | A | B | C | D | Agree when key=A | B | C | D |
|---|---|---|---|---|---|---|---|---|
| SciQ reference item (ceiling) | 32 | 37 | 31 | 50 | 87.5% | 94.6% | 93.5% | 96.0% |
| Base 3B, 0-shot | 15 | 73 | 38 | 14 | 53.3% | 75.3% | 81.6% | 85.7% |
| Base 3B, 2-shot | 19 | 57 | 22 | 23 | 73.7% | 71.9% | 100.0% | 60.9% |
| Base 3B + EduAI LoRA v1 | 104 | 28 | 9 | 9 | 67.3% | 67.9% | 66.7% | 44.4% |
| Base 3B + EduAI LoRA v2 | 61 | 44 | 30 | 15 | 73.8% | 68.2% | 90.0% | 73.3% |
| Base 3B + EduAI LoRA v2, 2-shot | 33 | 52 | 23 | 41 | 57.6% | 84.6% | 87.0% | 78.0% |

Structure problems among schema-valid items (an item can have several):

| Arm | all four options identical | near-identical options | stem gives away the answer | stimulus format without a stimulus | target misconception missing | wrong lo_id |
|---|---|---|---|---|---|---|
| Base 3B, 0-shot | 0 | 52 | 1 | 0 | 32 | 13 |
| Base 3B, 2-shot | 0 | 11 | 11 | 1 | 11 | 0 |
| Base 3B + EduAI LoRA v1 | 9 | 63 | 12 | 0 | 1 | 0 |
| Base 3B + EduAI LoRA v2 | 2 | 22 | 27 | 0 | 6 | 0 |
| Base 3B + EduAI LoRA v2, 2-shot | 1 | 9 | 29 | 0 | 2 | 0 |

Generation speed (greedy, one request at a time, MLX on the M1 Pro). No other training or benchmark job ran at the same time, but the machine wasn't idle: `eval_manifest.json` records a load average of 7 to 9.5, 6 running containers and about 12 GB of swap in use. Treat these speeds, and the gap between the adapter and base arms, as rough. The v2 arms were generated later, under the compute lease; `v2_test_manifest.json` records a load average of about 4.3 and no running containers, so their speeds aren't directly comparable with the v1 arms' either.

| Arm | Mean generation tok/s | Mean seconds per item | Peak memory (GB) |
|---|---|---|---|
| Base 3B, 0-shot | 80.3 | 3.0 | 2.50 |
| Base 3B, 2-shot | 76.1 | 3.8 | 2.74 |
| Base 3B + EduAI LoRA v1 | 48.0 | 3.6 | 3.05 |
| Base 3B + EduAI LoRA v2 | 46.4 | 3.9 | 3.05 |
| Base 3B + EduAI LoRA v2, 2-shot | 44.9 | 5.0 | 3.24 |
